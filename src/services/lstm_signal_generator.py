# services/lstm_signal_generator.py

import datetime
from typing import Any

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_preprocessor import DataPreprocessor
from services.base.schemas import TradingSignal
from services.base.services import SignalGenerator
from services.trading_strategy import TradingStrategy
from services.trend_analyzer import TrendAnalyzer


class LSTMSignalGenerator(SignalGenerator):
    """
    Gerador de sinais baseado em modelos LSTM.

    Implementa a interface SignalGenerator e utiliza modelos LSTM
    para prever os alvos de Take Profit e Stop Loss, gerando sinais
    de trading com base nessas previsões.
    """

    def __init__(
            self,
            tp_model: LSTMModel,
            sl_model: LSTMModel,
            strategy: TradingStrategy,
            sequence_length: int = 24
    ):
        """
        Inicializa o gerador de sinais LSTM.

        Args:
            tp_model: Modelo LSTM para previsão de take profit
            sl_model: Modelo LSTM para previsão de stop loss
            strategy: Estratégia de trading para tomada de decisões
            sequence_length: Comprimento da sequência para o modelo LSTM
        """
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.strategy = strategy
        self.sequence_length = sequence_length

        # Preprocessador para preparar dados para o modelo
        self.preprocessor: DataPreprocessor | None = None
        self.preprocessor_fitted = False

        # Registro de previsões para análise de desempenho
        self.prediction_history: list[tuple[float, float, float, str, datetime.datetime]] = []

        # Timestamp da última atualização de modelo
        self.last_model_update = datetime.datetime.now()

        logger.info(
            f"Signal Generator inicializado com modelos TP v{tp_model.config.version} e SL v{sl_model.config.version}")

    def update_models(self, tp_model: LSTMModel, sl_model: LSTMModel) -> bool:
        """
        Atualiza as referências dos modelos.

        Args:
            tp_model: Novo modelo LSTM para previsão de take profit
            sl_model: Novo modelo LSTM para previsão de stop loss

        Returns:
            bool: True se os modelos foram atualizados
        """
        try:
            # Verificar se realmente são modelos diferentes
            if (tp_model.config.version != self.tp_model.config.version or
                    sl_model.config.version != self.sl_model.config.version):
                # Atualizar modelos
                self.tp_model = tp_model
                self.sl_model = sl_model

                # Resetar preprocessador para garantir compatibilidade
                self.preprocessor = None
                self.preprocessor_fitted = False

                self.last_model_update = datetime.datetime.now()

                logger.info(
                    f"Signal Generator atualizado com novos modelos: "
                    f"TP v{tp_model.config.version}, SL v{sl_model.config.version}"
                )

                return True
            return False
        except Exception as e:
            logger.error(f"Erro ao atualizar modelos do Signal Generator: {e}", exc_info=True)
            return False

    def record_actual_values(self, signal_id: str, actual_tp_pct: float, actual_sl_pct: float,
                             retrainer: Any = None) -> None:
        """
        Registra os valores reais de TP/SL para uma determinada previsão.
        Esta informação é usada para avaliar a precisão do modelo ao longo do tempo.

        Args:
            signal_id: Identificador do sinal
            actual_tp_pct: Valor real do take profit percentual
            actual_sl_pct: Valor real do stop loss percentual
            retrainer: Instância opcional do ModelRetrainer para registrar erros
        """
        try:
            # Encontrar a previsão correspondente no histórico
            for i, (pred_tp, pred_sl, _, signal_id_history, timestamp) in enumerate(self.prediction_history):
                # Verificar se é o sinal correto
                if signal_id_history == signal_id:
                    # Calcular erros
                    tp_error = 0
                    sl_error = 0

                    # Registrar apenas erros relevantes (evitar divisão por zero)
                    if actual_tp_pct > 0 and pred_tp != 0:
                        tp_error = abs(pred_tp - actual_tp_pct)
                        logger.info(
                            f"TP: previsto={pred_tp:.2f}% vs real={actual_tp_pct:.2f}% "
                            f"(erro={tp_error:.2f}%)"
                        )
                        # Registrar no retrainer, especificando que é erro de TP
                        if retrainer:
                            retrainer.record_prediction_error(pred_tp, actual_tp_pct, "tp")

                    if actual_sl_pct > 0 and pred_sl != 0:
                        sl_error = abs(pred_sl - actual_sl_pct)
                        logger.info(
                            f"SL: previsto={pred_sl:.2f}% vs real={actual_sl_pct:.2f}% "
                            f"(erro={sl_error:.2f}%)"
                        )
                        # Registrar no retrainer, especificando que é erro de SL
                        if retrainer:
                            retrainer.record_prediction_error(pred_sl, actual_sl_pct, "sl")

                    # Remover essa entrada do histórico após processamento
                    self.prediction_history.pop(i)
                    logger.info(f"Processado resultado real do sinal {signal_id}")
                    break

        except Exception as e:
            logger.error(f"Erro ao registrar valores reais: {e}", exc_info=True)

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM, ajustando o tamanho
        conforme necessário para compatibilidade com o modelo.

        Args:
            df: DataFrame com dados históricos incluindo indicadores técnicos

        Returns:
            np.ndarray: Sequência formatada para o modelo LSTM ou None se houver erro
        """
        try:
            # Obter o comprimento da sequência esperado pelo modelo
            expected_sequence_length = self.tp_model.model.input_shape[1]

            # Ajustar o sequence_length se for diferente do esperado
            if self.sequence_length != expected_sequence_length:
                logger.warning(
                    f"Ajustando sequence_length: configurado={self.sequence_length}, "
                    f"esperado pelo modelo={expected_sequence_length}"
                )
                self.sequence_length = expected_sequence_length

            # Inicializar preprocessador (isso deve ser feito uma vez e salvo como atributo)
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor(
                    feature_columns=FEATURE_COLUMNS,
                    outlier_method='iqr',
                    scaling_method='robust'
                )
                # Ajustar o preprocessador nos dados históricos
                self.preprocessor.fit(df)

            # Usar o preprocessador para preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df,
                sequence_length=self.sequence_length
            )

            # Verificar se o formato é compatível com o esperado pelo modelo
            if x_pred is None:
                return None

            expected_shape = (None, self.tp_model.model.input_shape[1], len(FEATURE_COLUMNS))
            actual_shape = (x_pred.shape[0], x_pred.shape[1], x_pred.shape[2])

            logger.info(f"Shape da sequência preparada: {actual_shape}, esperado pelo modelo: {expected_shape}")

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência para LSTM: {e}", exc_info=True)
            return None

    async def generate_signal(self, df: pd.DataFrame, current_price: float,
                              current_strategy=None) -> TradingSignal | None:
        """
        Gera um sinal de trading baseado nos modelos LSTM.

        Args:
            df: DataFrame com dados históricos e indicadores técnicos
            current_price: Preço atual do ativo
            current_strategy: Estratégia atual selecionada pelo StrategyManager (opcional)

        Returns:
            TradingSignal: Sinal de trading gerado ou None se não houver sinal
        """
        try:
            X_seq = self._prepare_sequence(df)
            if X_seq is None:
                return None

            # Logs detalhados para diagnóstico de condições de mercado
            try:
                # Obter valores relevantes dos indicadores técnicos do último candle
                rsi_value = df['rsi'].iloc[-1] if 'rsi' in df.columns else None
                macd_value = df['macd'].iloc[-1] if 'macd' in df.columns else None
                macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else None
                atr_pct = (df['atr'].iloc[-1] / current_price * 100) if 'atr' in df.columns else None

                # Determinar tendência atual
                trend = "NEUTRO"
                if 'ema_short' in df.columns and 'ema_long' in df.columns:
                    ema_short = df['ema_short'].iloc[-1]
                    ema_long = df['ema_long'].iloc[-1]
                    if ema_short > ema_long:
                        trend = "ALTA"
                    elif ema_short < ema_long:
                        trend = "BAIXA"

                # Log das condições de mercado com formatação corrigida
                rsi_log = f"{rsi_value:.2f}" if rsi_value is not None else "N/A"
                macd_log = f"{macd_value:.6f}" if macd_value is not None else "N/A"
                atr_pct_log = f"{atr_pct:.2f}%" if atr_pct is not None else "N/A"

                logger.info(
                    f"Condições de mercado: Tendência={trend}, "
                    f"RSI={rsi_log}, "
                    f"MACD={macd_log}, "
                    f"ATR%={atr_pct_log}"
                )
            except Exception as e:
                logger.error(f"Erro ao registrar condições de mercado: {e}")

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

            # Gerar ID único para o sinal
            signal_id = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Guardar previsão no histórico com timestamp atual
            self.prediction_history.append((
                predicted_tp_pct,
                predicted_sl_pct,
                current_price,
                signal_id,
                datetime.datetime.now()
            ))

            # Limitar tamanho do histórico
            if len(self.prediction_history) > 100:
                self.prediction_history.pop(0)

            # Garantir valores positivos para SL
            predicted_sl_pct = abs(predicted_sl_pct)

            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Validar previsões - evitar valores absurdos ou muito pequenos
            if abs(predicted_tp_pct) > 20:
                logger.warning(f"TP previsto muito alto: {predicted_tp_pct:.2f}%. Limitando a 20%")
                predicted_tp_pct = 20.0 if predicted_tp_pct > 0 else -20.0

            if predicted_sl_pct > 10:
                logger.warning(f"SL previsto muito alto: {predicted_sl_pct:.2f}%. Limitando a 10%")
                predicted_sl_pct = 10.0

            # Ajustar SL dinamicamente se for muito pequeno (< 0.5%)
            if predicted_sl_pct < 0.5:  # Verificar se o SL é muito pequeno
                # Captura o R:R original antes de qualquer ajuste
                original_tp_sign = 1 if predicted_tp_pct > 0 else -1  # Preserva o sinal do TP
                original_rr = abs(predicted_tp_pct / predicted_sl_pct) if predicted_sl_pct > 0 else 0

                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
                if atr_value and original_rr > 0:
                    # Calcular o SL dinâmico baseado em ATR
                    dynamic_sl = self.strategy.risk_reward_manager.calculate_dynamic_sl(current_price, atr_value)

                    # Ajustar o TP proporcionalmente para manter o mesmo R:R
                    adjusted_tp = dynamic_sl * original_rr * original_tp_sign  # Mantém o sinal original

                    # Logs para depuração
                    logger.info(f"Ajustando SL: {predicted_sl_pct:.2f}% -> {dynamic_sl:.2f}% (baseado em ATR)")
                    logger.info(f"Ajustando TP proporcionalmente: {predicted_tp_pct:.2f}% -> {adjusted_tp:.2f}%")

                    # Atualizar os valores
                    predicted_sl_pct = dynamic_sl
                    predicted_tp_pct = adjusted_tp
                else:
                    # Mínimo de 0.5% se não tiver ATR
                    predicted_sl_pct = 0.5

            # Calcular e exibir a razão R:R atual
            rr_ratio = abs(predicted_tp_pct / predicted_sl_pct) if predicted_sl_pct > 0 else 0
            logger.info(f"Razão R:R calculada: {rr_ratio:.2f}")

            # Decidir direção baseada nas previsões LSTM e estratégia atual (se disponível)
            if current_strategy:
                # Obter o threshold da estratégia atual, com um ajuste para direcionalidade
                strategy_config = current_strategy.get_config()
                threshold = strategy_config.entry_threshold * 0.4  # Valor reduzido apenas para direcionalidade

                # Ajustar TP/SL antes de decidir direção, conforme fatores da estratégia
                adjusted_tp = predicted_tp_pct * strategy_config.tp_adjustment
                adjusted_sl = predicted_sl_pct * strategy_config.sl_adjustment

                direction = self.strategy.decide_direction(adjusted_tp, adjusted_sl, threshold=threshold)
                logger.info(f"Direção decidida com estratégia {strategy_config.name}: {direction or 'Neutro'}")
            else:
                # Usar método original para fallback
                direction = self.strategy.decide_direction(predicted_tp_pct, predicted_sl_pct, threshold=0.2)
                logger.info(f"Direção decidida com método padrão: {direction or 'Neutro'}")

            if direction is None:
                logger.info("Sinal neutro ou R:R insuficiente, não abrir trade.")
                return None

            # Ajustar TP para garantir razão R:R mínima, se necessário
            if abs(predicted_tp_pct) < abs(predicted_sl_pct * self.strategy.risk_reward_manager.min_rr_ratio):
                adjusted_tp_pct = predicted_sl_pct * self.strategy.risk_reward_manager.min_rr_ratio
                if direction == "SHORT":
                    adjusted_tp_pct = -adjusted_tp_pct
                logger.info(f"Ajustando TP para garantir R:R mínimo: {predicted_tp_pct:.2f}% -> {adjusted_tp_pct:.2f}%")
                predicted_tp_pct = adjusted_tp_pct

            # Mapear direção para parâmetros de ordem
            if direction == "LONG":
                side = "BUY"
                position_side = "LONG"
                tp_factor = 1 + max(abs(predicted_tp_pct) / 100, 0.02)
                sl_factor = 1 - max(abs(predicted_sl_pct) / 100, 0.005)
            else:  # SHORT
                side = "SELL"
                position_side = "SHORT"
                tp_factor = 1 - max(abs(predicted_tp_pct) / 100, 0.02)
                sl_factor = 1 + max(abs(predicted_sl_pct) / 100, 0.005)

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # Verificar se TP e SL estão muito próximos do preço atual (evitar trades sem sentido)
            min_price_move = current_price * 0.002  # Mínimo de 0.2% de movimento

            if abs(tp_price - current_price) < min_price_move:
                logger.warning(f"TP muito próximo do preço atual. TP: {tp_price}, Atual: {current_price}")
                return None

            if abs(sl_price - current_price) < min_price_move:
                logger.warning(f"SL muito próximo do preço atual. SL: {sl_price}, Atual: {current_price}")
                return None

            # Avaliar a qualidade da entrada
            should_enter, entry_score = self.strategy.evaluate_entry_quality(
                df, current_price, direction, predicted_tp_pct, predicted_sl_pct
            )

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Obter ATR para ajustes de quantidade
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Determinar tendência e força
            market_trend = TrendAnalyzer.ema_trend(df)
            market_strength = TrendAnalyzer.adx_trend(df)

            signal = TradingSignal(
                id=signal_id,
                direction=direction,
                side=side,
                position_side=position_side,
                predicted_tp_pct=predicted_tp_pct,
                predicted_sl_pct=predicted_sl_pct,
                tp_price=tp_price,
                sl_price=sl_price,
                current_price=current_price,
                tp_factor=tp_factor,
                sl_factor=sl_factor,
                atr_value=atr_value,
                entry_score=entry_score,
                rr_ratio=abs(predicted_tp_pct / predicted_sl_pct),
                market_trend=market_trend,
                market_strength=market_strength,
                timestamp=datetime.datetime.now()
            )

            return signal
        except Exception as e:
            logger.error(f"Erro na geração de sinal LSTM: {e}", exc_info=True)
            return None
