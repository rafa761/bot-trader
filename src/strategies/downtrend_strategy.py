# strategies\downtrend_strategy.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.schemas import TradingSignal
from strategies.base.model import BaseStrategy, StrategyConfig


class DowntrendStrategy(BaseStrategy):
    """
    Estratégia para mercados em tendência de baixa.
    Foca em capturar continuações da tendência de baixa com entradas em rallies.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """Inicializa a estratégia com configuração otimizada para mercados em baixa."""
        config = StrategyConfig(
            name="Downtrend Strategy",
            description="Estratégia otimizada para mercados em tendência de baixa",
            min_rr_ratio=1.8,  # Exigir R:R maior em tendência de baixa
            entry_threshold=0.55,  # Menos rigoroso na entrada por estar a favor da tendência
            tp_adjustment=1.2,  # Aumentar TP para capturar mais do movimento
            sl_adjustment=0.8,  # Stops mais apertados por estar a favor da tendência
            entry_aggressiveness=1.2,  # Mais agressivo nas entradas
            max_sl_percent=1.8,
            min_tp_percent=0.7,
            required_indicators=["ema_short", "ema_long", "adx", "rsi"]
        )
        super().__init__(config)
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.sequence_length = 24
        self.preprocessor = None
        self.preprocessor_fitted = False

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar EMAs no timeframe atual
        ema_short = df['ema_short'].iloc[-1]
        ema_long = df['ema_long'].iloc[-1]
        ema_downtrend = ema_short < ema_long

        # Verificar tendência multi-timeframe
        mtf_downtrend = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_downtrend = 'DOWNTREND' in mtf_data['consolidated_trend']

        # Verificar força da tendência (ADX)
        adx_strong = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            adx_strong = adx_value > 25

        # Ativar se tivermos confirmação em pelo menos 2 dos 3 indicadores
        confirmations = sum([ema_downtrend, mtf_downtrend, adx_strong])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de DOWNTREND ativada: EMA={ema_downtrend}, MTF={mtf_downtrend}, "
                f"ADX={adx_strong} (valor={df['adx'].iloc[-1]:.1f})"
            )

        return should_activate

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em baixa.
        Busca oportunidades de venda em rallies (correções de alta) da tendência de baixa.
        """
        # Na estratégia de tendência de baixa, queremos entrar principalmente em SHORTs
        # E queremos encontrar pontos de entrada em correções para cima (rallies)

        # Verificar se estamos em um rally usando RSI
        in_rally = False
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Rally = RSI subindo acima de 60 em tendência de baixa
            if rsi > 60 and rsi > prev_rsi:
                in_rally = True
                logger.info(f"Rally detectado: RSI={rsi:.1f} (anterior: {prev_rsi:.1f})")
            else:
                logger.info(f"Sem rally: RSI={rsi:.1f} (anterior: {prev_rsi:.1f}), "
                            f"condições: {rsi > 60} e {rsi > prev_rsi}")

        # Verificar resistência na média móvel
        near_resistance = False
        if 'ema_long' in df.columns:
            ema_long = df['ema_long'].iloc[-1]
            price_to_ema = abs(current_price - ema_long) / current_price * 100
            near_resistance = price_to_ema < 0.5  # Preço próximo da EMA longa

            if near_resistance:
                logger.info(f"Preço próximo da resistência em EMA longa: {price_to_ema:.2f}%")
            else:
                logger.info(f"Preço não está próximo da resistência: {price_to_ema:.2f}%")

        # Verificar Stochastic para confirmar overbought
        stoch_overbought = False
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            stoch_overbought = stoch_k > 80 and stoch_d > 80

            if stoch_overbought:
                logger.info(f"Stochastic em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f}")
            else:
                logger.info(f"Stochastic não está em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f}")

        # Verificar rejeição em resistência
        rejection_at_resistance = False
        if len(df) > 2 and 'high' in df.columns and 'close' in df.columns:
            current_high = df['high'].iloc[-1]
            current_close = df['close'].iloc[-1]

            # Range das últimas 20 barras
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            # Verificar se o preço tocou na resistência e foi rejeitado
            if current_high > current_close * 1.002 and (current_high - current_close) > (recent_range * 0.1):
                rejection_at_resistance = True
                logger.info(
                    f"Rejeição detectada em resistência: "
                    f"High={current_high}, Close={current_close}, Diferença={current_high - current_close:.2f}"
                )
            else:
                logger.info(f"Sem rejeição em resistência: "
                            f"High={current_high}, "
                            f"Close={current_close}, "
                            f"Diferença={current_high - current_close:.2f}, "
                            f"Threshold={recent_range * 0.1:.2f}"
                            )

        # Verificar tendência forte
        strong_trend = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            strong_trend = adx_value > 25
            if strong_trend:
                logger.info(f"Tendência forte detectada: ADX={adx_value:.1f}")

        # Verificar alinhamento multi-timeframe
        mtf_aligned = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']
            mtf_aligned = 'DOWNTREND' in mtf_trend
            if mtf_aligned:
                logger.info(f"Alinhamento multi-timeframe favorável: {mtf_trend}")

        # Decidir se geramos sinal
        conditions_met = sum(
            [in_rally, near_resistance, stoch_overbought, rejection_at_resistance, strong_trend, mtf_aligned])

        min_conditions = 2

        logger.info(
            f"Condições para SHORT: {conditions_met}/{min_conditions} atendidas - "
            f"Rally={in_rally}, EMA_Resistência={near_resistance}, "
            f"Stoch_Overbought={stoch_overbought}, Rejeição={rejection_at_resistance}, "
            f"Strong_Trend={strong_trend}, MTF_Aligned={mtf_aligned}"
        )

        if conditions_met >= min_conditions:
            logger.info(
                f"Condições SHORT em tendência de baixa: Rally={in_rally}, "
                f"Resistência={near_resistance}, Stoch_Overbought={stoch_overbought}, "
                f"Rejeição={rejection_at_resistance}, Strong_Trend={strong_trend}, MTF_Aligned={mtf_aligned}"
            )

            # Gerar previsões usando os modelos LSTM
            try:
                X_seq = self._prepare_sequence(df)
                if X_seq is None:
                    return None

                # Previsões com LSTM
                predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
                predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

                # Para SHORT, garantir que o TP seja negativo
                if predicted_tp_pct > 0:
                    predicted_tp_pct = -predicted_tp_pct

                # Garantir valores positivos para SL
                predicted_sl_pct = abs(predicted_sl_pct)

                logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

                # Validar previsões - evitar valores absurdos ou muito pequenos
                if abs(predicted_tp_pct) > 20:
                    predicted_tp_pct = -20.0  # Negativo para SHORT

                if predicted_sl_pct > 10:
                    predicted_sl_pct = 10.0

                # Ajustar SL dinamicamente se for muito pequeno
                if predicted_sl_pct < 0.5:
                    # Calcular o SL dinâmico baseado em ATR
                    atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
                    if atr_value:
                        predicted_sl_pct = (atr_value / current_price) * 100 * 1.5

                # Avaliar a qualidade da entrada
                should_enter, entry_score = self.evaluate_entry_quality(
                    df, current_price, "SHORT", predicted_tp_pct, predicted_sl_pct
                )

                if not should_enter:
                    logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                    return None

                # Vamos forçar operações SHORT em tendência de baixa
                side: Literal["SELL", "BUY"] = "SELL"
                position_side: Literal["LONG", "SHORT"] = "SHORT"
                tp_factor = 1 - abs(predicted_tp_pct) / 100
                sl_factor = 1 + abs(predicted_sl_pct) / 100

                # Calcular preços TP/SL
                tp_price = current_price * tp_factor
                sl_price = current_price * sl_factor

                # Gerar ID único para o sinal
                signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

                # Obter ATR para ajustes de quantidade
                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

                # Determinar tendência e força
                market_trend = "DOWNTREND"  # Já sabemos que estamos em downtrend
                market_strength = "STRONG_TREND" if df['adx'].iloc[-1] > 25 else "WEAK_TREND"

                signal = TradingSignal(
                    id=signal_id,
                    direction="SHORT",
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
                    timestamp=datetime.now()
                )

                return signal

            except Exception as e:
                logger.error(f"Erro na geração de sinal: {e}", exc_info=True)
                return None

        # Retornar None se não houver condições para gerar sinal
        return None

    def evaluate_entry_quality(
            self,
            df: pd.DataFrame,
            current_price: float,
            trade_direction: str,
            predicted_tp_pct: float = None,
            predicted_sl_pct: float = None,
            entry_threshold: float = None,
            mtf_alignment: float = None
    ) -> tuple[bool, float]:
        """
        Avalia a qualidade da entrada para DowntrendStrategy.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual do ativo
            trade_direction: "LONG" ou "SHORT"
            predicted_tp_pct: Take profit percentual previsto
            predicted_sl_pct: Stop loss percentual previsto
            entry_threshold: Limiar opcional para qualidade de entrada
            mtf_alignment: Score de alinhamento multi-timeframe (0-1)

        Returns:
            tuple[bool, float]: (Deve entrar, pontuação da entrada)
        """
        # Calcular razão risco-recompensa se tp e sl forem fornecidos
        if predicted_tp_pct is not None and predicted_sl_pct is not None and predicted_sl_pct > 0:
            rr_ratio = abs(predicted_tp_pct) / abs(predicted_sl_pct)

            # Verificar se RR é bom o suficiente
            should_enter = rr_ratio >= self.config.min_rr_ratio

            # Pontuação básica baseada em RR
            entry_score = min(1.0, rr_ratio / 3.0)  # Pontuação de 0 a 1
        else:
            # Valores padrão se tp e sl não forem fornecidos
            should_enter = False
            entry_score = 0.0

        # Verificações adicionais baseadas em indicadores
        if trade_direction == "SHORT" and self.calculate_trend_direction(df) == "DOWNTREND":
            # Bônus para trades SHORT em tendência de baixa
            entry_score = min(1.0, entry_score * 1.2)
        elif trade_direction == "LONG" and self.calculate_trend_direction(df) == "UPTREND":
            # Bônus para trades LONG em tendência de alta
            entry_score = min(1.0, entry_score * 1.2)

        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if trade_direction == "SHORT" and rsi > 60:
                # Bônus para trades SHORT quando RSI está alto (sobrecompra)
                entry_score = min(1.0, entry_score * 1.1)
            elif trade_direction == "LONG" and rsi < 40:
                # Bônus para trades LONG quando RSI está baixo (sobrevenda)
                entry_score = min(1.0, entry_score * 1.1)

        # Bônus para alinhamento multi-timeframe forte
        if mtf_alignment is not None and mtf_alignment > 0.5:
            entry_score = min(1.0, entry_score * (1.0 + (mtf_alignment - 0.5)))

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM.

        Args:
            df: DataFrame com dados históricos

        Returns:
            np.ndarray: Sequência formatada para o modelo LSTM ou None se houver erro
        """
        try:
            from core.constants import FEATURE_COLUMNS
            from repositories.data_preprocessor import DataPreprocessor

            # Verificar se temos dados suficientes
            if len(df) < self.sequence_length:
                return None

            # Inicializar preprocessador se necessário
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor(
                    feature_columns=FEATURE_COLUMNS,
                    outlier_method='iqr',
                    scaling_method='robust'
                )
                self.preprocessor.fit(df)

            # Preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df, sequence_length=self.sequence_length
            )

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência: {e}", exc_info=True)
            return None
    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em baixa.
        Em tendência de baixa, favorecemos sinais SHORT e ajustamos TP/SL.
        """
        if signal.direction != "SHORT":
            # Aumentar o threshold para trades contra a tendência (LONG em downtrend)
            logger.info(f"Sinal LONG em tendência de baixa: exigindo maior qualidade")
            # Reduzir a pontuação para dificultar a entrada
            if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                signal.entry_score = signal.entry_score * 0.8

        else:  # Para sinais SHORT em tendência de baixa
            # Ajustar TP para ser mais ambicioso
            signal.predicted_tp_pct = abs(signal.predicted_tp_pct) * self.config.tp_adjustment
            if signal.predicted_tp_pct > 0:  # Garantir que é negativo para SHORT
                signal.predicted_tp_pct = -signal.predicted_tp_pct

            # Ajustar SL para ser mais apertado
            signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment

            # Recalcular preços e fatores
            if signal.direction == "LONG":
                signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
            else:  # SHORT
                signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
                signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

            signal.tp_price = signal.current_price * signal.tp_factor
            signal.sl_price = signal.current_price * signal.sl_factor

            # Atualizar a razão R:R
            signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

            logger.info(
                f"Sinal SHORT ajustado para mercado em baixa: "
                f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
                f"R:R={signal.rr_ratio:.2f}"
            )

        return signal
