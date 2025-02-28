# services/trading_bot.py
import asyncio
import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.config import settings
from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_handler import DataHandler
from repositories.data_preprocessor import DataPreprocessor
from services.base.schemas import (
    TradingSignal,
    OrderResult,
    TPSLResult
)
from services.base.services import (
    MarketDataProvider,
    SignalGenerator,
    OrderExecutor
)
from services.binance_client import BinanceClient
from services.model_retrainer import ModelRetrainer
from services.trading_strategy import TradingStrategy
from services.trend_analyzer import TrendAnalyzer


class BinanceDataProvider(MarketDataProvider):
    """Provedor de dados de mercado da Binance."""

    def __init__(self, binance_client: BinanceClient, data_handler: DataHandler):
        self.client = binance_client
        self.data_handler = data_handler
        self._initialized = False
        self.min_candles_required = 100

    async def initialize(self) -> None:
        """Inicializa a conexão com a Binance e carrega os dados iniciais."""
        if not self._initialized:
            await self.client.initialize()
            self._initialized = True
            logger.info("Provedor de dados Binance inicializado")

    async def get_latest_data(self) -> pd.DataFrame:
        """Obtém os dados mais recentes da Binance."""
        # Se não há dados históricos, carrega um conjunto inicial maior
        if self.data_handler.historical_df.empty:
            logger.info(f"Carregando conjunto inicial de dados - {settings.INTERVAL}")
            large_df = await self.data_handler.get_latest_data(
                settings.SYMBOL, settings.INTERVAL, limit=1000
            )

            if large_df.empty:
                logger.error("Não foi possível obter dados históricos iniciais")
                return pd.DataFrame()

            # Verificar se há dados suficientes
            if len(large_df) < self.min_candles_required:
                logger.warning(
                    f"Dados históricos insuficientes: obtidos {len(large_df)} candles, "
                    f"necessários pelo menos {self.min_candles_required}"
                )
                return pd.DataFrame()

            try:
                with self.data_handler.data_lock:
                    self.data_handler.historical_df = (
                        self.data_handler.technical_indicator_adder.add_technical_indicators(large_df)
                    )

                # Verificar integridade dos indicadores
                missing_indicators = [
                    col for col in FEATURE_COLUMNS if col not in self.data_handler.historical_df.columns
                ]
                if missing_indicators:
                    logger.error(f"Indicadores ausentes após cálculo: {missing_indicators}")
                    return pd.DataFrame()

                # Remover linhas com valores NaN
                if self.data_handler.historical_df[FEATURE_COLUMNS].isna().any().any():
                    logger.warning("Existem valores NaN nos indicadores técnicos")
                    self.data_handler.historical_df.dropna(subset=FEATURE_COLUMNS, inplace=True)

                logger.info(
                    f"Dados históricos iniciais carregados: {len(self.data_handler.historical_df)} candles"
                )
                return self.data_handler.historical_df

            except Exception as e:
                logger.error(f"Erro ao processar dados históricos iniciais: {e}", exc_info=True)
                return pd.DataFrame()

        # Caso já tenha dados, apenas atualiza com os novos candles
        else:
            new_data = await self.data_handler.get_latest_data(
                settings.SYMBOL, settings.INTERVAL, limit=2
            )

            if not new_data.empty:
                for i in range(len(new_data)):
                    row = new_data.iloc[i]
                    new_row = {
                        "timestamp": row["timestamp"],
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"],
                        "volume": row["volume"]
                    }

                    # Verifica se o timestamp já existe
                    if not self.data_handler.historical_df.empty:
                        existing_timestamps = self.data_handler.historical_df["timestamp"].astype(str).tolist()
                        if str(row["timestamp"]) in existing_timestamps:
                            continue

                    # Atualiza o DataFrame histórico
                    self.data_handler.update_historical_data(new_row)

            return self.data_handler.historical_df

    def get_historical_data(self) -> pd.DataFrame:
        """Retorna os dados históricos armazenados."""
        return self.data_handler.historical_df


class LSTMSignalGenerator(SignalGenerator):
    """Gerador de sinais baseado em modelos LSTM."""

    def __init__(
            self,
            tp_model: LSTMModel,
            sl_model: LSTMModel,
            strategy: TradingStrategy,
            sequence_length: int = 24
    ):
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.strategy = strategy
        self.sequence_length = sequence_length

        # Adicionar preprocessador
        self.preprocessor: DataPreprocessor | None = None

        # Flag para rastrear se o preprocessador já foi ajustado
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

    def record_actual_values(self, signal_id: str, actual_tp_pct: float, actual_sl_pct: float, retrainer=None):
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

    async def generate_signal(self, df: pd.DataFrame, current_price: float) -> TradingSignal | None:
        """Gera um sinal de trading baseado nos modelos LSTM."""
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

            # Decidir direção baseada nas previsões LSTM (agora considera R:R)
            direction = self.strategy.decide_direction(predicted_tp_pct, predicted_sl_pct, threshold=0.2)
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
                side: Literal["BUY", "SELL"] = "BUY"
                position_side: Literal["LONG", "SHORT"] = "LONG"
                tp_factor = 1 + max(abs(predicted_tp_pct) / 100, 0.02)
                sl_factor = 1 - max(abs(predicted_sl_pct) / 100, 0.005)
            else:  # SHORT
                side: Literal["BUY", "SELL"] = "SELL"
                position_side: Literal["LONG", "SHORT"] = "SHORT"
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
                timestamp=datetime.datetime.now()
            )

            return signal
        except Exception as e:
            logger.error(f"Erro na geração de sinal LSTM: {e}", exc_info=True)
            return None


class BinanceOrderExecutor(OrderExecutor):
    """Executor de ordens na Binance."""

    def __init__(
            self,
            binance_client: BinanceClient,
            strategy: TradingStrategy,
            tick_size: float = 0.0,
            step_size: float = 0.0
    ):
        self.client = binance_client
        self.strategy = strategy
        self.tick_size = tick_size
        self.step_size = step_size

        # Histórico de ordens executadas
        self.executed_orders = []
        self.max_order_history = 100

    async def initialize_filters(self) -> None:
        """Inicializa os filtros de trading."""
        self.tick_size, self.step_size = await self.client.get_symbol_filters(settings.SYMBOL)

        if not self.tick_size or not self.step_size:
            logger.error("Não foi possível obter tickSize/stepSize.")
            return

        logger.info(f"{settings.SYMBOL} -> tickSize={self.tick_size}, stepSize={self.step_size}")

        # Define alavancagem
        await self.client.set_leverage(settings.SYMBOL, settings.LEVERAGE)

    async def check_positions(self) -> bool:
        """Verifica se existem posições abertas."""
        open_long = await self.client.get_open_position_by_side(settings.SYMBOL, "LONG")
        open_short = await self.client.get_open_position_by_side(settings.SYMBOL, "SHORT")

        has_position = open_long is not None or open_short is not None

        if has_position:
            logger.info("Já existe posição aberta. Aguardando fechamento para abrir novo trade.")

        return has_position

    async def place_tp_sl(
            self,
            direction: str,
            current_price: float,
            tp_price: float,
            sl_price: float
    ) -> TPSLResult:
        """Cria ordens de TP e SL."""
        position_side = direction
        if direction == "LONG":
            # Ajuste de TP
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj <= current_price:
                tp_price_adj = current_price + (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj >= current_price:
                sl_price_adj = current_price - (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "SELL"
        else:  # SHORT
            position_side = "SHORT"
            # Ajuste de TP
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj >= current_price:
                tp_price_adj = current_price - (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj <= current_price:
                sl_price_adj = current_price + (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "BUY"

        # Cria ordens TP e SL de forma assíncrona
        tp_order, sl_order = await self.client.place_tp_sl_orders(
            symbol=settings.SYMBOL,
            side=side_for_tp_sl,
            position_side=position_side,
            tp_price=tp_str,
            sl_price=sl_str
        )

        return TPSLResult(tp_order=tp_order, sl_order=sl_order)

    async def execute_order(self, signal: TradingSignal) -> OrderResult:
        """Executa uma ordem baseada no sinal fornecido."""
        try:
            # Obter o valor atual de ATR, se disponível
            current_atr = signal.atr_value
            min_notional = 100.0  # Mínimo exigido pela Binance

            # Calcula quantidade
            qty = self.strategy.calculate_trade_quantity(
                capital=settings.CAPITAL,
                current_price=signal.current_price,
                leverage=settings.LEVERAGE,
                risk_per_trade=settings.RISK_PER_TRADE,
                atr_value=current_atr,
                min_notional=min_notional,
            )

            # Ajusta quantidade
            qty_adj = self.strategy.adjust_quantity_to_step_size(qty, self.step_size)
            if qty_adj <= 0:
                logger.warning("Qty ajustada <= 0. Trade abortado.")
                return OrderResult(success=False, error_message="Quantidade ajustada <= 0")

            # Verificar valor notional mínimo da Binance
            notional_value = qty_adj * signal.current_price

            if notional_value < min_notional:
                # Recalcular quantidade para atender ao mínimo da Binance com margem de segurança
                import math
                min_qty = (min_notional * 1.1) / signal.current_price  # 10% acima para segurança

                # Garantir que seja múltiplo exato do step_size
                steps = math.ceil(min_qty / self.step_size)
                min_qty_adjusted = steps * self.step_size

                new_notional = min_qty_adjusted * signal.current_price

                logger.warning(
                    f"Valor notional calculado ({notional_value:.2f} USDT) abaixo do mínimo da Binance ({min_notional} USDT). "
                    f"Ajustando quantidade de {qty_adj:.3f} para {min_qty_adjusted:.3f} {settings.SYMBOL} "
                    f"(valor estimado: {new_notional:.2f} USDT)"
                )

                qty_adj = min_qty_adjusted

            logger.info(
                f"Abrindo {signal.direction} c/ qty={qty_adj:.3f}, lastPrice={signal.current_price:.2f}, "
                f"valor={qty_adj * signal.current_price:.2f} USDT (mínimo={min_notional} USDT)..."
            )

            # Coloca a ordem de abertura
            order_resp = await self.client.place_order_with_retry(
                symbol=settings.SYMBOL,
                side=signal.side,
                quantity=qty_adj,
                position_side=signal.position_side,
                step_size=self.step_size
            )

            if order_resp:
                logger.info(f"Ordem de abertura executada: {order_resp}")

                # Obter timestamp atual para comparações futuras
                current_time = datetime.datetime.now()

                # Adicionar ao histórico de ordens
                self.executed_orders.append({
                    "signal_id": signal.id,
                    "order_id": order_resp.get("orderId", "N/A"),
                    "direction": signal.direction,
                    "entry_price": signal.current_price,
                    "tp_price": signal.tp_price,
                    "sl_price": signal.sl_price,
                    "predicted_tp_pct": signal.predicted_tp_pct,
                    "predicted_sl_pct": signal.predicted_sl_pct,
                    "timestamp": current_time,
                    "filled": True,
                    "processed": False,
                    "position_side": signal.position_side
                })

                # Limitar tamanho do histórico
                if len(self.executed_orders) > self.max_order_history:
                    self.executed_orders.pop(0)

                # Coloca as ordens de TP e SL
                tp_sl_result = await self.place_tp_sl(
                    signal.direction,
                    signal.current_price,
                    signal.tp_price,
                    signal.sl_price
                )

                # Extrair order_id da resposta
                order_id = order_resp.get("orderId", "N/A")
                return OrderResult(success=True, order_id=str(order_id))
            else:
                logger.info("Não foi possível colocar ordem de abertura.")
                return OrderResult(success=False, error_message="Falha ao colocar ordem")

        except Exception as e:
            error_msg = f"Erro ao executar ordem: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(success=False, error_message=error_msg)


# Classe principal refatorada, removendo a análise de padrão de mercado
class TradingBot:
    """
    Classe principal do bot de trading, que coordena os demais componentes
    seguindo os princípios SOLID.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """
        Inicializa o bot de trading com os modelos LSTM.

        Args:
            tp_model: Modelo LSTM para previsão de Take Profit
            sl_model: Modelo LSTM para previsão de Stop Loss
        """
        # Instancia o cliente Binance
        binance_client = BinanceClient()

        # DataHandler
        self.data_handler = DataHandler(binance_client)

        # Estratégia
        self.strategy = TradingStrategy()

        # Componentes do SOLID
        self.data_provider = BinanceDataProvider(binance_client, self.data_handler)
        self.signal_generator = LSTMSignalGenerator(tp_model, sl_model, self.strategy)
        self.order_executor = BinanceOrderExecutor(binance_client, self.strategy)

        # Sistema de retreinamento - Com referência para o signal_generator
        self.model_retrainer = ModelRetrainer(
            tp_model=tp_model,
            sl_model=sl_model,
            get_data_callback=self.get_historical_data_for_retraining,
            signal_generator_ref=lambda: self.signal_generator,  # Fornece referência atualizada
            retraining_interval_hours=24,  # Retreinar a cada 24 horas
            min_data_points=1000,  # Mínimo de 1000 pontos de dados
            performance_threshold=0.15  # Limiar de erro para retreinamento
        )

        # Controle de estado interno
        self.last_retraining_check = datetime.datetime.now()
        self.retraining_check_interval = 300  # 5 minutos

        # Histórico de trades
        self.trades_history = []
        self.max_trades_history = 200

        # Controle de ciclos
        self.cycle_count = 0

        # Histórico de previsões
        self.predictions_history = []

        # Flag para verificação de retreinamento
        self.check_retraining_status_interval = 60  # Verificar a cada 60 ciclos

        # Valores padrão para parâmetros de trading
        self.default_entry_threshold = settings.ENTRY_THRESHOLD_DEFAULT
        self.default_tp_adjustment = 1.0
        self.default_sl_adjustment = 1.0

        logger.info("TradingBot SOLID inicializado com sucesso.")

    def get_historical_data_for_retraining(self) -> pd.DataFrame:
        """
        Retorna os dados históricos para retreinamento dos modelos.
        Esta função é usada como callback pelo ModelRetrainer.

        Returns:
            pd.DataFrame: DataFrame com dados históricos
        """
        try:
            df = self.data_handler.historical_df
            if df is not None and not df.empty:
                # Criar uma cópia para evitar modificar os dados originais
                df_copy = df.copy()
                logger.info(f"Fornecendo {len(df_copy)} registros para retreinamento")
                return df_copy
            logger.warning("Sem dados históricos disponíveis para retreinamento")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos para retreinamento: {e}", exc_info=True)
            return pd.DataFrame()

    async def check_model_updates(self) -> None:
        """
        Verifica se os modelos foram atualizados pelo retreinador e sincroniza se necessário.
        """
        try:
            # Verificar se já passou tempo suficiente desde a última verificação
            current_time = datetime.datetime.now()
            if (current_time - self.last_retraining_check).total_seconds() < self.retraining_check_interval:
                return

            self.last_retraining_check = current_time

            # Verificar status do retreinamento
            retraining_status = self.model_retrainer.get_retraining_status()

            # Se modelos foram atualizados, sincronizar
            if retraining_status.get("models_updated_flag", False):
                logger.info("Detectada atualização de modelos - Sincronizando signal_generator")

                # Aguardar um breve momento para garantir que os modelos estejam completamente atualizados
                await asyncio.sleep(2)

                # Atualizar modelos do signal_generator
                updated = self.signal_generator.update_models(
                    tp_model=self.model_retrainer.tp_model,
                    sl_model=self.model_retrainer.sl_model
                )

                if updated:
                    logger.info("Signal Generator sincronizado com os modelos retreinados")
                    # Limpar flag após sincronização bem-sucedida
                    self.model_retrainer.models_updated.clear()

                    # Log dos novos modelos
                    logger.info(
                        f"Modelos atualizados - TP: v{self.model_retrainer.tp_model.config.version}, "
                        f"SL: v{self.model_retrainer.sl_model.config.version}"
                    )

        except Exception as e:
            logger.error(f"Erro ao verificar atualizações de modelo: {e}", exc_info=True)

    async def _check_order_status(self, order_id: str) -> bool:
        """
        Verifica na Binance se uma ordem foi fechada.

        Args:
            order_id: ID da ordem a verificar

        Returns:
            bool: True se a ordem foi fechada (posição encerrada), False caso contrário
        """
        if not order_id or order_id == "N/A":
            return False

        try:
            # Verificar posições atuais
            positions = await self.order_executor.client.client.futures_position_information(
                symbol=settings.SYMBOL
            )

            # Verificar o histórico de ordens
            orders = await self.order_executor.client.client.futures_get_all_orders(
                symbol=settings.SYMBOL,
                orderId=order_id
            )

            # Se não encontrar a ordem no histórico, não foi fechada
            if not orders:
                return False

            target_order = next((o for o in orders if str(o.get('orderId')) == str(order_id)), None)

            if not target_order:
                return False

            # Verificar se esta ordem tem uma posição associada ainda aberta
            position_exists = False
            for pos in positions:
                # Se existe alguma posição aberta com quantidade não zero
                if abs(float(pos.get('positionAmt', 0))) > 0:
                    position_exists = True
                    break

            # Se a ordem está no status 'FILLED' mas não há posição aberta, então foi fechada
            if target_order.get('status') == 'FILLED' and not position_exists:
                return True

            # Adicionalmente, verificar se existem ordens TP/SL executadas para este símbolo
            # que possam indicar que a posição foi fechada
            recent_orders = await self.order_executor.client.client.futures_get_all_orders(
                symbol=settings.SYMBOL,
                limit=10  # Verificar apenas ordens recentes
            )

            for order in recent_orders:
                # Verifica se é uma ordem TAKE_PROFIT_MARKET ou STOP_MARKET que foi executada
                if (order.get('type') in ['TAKE_PROFIT_MARKET', 'STOP_MARKET'] and
                        order.get('status') == 'FILLED'):
                    logger.info(f"Encontrada ordem TP/SL executada: {order.get('orderId')}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Erro ao verificar status da ordem {order_id}: {e}", exc_info=True)
            return False

    async def _get_trade_result(self, order: dict) -> dict | None:
        """
        Obtém o resultado real de um trade fechado consultando a Binance.

        Args:
            order: Dicionário com detalhes da ordem

        Returns:
            dict: Resultado real do trade com informações de TP/SL e preços
            None: Se não foi possível obter os resultados
        """
        try:
            order_id = order.get("order_id")
            direction = order.get("direction")
            entry_price = order.get("entry_price")
            tp_price = order.get("tp_price")
            sl_price = order.get("sl_price")

            if not order_id or not entry_price:
                return None

            # Obter histórico de ordens para identificar a que fechou a posição
            trades = await self.order_executor.client.client.futures_account_trades(
                symbol=settings.SYMBOL,
                limit=50  # Limitar às 50 ordens mais recentes
            )

            # Filtrar apenas as ordens relacionadas a esta posição
            position_trades = []
            for trade in trades:
                # Verificar se o trade está relacionado a esta ordem (mesma posição ou ordens TP/SL relacionadas)
                if (trade.get('orderId') == order_id or
                        trade.get('positionSide') == direction or
                        (trade.get('time') > order.get('timestamp').timestamp() * 1000)):  # Trades após a abertura
                    position_trades.append(trade)

            if not position_trades:
                logger.warning(f"Não foram encontrados trades para a ordem {order_id}")
                return None

            # Ordenar por timestamp para obter o último trade (que fechou a posição)
            position_trades.sort(key=lambda x: x.get('time', 0))
            closing_trade = position_trades[-1]

            # Obter o preço de saída
            exit_price = float(closing_trade.get('price', 0))
            if exit_price <= 0:
                logger.warning(f"Preço de saída inválido: {exit_price}")
                return None

            # Determinar se foi TP ou SL com base no preço de saída e na direção
            if direction == "LONG":
                # Para LONG: se saiu acima do preço de entrada, foi TP, senão foi SL
                is_tp = exit_price > entry_price

                # Calcular o percentual real
                if is_tp:
                    actual_pct = (exit_price / entry_price - 1) * 100
                    result = {
                        "result": "TP",
                        "actual_tp_pct": actual_pct,
                        "actual_sl_pct": 0,
                        "exit_price": exit_price
                    }
                else:
                    actual_pct = (1 - exit_price / entry_price) * 100
                    result = {
                        "result": "SL",
                        "actual_tp_pct": 0,
                        "actual_sl_pct": actual_pct,
                        "exit_price": exit_price
                    }
            else:  # SHORT
                # Para SHORT: se saiu abaixo do preço de entrada, foi TP, senão foi SL
                is_tp = exit_price < entry_price

                # Calcular o percentual real
                if is_tp:
                    actual_pct = (1 - exit_price / entry_price) * 100
                    result = {
                        "result": "TP",
                        "actual_tp_pct": actual_pct,
                        "actual_sl_pct": 0,
                        "exit_price": exit_price
                    }
                else:
                    actual_pct = (exit_price / entry_price - 1) * 100
                    result = {
                        "result": "SL",
                        "actual_tp_pct": 0,
                        "actual_sl_pct": actual_pct,
                        "exit_price": exit_price
                    }

            # Verificação adicional: se a posição saiu por TP ou SL
            # Comparar com os preços configurados de TP e SL para maior precisão
            if direction == "LONG":
                if abs(exit_price - tp_price) < abs(exit_price - sl_price):
                    result["result"] = "TP"
                else:
                    result["result"] = "SL"
            else:  # SHORT
                if abs(exit_price - tp_price) < abs(exit_price - sl_price):
                    result["result"] = "TP"
                else:
                    result["result"] = "SL"

            return result

        except Exception as e:
            logger.error(f"Erro ao obter resultado do trade: {e}", exc_info=True)
            return None

    async def initialize(self) -> None:
        """Inicializa todos os componentes do bot."""
        await self.data_provider.initialize()
        await self.order_executor.initialize_filters()

        # Iniciar o sistema de retreinamento
        self.model_retrainer.start()

        logger.info("TradingBot inicializado e pronto para operar.")

    async def process_completed_trades(self):
        """
        Processa trades completados para registrar valores reais de TP/SL.
        Esta função busca informações de trades fechados diretamente da Binance
        para alimentar o sistema de retreinamento com dados reais.
        """
        try:
            # Obter todas as ordens executadas que não foram processadas ainda
            for order in self.order_executor.executed_orders:
                # Pular ordens já processadas
                if order.get("processed", False):
                    continue

                # Obter ID do sinal e da ordem
                signal_id = order.get("signal_id")
                order_id = order.get("order_id")
                if not signal_id or not order_id or order_id == "N/A":
                    continue

                # Verificar se a posição foi fechada na Binance
                is_closed = await self._check_order_status(order_id)

                if is_closed:
                    logger.info(f"Trade {signal_id} (ordem {order_id}) foi fechado. Obtendo resultados reais...")

                    # Obter os dados reais do trade
                    trade_result = await self._get_trade_result(order)

                    if trade_result:
                        result_type = trade_result["result"]

                        # Registrar os resultados com base no tipo de fechamento (TP ou SL)
                        if result_type == "TP":
                            actual_tp_pct = trade_result["actual_tp_pct"]
                            predicted_tp_pct = order.get("predicted_tp_pct", 0)

                            # Registrar no histórico de trades
                            self.trades_history.append({
                                "signal_id": signal_id,
                                "direction": order.get("direction"),
                                "result": "TP",
                                "predicted_tp_pct": predicted_tp_pct,
                                "actual_tp_pct": actual_tp_pct,
                                "close_time": datetime.datetime.now(),
                                "entry_price": order.get("entry_price"),
                                "exit_price": trade_result.get("exit_price")
                            })

                            # Registrar dados para o retreinamento
                            self.signal_generator.record_actual_values(
                                signal_id, actual_tp_pct, 0, self.model_retrainer
                            )

                            logger.info(
                                f"Trade {signal_id} atingiu TP: previsto={predicted_tp_pct:.2f}%, real={actual_tp_pct:.2f}%")

                        elif result_type == "SL":
                            actual_sl_pct = trade_result["actual_sl_pct"]
                            predicted_sl_pct = order.get("predicted_sl_pct", 0)

                            # Registrar no histórico de trades
                            self.trades_history.append({
                                "signal_id": signal_id,
                                "direction": order.get("direction"),
                                "result": "SL",
                                "predicted_sl_pct": predicted_sl_pct,
                                "actual_sl_pct": actual_sl_pct,
                                "close_time": datetime.datetime.now(),
                                "entry_price": order.get("entry_price"),
                                "exit_price": trade_result.get("exit_price")
                            })

                            # Registrar dados para o retreinamento
                            self.signal_generator.record_actual_values(
                                signal_id, 0, actual_sl_pct, self.model_retrainer
                            )

                            logger.info(
                                f"Trade {signal_id} atingiu SL: previsto={predicted_sl_pct:.2f}%, real={actual_sl_pct:.2f}%")

                        # Marcar ordem como processada para não processá-la novamente
                        order["processed"] = True

                    else:
                        logger.warning(f"Não foi possível obter resultados reais para o trade {signal_id}")

            # Limitar tamanho do histórico
            if len(self.trades_history) > self.max_trades_history:
                self.trades_history = self.trades_history[-self.max_trades_history:]

        except Exception as e:
            logger.error(f"Erro ao processar trades completados: {e}", exc_info=True)

    async def run(self) -> None:
        """
        Método principal do bot, refatorado para seguir os princípios SOLID.
        Coordena os componentes sem conter lógica de negócio diretamente.
        """
        try:
            await self.initialize()

            while True:
                self.cycle_count += 1
                logger.debug(f"Iniciando ciclo {self.cycle_count}")

                # A cada 10 ciclos, mostra um resumo do sistema
                if self.cycle_count % 10 == 0:
                    await self._log_system_summary()

                # Periodicamente verificar o status do retreinamento
                if self.cycle_count % self.check_retraining_status_interval == 0:
                    retraining_status = self.model_retrainer.get_retraining_status()
                    logger.info(f"Status do retreinamento: {retraining_status}")

                # Verificar atualizações de modelo
                await self.check_model_updates()

                # Processar trades completados para fornecer dados ao retreinador
                await self.process_completed_trades()

                # 1. Atualizar dados de mercado
                df = await self.data_provider.get_latest_data()
                if df.empty:
                    logger.warning("Sem dados disponíveis. Aguardando próximo ciclo.")
                    await asyncio.sleep(5)
                    continue

                # 2. Verificar posições abertas
                has_position = await self.order_executor.check_positions()
                if has_position:
                    await asyncio.sleep(5)
                    continue

                # 3. Obter preço atual
                current_price = await self.order_executor.client.get_futures_last_price(settings.SYMBOL)
                if current_price <= 0:
                    logger.warning("Falha ao obter preço atual. Aguardando próximo ciclo.")
                    await asyncio.sleep(5)
                    continue

                # 4. Gerar sinal de trading
                signal = await self.signal_generator.generate_signal(df, current_price)
                if signal:
                    # Armazenar previsões no histórico
                    self.predictions_history.append((signal.predicted_tp_pct, signal.predicted_sl_pct))

                    # Monitorar diversidade das previsões
                    self._monitor_predictions(self.predictions_history)

                if not signal:
                    await asyncio.sleep(5)
                    continue

                # 5. Analisar tendência de mercado atual usando TrendAnalyzer
                trend_direction = TrendAnalyzer.ema_trend(df)
                trend_strength = TrendAnalyzer.adx_trend(df)

                # 6. Determinar parâmetros de trading baseados na tendência
                # Substituindo a funcionalidade do MarketPatternAnalyzer e PatternBasedParameterAdjuster
                entry_threshold, tp_adjustment, sl_adjustment = self._adjust_parameters_based_on_trend(
                    trend_direction, trend_strength, signal.direction
                )

                # 7. Avaliar qualidade da entrada considerando parâmetros do sinal já gerado
                should_enter, entry_score = self.strategy.evaluate_entry_quality(
                    df,
                    signal.current_price,
                    signal.direction,
                    predicted_tp_pct=signal.predicted_tp_pct,
                    predicted_sl_pct=signal.predicted_sl_pct,
                    entry_threshold=entry_threshold
                )

                # 8. Log de análise técnica
                self._log_technical_analysis(
                    signal.direction, trend_direction, entry_score, entry_threshold
                )

                if not should_enter:
                    logger.info(
                        f"Sinal {signal.direction} gerado, mas condições de mercado não favoráveis. "
                        f"Score: {entry_score:.2f} < {entry_threshold:.2f}"
                    )
                    await asyncio.sleep(5)
                    continue

                # 9. Ajustar fatores de TP/SL baseado na tendência
                signal.tp_factor *= tp_adjustment
                signal.sl_factor *= sl_adjustment
                signal.tp_price = signal.current_price * signal.tp_factor
                signal.sl_price = signal.current_price * signal.sl_factor

                # Atualizar a razão R:R após ajustes
                new_tp_pct = (signal.tp_price / signal.current_price - 1) * 100
                new_sl_pct = (1 - signal.sl_price / signal.current_price) * 100 if signal.direction == "LONG" else \
                    (signal.sl_price / signal.current_price - 1) * 100

                # Atualizar signal com novos valores percentuais
                signal.predicted_tp_pct = new_tp_pct if signal.direction == "LONG" else -new_tp_pct
                signal.predicted_sl_pct = abs(new_sl_pct)
                signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

                logger.info(
                    f"Parâmetros ajustados: TP={signal.predicted_tp_pct:.2f}%, "
                    f"SL={signal.predicted_sl_pct:.2f}%, R:R={signal.rr_ratio:.2f}"
                )

                # Adicionar ATR para cálculo de tamanho da posição, se disponível
                if 'atr' in df.columns:
                    signal.atr_value = df['atr'].iloc[-1]

                # 10. Executar ordem
                order_result = await self.order_executor.execute_order(signal)
                if not order_result.success:
                    logger.warning(f"Falha na execução da ordem: {order_result.error_message}")

                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Erro no loop principal do bot: {e}", exc_info=True)
        finally:
            # Garantir que o cliente seja fechado corretamente
            await self.order_executor.client.close()
            logger.info("Conexões do bot fechadas corretamente.")

    def _adjust_parameters_based_on_trend(
            self, trend_direction: str, trend_strength: str, trade_direction: str
    ) -> tuple[float, float, float]:
        """
        Ajusta os parâmetros de trading baseado na tendência atual.
        Substitui a funcionalidade do PatternBasedParameterAdjuster.

        Args:
            trend_direction: Direção da tendência ("UPTREND", "DOWNTREND", "NEUTRAL")
            trend_strength: Força da tendência ("STRONG_TREND", "WEAK_TREND")
            trade_direction: Direção do trade ("LONG", "SHORT")

        Returns:
            tuple: (entry_threshold, tp_adjustment_factor, sl_adjustment_factor)
        """
        # Valores padrão
        entry_threshold = self.default_entry_threshold
        tp_adjustment = self.default_tp_adjustment
        sl_adjustment = self.default_sl_adjustment

        # Verificar se a tendência é forte
        is_strong_trend = trend_strength == "STRONG_TREND"

        # Ajustar com base na tendência e direção do trade
        if trend_direction == "UPTREND":
            if trade_direction == "LONG":
                # Trade a favor da tendência de alta
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_ALIGNED
                if is_strong_trend:
                    # Em tendência forte, podemos ser mais agressivos com TP
                    tp_adjustment = 1.2
                    sl_adjustment = 0.9
                logger.info(f"LONG alinhado com tendência de ALTA: menos seletivo, TP mais agressivo")
            else:  # SHORT
                # Trade contra a tendência de alta
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_AGAINST
                if is_strong_trend:
                    # Em tendência forte de alta, ser mais conservador com trades SHORT
                    tp_adjustment = 0.8
                    sl_adjustment = 0.7
                logger.info(f"SHORT contra tendência de ALTA: mais seletivo, alvos reduzidos")

        elif trend_direction == "DOWNTREND":
            if trade_direction == "SHORT":
                # Trade a favor da tendência de baixa
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_ALIGNED
                if is_strong_trend:
                    # Em tendência forte, podemos ser mais agressivos com TP
                    tp_adjustment = 1.2
                    sl_adjustment = 0.9
                logger.info(f"SHORT alinhado com tendência de BAIXA: menos seletivo, TP mais agressivo")
            else:  # LONG
                # Trade contra a tendência de baixa
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_AGAINST
                if is_strong_trend:
                    # Em tendência forte de baixa, ser mais conservador com trades LONG
                    tp_adjustment = 0.8
                    sl_adjustment = 0.7
                logger.info(f"LONG contra tendência de BAIXA: mais seletivo, alvos reduzidos")

        else:  # NEUTRAL
            # Em mercado sem tendência clara, usar configurações específicas para mercado em range
            entry_threshold = settings.ENTRY_THRESHOLD_RANGE
            tp_adjustment = settings.TP_ADJUSTMENT_RANGE
            sl_adjustment = settings.SL_ADJUSTMENT_RANGE
            logger.info(f"Mercado NEUTRO: ajustando parâmetros para operação em range")

        return entry_threshold, tp_adjustment, sl_adjustment

    async def _log_system_summary(self) -> None:
        """Log do resumo do sistema."""
        logger.info("=" * 50)
        logger.info(f"RESUMO DO SISTEMA - Ciclo {self.cycle_count}")
        logger.info(f"Símbolo: {settings.SYMBOL}, Interval: {settings.INTERVAL}")
        logger.info(f"Capital: {settings.CAPITAL}, Leverage: {settings.LEVERAGE}x")
        logger.info(f"Risco por Trade: {settings.RISK_PER_TRADE * 100}%")
        logger.info(
            f"Últimos candles processados: {len(self.data_handler.historical_df) if self.data_handler.historical_df is not None else 0}"
        )

        # Adicionar informações sobre o retreinamento
        retraining_status = self.model_retrainer.get_retraining_status()
        logger.info(f"Retreinamento: {'Em andamento' if retraining_status['retraining_in_progress'] else 'Inativo'}")
        logger.info(f"Último retreinamento: {retraining_status['last_retraining_time']}")
        logger.info(f"Horas desde último retreinamento: {retraining_status['hours_since_last_retraining']:.1f}")
        logger.info(f"Versão TP/SL: {retraining_status['tp_model_version']}/{retraining_status['sl_model_version']}")

        # Adicionar métricas de performance
        if self.trades_history:
            tp_count = sum(1 for trade in self.trades_history if trade['result'] == 'TP')
            total_trades = len(self.trades_history)
            win_rate = tp_count / total_trades * 100 if total_trades > 0 else 0
            logger.info(f"Win Rate: {win_rate:.1f}% ({tp_count}/{total_trades})")

        logger.info("=" * 50)

    def _log_technical_analysis(
            self, direction: str, trend_direction: str, entry_score: float, threshold: float
    ) -> None:
        """Log de análise técnica."""
        df_eval = self.data_handler.historical_df.copy()
        trend = TrendAnalyzer.ema_trend(df_eval)
        trend_strength = TrendAnalyzer.adx_trend(df_eval)

        logger.info(
            f"Análise Técnica: "
            f"Direção={direction}, "
            f"Tendência={trend_direction} ({trend_strength}), "
            f"Score de Entrada={entry_score:.2f}, "
            f"Threshold Ajustado={threshold:.2f}"
        )

    def _monitor_predictions(self, predictions_history: list[tuple[float, float]], window: int = 50) -> None:
        """
        Monitora a diversidade das previsões para detectar problemas de estagnação.

        Args:
            predictions_history: Lista com histórico de previsões (TP%, SL%)
            window: Tamanho da janela para análise
        """
        if len(predictions_history) < window:
            return

        # Analisar últimas previsões
        recent_predictions = predictions_history[-window:]
        tp_values = [tp for tp, _ in recent_predictions]
        sl_values = [sl for _, sl in recent_predictions]

        # Calcular estatísticas
        tp_std = np.std(tp_values)
        sl_std = np.std(sl_values)

        # Alertar se houver pouca variação
        if tp_std < 0.05 and sl_std < 0.05:  # Menos de 0.05% de desvio padrão
            logger.warning(
                f"ALERTA: Pouca variação nas previsões! TP std={tp_std:.4f}%, SL std={sl_std:.4f}%. "
                f"Considere retreinar o modelo ou verificar o pipeline de dados."
            )
