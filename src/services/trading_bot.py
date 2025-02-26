# services/trading_bot.py

import asyncio

import numpy as np
import pandas as pd

from core.config import settings
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_handler import DataHandler
from services.base.schemas import (
    MarketPatternResult,
    TradingParameters,
    TradingSignal,
    OrderResult,
    TPSLResult
)
from services.base.services import (
    MarketDataProvider,
    SignalGenerator,
    OrderExecutor,
    MarketPatternAnalyzer,
    ParameterAdjuster
)
from services.binance_client import BinanceClient
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
            sequence_length: int = 16
    ):
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.strategy = strategy
        self.sequence_length = sequence_length

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """Prepara uma sequência para previsão com modelo LSTM."""
        # Verifica se todos os FEATURE_COLUMNS existem no DataFrame
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.warning(f"Colunas ausentes no DataFrame: {missing_columns}")
            return None

        # Verifica se há dados suficientes
        if len(df) < self.sequence_length:
            logger.warning(
                f"Dados insuficientes para LSTM. Necessário: {self.sequence_length}, "
                f"Disponível: {len(df)}"
            )
            return None

        # Verifica se há valores NaN nas features requeridas
        if df[FEATURE_COLUMNS].isna().any().any():
            logger.warning("Existem valores NaN nas features necessárias para o LSTM")
            return None

        try:
            # Pegar as últimas 'sequence_length' entradas para a previsão
            last_sequence = df[FEATURE_COLUMNS].values[-self.sequence_length:]

            # Reformatar para o formato que o LSTM espera [samples, time steps, features]
            x_pred = np.array([last_sequence])

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

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])
            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Decidir direção baseada nas previsões LSTM
            direction = self.strategy.decide_direction(predicted_tp_pct, threshold=0.2)
            if direction is None:
                logger.info("Sinal neutro, não abrir trade.")
                return None

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

            # Criar sinal usando o schema Pydantic
            signal = TradingSignal(
                direction=direction,
                side=side,
                position_side=position_side,
                predicted_tp_pct=predicted_tp_pct,
                predicted_sl_pct=predicted_sl_pct,
                tp_price=tp_price,
                sl_price=sl_price,
                current_price=current_price,
                tp_factor=tp_factor,
                sl_factor=sl_factor
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

            # Calcula quantidade
            qty = self.strategy.calculate_trade_quantity(
                capital=settings.CAPITAL,
                current_price=signal.current_price,
                leverage=settings.LEVERAGE,
                risk_per_trade=settings.RISK_PER_TRADE,
                atr_value=current_atr,
            )

            # Ajusta quantidade
            qty_adj = self.strategy.adjust_quantity_to_step_size(qty, self.step_size)
            if qty_adj <= 0:
                logger.warning("Qty ajustada <= 0. Trade abortado.")
                return OrderResult(success=False, error_message="Quantidade ajustada <= 0")

            logger.info(
                f"Abrindo {signal.direction} c/ qty={qty_adj}, lastPrice={signal.current_price:.2f}..."
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


class ModelBasedPatternAnalyzer(MarketPatternAnalyzer):
    """Analisador de padrões de mercado baseado em modelos."""

    def __init__(self, pattern_classifier, sequence_length: int = 16):
        self.pattern_classifier = pattern_classifier
        self.sequence_length = sequence_length

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """Prepara uma sequência para o classificador de padrões."""
        # Verifica se todos os FEATURE_COLUMNS existem no DataFrame
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.warning(f"Colunas ausentes no DataFrame: {missing_columns}")
            return None

        # Verifica se há dados suficientes
        if len(df) < self.sequence_length:
            logger.warning(
                f"Dados insuficientes para classificação. Necessário: {self.sequence_length}, "
                f"Disponível: {len(df)}"
            )
            return None

        try:
            # Pegar as últimas 'sequence_length' entradas
            last_sequence = df[FEATURE_COLUMNS].values[-self.sequence_length:]

            # Reformatar para o formato esperado [samples, time steps, features]
            x_pred = np.array([last_sequence])

            return x_pred
        except Exception as e:
            logger.error(f"Erro ao preparar sequência para classificação: {e}", exc_info=True)
            return None

    def analyze_pattern(self, df: pd.DataFrame) -> MarketPatternResult:
        """Analisa e identifica o padrão atual do mercado."""
        if self.pattern_classifier is None:
            logger.warning("Classificador de padrões não carregado. Retornando padrão neutro.")
            return MarketPatternResult(pattern="UNKNOWN", confidence=0.0)

        try:
            # Preparar sequência para previsão
            X_seq = self._prepare_sequence(df)
            if X_seq is None:
                return MarketPatternResult(pattern="UNKNOWN", confidence=0.0)

            # Fazer previsão
            pattern, probabilities = self.pattern_classifier.predict(X_seq)
            confidence = float(np.max(probabilities))

            logger.info(f"Padrão de mercado detectado: {pattern} (confiança: {confidence:.2f})")
            return MarketPatternResult(pattern=pattern, confidence=confidence)

        except Exception as e:
            logger.error(f"Erro ao prever padrão de mercado: {e}", exc_info=True)
            return MarketPatternResult(pattern="UNKNOWN", confidence=0.0)


class PatternBasedParameterAdjuster(ParameterAdjuster):
    """Ajusta parâmetros de trading baseado no padrão de mercado."""

    def adjust_parameters(
            self, pattern: str, confidence: float, direction: str
    ) -> TradingParameters:
        """Ajusta os parâmetros de trading baseado no padrão de mercado."""
        # Valores padrão
        params = TradingParameters(
            entry_threshold=0.6,
            tp_adjustment_factor=1.0,
            sl_adjustment_factor=1.0
        )

        # Só ajustar se a confiança for razoável
        if confidence < 0.6:
            return params

        if pattern == "RANGE":
            # Em mercado lateralizado, ser mais conservador
            params.entry_threshold = 0.8  # Exigir score de entrada mais alto
            # Ajustar TP/SL para valores menores
            params.tp_adjustment_factor = 0.7
            params.sl_adjustment_factor = 0.7
            logger.info(
                "Ajustando estratégia para mercado em RANGE: trades mais seletivos, targets menores"
            )

        elif pattern == "VOLATILE":
            # Em mercado volátil, ser muito seletivo
            params.entry_threshold = 0.9  # Ser extremamente seletivo
            # Ampliar stops para evitar stop-outs
            params.tp_adjustment_factor = 1.2
            params.sl_adjustment_factor = 1.5
            logger.info(
                "Ajustando estratégia para mercado VOLÁTIL: muito seletivo, stops mais amplos"
            )

        elif pattern in ["UPTREND", "DOWNTREND"]:
            # Em tendência definida, verificar alinhamento
            if (direction == "LONG" and pattern == "UPTREND") or \
                    (direction == "SHORT" and pattern == "DOWNTREND"):
                # Trade a favor da tendência - ser mais agressivo
                params.entry_threshold = 0.65
                logger.info(
                    f"Trade {direction} alinhado com tendência {pattern}: menos seletivo"
                )
            else:
                # Trade contra a tendência - ser mais cauteloso
                params.entry_threshold = 0.85
                logger.info(
                    f"Trade {direction} contra tendência {pattern}: mais seletivo"
                )

        return params


# Classe principal refatorada
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

        # Classificador de padrões
        pattern_classifier = self._load_pattern_classifier()
        self.pattern_analyzer = ModelBasedPatternAnalyzer(pattern_classifier)

        # Ajustador de parâmetros
        self.parameter_adjuster = PatternBasedParameterAdjuster()

        # Controle de ciclos
        self.cycle_count = 0

        logger.info("TradingBot SOLID inicializado com sucesso.")

    def _load_pattern_classifier(self):
        """Carrega o modelo classificador de padrões de mercado."""
        pattern_classifier_path = TRAINED_MODELS_DIR / "market_pattern_classifier.keras"

        if pattern_classifier_path.exists():
            try:
                from models.market_pattern.model import MarketPatternClassifier
                from models.market_pattern.schemas import MarketPatternConfig

                pattern_config = MarketPatternConfig(
                    model_name="market_pattern_classifier",
                    num_classes=4,
                    class_names=["UPTREND", "DOWNTREND", "RANGE", "VOLATILE"]
                )

                classifier = MarketPatternClassifier.load(pattern_classifier_path, pattern_config)
                logger.info("Classificador de padrões de mercado carregado com sucesso")
                return classifier

            except Exception as e:
                logger.error(f"Erro ao carregar classificador de padrões: {e}", exc_info=True)
                return None
        else:
            logger.warning(f"Classificador de padrões não encontrado em {pattern_classifier_path}")
            return None

    async def initialize(self) -> None:
        """Inicializa todos os componentes do bot."""
        await self.data_provider.initialize()
        await self.order_executor.initialize_filters()
        logger.info("TradingBot inicializado e pronto para operar.")

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
                if not signal:
                    await asyncio.sleep(5)
                    continue

                # 5. Analisar padrão de mercado
                pattern_result = self.pattern_analyzer.analyze_pattern(df)

                # 6. Ajustar parâmetros baseado no padrão
                params = self.parameter_adjuster.adjust_parameters(
                    pattern_result.pattern, pattern_result.confidence, signal.direction
                )

                # 7. Avaliar qualidade da entrada
                should_enter, entry_score = self.strategy.evaluate_entry_quality(
                    df,
                    signal.current_price,
                    signal.direction,
                    entry_threshold=params.entry_threshold
                )

                # 8. Log de análise técnica
                self._log_technical_analysis(
                    signal.direction, pattern_result.pattern, entry_score, params.entry_threshold
                )

                if not should_enter:
                    logger.info(
                        f"Sinal {signal.direction} gerado, mas condições de mercado não favoráveis. "
                        f"Score: {entry_score:.2f} < {params.entry_threshold:.2f}"
                    )
                    await asyncio.sleep(5)
                    continue

                # 9. Ajustar fatores de TP/SL baseado no padrão
                signal.tp_factor *= params.tp_adjustment_factor
                signal.sl_factor *= params.sl_adjustment_factor
                signal.tp_price = signal.current_price * signal.tp_factor
                signal.sl_price = signal.current_price * signal.sl_factor

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
        logger.info(
            f"Classificador de padrões: {'Operacional' if self.pattern_analyzer.pattern_classifier is not None else 'Não disponível'}"
        )
        logger.info("=" * 50)

    def _log_technical_analysis(
            self, direction: str, pattern: str, entry_score: float, threshold: float
    ) -> None:
        """Log de análise técnica."""
        df_eval = self.data_handler.historical_df.copy()
        trend = TrendAnalyzer.ema_trend(df_eval)
        trend_strength = TrendAnalyzer.adx_trend(df_eval)

        logger.info(
            f"Análise Técnica: "
            f"Direção={direction}, "
            f"Padrão Mercado={pattern}, "
            f"Tendência Local={trend} ({trend_strength}), "
            f"Score de Entrada={entry_score:.2f}, "
            f"Threshold Ajustado={threshold:.2f}"
        )
