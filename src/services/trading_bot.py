# services/trading_bot.py
import asyncio
import datetime

import numpy as np

from core.config import settings
from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.interfaces import IOrderExecutor, IPerformanceMonitor
# Importando as classes extraídas
from services.base.services import MarketDataProvider, SignalGenerator
from services.binance_client import BinanceClient
from services.cleanup_handler import CleanupHandler
from services.market_analyzers import MarketTrendAnalyzer
from services.model_retrainer import ModelRetrainer
from services.performance_monitor import TradePerformanceMonitor
from services.trade_processor import TradeProcessor
from services.trading_strategy import TradingStrategy
from services.trend_analyzer import MultiTimeFrameTrendAnalyzer
from strategies.strategy_manager import StrategyManager


class TradingBot:
    """
    Classe principal do bot de trading, que coordena os demais componentes
    seguindo os princípios SOLID.

    Esta classe atua como orquestradora, delegando responsabilidades específicas
    a componentes especializados.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """
        Inicializa o bot de trading com os modelos LSTM.

        Args:
            tp_model: Modelo LSTM para previsão de Take Profit
            sl_model: Modelo LSTM para previsão de Stop Loss
        """
        # Cliente Binance
        self.binance_client = BinanceClient()

        # Componentes do sistema
        from repositories.data_handler import DataHandler
        self.data_handler = DataHandler(self.binance_client)
        self.performance_monitor: IPerformanceMonitor = TradePerformanceMonitor()
        self.strategy = TradingStrategy()

        # Analisador de tendências de mercado
        self.market_analyzer = MarketTrendAnalyzer()
        self.multi_tf_analyzer = None  # Será inicializado no método initialize()

        # Implementações SOLID com interfaces
        from services.binance_data_provider import BinanceDataProvider
        from services.lstm_signal_generator import LSTMSignalGenerator
        from services.binance_order_executor import BinanceOrderExecutor

        # Componentes seguindo injeção de dependência
        self.data_provider: MarketDataProvider = BinanceDataProvider(
            binance_client=self.binance_client,
            data_handler=self.data_handler
        )

        self.signal_generator: SignalGenerator = LSTMSignalGenerator(
            tp_model=tp_model,
            sl_model=sl_model,
            strategy=self.strategy,
            sequence_length=24
        )

        self.order_executor: IOrderExecutor = BinanceOrderExecutor(
            binance_client=self.binance_client,
            strategy=self.strategy,
            performance_monitor=self.performance_monitor
        )

        # Processador de trades - Corrigido para receber o order_executor
        self.trade_processor = TradeProcessor(
            binance_client=self.binance_client,
            signal_generator=self.signal_generator,
            performance_monitor=self.performance_monitor,
            order_executor=self.order_executor
        )

        # Sistema de retreinamento com referência para o signal_generator
        self.model_retrainer = ModelRetrainer(
            tp_model=tp_model,
            sl_model=sl_model,
            get_data_callback=self.get_historical_data_for_retraining,
            signal_generator_ref=lambda: self.signal_generator,
            retraining_interval_hours=24,
            min_data_points=1000,
            performance_threshold=0.15
        )

        # Gerenciador de estratégias centralizado
        self.strategy_manager = StrategyManager()

        # Estratégia atual (mantido para compatibilidade com os logs)
        self.current_strategy_name = "Não definida"

        # Manipulador de limpeza para interrupções
        self.cleanup_handler = CleanupHandler(self.binance_client, settings.SYMBOL)
        self.cleanup_handler.register()

        # Controle de ciclos
        self.cycle_count = 0

        # Controle de verificação de retreinamento
        self.last_retraining_check = datetime.datetime.now()
        self.retraining_check_interval = 300  # 5 minutos
        self.check_retraining_status_interval = 60  # Verificar a cada 60 ciclos

        logger.info("TradingBot SOLID inicializado com sucesso.")

    def get_historical_data_for_retraining(self) -> np.ndarray:
        """
        Retorna os dados históricos para retreinamento dos modelos.
        Esta função é usada como callback pelo ModelRetrainer.

        Returns:
            np.ndarray: Array com dados históricos
        """
        try:
            df = self.data_handler.historical_df
            if df is not None and not df.empty:
                df_copy = df.copy()
                logger.info(f"Fornecendo {len(df_copy)} registros para retreinamento")
                return df_copy
            logger.warning("Sem dados históricos disponíveis para retreinamento")
            return np.array([])
        except Exception as e:
            logger.error(f"Erro ao obter dados históricos para retreinamento: {e}", exc_info=True)
            return np.array([])

    async def initialize(self) -> None:
        """Inicializa todos os componentes do bot."""
        await self.data_provider.initialize()
        await self.order_executor.initialize_filters()

        # Inicializar analisador multi-timeframe
        self.multi_tf_analyzer = MultiTimeFrameTrendAnalyzer(
            binance_client=self.binance_client,
            symbol=settings.SYMBOL
        )

        # Iniciar o sistema de retreinamento
        self.model_retrainer.start()

        logger.info("TradingBot inicializado e pronto para operar.")

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
                if hasattr(self.signal_generator, 'update_models'):
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
                else:
                    logger.warning("Signal Generator não possui método update_models")

        except Exception as e:
            logger.error(f"Erro ao verificar atualizações de modelo: {e}", exc_info=True)

    async def _log_system_summary(self) -> None:
        """Log do resumo do sistema com informações multi-timeframe."""
        logger.info("=" * 50)
        logger.info(f"RESUMO DO SISTEMA - Ciclo {self.cycle_count}")
        logger.info(f"Símbolo: {settings.SYMBOL}, Interval: {settings.INTERVAL}")
        logger.info(f"Capital: {settings.CAPITAL}, Leverage: {settings.LEVERAGE}x")
        logger.info(f"Risco por Trade: {settings.RISK_PER_TRADE * 100}%")
        logger.info(
            f"Últimos candles processados: {len(self.data_handler.historical_df) if self.data_handler.historical_df is not None else 0}"
        )

        # Adicionar análise multi-timeframe
        try:
            mtf_trend, confidence, details = await self.multi_tf_analyzer.analyze_multi_timeframe_trend()
            logger.info("-" * 30)
            logger.info("ANÁLISE MULTI-TIMEFRAME")
            logger.info(f"Tendência Consolidada: {mtf_trend}")
            logger.info(f"Confiança: {confidence:.2f}%")

            # Tendência por timeframe
            logger.info("Tendências por timeframe:")
            for tf, info in details["tf_summary"].items():
                logger.info(f"  {tf}: {info['strength']} (score: {info['score']:.2f})")
        except Exception as e:
            logger.error(f"Erro ao mostrar análise multi-timeframe: {e}")

        # Adicionar informações sobre o retreinamento
        retraining_status = self.model_retrainer.get_retraining_status()
        logger.info("-" * 30)
        logger.info("STATUS DO RETREINAMENTO")
        logger.info(f"Status: {'Em andamento' if retraining_status['retraining_in_progress'] else 'Inativo'}")
        logger.info(f"Último retreinamento: {retraining_status['last_retraining_time']}")
        logger.info(f"Horas desde último retreinamento: {retraining_status['hours_since_last_retraining']:.1f}")
        logger.info(f"Versão TP/SL: {retraining_status['tp_model_version']}/{retraining_status['sl_model_version']}")

        # Adicionar métricas de performance
        try:
            metrics = self.performance_monitor.metrics
            if metrics.total_trades > 0:
                logger.info("-" * 30)
                logger.info("MÉTRICAS DE PERFORMANCE")
                logger.info(f"Total de trades: {metrics.total_trades}")
                logger.info(f"Win rate: {metrics.win_rate:.2%}")
                logger.info(f"P&L total: ${metrics.total_profit_loss:.2f}")
                logger.info(f"Expectancy: ${metrics.expectancy:.2f}")

                if metrics.long_trades > 0:
                    logger.info(
                        f"LONG win rate: {metrics.long_win_rate:.2%} ({metrics.long_wins}/{metrics.long_trades})")
                if metrics.short_trades > 0:
                    logger.info(
                        f"SHORT win rate: {metrics.short_win_rate:.2%} ({metrics.short_wins}/{metrics.short_trades})")
        except Exception as e:
            logger.error(f"Erro ao incluir métricas de performance no resumo: {e}")

        # Adicionar informações sobre a estratégia atual
        logger.info(f"Estratégia atual: {self.current_strategy_name}")
        if hasattr(self, 'strategy_manager'):
            strategy_details = self.strategy_manager.get_strategy_details()
            if strategy_details['active']:
                config = strategy_details['config']
                logger.info(f"Configuração da estratégia: "
                            f"TP ajuste={config.get('tp_adjustment', 1.0)}, "
                            f"SL ajuste={config.get('sl_adjustment', 1.0)}")
                logger.info(
                    f"Min R:R={config.get('min_rr_ratio', 1.5)}, Threshold={config.get('entry_threshold', 0.6)}")

        logger.info("=" * 50)

    async def add_multi_timeframe_analysis(self, signal, df):
        """
        Adiciona análise multi-timeframe ao sinal de trading.

        Args:
            signal: Sinal de trading gerado
            df: DataFrame do timeframe atual

        Returns:
            Sinal enriquecido com análise multi-timeframe e booleano indicando se deve prosseguir
        """
        if not signal:
            return None, False

        try:
            # Executar análise multi-timeframe
            mtf_trend, confidence, details = await self.multi_tf_analyzer.analyze_multi_timeframe_trend()

            # Verificar alinhamento do sinal com a tendência multi-timeframe
            trade_direction = signal.direction
            alignment_score, confidence = await self.multi_tf_analyzer.get_trend_alignment(trade_direction)

            # Adicionar informações ao sinal
            signal.mtf_trend = mtf_trend
            signal.mtf_alignment = alignment_score
            signal.mtf_confidence = confidence
            signal.mtf_details = details["tf_summary"]

            # Verificar se o alinhamento é suficiente para executar o trade
            MIN_MTF_ALIGNMENT = 0.3  # Mínimo alinhamento para prosseguir
            should_proceed = alignment_score >= MIN_MTF_ALIGNMENT

            if not should_proceed:
                logger.info(
                    f"Sinal {trade_direction} rejeitado - Baixo alinhamento multi-timeframe: "
                    f"{alignment_score:.2f} < {MIN_MTF_ALIGNMENT}"
                )
            else:
                logger.info(
                    f"Análise multi-timeframe para sinal {trade_direction}: "
                    f"Tendência MTF={mtf_trend}, Alinhamento={alignment_score:.2f}, "
                    f"Confiança={confidence:.2f}"
                )

                # Logar detalhes dos timeframes para análise
                for tf, info in details["tf_summary"].items():
                    logger.info(f"  Timeframe {tf}: {info['strength']}, Score: {info['score']:.2f}")

            return signal, should_proceed

        except Exception as e:
            logger.error(f"Erro na análise multi-timeframe: {e}", exc_info=True)
            # Em caso de erro, prosseguir normalmente sem a análise MTF
            return signal, True

    async def run(self) -> None:
        """
        Método principal do bot, refatorado para incluir análise multi-timeframe.
        Coordena os componentes sem conter lógica de negócio diretamente.
        """
        try:
            await self.initialize()

            # Inicializar o gerenciador de estratégias (se não foi feito no __init__)
            if not hasattr(self, 'strategy_manager'):
                from strategies.strategy_manager import StrategyManager
                self.strategy_manager = StrategyManager()
                logger.info("Gerenciador de estratégias inicializado")

            while True:
                self.cycle_count += 1
                logger.debug(f"Iniciando ciclo {self.cycle_count}")

                # A cada 10 ciclos, mostra um resumo do sistema
                if self.cycle_count % 10 == 0:
                    await self._log_system_summary()

                    # Adicionar resumo de performance a cada 30 ciclos
                    if self.cycle_count % 30 == 0:
                        self.performance_monitor.log_performance_summary()

                # Periodicamente verificar o status do retreinamento
                if self.cycle_count % self.check_retraining_status_interval == 0:
                    retraining_status = self.model_retrainer.get_retraining_status()
                    logger.info(f"Status do retreinamento: {retraining_status}")

                # Verificar atualizações de modelo
                await self.check_model_updates()

                # Processar trades completados para fornecer dados ao retreinador
                await self.trade_processor.process_completed_trades()

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
                current_price = await self.binance_client.get_futures_last_price(settings.SYMBOL)
                if current_price <= 0:
                    logger.warning("Falha ao obter preço atual. Aguardando próximo ciclo.")
                    await asyncio.sleep(5)
                    continue

                # 4. Processar análise de mercado UNIFICADA com o gerenciador de estratégias
                market_analysis = await self.strategy_manager.process_market_data(df, self.multi_tf_analyzer)

                # Atualizar a estratégia atual para o resumo do sistema
                self.current_strategy_name = market_analysis.get("strategy_name", "Não definida")

                # 5. Gerar sinal de trading (isso não mudou)
                signal = await self.signal_generator.generate_signal(df, current_price)
                if not signal:
                    await asyncio.sleep(5)
                    continue

                # 6. Avaliar e ajustar o sinal usando o gerenciador de estratégias
                signal, should_execute = await self.strategy_manager.evaluate_signal(
                    signal, df, self.multi_tf_analyzer
                )

                # Se o sinal não for aprovado, continuar para o próximo ciclo
                if not should_execute:
                    logger.info("Sinal rejeitado pelo gerenciador de estratégias.")
                    await asyncio.sleep(5)
                    continue

                # 7. Executar ordem (só chegamos aqui se o sinal foi aprovado)
                order_result = await self.order_executor.execute_order(signal)
                if not order_result.success:
                    logger.warning(f"Falha na execução da ordem: {order_result.error_message}")

                await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Tarefa do bot cancelada. Realizando limpeza...")
            await self.cleanup()
            raise
        except KeyboardInterrupt:
            logger.info("Interrupção do teclado detectada no método run. Realizando limpeza...")
            await self.cleanup()
        except Exception as e:
            logger.error(f"Erro no loop principal do bot: {e}", exc_info=True)
        finally:
            # Garantir que o cliente seja fechado corretamente mesmo sem execução do cleanup
            try:
                await self.binance_client.close()
                logger.info("Conexões do bot fechadas corretamente.")
            except Exception as e:
                logger.error(f"Erro ao fechar conexões: {e}")

    async def cleanup(self) -> None:
        """
        Limpa todas as ordens e posições abertas.
        Este método é chamado quando o bot está sendo encerrado.
        """
        logger.info("Iniciando limpeza de ordens e posições...")
        try:
            # Executar limpeza
            if hasattr(self, 'cleanup_handler') and self.cleanup_handler:
                await self.cleanup_handler.execute_cleanup()
            else:
                logger.warning("CleanupHandler não encontrado. Tentando limpeza direta...")

                # Código de fallback para limpeza básica
                if not self.binance_client._initialized:
                    await self.binance_client.initialize()

                # Cancelar ordens abertas
                try:
                    result = await self.binance_client.client.futures_cancel_all_open_orders(symbol=settings.SYMBOL)
                    logger.info(f"Ordens canceladas: {result}")
                except Exception as e:
                    logger.error(f"Erro ao cancelar ordens: {e}")

                # Fechar posições (simplificado)
                try:
                    positions = await self.binance_client.client.futures_position_information(symbol=settings.SYMBOL)
                    for position in positions:
                        position_amt = float(position['positionAmt'])
                        if position_amt == 0:
                            continue

                        # Lógica mínima para fechamento
                        if position_amt > 0:  # LONG
                            await self.binance_client.client.futures_create_order(
                                symbol=settings.SYMBOL,
                                side="SELL",
                                positionSide="LONG",
                                type="MARKET",
                                quantity=abs(position_amt)
                            )
                        else:  # SHORT
                            await self.binance_client.client.futures_create_order(
                                symbol=settings.SYMBOL,
                                side="BUY",
                                positionSide="SHORT",
                                type="MARKET",
                                quantity=abs(position_amt)
                            )
                except Exception as e:
                    logger.error(f"Erro ao fechar posições: {e}")

            logger.info("Limpeza concluída.")
        except Exception as e:
            logger.error(f"Erro durante limpeza: {e}", exc_info=True)
        finally:
            # Garantir que o cliente seja fechado
            try:
                await self.binance_client.close()
                logger.info("Cliente Binance fechado.")
            except Exception as e:
                logger.error(f"Erro ao fechar cliente Binance: {e}")
