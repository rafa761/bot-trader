# services/trading_bot.py
import asyncio

from core.config import settings
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_handler import DataHandler
from services.base.interfaces import IOrderExecutor
from services.base.services import MarketDataProvider
from services.binance.binance_client import BinanceClient
from services.binance.binance_data_provider import BinanceDataProvider
from services.binance.binance_order_executor import BinanceOrderExecutor
from services.cleanup_handler import CleanupHandler
from services.order_calculator import OrderCalculator
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
        self.data_handler = DataHandler(self.binance_client)
        self.order_calculator = OrderCalculator()

        # Analisador de tendências de mercado
        self.multi_tf_analyzer = None  # Será inicializado no método initialize()

        # Componentes seguindo injeção de dependência
        self.data_provider: MarketDataProvider = BinanceDataProvider(
            binance_client=self.binance_client,
            data_handler=self.data_handler
        )

        self.order_executor: IOrderExecutor = BinanceOrderExecutor(
            binance_client=self.binance_client,
            order_calculator=self.order_calculator,
        )

        # Gerenciador de estratégias centralizado
        self.strategy_manager = StrategyManager(tp_model=tp_model, sl_model=sl_model)

        # Estratégia atual (mantido para compatibilidade com os logs)
        self.current_strategy_name = "Não definida"

        # Manipulador de limpeza para interrupções
        self.cleanup_handler = CleanupHandler(self.binance_client, settings.SYMBOL)
        self.cleanup_handler.register()

        # Controle de ciclos
        self.cycle_count = 0

        logger.info("TradingBot SOLID inicializado com sucesso.")

    async def initialize(self) -> None:
        """Inicializa todos os componentes do bot."""
        await self.data_provider.initialize()
        await self.order_executor.initialize_filters()

        # Inicializar analisador multi-timeframe
        self.multi_tf_analyzer = MultiTimeFrameTrendAnalyzer(
            binance_client=self.binance_client,
            symbol=settings.SYMBOL
        )

        logger.info("TradingBot inicializado e pronto para operar.")

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

        # Adicionar informações sobre a estratégia atual
        logger.info(f"Estratégia atual: {self.current_strategy_name}")
        if hasattr(self, 'strategy_manager'):
            strategy_details = self.strategy_manager.get_strategy_details()
            if strategy_details.active:
                config = strategy_details.config
                logger.info(
                    f"Configuração da estratégia: "
                    f"TP ajuste={config.tp_adjustment}, "
                    f"SL ajuste={config.sl_adjustment}"
                )
                logger.info(f"Min R:R={config.min_rr_ratio}, Threshold={config.entry_threshold}")

        logger.info("=" * 50)

    async def run(self) -> None:
        """
        Método principal do bot, refatorado para incluir análise multi-timeframe.
        Coordena os componentes sem conter lógica de negócio diretamente.
        """
        try:
            await self.initialize()

            while True:
                self.cycle_count += 1
                logger.info(f"Iniciando ciclo {self.cycle_count}")

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
                    logger.info("Posição existente detectada. Aguardando fechamento.")
                    await asyncio.sleep(5)
                    continue

                # 3. Obter preço atual
                current_price = await self.binance_client.get_futures_last_price(settings.SYMBOL)
                if current_price <= 0:
                    logger.warning("Falha ao obter preço atual. Aguardando próximo ciclo.")
                    await asyncio.sleep(5)
                    continue

                # 4. Processar análise de mercado
                market_analysis = await self.strategy_manager.process_market_data(df, self.multi_tf_analyzer)

                # Atualizar a estratégia atual para o resumo do sistema
                self.current_strategy_name = market_analysis.strategy_name

                # 5. Tentar gerar sinal de trading usando o gerenciador de estratégias
                logger.info(
                    f"Tentando gerar sinal com estratégia: {self.current_strategy_name}, "
                    f"preço: {current_price}"
                )
                signal = await self.strategy_manager.generate_signal(df, current_price)
                if signal:
                    logger.info(
                        f"Sinal gerado: {signal.direction}, TP: {signal.predicted_tp_pct:.2f}%, "
                        f"SL: {signal.predicted_sl_pct:.2f}%, R:R: {signal.rr_ratio:.2f}"
                    )
                else:
                    logger.info("Nenhum sinal gerado neste ciclo")
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
                logger.info(f"Executando ordem: {signal.direction} em {current_price}")
                order_result = await self.order_executor.execute_order(signal)

                if order_result:
                    if order_result.success:
                        logger.info(f"Ordem executada com sucesso! ID: {order_result.order_id}")
                    else:
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
                if not self.binance_client.is_client_initialized():
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
