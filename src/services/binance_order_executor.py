# services/binance_order_executor.py

import datetime
from typing import Any, Literal

from core.config import settings
from core.logger import logger
from services.base.interfaces import IOrderExecutor
from services.base.schemas import (
    TradingSignal,
    OrderResult,
    TPSLResult
)
from services.binance_client import BinanceClient
from services.performance_monitor import TradePerformanceMonitor
from services.trading_strategy import TradingStrategy


class BinanceOrderExecutor(IOrderExecutor):
    """
    Executor de ordens na Binance.

    Implementa a interface IOrderExecutor, sendo responsável por executar
    ordens, verificar posições e gerenciar ordens de TP/SL na Binance.
    """

    def __init__(
            self,
            binance_client: BinanceClient,
            strategy: TradingStrategy,
            performance_monitor: TradePerformanceMonitor | None = None,
            tick_size: float = 0.0,
            step_size: float = 0.0
    ):
        """
        Inicializa o executor de ordens.

        Args:
            binance_client: Cliente da Binance para operações na exchange
            strategy: Estratégia de trading para cálculos de ordens
            performance_monitor: Monitor de performance para registrar trades
            tick_size: Tamanho do tick para o par de trading
            step_size: Tamanho do step para o par de trading
        """
        self.client = binance_client
        self.strategy = strategy
        self.performance_monitor = performance_monitor
        self.tick_size = tick_size
        self.step_size = step_size

        # Histórico de ordens executadas
        self.executed_orders: list[dict[str, Any]] = []
        self.max_order_history = 100

    def get_executed_orders(self) -> list[dict[str, Any]]:
        """
        Retorna uma cópia da lista de ordens executadas.

        Returns:
            list: Lista de dicionários contendo informações das ordens executadas
        """
        return self.executed_orders.copy()

    def mark_order_as_processed(self, order_id: str) -> bool:
        """
        Marca uma ordem como processada.

        Args:
            order_id: ID da ordem a ser marcada como processada

        Returns:
            bool: True se a ordem foi encontrada e marcada, False caso contrário
        """
        for order in self.executed_orders:
            if order.get("order_id") == order_id:
                order["processed"] = True
                logger.info(f"Ordem {order_id} marcada como processada")
                return True
        logger.warning(f"Ordem {order_id} não encontrada para marcar como processada")
        return False

    async def get_unprocessed_orders(self) -> list[dict[str, Any]]:
        """
        Retorna uma lista com ordens que ainda não foram processadas.

        Returns:
            list: Lista de dicionários contendo ordens não processadas
        """
        return [order for order in self.executed_orders if not order.get("processed", False)]

    async def initialize_filters(self) -> None:
        """
        Inicializa os filtros de trading e configura a alavancagem.
        """
        self.tick_size, self.step_size = await self.client.get_symbol_filters(settings.SYMBOL)

        if not self.tick_size or not self.step_size:
            logger.error("Não foi possível obter tickSize/stepSize.")
            return

        logger.info(f"{settings.SYMBOL} -> tickSize={self.tick_size}, stepSize={self.step_size}")

        # Define alavancagem
        await self.client.set_leverage(settings.SYMBOL, settings.LEVERAGE)

    async def check_positions(self) -> bool:
        """
        Verifica se existem posições abertas.

        Returns:
            bool: True se existir alguma posição aberta, False caso contrário
        """
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
        """
        Cria ordens de TP e SL.

        Args:
            direction: Direção do trade ("LONG" ou "SHORT")
            current_price: Preço atual do ativo
            tp_price: Preço alvo para take profit
            sl_price: Preço alvo para stop loss

        Returns:
            TPSLResult: Resultado das ordens TP/SL
        """
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

            side_for_tp_sl: Literal["SELL", "BUY"] = "SELL"
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

            side_for_tp_sl: Literal["SELL", "BUY"] = "BUY"

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
        """
        Executa uma ordem baseada no sinal fornecido.

        Args:
            signal: Sinal de trading contendo parâmetros da ordem

        Returns:
            OrderResult: Resultado da execução da ordem
        """
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

                # Registrar o trade no monitor de performance
                if self.performance_monitor is not None:
                    try:
                        # Garantir que temos todos os dados necessários
                        market_trend = getattr(signal, 'market_trend', None)
                        market_strength = getattr(signal, 'market_strength', None)

                        # Calcular volatilidade
                        volatility = None
                        if signal.atr_value:
                            volatility = (signal.atr_value / signal.current_price) * 100

                        self.performance_monitor.register_trade_from_signal(
                            signal_id=signal.id,
                            direction=signal.direction,
                            entry_price=signal.current_price,
                            quantity=qty_adj,
                            tp_target_price=signal.tp_price,
                            sl_target_price=signal.sl_price,
                            predicted_tp_pct=signal.predicted_tp_pct,
                            predicted_sl_pct=signal.predicted_sl_pct,
                            market_trend=market_trend,
                            market_volatility=volatility,
                            market_strength=market_strength,
                            entry_score=signal.entry_score if hasattr(signal, 'entry_score') else None,
                            rr_ratio=signal.rr_ratio if hasattr(signal, 'rr_ratio') else None,
                            leverage=settings.LEVERAGE,
                            trade_id=str(order_resp.get("orderId"))
                        )
                        logger.info(f"Trade {signal.id} registrado no monitor de performance")
                    except Exception as e:
                        logger.error(f"Erro ao registrar trade no monitor de performance: {e}")

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
