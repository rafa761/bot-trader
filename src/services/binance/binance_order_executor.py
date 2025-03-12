# services/binance/binance_order_executor.py
import datetime
import math
from typing import Any, Literal

from core.config import settings
from core.logger import logger
from services.base.interfaces import IOrderCalculator, IOrderExecutor
from services.base.schemas import ExecutedOrder, OrderResult, TPSLResult, TradingSignal
from services.binance.binance_client import BinanceClient


class BinanceOrderExecutor(IOrderExecutor):
    """
    Executor de ordens na Binance.

    Implementa a interface IOrderExecutor, sendo responsável por executar
    ordens, verificar posições e gerenciar ordens de TP/SL na Binance.
    """

    def __init__(
            self,
            binance_client: BinanceClient,
            order_calculator: IOrderCalculator,
            tick_size: float = 0.0,
            step_size: float = 0.0
    ):
        """
        Inicializa o executor de ordens.

        Args:
            binance_client: Cliente da Binance para operações na exchange
            order_calculator: Calculador para parâmetros de ordem
            tick_size: Tamanho do tick para o par de trading
            step_size: Tamanho do step para o par de trading
        """
        self.client = binance_client
        self.order_calculator = order_calculator
        self.tick_size = tick_size
        self.step_size = step_size

        # Histórico de ordens executadas
        self.executed_orders: list[dict[str, Any]] = []
        self.max_order_history = 100

    def get_executed_orders(self) -> list[ExecutedOrder]:
        """
        Retorna uma cópia da lista de ordens executadas.

        Returns:
            list[ExecutedOrder]: Lista de objetos Pydantic contendo informações das ordens executadas
        """
        # Converte cada dicionário para um objeto ExecutedOrder
        return [ExecutedOrder(**order) for order in self.executed_orders]

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

    async def get_unprocessed_orders(self) -> list[ExecutedOrder]:
        """
        Retorna uma lista com ordens que ainda não foram processadas.

        Returns:
            list[ExecutedOrder]: Lista de objetos Pydantic contendo ordens não processadas
        """
        return [ExecutedOrder(**order) for order in self.executed_orders if not order.get("processed", False)]

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
            tp_price_adj = self.order_calculator.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj <= current_price:
                tp_price_adj = current_price + (self.tick_size * 10)
            tp_str = self.order_calculator.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.order_calculator.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj >= current_price:
                sl_price_adj = current_price - (self.tick_size * 10)
            sl_str = self.order_calculator.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl: Literal["SELL", "BUY"] = "SELL"
        else:  # SHORT
            position_side = "SHORT"
            # Ajuste de TP
            tp_price_adj = self.order_calculator.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj >= current_price:
                tp_price_adj = current_price - (self.tick_size * 10)
            tp_str = self.order_calculator.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.order_calculator.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj <= current_price:
                sl_price_adj = current_price + (self.tick_size * 10)
            sl_str = self.order_calculator.format_price_for_tick_size(sl_price_adj, self.tick_size)

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
            qty = self.order_calculator.calculate_trade_quantity(
                capital=settings.CAPITAL,
                current_price=signal.current_price,
                leverage=settings.LEVERAGE,
                risk_per_trade=settings.RISK_PER_TRADE,
                atr_value=current_atr,
                min_notional=min_notional,
            )

            # Ajusta quantidade
            logger.info(f"Quantidade calculada: {qty}. Step size: {self.step_size}")
            qty_adj = self.order_calculator.adjust_quantity_to_step_size(qty, self.step_size)
            if qty_adj <= 0:
                logger.warning("Qty ajustada <= 0. Trade abortado.")
                return OrderResult(success=False, error_message="Quantidade ajustada <= 0")

            # Verificar valor notional mínimo da Binance
            notional_value = qty_adj * signal.current_price

            if notional_value < min_notional:
                # Recalcular quantidade para atender ao mínimo da Binance com margem de segurança
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
                    "order_id": str(order_resp.get("orderId", "N/A")),
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

        except Exception as e:
            error_msg = f"Erro ao executar ordem: {e}"
            logger.error(error_msg, exc_info=True)
            return OrderResult(success=False, error_message=error_msg)
