# services/trade_processor.py

import datetime
from typing import Dict, Any, Optional

from core.config import settings
from core.logger import logger
from services.binance_client import BinanceClient
from services.lstm_signal_generator import LSTMSignalGenerator
from services.performance_monitor import TradePerformanceMonitor


class TradeProcessor:
    """
    Processador de trades completados.

    Esta classe é responsável por detectar trades completados,
    obter seus resultados da Binance e fornecer feedback ao sistema
    de retreinamento e monitoramento de performance.
    """

    def __init__(
            self,
            binance_client: BinanceClient,
            signal_generator: LSTMSignalGenerator,
            performance_monitor: TradePerformanceMonitor | None = None
    ):
        """
        Inicializa o processador de trades.

        Args:
            binance_client: Cliente da Binance para consulta de trades
            signal_generator: Gerador de sinais para registrar resultados reais
            performance_monitor: Monitor de performance para atualizar métricas
        """
        self.client = binance_client
        self.signal_generator = signal_generator
        self.performance_monitor = performance_monitor

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
            positions = await self.client.client.futures_position_information(
                symbol=settings.SYMBOL
            )

            # Verificar o histórico de ordens
            orders = await self.client.client.futures_get_all_orders(
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
            recent_orders = await self.client.client.futures_get_all_orders(
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

    async def _get_trade_result(self, order: dict[str, Any]) -> dict[str, Any] | None:
        """
        Obtém o resultado real de um trade fechado consultando a Binance.

        Args:
            order: Dicionário com detalhes da ordem

        Returns:
            Dict[str, Any]: Resultado real do trade com informações de TP/SL e preços
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
            trades = await self.client.client.futures_account_trades(
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

    async def process_completed_trades(self) -> None:
        """
        Processa trades completados para registrar valores reais de TP/SL.
        Busca informações de trades fechados diretamente da Binance para alimentar
        o sistema de retreinamento e monitor de performance com dados reais.
        """
        try:
            # Obter ordens executadas que não foram processadas do executor
            # Vamos supor que ordem_executor tem um atributo executed_orders para acessar o histórico de ordens
            if not hasattr(self.signal_generator, "order_executor"):
                # Neste caso, assumimos que o executor está em outro lugar
                # Será necessário implementar uma maneira de acessar essas ordens
                # Por enquanto, skip
                return

            executed_orders = getattr(self.signal_generator, "order_executor", {}).executed_orders or []

            for order in executed_orders:
                if order.get("processed", False):
                    continue

                signal_id = order.get("signal_id")
                order_id = order.get("order_id")

                if not signal_id or not order_id or order_id == "N/A":
                    continue

                # Verificar se a posição foi fechada
                is_closed = await self._check_order_status(order_id)

                if is_closed:
                    logger.info(f"Trade {signal_id} (ordem {order_id}) foi fechado. Obtendo resultados reais...")

                    # Obter os dados reais do trade
                    trade_result = await self._get_trade_result(order)

                    if trade_result:
                        result_type = trade_result["result"]
                        exit_price = trade_result.get("exit_price")

                        if exit_price and self.performance_monitor:
                            # Registrar saída no monitor de performance
                            try:
                                # Buscar o trade pelo signal_id
                                trade = self.performance_monitor.get_trade_by_signal_id(signal_id)

                                if trade:
                                    # Registrar saída do trade
                                    self.performance_monitor.register_trade_exit(
                                        trade_id=trade.trade_id,
                                        exit_price=exit_price,
                                        exit_time=datetime.datetime.now()
                                    )
                                    logger.info(f"Saída de trade registrada no monitor: {signal_id}")
                                else:
                                    logger.warning(f"Trade com signal_id {signal_id} não encontrado no monitor")
                            except Exception as e:
                                logger.error(f"Erro ao registrar saída de trade no monitor: {e}")

                        # Registrar dados para o retreinamento
                        if result_type == "TP":
                            actual_tp_pct = trade_result["actual_tp_pct"]
                            predicted_tp_pct = order.get("predicted_tp_pct", 0)

                            # Usar None como placeholder para retrainer
                            # O verdadeiro retrainer seria injetado pelo TradingBot
                            self.signal_generator.record_actual_values(signal_id, actual_tp_pct, 0, None)

                        elif result_type == "SL":
                            actual_sl_pct = trade_result["actual_sl_pct"]
                            predicted_sl_pct = order.get("predicted_sl_pct", 0)

                            self.signal_generator.record_actual_values(signal_id, 0, actual_sl_pct, None)

                        # Marcar ordem como processada
                        order["processed"] = True

                    else:
                        logger.warning(f"Não foi possível obter resultados reais para o trade {signal_id}")

        except Exception as e:
            logger.error(f"Erro ao processar trades completados: {e}", exc_info=True)
