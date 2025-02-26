# services\trading_strategy.py

import math


class TradingStrategy:
    """
    Classe responsável pela lógica de decisão (LONG, SHORT ou neutro),
    bem como pelo cálculo de quantidade e ajuste de preços.
    """

    def decide_direction(self, predicted_tp_pct: float, threshold: float = 0.2) -> str | None:
        """
        Decide se vamos abrir uma posição LONG, SHORT ou permanecer neutro,
        com base no valor previsto de TP.

        :param predicted_tp_pct: Previsão de variação percentual para TP
        :param threshold: Limiar para decidir se é LONG/SHORT
        :return: "LONG", "SHORT" ou None
        """
        if predicted_tp_pct > threshold:
            return "LONG"
        elif predicted_tp_pct < -threshold:
            return "SHORT"
        else:
            return None

    def calculate_trade_quantity(
            self,
            capital: float,
            current_price: float,
            leverage: float,
            risk_per_trade: float
    ) -> float:
        """
        Calcula a quantidade a ser negociada:
        (capital * risco_por_trade) / current_price * leverage.

        :param capital: Capital total disponível
        :param current_price: Preço atual do ativo
        :param leverage: Alavancagem utilizada
        :param risk_per_trade: Porcentagem do capital a ser arriscado
        :return: Quantidade calculada em float
        """
        risk_amount = capital * risk_per_trade
        quantity = (risk_amount / current_price) * leverage
        return quantity

    def adjust_price_to_tick_size(self, price: float, tick_size: float) -> float:
        """
        Arredonda 'price' para baixo (floor) ao múltiplo de tick_size.

        :param price: Preço original
        :param tick_size: Valor de tick size
        :return: Preço arredondado
        """
        return math.floor(price / tick_size) * tick_size

    def format_price_for_tick_size(self, price: float, tick_size: float) -> str:
        """
        Formata 'price' com a quantidade correta de casas decimais
        baseada no tick_size.

        :param price: Valor do preço
        :param tick_size: Tick size do símbolo
        :return: Preço formatado em string
        """
        decimals = 0
        if '.' in str(tick_size):
            decimals = len(str(tick_size).split('.')[-1])
        return f"{price:.{decimals}f}"

    def adjust_quantity_to_step_size(self, qty: float, step_size: float) -> float:
        """
        Arredonda 'qty' para o múltiplo do step_size.

        :param qty: Quantidade original
        :param step_size: Step size do símbolo
        :return: Quantidade arredondada
        """
        return math.floor(qty / step_size) * step_size
