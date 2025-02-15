# trading_strategy.py

import pandas as pd

from config import (
    CAPITAL_INICIAL, DAILY_LOSS_LIMIT
)
from logger import logger
from risk_manager import dynamic_risk_adjustment


class TradingStrategy:
    """
    Classe para encapsular a lógica de quando comprar/vender
    com base em indicadores e nas previsões dos modelos.
    """

    def __init__(self):
        self.position: str | None = None  # "long" ou "short"
        self.entry_price: float = 0.0
        self.take_profit_target: float | None = None
        self.trailing_stop_loss: float | None = None
        self.quantity: float = 0.0
        self.entry_time: pd.Timestamp | None = None
        self.capital: float = CAPITAL_INICIAL
        self.daily_profit: float = 0.0

        # Pode usar como default (se não usar a abordagem dinâmica):
        self.base_risk_per_trade: float = 0.20
        self.leverage: int = 5

    def reset_position(self) -> None:
        """Reseta todos os parâmetros de posição."""
        self.position = None
        self.entry_price = 0.0
        self.take_profit_target = None
        self.trailing_stop_loss = None
        self.quantity = 0.0
        self.entry_time = None

    @staticmethod
    def should_buy(sma_short: float, sma_long: float) -> bool:
        """Verifica se a condição de compra é atendida com base nas médias móveis."""
        return sma_short > sma_long

    @staticmethod
    def should_sell(sma_short: float, sma_long: float) -> bool:
        """Verifica se a condição de venda é atendida com base nas médias móveis."""
        return sma_short < sma_long

    def check_daily_loss_limit(self) -> bool:
        """Verifica se o limite diário de perda foi atingido e encerra operações caso necessário."""
        if self.daily_profit <= DAILY_LOSS_LIMIT:
            logger.warning("Limite de perda diária atingido. Operações encerradas.")
            return True
        return False

    def calculate_quantity_dynamic(
            self, df: pd.DataFrame, current_volatility: float, real_price: float, available_balance: float
    ) -> tuple[float, float]:
        """
        Calcula a quantidade de ativos a serem negociados com base em um ajuste dinâmico de risco.
        Retorna a quantidade calculada e a alavancagem ajustada.

        :param df: DataFrame contendo dados do mercado.
        :param current_volatility: Volatilidade atual do mercado.
        :param real_price: Preço atual do ativo.
        :param available_balance: Saldo disponível para operações.
        :return: Tuple contendo a quantidade calculada e a alavancagem ajustada.
        """
        adjusted_risk, adjusted_leverage = dynamic_risk_adjustment(df, current_volatility)

        # O risco base agora é 'adjusted_risk' ao invés de um valor fixo.
        risk_amount = self.capital * adjusted_risk

        # Exemplo simples: Stop loss fixo de 1%
        effective_sl_percent = 1.0
        calc_qty = (risk_amount / (real_price * (effective_sl_percent / 100))) * adjusted_leverage
        calc_qty = round(calc_qty, 3)

        # Calcula quantidade máxima com base no saldo disponível e alavancagem ajustada
        max_quantity = (available_balance * adjusted_leverage) / real_price
        max_quantity = round(max_quantity, 3)

        return min(calc_qty, max_quantity), adjusted_leverage
