# trading_strategy.py

import numpy as np
import pandas as pd
from logger import logger
from config import (
    CAPITAL_INICIAL, RISK_PER_TRADE, LEVERAGE,
    TRANSACTION_COST, SLIPPAGE, DAILY_LOSS_LIMIT,
    SMA_WINDOW_SHORT, SMA_WINDOW_LONG
)

class TradingStrategy:
    """
    Classe para encapsular a lógica de quando comprar/vender
    com base em indicadores e nas previsões dos modelos.
    """

    def __init__(self):
        self.position = None
        self.entry_price = 0.0
        self.take_profit_target = None
        self.trailing_stop_loss = None
        self.quantity = 0.0
        self.entry_time = None
        self.capital = CAPITAL_INICIAL
        self.daily_profit = 0.0  # Soma dos lucros/perdas do dia

    def reset_position(self):
        """Reseta todos os parâmetros de posição."""
        self.position = None
        self.entry_price = 0.0
        self.take_profit_target = None
        self.trailing_stop_loss = None
        self.quantity = 0.0
        self.entry_time = None

    def calculate_quantity(self, real_price, predicted_sl_percent, available_balance):
        """
        Cálculo de quantidade baseada em risco.
        """
        risk_amount = self.capital * RISK_PER_TRADE
        effective_sl_percent = max(predicted_sl_percent, 0.1)  # Evitar zero
        calc_qty = (risk_amount / (real_price * (effective_sl_percent / 100))) * LEVERAGE
        calc_qty = round(calc_qty, 3)

        # Calcula quantidade máxima com base no saldo disponível
        max_quantity = (available_balance * LEVERAGE) / real_price
        max_quantity = round(max_quantity, 3)

        return min(calc_qty, max_quantity)

    def should_buy(self, sma_short, sma_long):
        return sma_short > sma_long

    def should_sell(self, sma_short, sma_long):
        return sma_short < sma_long

    def check_daily_loss_limit(self):
        """
        Verifica se o limite de perda diária foi atingido.
        """
        if self.daily_profit <= DAILY_LOSS_LIMIT:
            logger.warning("Limite de perda diária atingido. Operações encerradas.")
            return True
        return False
