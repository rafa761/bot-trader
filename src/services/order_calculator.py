# services/order_calculator.py
import math

from core.config import settings
from core.logger import logger
from services.base.interfaces import IOrderCalculator


class OrderCalculator(IOrderCalculator):
    """
    Implementação padrão para cálculo de parâmetros de ordem.
    """

    def calculate_trade_quantity(
            self,
            capital: float,
            current_price: float,
            leverage: float,
            risk_per_trade: float,
            atr_value: float = None,
            min_notional: float = 100.0
    ) -> float:
        """
        Calcula a quantidade a ser negociada com ajuste de volatilidade.
        """
        risk_amount = capital * risk_per_trade
        original_risk = risk_amount

        # Ajuste baseado em ATR
        if atr_value is not None:
            atr_percentage = atr_value / current_price * 100

            if atr_percentage > settings.VOLATILITY_HIGH_THRESHOLD:  # Alta volatilidade
                volatility_factor = 0.7
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR alto ({atr_percentage:.2f}%) - "
                    f"Reduzindo risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            elif atr_percentage < settings.VOLATILITY_LOW_THRESHOLD:  # Baixa volatilidade
                volatility_factor = 1.3
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR baixo ({atr_percentage:.2f}%) - "
                    f"Aumentando risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            else:
                logger.info(f"ATR normal ({atr_percentage:.2f}%) - Mantendo risco padrão")

        # Calcular quantidade básica
        base_quantity = (risk_amount / current_price) * leverage

        # Verificar se excede o tamanho máximo permitido
        max_quantity = (capital * settings.MAX_POSITION_SIZE_PCT) / current_price * leverage

        # Usar o menor valor entre a quantidade calculada e o máximo permitido
        quantity = min(base_quantity, max_quantity)

        if quantity < base_quantity:
            logger.info(f"Quantidade ajustada para limite máximo: {quantity:.4f} (era {base_quantity:.4f})")

            # Verificar se atende ao valor mínimo notional da Binance
            notional_value = quantity * current_price
            if notional_value < min_notional:
                # Ajustar para o mínimo requerido com margem de segurança
                min_quantity = (min_notional * 1.05) / current_price
                logger.warning(
                    f"Quantidade calculada ({quantity:.4f} BTC, ${notional_value:.2f}) abaixo do valor mínimo da Binance. "
                    f"Ajustando para {min_quantity:.4f} BTC (${min_quantity * current_price:.2f})"
                )
                quantity = min_quantity

        return quantity

    def adjust_price_to_tick_size(self, price: float, tick_size: float) -> float:
        """
        Arredonda 'price' para baixo (floor) ao múltiplo de tick_size.
        """
        return math.floor(price / tick_size) * tick_size

    def format_price_for_tick_size(self, price: float, tick_size: float) -> str:
        """
        Formata 'price' com a quantidade correta de casas decimais
        baseada no tick_size.
        """
        decimals = 0
        if '.' in str(tick_size):
            decimals = len(str(tick_size).split('.')[-1])
        return f"{price:.{decimals}f}"

    def adjust_quantity_to_step_size(self, qty: float, step_size: float) -> float:
        """
        Arredonda 'qty' para o múltiplo do step_size.
        """
        return math.floor(qty / step_size) * step_size
