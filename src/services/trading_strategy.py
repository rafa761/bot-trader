# services\trading_strategy.py

import math

from core.config import settings
from core.logger import logger


class TradingStrategy:
    """
    Classe responsável pela lógica de decisão (LONG, SHORT ou neutro),
    bem como pelo cálculo de quantidade e ajuste de preços.
    """


    def decide_direction(self, predicted_tp_pct: float, predicted_sl_pct: float,
                         threshold: float = 0.2) -> str | None:
        """
        Decide se vamos abrir uma posição LONG, SHORT ou permanecer neutro,
        com base nos valores previstos de TP e SL, considerando a relação R:R.

        Args:
            predicted_tp_pct: Previsão de variação percentual para TP
            predicted_sl_pct: Previsão de variação percentual para SL
            threshold: Limiar para decidir se é LONG/SHORT

        Returns:
            "LONG", "SHORT" ou None
        """
        # Verificar se os valores são válidos
        if not isinstance(predicted_tp_pct, (int, float)) or not isinstance(predicted_sl_pct, (int, float)):
            logger.warning(f"Valores de previsão inválidos: TP={predicted_tp_pct}, SL={predicted_sl_pct}")
            return None

        # Assegurar que SL é positivo
        predicted_sl_pct = abs(predicted_sl_pct)

        # Calcular a razão RR para esta previsão
        if predicted_sl_pct <= 0.1:  # Evitar divisão por zero ou SL muito pequeno
            logger.warning(f"SL previsto muito pequeno ou inválido: {predicted_sl_pct}")
            return None

        rr_ratio = abs(predicted_tp_pct / predicted_sl_pct)

        # Decisão de direção baseada no TP previsto
        if predicted_tp_pct > threshold:
            logger.info(
                f"Sinal LONG gerado: TP={predicted_tp_pct:.2f}%, SL={predicted_sl_pct:.2f}%, R:R={rr_ratio:.2f}")
            return "LONG"
        elif predicted_tp_pct < -threshold:
            logger.info(
                f"Sinal SHORT gerado: TP={predicted_tp_pct:.2f}%, SL={predicted_sl_pct:.2f}%, R:R={rr_ratio:.2f}")
            return "SHORT"
        else:
            logger.info(f"Sinal neutro: TP={predicted_tp_pct:.2f}% dentro do threshold ({threshold})")
            return None

    @staticmethod
    def calculate_trade_quantity(
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

    @staticmethod
    def adjust_price_to_tick_size(price: float, tick_size: float) -> float:
        """
        Arredonda 'price' para baixo (floor) ao múltiplo de tick_size.

        :param price: Preço original
        :param tick_size: Valor de tick size
        :return: Preço arredondado
        """
        return math.floor(price / tick_size) * tick_size

    @staticmethod
    def format_price_for_tick_size(price: float, tick_size: float) -> str:
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

    @staticmethod
    def adjust_quantity_to_step_size(qty: float, step_size: float) -> float:
        """
        Arredonda 'qty' para o múltiplo do step_size.

        :param qty: Quantidade original
        :param step_size: Step size do símbolo
        :return: Quantidade arredondada
        """
        return math.floor(qty / step_size) * step_size
