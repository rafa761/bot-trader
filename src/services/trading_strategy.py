# services\trading_strategy.py

import math

from core.logger import logger
from services.entry_scorer import EntryScorer
from services.trend_analyzer import TrendAnalyzer


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
            risk_per_trade: float,
            atr_value: float = None
    ) -> float:
        """
        Calcula a quantidade a ser negociada com ajuste de volatilidade.
        """
        risk_amount = capital * risk_per_trade
        original_risk = risk_amount

        # Ajuste baseado em ATR
        if atr_value is not None:
            atr_percentage = atr_value / current_price * 100

            if atr_percentage > 1.5:  # Alta volatilidade
                volatility_factor = 0.7
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR alto ({atr_percentage:.2f}%) - "
                    f"Reduzindo risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            elif atr_percentage < 0.5:  # Baixa volatilidade
                volatility_factor = 1.3
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR baixo ({atr_percentage:.2f}%) - "
                    f"Aumentando risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            else:
                logger.info(f"ATR normal ({atr_percentage:.2f}%) - Mantendo risco padrão")

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

    def evaluate_entry_quality(
            self,
            df,
            current_price: float,
            trade_direction: str,
            entry_threshold: float = 0.7
    ) -> tuple[bool, float]:
        """
        Avalia a qualidade da entrada potencial usando múltiplos critérios.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual
            trade_direction: "LONG" ou "SHORT"
            entry_threshold: Pontuação mínima para considerar a entrada

        Returns:
            tuple[bool, float]: (Deve entrar, pontuação da entrada)
        """
        # 1. Análise de tendência
        trend = TrendAnalyzer.ema_trend(df)
        trend_strength = TrendAnalyzer.adx_trend(df)

        # 2. Calcular componentes de score individuais
        trend_score = EntryScorer.score_trend_alignment(trend, trade_direction)
        rsi_score = EntryScorer.score_rsi_condition(df, trade_direction)
        macd_score = EntryScorer.score_macd_signal(df, trade_direction)

        # Logar componentes
        logger.info(
            f"Componentes do Score - "
            f"Alinhamento de Tendência: {trend_score:.2f}, "
            f"Condição RSI: {rsi_score:.2f}, "
            f"Sinal MACD: {macd_score:.2f}"
        )

        # 3. Calcular pontuação geral
        entry_score = EntryScorer.calculate_entry_score(df, current_price, trade_direction, trend)

        # 4. Penalizar para tendência forte na direção oposta
        if (trend == "UPTREND" and trade_direction == "SHORT") or \
                (trend == "DOWNTREND" and trade_direction == "LONG"):
            if trend_strength == "STRONG_TREND":
                entry_score *= 0.7  # Reduz score em 30%
                logger.info(
                    f"Score penalizado por trade contra tendência forte: {entry_score:.2f} (após redução de 30%)")

        # Decidir se deve entrar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score
