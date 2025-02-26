# services/entry_scorer.py
from typing import Dict

import pandas as pd


class EntryScorer:
    """
    Sistema de pontuação para avaliar a qualidade de potenciais entradas de trading
    com base em múltiplos fatores técnicos.
    """

    @staticmethod
    def score_trend_alignment(trend_direction: str, trade_direction: str) -> float:
        """
        Pontua o alinhamento entre a direção da tendência e a direção pretendida do trade.

        Args:
            trend_direction: Direção da tendência ("UPTREND", "DOWNTREND", "NEUTRAL", etc.)
            trade_direction: Direção pretendida do trade ("LONG" ou "SHORT")

        Returns:
            float: Pontuação entre 0.0 e 1.0
        """
        if trend_direction == "NEUTRAL":
            return 0.5

        if (trend_direction == "UPTREND" and trade_direction == "LONG") or \
                (trend_direction == "DOWNTREND" and trade_direction == "SHORT"):
            return 1.0
        else:
            return 0.0

    @staticmethod
    def score_rsi_condition(df: pd.DataFrame, trade_direction: str) -> float:
        """
        Pontua as condições do RSI para a entrada.

        Args:
            df: DataFrame com dados históricos incluindo 'rsi'
            trade_direction: Direção pretendida do trade ("LONG" ou "SHORT")

        Returns:
            float: Pontuação entre 0.0 e 1.0
        """
        if 'rsi' not in df.columns:
            return 0.5

        current_rsi = df['rsi'].iloc[-1]

        if trade_direction == "LONG":
            # Para LONG, RSI baixo é melhor (condição de sobrevenda)
            if current_rsi < 30:
                return 1.0
            elif current_rsi < 40:
                return 0.8
            elif current_rsi < 50:
                return 0.6
            else:
                return 0.4
        else:
            # Para SHORT, RSI alto é melhor (condição de sobrecompra)
            if current_rsi > 70:
                return 1.0
            elif current_rsi > 60:
                return 0.8
            elif current_rsi > 50:
                return 0.6
            else:
                return 0.4

    @staticmethod
    def score_macd_signal(df: pd.DataFrame, trade_direction: str) -> float:
        """
        Pontua o sinal do MACD para a entrada.

        Args:
            df: DataFrame com dados históricos incluindo 'macd'
            trade_direction: Direção pretendida do trade ("LONG" ou "SHORT")

        Returns:
            float: Pontuação entre 0.0 e 1.0
        """
        if 'macd' not in df.columns or len(df) < 3:
            return 0.5

        # Verificar cruzamento recente
        current_macd = df['macd'].iloc[-1]
        prev_macd = df['macd'].iloc[-2]

        if trade_direction == "LONG":
            # Para LONG, queremos MACD cruzando para cima
            if current_macd > 0 and prev_macd <= 0:
                return 1.0  # Cruzamento da linha zero
            elif current_macd > prev_macd:
                return 0.8  # Em tendência de alta
            else:
                return 0.3
        else:
            # Para SHORT, queremos MACD cruzando para baixo
            if current_macd < 0 and prev_macd >= 0:
                return 1.0  # Cruzamento da linha zero
            elif current_macd < prev_macd:
                return 0.8  # Em tendência de baixa
            else:
                return 0.3

    @staticmethod
    def calculate_entry_score(
            df: pd.DataFrame,
            current_price: float,
            trade_direction: str,
            trend_direction: str,
            weights: dict[str, float] = None
    ) -> float:
        """
        Calcula uma pontuação geral para a qualidade da entrada.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual
            trade_direction: Direção pretendida do trade ("LONG" ou "SHORT")
            trend_direction: Direção da tendência atual
            weights: Pesos para cada componente da pontuação

        Returns:
            float: Pontuação geral entre 0.0 e 1.0
        """
        if weights is None:
            weights = {
                'trend': 0.35,
                'rsi': 0.35,
                'macd': 0.30
            }

        # Calcular pontuações individuais
        trend_score = EntryScorer.score_trend_alignment(trend_direction, trade_direction)
        rsi_score = EntryScorer.score_rsi_condition(df, trade_direction)
        macd_score = EntryScorer.score_macd_signal(df, trade_direction)

        # Calcular pontuação ponderada
        total_score = (
                trend_score * weights['trend'] +
                rsi_score * weights['rsi'] +
                macd_score * weights['macd']
        )

        return min(total_score, 1.0)  # Garantir que não exceda 1.0
