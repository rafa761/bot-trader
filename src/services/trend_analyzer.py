# services/trend_analyzer.py
import pandas as pd


class TrendAnalyzer:
    """
    Classe responsável por analisar e identificar tendências no mercado
    usando diferentes métodos e timeframes.
    """

    @staticmethod
    def ema_trend(df: pd.DataFrame, short_period: int = 9, long_period: int = 21) -> str:
        """
        Identifica a tendência com base no cruzamento de EMAs.

        Args:
            df: DataFrame com dados históricos incluindo 'close'
            short_period: Período para EMA curta
            long_period: Período para EMA longa

        Returns:
            str: "UPTREND", "DOWNTREND" ou "NEUTRAL"
        """
        if len(df) < long_period or 'close' not in df.columns:
            return "NEUTRAL"

        # Verificar se já temos EMAs existentes com nome similar
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            # Usar EMAs existentes
            last_short_ema = df['ema_short'].iloc[-1]
            last_long_ema = df['ema_long'].iloc[-1]
        else:
            # Calcular EMAs temporariamente
            short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
            long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
            last_short_ema = short_ema.iloc[-1]
            last_long_ema = long_ema.iloc[-1]

        # Determinar tendência
        if last_short_ema > last_long_ema:
            return "UPTREND"
        elif last_short_ema < last_long_ema:
            return "DOWNTREND"
        else:
            return "NEUTRAL"

    @staticmethod
    def adx_trend(df: pd.DataFrame, period: int = 14, threshold: int = 25) -> str:
        """
        Identifica a força da tendência usando ADX.

        Args:
            df: DataFrame com dados históricos incluindo 'adx'
            period: Período para ADX
            threshold: Limiar para considerar uma tendência forte

        Returns:
            str: "STRONG_TREND" ou "WEAK_TREND"
        """
        if 'adx' not in df.columns or len(df) < period:
            return "WEAK_TREND"

        last_adx = df['adx'].iloc[-1]

        if last_adx > threshold:
            return "STRONG_TREND"
        else:
            return "WEAK_TREND"
