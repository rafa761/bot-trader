# risk_manager.py

import numpy as np
import pandas as pd

from logger import logger
from sentiment_analysis import get_market_sentiment


def dynamic_risk_adjustment(df: pd.DataFrame, current_volatility: float, base_risk: float = 0.02) -> (float, int):
    """
    Ajusta risco e alavancagem dinamicamente com base no ATR médio do DF,
    volatilidade atual e sentimento de mercado.

    df: DataFrame que deve conter alguma coluna (ex: 'atr')
    current_volatility: valor de volatilidade atual (ex: pode ser último ATR)
    base_risk: risco base inicial (ex: 2%)

    Retorna (adjusted_risk, adjusted_leverage).
    """
    try:
        # Exemplo: supõe que df['atr'] existe
        if 'atr' not in df.columns or df['atr'].isnull().all():
            logger.warning("DF não contém coluna 'atr'. Usando approach default para ATR = 1.")
            df['atr'] = 1.0

        sentiment = get_market_sentiment(query='bitcoin')  # ou outro termo
        atr_mean = df['atr'].mean() if not df['atr'].empty else 1.0

        # Calcula um multiplicador que leva em conta volatilidade e sentimento
        # Exemplo: multiplica volatilidade normalizada * (sentimento + 0.5)
        # e faz um clip entre 0.5 e 2.0
        risk_multiplier = np.clip(
            (current_volatility / atr_mean) * (sentiment + 0.5),
            0.5, 2.0
        )
        adjusted_risk = base_risk * risk_multiplier

        # Exemplo: se o risk_multiplier é grande, reduz alavancagem
        # (quanto mais arriscado, menos alavancagem)
        possible_leverage = int(25 * (1 / risk_multiplier))
        adjusted_leverage = min(25, max(1, possible_leverage))

        logger.info(
            f"[RISK] Sentiment={sentiment:.2f}, Volatility={current_volatility:.4f}, "
            f"ATR_mean={atr_mean:.4f}, RiskMultiplier={risk_multiplier:.2f}, "
            f"AdjustedRisk={adjusted_risk:.4f}, AdjustedLeverage={adjusted_leverage}"
        )
        return adjusted_risk, adjusted_leverage
    except Exception as e:
        logger.error(f"Erro ao ajustar risco dinamicamente: {e}", exc_info=True)
        # Retorna valores default em caso de falha
        return base_risk, 25
