# core\constants.py

import tempfile
from pathlib import Path

FEATURE_COLUMNS = [
    # Dados OHLCV básicos
    "open",
    "high",
    "low",
    "close",
    "volume",

    # Médias móveis
    "sma_short",
    "sma_long",
    "ema_short",
    "ema_long",
    "hma",  # Nova média móvel Hull

    # Indicadores de tendência
    "parabolic_sar",
    "supertrend",  # Novo indicador
    "supertrend_direction",  # Direção do SuperTrend (-1 ou 1)

    # Velas Heikin Ashi
    "ha_open",
    "ha_high",
    "ha_low",
    "ha_close",

    # Osciladores
    "rsi",
    "stoch_k",
    "stoch_d",
    "stoch_rsi",  # Novo - Stochastic RSI
    "cci",
    "macd",
    "macd_signal",  # Novo - linha de sinal do MACD
    "macd_histogram",  # Novo - histograma do MACD
    "williams_r",  # Novo - Williams %R
    "ultimate_osc",  # Novo - Ultimate Oscillator

    # Divergências de RSI
    "rsi_divergence_bull",  # Novo - divergência de alta
    "rsi_divergence_bear",  # Novo - divergência de baixa

    # Indicadores de volatilidade
    "atr",
    "atr_pct",  # Novo - ATR como percentual do preço
    "boll_hband",
    "boll_lband",
    "boll_mavg",  # Novo - média móvel das Bollinger
    "boll_width",
    "boll_pct_b",  # Novo - %B de Bollinger (posição relativa)
    "keltner_hband",
    "keltner_lband",
    "keltner_mband",  # Novo - média das Keltner
    "keltner_width",  # Novo - largura das Keltner
    "squeeze",  # Novo - indicador de squeeze
    "ttm_squeeze",  # Novo - momentum TTM squeeze

    # Indicadores de volume
    "obv",
    "cmf",  # Novo - Chaikin Money Flow
    "vwap",
    "vwap_distance",  # Novo - distância ao VWAP
    "volume_macd",
    "zone_volume",  # Novo - volume na zona de preço atual
    "vol_zone_1",  # Novo - preço da zona de alto volume 1
    "vol_zone_2",  # Novo - preço da zona de alto volume 2
    "vol_zone_3",  # Novo - preço da zona de alto volume 3

    # Indicadores direcionais
    "adx",
    "di_plus",  # Novo - DI+ do ADX
    "di_minus",  # Novo - DI- do ADX
    "roc",

    # Ichimoku Cloud
    "tenkan_sen",  # Novo
    "kijun_sen",  # Novo
    "senkou_span_a",  # Novo
    "senkou_span_b",  # Novo
    "cloud_green",  # Novo - cloud é verde (bullish)
    "price_above_cloud",  # Novo
    "price_below_cloud",  # Novo
    "tk_cross_bull",  # Novo - cruzamento TK de alta
    "tk_cross_bear",  # Novo - cruzamento TK de baixa

    # Níveis de pivô
    "pivot",  # Novo
    "pivot_r1",  # Novo - resistência 1
    "pivot_r2",  # Novo - resistência 2
    "pivot_s1",  # Novo - suporte 1
    "pivot_s2",  # Novo - suporte 2
    "pivot_resistance",  # Novo - flag de rejeição na resistência
    "pivot_support",  # Novo - flag de rejeição no suporte

    # Classificação de mercado
    "market_phase",  # Novo - fase do mercado (range, tendência alta, tendência baixa)
    "volatility_class",  # Novo - classificação de volatilidade
    "volume_ratio",  # Novo - volume comparado à média
    "volume_class",  # Novo - classificação de volume
]

# Definir também quais indicadores são categóricos (para one-hot encoding)
CATEGORICAL_FEATURES = [
    "market_phase",
    "volatility_class",
    "volume_class",
    "trend_strength",
    "supertrend_direction"
]

# Define o caminho absoluto para a pasta de modelos treinados
# Explicação:
# - Path(__file__): Cria um objeto Path para o arquivo atual (constants.py).
# - .resolve(): Resolve o caminho absoluto.
# - .parent.parent: Volta dois níveis na hierarquia de pastas (de `core` para `src`).
# - / "trained_models": Adiciona a pasta "trained_models" ao caminho.
TRAINED_MODELS_DIR = Path(__file__).resolve().parent.parent / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Define o caminho absoluto para a pasta de dados de treino
TRAIN_DATA_DIR = Path(__file__).resolve().parent.parent / "train_data"
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define o caminho absoluto para a pasta de cache
CACHE_DIR = Path(tempfile.gettempdir()) / 'traderbot_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
