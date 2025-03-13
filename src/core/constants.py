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
    "ema_short",
    "ema_long",
    "ema_50",
    "hma",  # Hull MA - excelente para reduzir lag em timeframes curtos

    # Indicadores de tendência
    "parabolic_sar",
    "supertrend",
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
    "stoch_rsi",
    "macd",
    "macd_signal",
    "macd_histogram",

    # Divergências de RSI
    "rsi_divergence_bull",  # divergência de alta
    "rsi_divergence_bear",  # divergência de baixa

    # Indicadores de volatilidade
    "atr",
    "atr_pct",  # ATR como percentual do preço
    "boll_hband",
    "boll_lband",
    "boll_mavg",  # média móvel das Bollinger
    "boll_width",
    "boll_pct_b",  # %B de Bollinger (posição relativa)
    "squeeze",  # indicador de squeeze
    "ttm_squeeze",  # momentum TTM squeeze

    # Indicadores de volume
    "obv",
    "cmf",  # Chaikin Money Flow
    "vwap",
    "vwap_distance",

    # Indicadores direcionais
    "adx",
    "di_plus",  # DI+ do ADX
    "di_minus",  # DI- do ADX

    # Níveis de pivô
    "pivot",
    "pivot_r1",  # resistência 1
    "pivot_r2",  # resistência 2
    "pivot_s1",  # suporte 1
    "pivot_s2",  # suporte 2
    "pivot_resistance",  # flag de rejeição na resistência
    "pivot_support",  # flag de rejeição no suporte

    # Classificação de mercado
    "market_phase",  # fase do mercado (range, tendência alta, tendência baixa)
    "volatility_class",  # classificação de volatilidade
    "volume_ratio",  # volume comparado à média
    "volume_class",  # classificação de volume
]

# Define o caminho absoluto para a pasta de modelos treinados
# Explicação:
# - Path(__file__): Cria um objeto Path para o arquivo atual (constants.py).
# - .resolve(): Resolve o caminho absoluto.
# - .parent.parent: Volta dois níveis na hierarquia de pastas (de `core` para `src`).
# - / "trained_models": Adiciona a pasta "trained_models" ao caminho.
TRAINED_MODELS_DIR = Path(__file__).resolve().parent.parent / "trained_models"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS_CHECKPOINTS_DIR = TRAINED_MODELS_DIR / "checkpoints"
TRAINED_MODELS_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS_BACKUP_DIR = TRAINED_MODELS_DIR / "backups"
TRAINED_MODELS_BACKUP_DIR.mkdir(parents=True, exist_ok=True)

TRAINED_MODELS_TEMP_DIR = TRAINED_MODELS_DIR / "temp"
TRAINED_MODELS_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Define o caminho absoluto para a pasta de dados de treino
TRAIN_DATA_DIR = Path(__file__).resolve().parent.parent / "train_data"
TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Define o caminho absoluto para a pasta de cache
CACHE_DIR = Path(tempfile.gettempdir()) / 'traderbot_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
