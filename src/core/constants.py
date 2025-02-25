# core\constants.py

import tempfile
from pathlib import Path

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sma_short",
    "sma_long",
    "ema_short",
    "ema_long",
    "parabolic_sar",
    "rsi",
    "stoch_k",
    "stoch_d",
    "cci",
    "macd",
    "volume_macd",
    "atr",
    "boll_hband",
    "boll_lband",
    "boll_width",
    "keltner_hband",
    "keltner_lband",
    "obv",
    "vwap",
    "adx",
    "roc",
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
