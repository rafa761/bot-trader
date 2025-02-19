from pathlib import Path

FEATURE_COLUMNS = [
    'sma_short',
    'sma_long',
    'rsi',
    'atr',
    'volume',
    'macd',
    'boll_hband',
    'boll_lband'
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
