# core\config.py

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    ENVIRONMENT: str = Field(
        "development", description="Ambiente para execução da aplicação. production ou development"
        )
    BINANCE_API_KEY: str = Field(..., description="Chave de API para conexão com a Binance Testnet")
    BINANCE_API_SECRET: str = Field(..., description="Secret de API para conexão com a Binance Testnet")
    BINANCE_API_KEY_TESTNET: str = Field(..., description="Chave de API para conexão com a Binance Testnet")
    BINANCE_API_SECRET_TESTNET: str = Field(..., description="Secret de API para conexão com a Binance Testnet")
    SYMBOL: str = Field("BTCUSDT", description="Par de trading")
    INTERVAL: str = Field("15m", description="Intervalo do candle, ex: 1m, 15m, etc.")
    MODEL_DATA_TRAINING_START_DATE: str = Field(
        "1 Jan, 2020",
                                                description="Data de inicio para treino dos modelos de machine learning")
    MODEL_DATA_PREDICTION_HORIZON: int = Field(6, description="Quantos períodos de previsão do modelo")
    CAPITAL: float = Field(100.0, description="Capital disponível para trade")
    LEVERAGE: int = Field(5, description="Alavancagem utilizada")
    RISK_PER_TRADE: float = Field(0.02, description="Percentual de risco por trade (ex: 0.02 para 2%)")

    # Novos parâmetros para thresholds de entrada
    ENTRY_THRESHOLD_DEFAULT: float = Field(0.65, description="Threshold padrão para qualidade de entrada")

    # Novos parâmetros para risk/reward
    MIN_RR_RATIO: float = Field(1.5, description="Razão mínima entre take profit e stop loss")
    ATR_MULTIPLIER: float = Field(1.5, description="Multiplicador do ATR para stops dinâmicos")

    # Parâmetros adicionais para day trading
    VOLATILITY_HIGH_THRESHOLD: float = Field(1.2, description="Limite para considerar alta volatilidade (percentual)")
    VOLATILITY_LOW_THRESHOLD: float = Field(0.5, description="Limite para considerar baixa volatilidade (percentual)")
    MAX_POSITION_SIZE_PCT: float = Field(0.10, description="Tamanho máximo da posição como percentual do capital")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
