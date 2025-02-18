from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    BINANCE_API_KEY: str = Field(..., description="Chave de API para conexão com a Binance Testnet")
    BINANCE_API_SECRET: str = Field(..., description="Secret de API para conexão com a Binance Testnet")
    BINANCE_API_KEY_TESTNET: str = Field(..., description="Chave de API para conexão com a Binance Testnet")
    BINANCE_API_SECRET_TESTNET: str = Field(..., description="Secret de API para conexão com a Binance Testnet")
    NEWS_API_KEY: str = Field(..., description="Chave de API para busca de notícias")
    SENTIMENT_ANALYSIS_ENABLED: bool = Field(False, description="Indica se deve habilitar a análise de sentimento")
    SYMBOL: str = Field("BTCUSDT", description="Par de trading")
    INTERVAL: str = Field("15m", description="Intervalo do candle, ex: 1m, 15m, etc.")
    MODEL_DATA_TRAINING_START_DATE: str = Field("1 Jan, 2023",
                                                description="Data de inicio para treino dos modelos de machine learning")
    MODEL_DATA_PREDICTION_HORIZON: int = Field(12, description="Quantos períodos de previsão do modelo")
    CAPITAL: float = Field(1000.0, description="Capital disponível para trade")
    LEVERAGE: int = Field(5, description="Alavancagem utilizada")
    RISK_PER_TRADE: float = Field(0.20, description="Percentual de risco por trade (ex: 0.01 para 1%)")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


config = get_settings()
