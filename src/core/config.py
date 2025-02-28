# core\config.py

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
    MODEL_DATA_PREDICTION_HORIZON: int = Field(6, description="Quantos períodos de previsão do modelo")
    CAPITAL: float = Field(1000.0, description="Capital disponível para trade")
    LEVERAGE: int = Field(5, description="Alavancagem utilizada")
    RISK_PER_TRADE: float = Field(0.02, description="Percentual de risco por trade (ex: 0.02 para 2%)")

    # Novos parâmetros para thresholds de entrada
    ENTRY_THRESHOLD_DEFAULT: float = Field(0.60, description="Threshold padrão para qualidade de entrada")
    ENTRY_THRESHOLD_RANGE: float = Field(0.75, description="Threshold para mercado em range")
    ENTRY_THRESHOLD_VOLATILE: float = Field(0.85, description="Threshold para mercado volátil")
    ENTRY_THRESHOLD_TREND_ALIGNED: float = Field(0.55, description="Threshold quando trade alinhado com tendência")
    ENTRY_THRESHOLD_TREND_AGAINST: float = Field(0.80, description="Threshold quando trade contra tendência")

    # Novos parâmetros para risk/reward
    MIN_RR_RATIO: float = Field(1.25, description="Razão mínima entre take profit e stop loss")
    ATR_MULTIPLIER: float = Field(1.3, description="Multiplicador do ATR para stops dinâmicos")

    # Parâmetros de ajuste para diferentes condições de mercado
    TP_ADJUSTMENT_RANGE: float = Field(0.7, description="Fator de ajuste TP em mercado lateralizado")
    SL_ADJUSTMENT_RANGE: float = Field(0.7, description="Fator de ajuste SL em mercado lateralizado")
    TP_ADJUSTMENT_VOLATILE: float = Field(1.2, description="Fator de ajuste TP em mercado volátil")
    SL_ADJUSTMENT_VOLATILE: float = Field(1.3, description="Fator de ajuste SL em mercado volátil")

    # Parâmetros adicionais para day trading
    VOLATILITY_HIGH_THRESHOLD: float = Field(1.5, description="Limite para considerar alta volatilidade (percentual)")
    VOLATILITY_LOW_THRESHOLD: float = Field(0.4, description="Limite para considerar baixa volatilidade (percentual)")
    MAX_POSITION_SIZE_PCT: float = Field(0.30, description="Tamanho máximo da posição como percentual do capital")

    # Parâmetros de trailing stop
    TRAILING_STOP_ENABLED: bool = Field(True, description="Habilitar trailing stop")
    TRAILING_STOP_ACTIVATION_PCT: float = Field(1.0, description="Percentual de lucro para ativar trailing stop")
    TRAILING_STOP_DISTANCE_PCT: float = Field(0.5, description="Distância do trailing stop em percentual")

    # Parâmetros de take profit parcial
    PARTIAL_TP_ENABLED: bool = Field(True, description="Habilitar take profit parcial")
    PARTIAL_TP_PCT: float = Field(0.5, description="Percentual do alvo para primeiro take profit")
    PARTIAL_TP_POSITION_PCT: float = Field(0.5, description="Percentual da posição a fechar no primeiro TP")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
