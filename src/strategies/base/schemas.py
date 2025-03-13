# strategies/base/schemas.py

from enum import Enum

from pydantic import BaseModel, Field


class StrategyConfigSummary(BaseModel):
    min_rr_ratio: float = Field(..., description="Razão mínima entre recompensa e risco para operações válidas")
    entry_threshold: float = Field(
        ...,
        description="Limiar para pontuação de entrada que determina se o trade deve ser executado"
    )
    tp_adjustment: float = Field(..., description="Fator de ajuste para o take profit")
    sl_adjustment: float = Field(..., description="Fator de ajuste para o stop loss")
    entry_aggressiveness: float = Field(
        ...,
        description="Fator que determina quão agressivo o bot deve ser ao entrar em operações"
    )


class StrategyDetails(BaseModel):
    active: bool = Field(..., description="Indica se a estratégia está ativa ou não")
    name: str = Field(..., description="Nome da estratégia para identificação")
    condition: str = Field(..., description="Condição de mercado em que a estratégia é aplicável")
    config: StrategyConfigSummary = Field(..., description="Configuração da estratégia com parâmetros ajustáveis")


class TrendStrength(str, Enum):
    """Enumeração da força e direção das tendências."""
    STRONG_UP = "STRONG_UPTREND"
    MODERATE_UP = "MODERATE_UPTREND"
    WEAK_UP = "WEAK_UPTREND"
    NEUTRAL = "NEUTRAL"
    WEAK_DOWN = "WEAK_DOWNTREND"
    MODERATE_DOWN = "MODERATE_DOWNTREND"
    STRONG_DOWN = "STRONG_DOWNTREND"


class MTFAnalysisInfo(BaseModel):
    """Schema para informações extraídas da análise multi-timeframe."""
    trend: str = Field(default="NEUTRAL", description="Tendência consolidada detectada")
    strength: float = Field(default=0.0, description="Força da tendência (0-1)")
    is_strong_uptrend: bool = Field(default=False, description="Se é uma tendência forte de alta")
    is_strong_downtrend: bool = Field(default=False, description="Se é uma tendência forte de baixa")
    is_uptrend: bool = Field(default=False, description="Se é uma tendência de alta (qualquer força)")
    is_downtrend: bool = Field(default=False, description="Se é uma tendência de baixa (qualquer força)")
    is_neutral: bool = Field(default=True, description="Se é uma tendência neutra")
    confidence: float = Field(default=0.0, description="Nível de confiança na análise (0-100)")

    @classmethod
    def from_mtf_data(cls, mtf_data: dict) -> "MTFAnalysisInfo":
        """Cria uma instância a partir dos dados MTF."""
        if not mtf_data:
            return cls()

        trend = mtf_data.get('consolidated_trend', "NEUTRAL")
        strength = mtf_data.get('confidence', 0) / 100

        return cls(
            trend=trend,
            strength=strength,
            is_strong_uptrend="STRONG_UPTREND" in trend,
            is_strong_downtrend="STRONG_DOWNTREND" in trend,
            is_uptrend="UPTREND" in trend,
            is_downtrend="DOWNTREND" in trend,
            is_neutral="NEUTRAL" in trend,
            confidence=mtf_data.get('confidence', 0)
        )


class MarketConditionResult(BaseModel):
    """Schema para os resultados da avaliação de condições de mercado."""
    high_volatility: bool = Field(default=False, description="Se o mercado está em alta volatilidade")
    low_volatility: bool = Field(default=False, description="Se o mercado está em baixa volatilidade")
    uptrend: bool = Field(default=False, description="Se o mercado está em tendência de alta")
    downtrend: bool = Field(default=False, description="Se o mercado está em tendência de baixa")
    range_market: bool = Field(default=False, description="Se o mercado está em range")
    mtf_alignment: float = Field(default=0.0, description="Alinhamento com análise MTF (0-1)")


class StrategySelectionContext(BaseModel):
    """Schema para o contexto completo da seleção de estratégia."""
    mtf_info: MTFAnalysisInfo
    market_conditions: MarketConditionResult
    condition: str | None = Field(default=None, description="Condição selecionada")
    log_message: str = Field(default="", description="Mensagem para log")
