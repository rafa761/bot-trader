# strategies/schemas.py

from pydantic import BaseModel, Field


class StrategyConfigSummary(BaseModel):
    min_rr_ratio: float = Field(..., description="Razão mínima entre recompensa e risco para operações válidas")
    entry_threshold: float = Field(...,
                                   description="Limiar para pontuação de entrada que determina se o trade deve ser executado")
    tp_adjustment: float = Field(..., description="Fator de ajuste para o take profit")
    sl_adjustment: float = Field(..., description="Fator de ajuste para o stop loss")
    entry_aggressiveness: float = Field(...,
                                        description="Fator que determina quão agressivo o bot deve ser ao entrar em operações")


class StrategyDetails(BaseModel):
    active: bool = Field(..., description="Indica se a estratégia está ativa ou não")
    name: str = Field(..., description="Nome da estratégia para identificação")
    condition: str = Field(..., description="Condição de mercado em que a estratégia é aplicável")
    config: StrategyConfigSummary = Field(..., description="Configuração da estratégia com parâmetros ajustáveis")
