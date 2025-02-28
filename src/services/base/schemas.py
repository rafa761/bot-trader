# services/base/schemas.py
from typing import Literal

from pydantic import BaseModel, Field


class TradingParameters(BaseModel):
    """Parâmetros ajustados para trading com base no padrão de mercado."""
    entry_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Limiar para score de entrada")
    tp_adjustment_factor: float = Field(1.0, gt=0.0, description="Fator de ajuste para take profit")
    sl_adjustment_factor: float = Field(1.0, gt=0.0, description="Fator de ajuste para stop loss")


class TradingSignal(BaseModel):
    """Sinal de trading completo gerado pelos modelos."""
    id: str = Field(..., description="Identificador único do sinal")
    direction: Literal["LONG", "SHORT"] = Field(..., description="Direção do trade")
    side: Literal["BUY", "SELL"] = Field(..., description="Lado da ordem (BUY/SELL)")
    position_side: Literal["LONG", "SHORT"] = Field(..., description="Lado da posição (LONG/SHORT)")
    predicted_tp_pct: float = Field(..., description="Previsão percentual de take profit")
    predicted_sl_pct: float = Field(..., description="Previsão percentual de stop loss")
    tp_price: float = Field(..., gt=0, description="Preço calculado para take profit")
    sl_price: float = Field(..., gt=0, description="Preço calculado para stop loss")
    current_price: float = Field(..., gt=0, description="Preço atual do ativo")
    tp_factor: float = Field(..., gt=0, description="Fator multiplicador para take profit")
    sl_factor: float = Field(..., gt=0, description="Fator multiplicador para stop loss")
    atr_value: float | None = Field(None, description="Valor atual do ATR, se disponível")
    entry_score: float | None = Field(None, ge=0, le=1, description="Pontuação de qualidade da entrada (0-1)")
    rr_ratio: float | None = Field(None, gt=0, description="Razão risco/recompensa calculada")
    timestamp: object = Field(None, description="Timestamp de criação do sinal")


class OrderResult(BaseModel):
    """Resultado da execução de uma ordem."""
    success: bool = Field(..., description="Se a ordem foi executada com sucesso")
    order_id: str | None = Field(None, description="ID da ordem, se bem-sucedida")
    error_message: str | None = Field(None, description="Mensagem de erro, se falhou")


class TPSLResult(BaseModel):
    """Resultado da colocação de ordens TP/SL."""
    tp_order: dict | None = Field(None, description="Resultado da ordem TP")
    sl_order: dict | None = Field(None, description="Resultado da ordem SL")
