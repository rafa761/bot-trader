# services/base/schemas.py

from datetime import datetime
from typing import Literal, Any

import numpy as np
from pydantic import BaseModel, Field

rng = np.random.default_rng(seed=42)


class TradingParameters(BaseModel):
    """Parâmetros ajustados para trading com base no padrão de mercado."""
    entry_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Limiar para score de entrada")
    tp_adjustment_factor: float = Field(1.0, gt=0.0, description="Fator de ajuste para take profit")
    sl_adjustment_factor: float = Field(1.0, gt=0.0, description="Fator de ajuste para stop loss")


class TradingSignal(BaseModel):
    """Sinal de trading completo gerado pelos modelos."""
    id: str = Field(
        default_factory=lambda: f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{rng.integers(1000, 9999)}")
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
    mtf_trend: str | None = Field(None, description="Tendência consolidada multi-timeframe")
    mtf_alignment: float | None = Field(None, ge=0.0, le=1.0, description="Score de alinhamento multi-timeframe")
    mtf_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confiança na análise multi-timeframe")
    mtf_details: dict[str, Any] | None = Field(None, description="Detalhes da análise multi-timeframe por timeframe")
    atr_value: float | None = Field(None, description="Valor atual do ATR, se disponível")
    entry_score: float | None = Field(None, ge=0, le=1, description="Pontuação de qualidade da entrada (0-1)")
    rr_ratio: float | None = Field(None, gt=0, description="Razão risco/recompensa calculada")
    market_trend: str | None = Field(None, description="Tendência do mercado (UPTREND, DOWNTREND, NEUTRAL)")
    market_strength: str | None = Field(None, description="Força da tendência (STRONG_TREND, WEAK_TREND)")
    timestamp: datetime = Field(default_factory=datetime.now)


class OrderResult(BaseModel):
    """Resultado da execução de uma ordem."""
    success: bool = Field(..., description="Se a ordem foi executada com sucesso")
    order_id: str | None = Field(None, description="ID da ordem, se bem-sucedida")
    error_message: str | None = Field(None, description="Mensagem de erro, se falhou")


class TPSLResult(BaseModel):
    """Resultado da colocação de ordens TP/SL."""
    tp_order: dict | None = Field(None, description="Resultado da ordem TP")
    sl_order: dict | None = Field(None, description="Resultado da ordem SL")


# Schemas para análise de mercado e estratégias
class TimeFrameSummary(BaseModel):
    strength: str = Field(..., description="Força e direção da tendência (ex: 'STRONG_UPTREND', 'WEAK_DOWNTREND')")
    score: float = Field(..., description="Pontuação numérica da tendência, de -1.0 (forte baixa) a 1.0 (forte alta)")


class MultiTimeFrameDetails(BaseModel):
    consolidated_trend: str = Field(..., description="Tendência consolidada considerando múltiplos timeframes")
    trend_score: float = Field(..., description="Pontuação consolidada da tendência, de -1.0 a 1.0")
    confidence: float = Field(..., description="Nível de confiança na análise, de 0.0 a 100.0")
    tf_summary: dict[str, TimeFrameSummary] = Field(..., description="Resumo da análise por timeframe")
    tf_details: dict[str, dict[str, Any]] = Field(..., description="Detalhes técnicos da análise por timeframe")
    tf_agreement: float = Field(..., description="Grau de concordância entre os diferentes timeframes, de 0.0 a 1.0")


class MarketAnalysisResult(BaseModel):
    mtf_trend: str = Field(..., description="Tendência identificada pela análise multi-timeframe")
    mtf_confidence: float = Field(..., description="Confiança na análise multi-timeframe, de 0.0 a 100.0")
    mtf_details: MultiTimeFrameDetails = Field(..., description="Detalhes completos da análise multi-timeframe")
    strategy_name: str = Field(..., description="Nome da estratégia selecionada com base na análise")
    strategy: Any | None = Field(None,
                                 description="Instância da estratégia selecionada ou None se nenhuma for aplicável")


# Schemas para avaliação de trades
class TradeEvaluation(BaseModel):
    direction: str = Field(..., description="Direção do trade avaliado ('LONG' ou 'SHORT')")
    alignment_score: float = Field(..., description="Pontuação de alinhamento com a tendência de mercado, de 0.0 a 1.0")
    confidence: float = Field(..., description="Nível de confiança na avaliação, de 0.0 a 1.0")
    quality_score: float = Field(..., description="Pontuação geral de qualidade do trade, de 0.0 a 1.0")
    alignment_category: str = Field(...,
                                    description="Categoria de alinhamento ('EXCELLENT', 'GOOD', 'MODERATE', 'POOR', 'VERY_POOR')")
    favorable: bool = Field(..., description="Se o trade é favorável considerando as condições atuais de mercado")
    strong_signal: bool = Field(..., description="Se o sinal é considerado forte (score >= 0.7)")


# Schemas para results de trades
class TradeResultDetails(BaseModel):
    result: str = Field(..., description="Resultado do trade ('TP' para take profit ou 'SL' para stop loss)")
    actual_tp_pct: float = Field(..., description="Percentual real de take profit atingido")
    actual_sl_pct: float = Field(..., description="Percentual real de stop loss atingido")
    exit_price: float = Field(..., description="Preço de saída do trade")


# Schema para ordens executadas
class ExecutedOrder(BaseModel):
    signal_id: str = Field(..., description="ID do sinal que gerou a ordem")
    order_id: str = Field(..., description="ID da ordem executada na corretora")
    direction: str = Field(..., description="Direção do trade ('LONG' ou 'SHORT')")
    entry_price: float = Field(..., description="Preço de entrada do trade")
    tp_price: float = Field(..., description="Preço alvo para take profit")
    sl_price: float = Field(..., description="Preço alvo para stop loss")
    predicted_tp_pct: float = Field(..., description="Previsão percentual de take profit")
    predicted_sl_pct: float = Field(..., description="Previsão percentual de stop loss")
    timestamp: datetime = Field(..., description="Data e hora da execução da ordem")
    filled: bool = Field(..., description="Se a ordem foi preenchida completamente")
    processed: bool = Field(..., description="Se a ordem já foi processada pelo sistema")
    position_side: str = Field(..., description="Lado da posição ('LONG' ou 'SHORT')")
