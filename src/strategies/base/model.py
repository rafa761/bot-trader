# strategies/base/model.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel, Field

from core.logger import logger

if TYPE_CHECKING:
    from services.base.schemas import TradingSignal


class MarketCondition(str, Enum):
    """Enumeração das possíveis condições de mercado."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"


class StrategyConfig(BaseModel):
    """Configuração base para estratégias de mercado."""
    name: str = Field(..., description="Nome da estratégia")
    description: str = Field("", description="Descrição da estratégia")
    min_rr_ratio: float = Field(1.5, description="Razão mínima entre recompensa e risco")
    entry_threshold: float = Field(0.6, description="Pontuação mínima para entrada")

    # Parâmetros específicos para ajustes de ordens
    tp_adjustment: float = Field(1.0, description="Fator de ajuste do take profit")
    sl_adjustment: float = Field(1.0, description="Fator de ajuste do stop loss")
    entry_aggressiveness: float = Field(1.0, description="Fator de agressividade na entrada (1.0 = neutro)")

    # Limites de proteção
    max_sl_percent: float = Field(1.5, description="Percentual máximo permitido para stop loss")
    min_tp_percent: float = Field(0.5, description="Percentual mínimo exigido para take profit")

    # Indicadores requeridos
    required_indicators: list[str] = Field(
        default_factory=list,
        description="Lista de indicadores técnicos necessários para esta estratégia"
    )


class IMarketStrategy(ABC):
    """Interface para estratégias de mercado."""

    @abstractmethod
    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.

        Args:
            df: DataFrame com dados históricos
            mtf_data: Dados multi-timeframe

        Returns:
            bool: True se a estratégia deve ser ativada
        """
        pass

    @abstractmethod
    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> "TradingSignal | None":
        """
        Gera um sinal de trading com base na estratégia.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual
            mtf_data: Dados multi-timeframe

        Returns:
            Optional[TradingSignal]: Sinal de trading ou None
        """
        pass

    @abstractmethod
    def adjust_signal(self, signal: "TradingSignal", df: pd.DataFrame, mtf_data: dict) -> "TradingSignal":
        """
        Ajusta um sinal existente baseado nas condições de mercado.

        Args:
            signal: Sinal a ser ajustado
            df: DataFrame com dados históricos
            mtf_data: Dados multi-timeframe

        Returns:
            TradingSignal: Sinal ajustado
        """
        pass

    @abstractmethod
    def get_config(self) -> StrategyConfig:
        """
        Obtém a configuração da estratégia.

        Returns:
            StrategyConfig: Configuração atual
        """
        pass


class BaseStrategy(IMarketStrategy):
    """
    Implementação base para estratégias de mercado.
    Contém métodos comuns que podem ser reutilizados em estratégias específicas.
    """

    def __init__(self, config: StrategyConfig):
        """
        Inicializa a estratégia base.

        Args:
            config: Configuração da estratégia
        """
        self.config = config

    def get_config(self) -> StrategyConfig:
        """Retorna a configuração da estratégia."""
        return self.config

    def verify_indicators(self, df: pd.DataFrame) -> bool:
        """
        Verifica se todos os indicadores necessários estão presentes no DataFrame.

        Args:
            df: DataFrame a ser verificado

        Returns:
            bool: True se todos os indicadores estão presentes
        """
        missing_indicators = [i for i in self.config.required_indicators if i not in df.columns]
        if missing_indicators:
            logger.warning(
                f"Indicadores ausentes para estratégia {self.config.name}: {missing_indicators}"
            )
            return False
        return True

    @staticmethod
    def calculate_trend_direction(df: pd.DataFrame) -> str:
        """
        Calcula a direção atual da tendência com base nas médias móveis.

        Args:
            df: DataFrame com os dados históricos

        Returns:
            str: "UPTREND", "DOWNTREND" ou "NEUTRAL"
        """
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]

            if ema_short > ema_long:
                return "UPTREND"
            elif ema_short < ema_long:
                return "DOWNTREND"

        return "NEUTRAL"

    @staticmethod
    def calculate_volatility_level(df: pd.DataFrame) -> float:
        """
        Calcula o nível de volatilidade atual com base no ATR percentual.

        Args:
            df: DataFrame com os dados históricos

        Returns:
            float: Nível de volatilidade (0.0 a 1.0)
        """
        if 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100

            # Normalizar para uma escala de 0 a 1
            # Considerando que ATR acima de 3% é volatilidade extrema (1.0)
            # e ATR abaixo de 0.3% é volatilidade mínima (0.0)
            volatility = (atr_pct - 0.3) / (3.0 - 0.3)
            return max(0.0, min(1.0, volatility))

        return 0.5  # Valor padrão médio se ATR não estiver disponível

    def evaluate_entry_quality(
            self,
            df: pd.DataFrame,
            current_price: float,
            trade_direction: str,
            predicted_tp_pct: float = None,
            predicted_sl_pct: float = None,
            entry_threshold: float = None,
            mtf_alignment: float = None
    ) -> tuple[bool, float]:
        """
        Avalia a qualidade da entrada baseada em múltiplos critérios.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual do ativo
            trade_direction: "LONG" ou "SHORT"
            predicted_tp_pct: Take profit percentual previsto
            predicted_sl_pct: Stop loss percentual previsto
            entry_threshold: Limiar opcional para qualidade de entrada
            mtf_alignment: Score de alinhamento multi-timeframe (0-1)

        Returns:
            tuple[bool, float]: (Deve entrar, pontuação da entrada)
        """
        # Calcular razão risco-recompensa se tp e sl forem fornecidos
        if predicted_tp_pct is not None and predicted_sl_pct is not None and predicted_sl_pct > 0:
            rr_ratio = abs(predicted_tp_pct) / abs(predicted_sl_pct)

            # Verificar se RR é bom o suficiente
            should_enter = rr_ratio >= self.config.min_rr_ratio

            # Pontuação básica baseada em RR
            entry_score = min(1.0, rr_ratio / 3.0)  # Pontuação de 0 a 1
        else:
            # Valores padrão se tp e sl não forem fornecidos
            should_enter = False
            entry_score = 0.0

        # Verificações adicionais baseadas em indicadores
        trend_direction = self.calculate_trend_direction(df)
        if (trade_direction == "LONG" and trend_direction == "UPTREND") or \
                (trade_direction == "SHORT" and trend_direction == "DOWNTREND"):
            # Bônus para trades alinhados com a tendência
            entry_score = min(1.0, entry_score * 1.2)

        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if trade_direction == "LONG" and rsi < 40:
                # Bônus para trades LONG quando RSI está baixo (sobrevenda)
                entry_score = min(1.0, entry_score * 1.1)
            elif trade_direction == "SHORT" and rsi > 60:
                # Bônus para trades SHORT quando RSI está alto (sobrecompra)
                entry_score = min(1.0, entry_score * 1.1)

        # Bônus para alinhamento multi-timeframe forte
        if mtf_alignment is not None and mtf_alignment > 0.5:
            entry_score = min(1.0, entry_score * (1.0 + (mtf_alignment - 0.5)))

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> "TradingSignal | None":
        """
        Implementação padrão do método generate_signal que retorna None.
        Deve ser sobrescrito por estratégias concretas.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual
            mtf_data: Dados multi-timeframe

        Returns:
            TradingSignal ou None
        """
        return None
