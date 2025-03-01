# services\base\services.py

from abc import ABC, abstractmethod

import pandas as pd

from services.base.schemas import OrderResult
from services.base.schemas import TradingSignal


class MarketDataProvider(ABC):
    """Interface para provedores de dados de mercado."""

    @abstractmethod
    async def initialize(self) -> None:
        """Inicializa o provedor de dados."""
        pass

    @abstractmethod
    async def get_latest_data(self) -> pd.DataFrame:
        """Obtém os dados mais recentes do mercado."""
        pass

    @abstractmethod
    def get_historical_data(self) -> pd.DataFrame:
        """Retorna os dados históricos armazenados."""
        pass


class SignalGenerator(ABC):
    """Interface para geradores de sinais de trading."""

    @abstractmethod
    async def generate_signal(self, df: pd.DataFrame, current_price: float) -> TradingSignal | None:
        """Gera um sinal de trading baseado nos dados fornecidos."""
        pass


class OrderExecutor(ABC):
    """Interface para executores de ordens."""

    @abstractmethod
    async def check_positions(self) -> bool:
        """Verifica se existem posições abertas."""
        pass

    @abstractmethod
    async def execute_order(self, signal: TradingSignal) -> OrderResult:
        """Executa uma ordem baseada no sinal fornecido."""
        pass
