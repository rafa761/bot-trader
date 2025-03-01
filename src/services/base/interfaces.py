# services/base/interfaces.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from services.base.schemas import TradingSignal, OrderResult
    from services.performance_monitor import Trade, PerformanceMetrics


class IOrderExecutor(ABC):
    """
    Interface para executores de ordens seguindo o princípio ISP (Interface Segregation Principle).
    """

    @abstractmethod
    async def check_positions(self) -> bool:
        """Verifica se existem posições abertas."""
        pass

    @abstractmethod
    async def execute_order(self, signal: "TradingSignal") -> "OrderResult":
        """Executa uma ordem com base no sinal fornecido."""
        pass

    @abstractmethod
    def get_executed_orders(self) -> list[dict[str, Any]]:
        """Retorna a lista de ordens executadas."""
        pass

    @abstractmethod
    def mark_order_as_processed(self, order_id: str) -> bool:
        """Marca uma ordem como processada."""
        pass


class ITradeProcessor(ABC):
    """
    Interface para processadores de trades seguindo o princípio ISP.
    """

    @abstractmethod
    async def process_completed_trades(self) -> None:
        """Processa trades completados e atualiza o monitor de performance."""
        pass


class IPerformanceMonitor(ABC):
    """
    Interface para monitores de performance seguindo o princípio ISP.
    """

    @abstractmethod
    def add_trade(self, trade: "Trade") -> None:
        """Adiciona um novo trade ao monitor."""
        pass

    @abstractmethod
    def update_trade(self, trade: "Trade") -> None:
        """Atualiza um trade existente."""
        pass

    @abstractmethod
    def get_trade(self, trade_id: str) -> "Trade | None":
        """Recupera um trade pelo ID."""
        pass

    @abstractmethod
    def get_trade_by_signal_id(self, signal_id: str) -> "Trade | None":
        """Recupera um trade pelo signal_id."""
        pass

    @abstractmethod
    def register_trade_exit(self, trade_id: str, exit_price: float, exit_time: datetime = None) -> None:
        """Registra a saída de um trade."""
        pass

    @abstractmethod
    def calculate_metrics(self) -> "PerformanceMetrics":
        """Calcula métricas de performance com base nos trades registrados."""
        pass

    @abstractmethod
    def get_metrics(self) -> "PerformanceMetrics":
        """Retorna as métricas atuais de performance."""
        pass

    @abstractmethod
    def get_trades_dataframe(self) -> Any:  # Retorna um pandas.DataFrame
        """Converte todos os trades para um DataFrame do Pandas."""
        pass
