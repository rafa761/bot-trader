# services/strategy_selector.py

import pandas as pd

from core.logger import logger
from models.lstm.model import LSTMModel
from strategies.base.model import IMarketStrategy, MarketCondition
from strategies.downtrend_strategy import DowntrendStrategy
from strategies.range_strategy import RangeStrategy
from strategies.uptrend_strategy import UptrendStrategy
from strategies.volatility_strategies import HighVolatilityStrategy, LowVolatilityStrategy


class StrategySelector:
    """
    Responsável por selecionar a estratégia mais adequada com base nas condições atuais do mercado.
    Implementa o padrão Strategy do GoF para permitir a troca dinâmica da estratégia em tempo de execução.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """Inicializa o seletor de estratégias com todas as estratégias disponíveis."""
        # Inicializar todas as estratégias
        self.strategies: dict[MarketCondition, IMarketStrategy] = {
            MarketCondition.UPTREND: UptrendStrategy(tp_model, sl_model),
            MarketCondition.DOWNTREND: DowntrendStrategy(tp_model, sl_model),
            MarketCondition.RANGE: RangeStrategy(tp_model, sl_model),
            MarketCondition.HIGH_VOLATILITY: HighVolatilityStrategy(tp_model, sl_model),
            MarketCondition.LOW_VOLATILITY: LowVolatilityStrategy(tp_model, sl_model)
        }

        # Estratégia atual
        self.current_strategy: IMarketStrategy | None = None
        self.current_condition: MarketCondition | None = None
        self.strategy_list: list[IMarketStrategy] = list(self.strategies.values())

        logger.info(f"Seletor de estratégias inicializado com {len(self.strategies)} estratégias")

    def select_strategy(self, df: pd.DataFrame, mtf_data: dict) -> IMarketStrategy | None:
        """
        Seleciona a estratégia mais adequada com base nas condições atuais do mercado.

        Args:
            df: DataFrame com dados históricos
            mtf_data: Dados multi-timeframe

        Returns:
            A estratégia selecionada ou None se nenhuma estratégia for adequada
        """
        # Verificar volatilidade primeiro (tem precedência)
        if self.strategies[MarketCondition.HIGH_VOLATILITY].should_activate(df, mtf_data):
            selected_condition = MarketCondition.HIGH_VOLATILITY
            selected_strategy = self.strategies[selected_condition]

        elif self.strategies[MarketCondition.LOW_VOLATILITY].should_activate(df, mtf_data):
            selected_condition = MarketCondition.LOW_VOLATILITY
            selected_strategy = self.strategies[selected_condition]

        # Depois verificar tendência
        elif self.strategies[MarketCondition.UPTREND].should_activate(df, mtf_data):
            selected_condition = MarketCondition.UPTREND
            selected_strategy = self.strategies[selected_condition]

        elif self.strategies[MarketCondition.DOWNTREND].should_activate(df, mtf_data):
            selected_condition = MarketCondition.DOWNTREND
            selected_strategy = self.strategies[selected_condition]

        # Por fim, verificar consolidação
        elif self.strategies[MarketCondition.RANGE].should_activate(df, mtf_data):
            selected_condition = MarketCondition.RANGE
            selected_strategy = self.strategies[selected_condition]

        else:
            # Se nenhuma estratégia for ativada, manter a atual ou usar a de uptrend como padrão
            if self.current_strategy is not None:
                logger.info("Mantendo estratégia atual, nenhuma nova condição detectada")
                return self.current_strategy

            logger.info("Nenhuma estratégia ativada, usando UPTREND como padrão")
            selected_condition = MarketCondition.UPTREND
            selected_strategy = self.strategies[selected_condition]

        # Verificar se houve mudança de estratégia
        if self.current_condition != selected_condition:
            logger.info(
                f"Mudança de estratégia: {self.current_condition} -> {selected_condition} "
                f"({selected_strategy.get_config().name})"
            )

            # Atualizar a estratégia atual
            self.current_strategy = selected_strategy
            self.current_condition = selected_condition

        return selected_strategy

    def get_current_strategy(self) -> IMarketStrategy | None:
        """
        Retorna a estratégia atualmente selecionada.

        Returns:
            A estratégia atual ou None se nenhuma estiver selecionada
        """
        return self.current_strategy

    def get_current_condition(self) -> MarketCondition | None:
        """
        Retorna a condição de mercado atualmente detectada.

        Returns:
            A condição atual ou None se nenhuma estiver detectada
        """
        return self.current_condition
