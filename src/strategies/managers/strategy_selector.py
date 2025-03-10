# services/managers/strategy_selector.py

import pandas as pd

from core.logger import logger
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

    def __init__(self):
        """Inicializa o seletor de estratégias com todas as estratégias disponíveis."""
        # Inicializar todas as estratégias
        self.strategies: dict[MarketCondition, IMarketStrategy] = {
            MarketCondition.UPTREND: UptrendStrategy(),
            MarketCondition.DOWNTREND: DowntrendStrategy(),
            MarketCondition.RANGE: RangeStrategy(),
            MarketCondition.HIGH_VOLATILITY: HighVolatilityStrategy(),
            MarketCondition.LOW_VOLATILITY: LowVolatilityStrategy()
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
        high_volatility = self.strategies[MarketCondition.HIGH_VOLATILITY].should_activate(df, mtf_data)
        low_volatility = self.strategies[MarketCondition.LOW_VOLATILITY].should_activate(df, mtf_data)
        uptrend = self.strategies[MarketCondition.UPTREND].should_activate(df, mtf_data)
        downtrend = self.strategies[MarketCondition.DOWNTREND].should_activate(df, mtf_data)
        range_market = self.strategies[MarketCondition.RANGE].should_activate(df, mtf_data)

        # Lógica de decisão com prioridade e casos específicos
        if high_volatility:
            selected_condition = MarketCondition.HIGH_VOLATILITY
            selected_strategy = self.strategies[selected_condition]
            logger.info(f"Alta volatilidade detectada - ativando estratégia específica")
        elif low_volatility and (uptrend or downtrend):
            # Em baixa volatilidade, se houver tendência, priorizar a estratégia de tendência
            if uptrend:
                selected_condition = MarketCondition.UPTREND
            else:
                selected_condition = MarketCondition.DOWNTREND
            selected_strategy = self.strategies[selected_condition]
            logger.info(f"Tendência em baixa volatilidade - priorizando estratégia de tendência")
        elif low_volatility:
            selected_condition = MarketCondition.LOW_VOLATILITY
            selected_strategy = self.strategies[selected_condition]
            logger.info(f"Baixa volatilidade detectada - ativando estratégia específica")
        elif uptrend:
            selected_condition = MarketCondition.UPTREND
            selected_strategy = self.strategies[selected_condition]
        elif downtrend:
            selected_condition = MarketCondition.DOWNTREND
            selected_strategy = self.strategies[selected_condition]
        elif range_market:
            selected_condition = MarketCondition.RANGE
            selected_strategy = self.strategies[selected_condition]
        else:
            # Se nenhuma estratégia for ativada, usar a de range como padrão
            # (mais conservadora e adequada para mercados indefinidos)
            if self.current_strategy is not None:
                logger.info("Mantendo estratégia atual, nenhuma nova condição detectada")
                return self.current_strategy

            logger.info("Nenhuma estratégia ativada, usando RANGE como padrão")
            selected_condition = MarketCondition.RANGE
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
