# strategies/strategy_selector.py
from typing import Optional

import pandas as pd

from core.logger import logger
from strategies.base.model import IMarketStrategy, MarketCondition
from strategies.base.schemas import MTFAnalysisInfo, MarketConditionResult, StrategySelectionContext
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
        # Extrair informações do MTF para uso em todas as decisões
        mtf_info = MTFAnalysisInfo.from_mtf_data(mtf_data)

        # Avaliar condições de mercado
        market_conditions = self._evaluate_market_conditions(df, mtf_data, mtf_info)

        # Criar contexto para seleção
        context = StrategySelectionContext(
            mtf_info=mtf_info,
            market_conditions=market_conditions
        )

        # Selecionar estratégia com base no contexto
        selected_context = self._select_based_on_context(context)

        # Atualizar e retornar a estratégia selecionada
        return self._update_current_strategy(selected_context)

    def _evaluate_market_conditions(
            self, df: pd.DataFrame, mtf_data: dict, mtf_info: MTFAnalysisInfo
    ) -> MarketConditionResult:
        """Avalia as diferentes condições de mercado."""
        # Verificar ativação de cada estratégia com filtro MTF
        high_vol = self.strategies[MarketCondition.HIGH_VOLATILITY].should_activate(df, mtf_data)
        low_vol = self.strategies[MarketCondition.LOW_VOLATILITY].should_activate(df, mtf_data)
        uptrend = (self.strategies[MarketCondition.UPTREND].should_activate(df, mtf_data) and
                   not mtf_info.is_downtrend)
        downtrend = (self.strategies[MarketCondition.DOWNTREND].should_activate(df, mtf_data) and
                     not mtf_info.is_uptrend)
        range_market = self.strategies[MarketCondition.RANGE].should_activate(df, mtf_data)

        # Calcular alinhamento MTF (0-1)
        mtf_alignment = 0.0
        if uptrend and mtf_info.is_uptrend:
            mtf_alignment = mtf_info.strength
        elif downtrend and mtf_info.is_downtrend:
            mtf_alignment = mtf_info.strength
        elif range_market and mtf_info.is_neutral:
            mtf_alignment = 0.7  # Bom alinhamento para range em mercado neutro

        return MarketConditionResult(
            high_volatility=high_vol,
            low_volatility=low_vol,
            uptrend=uptrend,
            downtrend=downtrend,
            range_market=range_market,
            mtf_alignment=mtf_alignment
        )

    def _select_based_on_context(self, context: StrategySelectionContext) -> StrategySelectionContext:
        """Seleciona a estratégia com base no contexto avaliado."""
        conditions = context.market_conditions
        mtf_info = context.mtf_info

        # Volatilidade alta tem precedência
        if conditions.high_volatility:
            context.condition = MarketCondition.HIGH_VOLATILITY.value
            context.log_message = "Alta volatilidade detectada - ativando estratégia específica"
            return context

        # Tendência forte no MTF tem precedência secundária
        if mtf_info.is_strong_downtrend and mtf_info.strength > 0.6:
            context.condition = MarketCondition.DOWNTREND.value
            context.log_message = "Tendência de baixa forte no MTF - forçando estratégia de baixa"
            return context

        if mtf_info.is_strong_uptrend and mtf_info.strength > 0.6:
            context.condition = MarketCondition.UPTREND.value
            context.log_message = "Tendência de alta forte no MTF - forçando estratégia de alta"
            return context

        # Combinações de baixa volatilidade com tendência
        if conditions.low_volatility:
            return self._handle_low_volatility_case(context)

        # Tendências simples
        if conditions.uptrend:
            context.condition = MarketCondition.UPTREND.value
            context.log_message = "Tendência de alta alinhada com MTF"
            return context

        if conditions.downtrend:
            context.condition = MarketCondition.DOWNTREND.value
            context.log_message = "Tendência de baixa alinhada com MTF"
            return context

        # Range
        if conditions.range_market:
            context.condition = MarketCondition.RANGE.value
            context.log_message = "Mercado em range detectado"
            return context

        # Fallback para MTF ou estratégia atual
        return self._handle_fallback_case(context)

    def _handle_low_volatility_case(self, context: StrategySelectionContext) -> StrategySelectionContext:
        """Lida com o caso específico de baixa volatilidade."""
        conditions = context.market_conditions
        mtf_info = context.mtf_info

        if conditions.uptrend and (mtf_info.is_uptrend or mtf_info.is_neutral):
            context.condition = MarketCondition.UPTREND.value
            context.log_message = "Tendência de alta em baixa volatilidade - alinhada com MTF"
            return context

        if conditions.downtrend and (mtf_info.is_downtrend or mtf_info.is_neutral):
            context.condition = MarketCondition.DOWNTREND.value
            context.log_message = "Tendência de baixa em baixa volatilidade - alinhada com MTF"
            return context

        context.condition = MarketCondition.LOW_VOLATILITY.value
        context.log_message = "Baixa volatilidade detectada - ativando estratégia específica"
        return context

    def _handle_fallback_case(self, context: StrategySelectionContext) -> StrategySelectionContext:
        """Lida com o caso quando nenhuma condição principal é atendida."""
        mtf_info = context.mtf_info

        if mtf_info.is_downtrend and mtf_info.strength > 0.4:
            context.condition = MarketCondition.DOWNTREND.value
            context.log_message = "Usando estratégia de baixa baseada apenas no MTF"
            return context

        if mtf_info.is_uptrend and mtf_info.strength > 0.4:
            context.condition = MarketCondition.UPTREND.value
            context.log_message = "Usando estratégia de alta baseada apenas no MTF"
            return context

        if self.current_strategy is not None:
            context.condition = self.current_condition.value if self.current_condition else None
            context.log_message = "Mantendo estratégia atual, nenhuma nova condição detectada"
            return context

        context.condition = MarketCondition.RANGE.value
        context.log_message = "Nenhuma estratégia ativada, usando RANGE como padrão"
        return context

    def _update_current_strategy(self, context: StrategySelectionContext) -> IMarketStrategy | None:
        """Atualiza a estratégia atual se houver mudança."""
        if not context.condition:
            return self.current_strategy

        try:
            # Converter string para enum
            selected_condition = MarketCondition(context.condition)
            selected_strategy = self.strategies[selected_condition]

            logger.info(context.log_message)

            if self.current_condition != selected_condition:
                logger.info(
                    f"Mudança de estratégia: {self.current_condition} -> {selected_condition} "
                    f"({selected_strategy.get_config().name})"
                )

                self.current_strategy = selected_strategy
                self.current_condition = selected_condition

            return selected_strategy

        except (ValueError, KeyError) as e:
            logger.error(f"Erro ao selecionar estratégia: {e}")
            return self.current_strategy

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
