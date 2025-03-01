# strategies/strategy_manager.py

from typing import Tuple, Dict, Any

import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from services.trend_analyzer import MultiTimeFrameTrendAnalyzer
from strategies.strategy_selector import StrategySelector


class StrategyManager:
    """
    Gerenciador unificado para estratégias adaptativas que coordena todo o
    processo de análise, seleção de estratégia e ajuste de sinais.

    Esta classe centraliza todas as decisões relacionadas à estratégia,
    evitando duplicação de análises e conflitos entre componentes.
    """

    def __init__(self):
        """Inicializa o gerenciador de estratégias."""
        self.strategy_selector = StrategySelector()
        logger.info("Gerenciador de estratégias inicializado")

        # Armazenar dados da última análise multi-timeframe
        self.last_mtf_data = {}
        self.last_strategy_name = "Nenhuma"

    async def process_market_data(self, df: pd.DataFrame, mtf_analyzer: MultiTimeFrameTrendAnalyzer) -> dict[str, Any]:
        """
        Processa os dados de mercado, realizando análise multi-timeframe e selecionando a estratégia adequada.

        Args:
            df: DataFrame com dados históricos
            mtf_analyzer: Analisador multi-timeframe

        Returns:
            Dicionário com dados da análise incluindo a estratégia selecionada
        """
        try:
            # Realizar análise multi-timeframe
            mtf_trend, confidence, details = await mtf_analyzer.analyze_multi_timeframe_trend()

            # Armazenar para uso posterior
            self.last_mtf_data = details

            # Selecionar estratégia com base nos dados analisados
            strategy = self.strategy_selector.select_strategy(df, details)

            if strategy:
                self.last_strategy_name = strategy.get_config().name
                logger.info(f"Estratégia selecionada: {self.last_strategy_name}")
            else:
                logger.warning("Nenhuma estratégia selecionada")
                self.last_strategy_name = "Nenhuma"

            result = {
                "mtf_trend": mtf_trend,
                "mtf_confidence": confidence,
                "mtf_details": details,
                "strategy_name": self.last_strategy_name,
                "strategy": strategy
            }

            return result
        except Exception as e:
            logger.error(f"Erro ao processar dados de mercado: {e}", exc_info=True)
            return {
                "mtf_trend": "NEUTRAL",
                "mtf_confidence": 0.0,
                "mtf_details": {},
                "strategy_name": "Fallback",
                "strategy": None
            }

    async def evaluate_signal(self,
                              signal: TradingSignal,
                              df: pd.DataFrame,
                              mtf_analyzer: MultiTimeFrameTrendAnalyzer) -> tuple[TradingSignal, bool]:
        """
        Avalia e aprimora um sinal de trading usando a estratégia mais adequada.
        Centraliza toda a lógica de decisão e ajuste em um único lugar.

        Args:
            signal: Sinal original
            df: DataFrame com dados históricos
            mtf_analyzer: Analisador multi-timeframe

        Returns:
            Tupla com (sinal ajustado, deve executar)
        """
        if signal is None:
            logger.info("Nenhum sinal para avaliar")
            return None, False

        try:
            # Enriquecer sinal com análise multi-timeframe
            alignment_score = 0.5  # Valor padrão

            if hasattr(signal, 'direction') and signal.direction:
                alignment_score, confidence = await mtf_analyzer.get_trend_alignment(signal.direction)

                # Adicionar dados MTF ao sinal
                signal.mtf_alignment = alignment_score
                signal.mtf_confidence = confidence

                # Se tivermos dados MTF da última análise, adicionar ao sinal
                if self.last_mtf_data:
                    if 'consolidated_trend' in self.last_mtf_data:
                        signal.mtf_trend = self.last_mtf_data['consolidated_trend']
                    if 'tf_summary' in self.last_mtf_data:
                        signal.mtf_details = self.last_mtf_data['tf_summary']

            # Obter a estratégia selecionada
            strategy = self.strategy_selector.get_current_strategy()

            if not strategy:
                # Selecionar uma estratégia agora se não tiver uma
                strategy = self.strategy_selector.select_strategy(df, self.last_mtf_data)

            if strategy:
                # Ajustar o sinal com base na estratégia
                signal = strategy.adjust_signal(signal, df, self.last_mtf_data)
                logger.info(f"Sinal ajustado pela estratégia: {strategy.get_config().name}")

                # Verificar se o sinal atende aos critérios mínimos da estratégia
                config = strategy.get_config()

                # Se o sinal tem um score de entrada, usar para decisão
                if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                    should_execute = signal.entry_score >= config.entry_threshold

                    if not should_execute:
                        logger.info(
                            f"Sinal rejeitado: score ({signal.entry_score:.2f}) < threshold ({config.entry_threshold:.2f})"
                        )
                        return signal, False

                # Verificar se a razão risk:reward é adequada
                if hasattr(signal, 'rr_ratio') and signal.rr_ratio is not None:
                    if signal.rr_ratio < config.min_rr_ratio:
                        logger.info(
                            f"Sinal rejeitado: R:R ({signal.rr_ratio:.2f}) < mínimo ({config.min_rr_ratio:.2f})"
                        )
                        return signal, False

                # Verificar alinhamento com MTF se disponível
                min_alignment = 0.3  # Valor mínimo para alinhamento MTF
                if hasattr(signal, 'mtf_alignment') and signal.mtf_alignment is not None:
                    if signal.mtf_alignment < min_alignment:
                        logger.info(
                            f"Sinal rejeitado: baixo alinhamento MTF ({signal.mtf_alignment:.2f} < {min_alignment})"
                        )
                        return signal, False

                # Se passou por todas as verificações, sinal aprovado
                logger.info(
                    f"Sinal APROVADO pela estratégia {strategy.get_config().name}: "
                    f"Score={getattr(signal, 'entry_score', 'N/A')}, "
                    f"R:R={getattr(signal, 'rr_ratio', 'N/A')}, "
                    f"MTF={getattr(signal, 'mtf_alignment', 'N/A')}"
                )
                return signal, True
            else:
                logger.warning("Nenhuma estratégia disponível para avaliar o sinal")
                return signal, False

        except Exception as e:
            logger.error(f"Erro ao avaliar sinal: {e}", exc_info=True)
            return signal, False

    def get_strategy_details(self) -> dict:
        """
        Retorna detalhes sobre a estratégia atual para logging e monitoramento.

        Returns:
            Dicionário com detalhes da estratégia atual
        """
        strategy = self.strategy_selector.get_current_strategy()
        condition = self.strategy_selector.get_current_condition()

        if strategy is None or condition is None:
            return {
                "active": False,
                "name": "Nenhuma",
                "condition": "Desconhecida",
                "config": {}
            }

        config = strategy.get_config()

        return {
            "active": True,
            "name": config.name,
            "condition": condition.value,
            "config": {
                "min_rr_ratio": config.min_rr_ratio,
                "entry_threshold": config.entry_threshold,
                "tp_adjustment": config.tp_adjustment,
                "sl_adjustment": config.sl_adjustment,
                "entry_aggressiveness": config.entry_aggressiveness
            }
        }
