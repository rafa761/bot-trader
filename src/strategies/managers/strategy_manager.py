# strategies/managers/strategy_manager.py

import pandas as pd

from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.schemas import MarketAnalysisResult, MultiTimeFrameDetails, TimeFrameSummary, TradingSignal
from services.trend_analyzer import MultiTimeFrameTrendAnalyzer
from strategies.base.schemas import StrategyConfigSummary, StrategyDetails
from strategies.managers.strategy_selector import StrategySelector


class StrategyManager:
    """
    Gerenciador unificado para estratégias adaptativas que coordena todo o
    processo de análise, seleção de estratégia e ajuste de sinais.

    Esta classe centraliza todas as decisões relacionadas à estratégia,
    evitando duplicação de análises e conflitos entre componentes.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """Inicializa o gerenciador de estratégias."""
        self.strategy_selector = StrategySelector(tp_model, sl_model)
        logger.info("Gerenciador de estratégias inicializado")

        # Armazenar dados da última análise multi-timeframe
        self.last_mtf_data = {}
        self.last_strategy_name = "Nenhuma"

    async def process_market_data(self, df: pd.DataFrame,
                                  mtf_analyzer: MultiTimeFrameTrendAnalyzer) -> MarketAnalysisResult:
        """
        Processa os dados de mercado, realizando análise multi-timeframe e selecionando a estratégia adequada.

        Args:
            df: DataFrame com dados históricos
            mtf_analyzer: Analisador multi-timeframe

        Returns:
            MarketAnalysisResult com dados da análise incluindo a estratégia selecionada
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

            # Converter os detalhes da análise multi-timeframe para schema Pydantic
            mtf_details = MultiTimeFrameDetails(
                consolidated_trend=details["consolidated_trend"],
                trend_score=details["trend_score"],
                confidence=details["confidence"],
                tf_summary={
                    k: TimeFrameSummary(strength=v["strength"], score=v["score"])
                    for k, v in details["tf_summary"].items()
                },
                tf_details=details["tf_details"],
                tf_agreement=details["tf_agreement"]
            )

            # Criar e retornar o resultado da análise usando schema Pydantic
            return MarketAnalysisResult(
                mtf_trend=mtf_trend,
                mtf_confidence=confidence,
                mtf_details=mtf_details,
                strategy_name=self.last_strategy_name,
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Erro ao processar dados de mercado: {e}", exc_info=True)
            # Em caso de erro, retornar um resultado padrão
            fallback_details = MultiTimeFrameDetails(
                consolidated_trend="NEUTRAL",
                trend_score=0.0,
                confidence=0.0,
                tf_summary={},
                tf_details={},
                tf_agreement=0.0
            )

            return MarketAnalysisResult(
                mtf_trend="NEUTRAL",
                mtf_confidence=0.0,
                mtf_details=fallback_details,
                strategy_name="Fallback",
                strategy=None
            )

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

                # Adicionar log detalhado sobre alinhamento MTF
                logger.info(
                    f"Sinal {signal.direction} avaliado - Alinhamento MTF: {alignment_score:.2f}, "
                    f"Confiança: {confidence:.2f}. Tendência: {self.last_mtf_data.get('consolidated_trend', 'N/A')}"
                )

            # Obter a estratégia selecionada
            strategy = self.strategy_selector.get_current_strategy()

            if not strategy:
                # Selecionar uma estratégia agora se não tiver uma
                strategy = self.strategy_selector.select_strategy(df, self.last_mtf_data)

            if strategy:
                # Ajustar o sinal com base na estratégia
                signal = strategy.adjust_signal(signal, df, self.last_mtf_data)
                logger.info(
                    f"Sinal ajustado pela estratégia: {strategy.get_config().name} - "
                    f"TP: {signal.predicted_tp_pct:.2f}%, SL: {signal.predicted_sl_pct:.2f}%, "
                    f"R:R: {signal.rr_ratio:.2f}"
                )

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
                    else:
                        logger.info(
                            f"Sinal aprovado: score ({signal.entry_score:.2f}) >= threshold ({config.entry_threshold:.2f})"
                        )

                # Verificar se a razão risk:reward é adequada
                if hasattr(signal, 'rr_ratio') and signal.rr_ratio is not None:
                    if signal.rr_ratio < config.min_rr_ratio:
                        logger.info(
                            f"Sinal rejeitado: R:R ({signal.rr_ratio:.2f}) < mínimo ({config.min_rr_ratio:.2f})"
                        )
                        return signal, False
                    else:
                        logger.info(
                            f"Sinal aprovado: R:R ({signal.rr_ratio:.2f}) >= mínimo ({config.min_rr_ratio:.2f})"
                        )

                # Verificar alinhamento com MTF se disponível
                # Usar um alinhamento mínimo mais baixo para estratégias em alta volatilidade
                min_alignment = 0.2 if config.name.startswith("High Volatility") else 0.3
                if hasattr(signal, 'mtf_alignment') and signal.mtf_alignment is not None:
                    if signal.mtf_alignment < min_alignment:
                        logger.info(
                            f"Sinal rejeitado: baixo alinhamento MTF ({signal.mtf_alignment:.2f} < {min_alignment})"
                        )
                        return signal, False
                    else:
                        logger.info(
                            f"Sinal aprovado: alinhamento MTF adequado ({signal.mtf_alignment:.2f} >= {min_alignment})"
                        )

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

    async def generate_signal(self, df: pd.DataFrame, current_price: float) -> TradingSignal | None:
        """
        Gera um sinal de trading utilizando a estratégia mais adequada.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual do ativo

        Returns:
            TradingSignal: Sinal de trading gerado ou None se não houver sinal
        """
        # Obter a estratégia atual
        strategy = self.strategy_selector.get_current_strategy()

        if not strategy:
            strategy = self.strategy_selector.select_strategy(df, self.last_mtf_data)

        if not strategy:
            logger.warning("Nenhuma estratégia disponível para gerar sinal")
            return None

        # Deixar a estratégia gerar o sinal
        signal = await strategy.generate_signal(df, current_price, self.last_mtf_data)
        return signal


    def get_strategy_details(self) -> StrategyDetails:
        """
        Retorna detalhes sobre a estratégia atual para logging e monitoramento.

        Returns:
            StrategyDetails com detalhes da estratégia atual
        """
        strategy = self.strategy_selector.get_current_strategy()
        condition = self.strategy_selector.get_current_condition()

        if strategy is None or condition is None:
            return StrategyDetails(
                active=False,
                name="Nenhuma",
                condition="Desconhecida",
                config=StrategyConfigSummary(
                    min_rr_ratio=1.5,
                    entry_threshold=0.6,
                    tp_adjustment=1.0,
                    sl_adjustment=1.0,
                    entry_aggressiveness=1.0
                )
            )

        config = strategy.get_config()

        return StrategyDetails(
            active=True,
            name=config.name,
            condition=condition.value,
            config=StrategyConfigSummary(
                min_rr_ratio=config.min_rr_ratio,
                entry_threshold=config.entry_threshold,
                tp_adjustment=config.tp_adjustment,
                sl_adjustment=config.sl_adjustment,
                entry_aggressiveness=config.entry_aggressiveness
            )
        )
