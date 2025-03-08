# strategies\patterns\entry_evaluator.py

import pandas as pd

from core.logger import logger
from strategies.base.model import StrategyConfig


class EntryEvaluator:
    """
    Classe base para avaliação de qualidade de entradas em diferentes condições de mercado.

    Esta classe encapsula a lógica comum para avaliar a qualidade da entrada
    em diferentes condições de mercado, permitindo customização
    através de parâmetros e substituição de métodos específicos.
    """

    def __init__(self, config: StrategyConfig):
        """
        Inicializa o avaliador de entradas.

        Args:
            config: Configuração da estratégia contendo parâmetros de avaliação
        """
        self.config = config

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
        Avalia a qualidade da entrada com base em diversos fatores.

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

            # Pontuação básica baseada em R:R
            entry_score = min(1.0, rr_ratio / 3.0)  # Pontuação de 0 a 1
        else:
            # Valores padrão se tp e sl não forem fornecidos
            should_enter = False
            entry_score = 0.0

        # Verificar tendência vs direção do trade
        trend_direction = self._calculate_trend_direction(df)
        entry_score = self._adjust_score_by_trend(entry_score, trend_direction, trade_direction)

        # Verificar indicadores técnicos
        entry_score = self._evaluate_technical_indicators(df, entry_score, trade_direction)

        # Verificar níveis de volatilidade
        entry_score = self._adjust_score_by_volatility(df, entry_score)

        # Ajustar por alinhamento multi-timeframe
        if mtf_alignment is not None:
            entry_score = self._adjust_score_by_mtf(entry_score, mtf_alignment, trade_direction)

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def _calculate_trend_direction(self, df: pd.DataFrame) -> str:
        """
        Calcula a direção atual da tendência com base nas médias móveis.
        """
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]

            if ema_short > ema_long:
                return "UPTREND"
            elif ema_short < ema_long:
                return "DOWNTREND"

        return "NEUTRAL"

    def _adjust_score_by_trend(self, entry_score: float, trend_direction: str, trade_direction: str) -> float:
        """
        Ajusta a pontuação de entrada com base na relação entre tendência e direção do trade.
        """
        # Bônus para trades alinhados com a tendência, penalidade para contra-tendência
        if (trade_direction == "LONG" and trend_direction == "UPTREND") or \
                (trade_direction == "SHORT" and trend_direction == "DOWNTREND"):
            entry_score = min(1.0, entry_score * 1.2)
        elif (trade_direction == "LONG" and trend_direction == "DOWNTREND") or \
                (trade_direction == "SHORT" and trend_direction == "UPTREND"):
            entry_score = entry_score * 0.7

        return entry_score

    def _evaluate_technical_indicators(self, df: pd.DataFrame, entry_score: float, trade_direction: str) -> float:
        """
        Avalia indicadores técnicos para refinar a pontuação de entrada.
        """
        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]

            # Bônus para LONG com RSI abaixo de 40 (sobrevenda)
            if trade_direction == "LONG" and rsi < 40:
                rsi_bonus = min((40 - rsi) / 30, 1.0) * 0.2
                entry_score = min(1.0, entry_score + rsi_bonus)
                logger.info(f"Bônus para LONG com RSI em sobrevenda ({rsi:.1f}): +{rsi_bonus:.2f}")

            # Bônus para SHORT com RSI acima de 60 (sobrecompra)
            elif trade_direction == "SHORT" and rsi > 60:
                rsi_bonus = min((rsi - 60) / 30, 1.0) * 0.2
                entry_score = min(1.0, entry_score + rsi_bonus)
                logger.info(f"Bônus para SHORT com RSI em sobrecompra ({rsi:.1f}): +{rsi_bonus:.2f}")

            # Penalidades para entradas contra o momento do RSI
            elif (trade_direction == "LONG" and rsi > 75) or (trade_direction == "SHORT" and rsi < 25):
                entry_score = entry_score * 0.8
                logger.info(f"Penalidade para trade contra extremo de RSI ({rsi:.1f})")

        # Verificar ADX para força da tendência
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]

            # Bônus para trades em tendência forte
            if adx > 30:
                adx_bonus = min((adx - 25) / 25, 0.2)
                entry_score = min(1.0, entry_score + adx_bonus)
                logger.info(f"Bônus para trade em tendência forte (ADX={adx:.1f}): +{adx_bonus:.2f}")

            # Penalidade para trades em tendência fraca
            elif adx < 20:
                entry_score = entry_score * 0.9
                logger.info(f"Penalidade para trade em tendência fraca (ADX={adx:.1f})")

        return entry_score

    def _adjust_score_by_volatility(self, df: pd.DataFrame, entry_score: float) -> float:
        """
        Ajusta a pontuação da entrada com base na volatilidade atual.
        """
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            # Condições de alta volatilidade requerem mais cautela
            if atr_pct > 1.5:
                entry_score = entry_score * 0.9
                logger.info(f"Ajuste para alta volatilidade (ATR={atr_pct:.2f}%): reduzindo score")

            # Baixa volatilidade pode permitir entradas mais flexíveis
            elif atr_pct < 0.5:
                entry_score = min(1.0, entry_score * 1.1)
                logger.info(f"Ajuste para baixa volatilidade (ATR={atr_pct:.2f}%): aumentando score")

        return entry_score

    def _adjust_score_by_mtf(self, entry_score: float, mtf_alignment: float, trade_direction: str) -> float:
        """
        Ajusta a pontuação da entrada com base no alinhamento multi-timeframe.
        """
        # mtf_alignment próximo de 1.0 é favorável para LONG
        # mtf_alignment próximo de 0.0 é favorável para SHORT
        # mtf_alignment próximo de 0.5 é neutro
        if trade_direction == "LONG":
            if mtf_alignment > 0.7:  # Forte alinhamento para LONG
                mtf_bonus = (mtf_alignment - 0.5) * 0.5
                entry_score = min(1.0, entry_score + mtf_bonus)
                logger.info(f"Bônus para LONG com forte alinhamento MTF ({mtf_alignment:.2f}): +{mtf_bonus:.2f}")
            elif mtf_alignment < 0.3:  # Contra o MTF
                entry_score = entry_score * 0.8
                logger.info(f"Penalidade para LONG contra o MTF ({mtf_alignment:.2f})")
        else:  # SHORT
            if mtf_alignment < 0.3:  # Forte alinhamento para SHORT
                mtf_bonus = (0.5 - mtf_alignment) * 0.5
                entry_score = min(1.0, entry_score + mtf_bonus)
                logger.info(f"Bônus para SHORT com forte alinhamento MTF ({mtf_alignment:.2f}): +{mtf_bonus:.2f}")
            elif mtf_alignment > 0.7:  # Contra o MTF
                entry_score = entry_score * 0.8
                logger.info(f"Penalidade para SHORT contra o MTF ({mtf_alignment:.2f})")

        return entry_score


class VolatilityEntryEvaluator(EntryEvaluator):
    """
    Avaliador de entradas especializado para condições de alta volatilidade.
    """

    def _adjust_score_by_volatility(self, df: pd.DataFrame, entry_score: float) -> float:
        """
        Implementação especializada para entradas em alta volatilidade.
        Exige maior qualidade e promove setups mais limpos.
        """
        # Em alta volatilidade usamos critérios mais rigorosos
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            # Em volatilidade muito alta, ser extremamente cauteloso
            if atr_pct > 2.5:
                entry_score = entry_score * 0.7
                logger.info(
                    f"Ajuste para volatilidade extrema (ATR={atr_pct:.2f}%): reduzindo score significativamente"
                )
            # Em volatilidade alta, ser moderadamente cauteloso
            elif atr_pct > 1.5:
                entry_score = entry_score * 0.85
                logger.info(f"Ajuste para alta volatilidade (ATR={atr_pct:.2f}%): reduzindo score")

        # Em alta volatilidade, dar maior ênfase à direção do breakout
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]
            if pct_b > 0.9:  # Próximo/acima da banda superior
                logger.info(f"Alta volatilidade: preço próximo à banda superior (%B={pct_b:.2f})")

            elif pct_b < 0.1:  # Próximo/abaixo da banda inferior
                logger.info(f"Alta volatilidade: preço próximo à banda inferior (%B={pct_b:.2f})")

        return entry_score


class RangeEntryEvaluator(EntryEvaluator):
    """
    Avaliador de entradas especializado para condições de mercado em range (lateralização).
    """

    def _adjust_score_by_trend(self, entry_score: float, trend_direction: str, trade_direction: str) -> float:
        """
        Em mercados em range, a direção da tendência tem menos importância.
        */
        """
        # Em mercados de range, tendência tem menos importância
        # Aplicamos apenas um pequeno ajuste
        if (trade_direction == "LONG" and trend_direction == "UPTREND") or \
                (trade_direction == "SHORT" and trend_direction == "DOWNTREND"):
            entry_score = min(1.0, entry_score * 1.1)  # Bônus menor
        elif (trade_direction == "LONG" and trend_direction == "DOWNTREND") or \
                (trade_direction == "SHORT" and trend_direction == "UPTREND"):
            entry_score = entry_score * 0.9  # Penalidade menor

        return entry_score

    def _evaluate_technical_indicators(self, df: pd.DataFrame, entry_score: float, trade_direction: str) -> float:
        """
        Em range, os níveis extremos de osciladores são mais relevantes.
        """
        # Verificar RSI para níveis extremos - importante em mercados de range
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]

            # Bônus para LONG com RSI baixo (sobrevenda)
            if trade_direction == "LONG" and rsi < 30:
                rsi_bonus = min((30 - rsi) / 20, 1.0) * 0.3  # Bônus maior em range
                entry_score = min(1.0, entry_score + rsi_bonus)
                logger.info(f"Bônus para LONG com RSI em sobrevenda ({rsi:.1f}): +{rsi_bonus:.2f}")

            # Bônus para SHORT com RSI alto (sobrecompra)
            elif trade_direction == "SHORT" and rsi > 70:
                rsi_bonus = min((rsi - 70) / 20, 1.0) * 0.3  # Bônus maior em range
                entry_score = min(1.0, entry_score + rsi_bonus)
                logger.info(f"Bônus para SHORT com RSI em sobrecompra ({rsi:.1f}): +{rsi_bonus:.2f}")

        # Verificar %B para identificar extremos no range
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]

            # Bônus para LONG próximo da banda inferior
            if trade_direction == "LONG" and pct_b < 0.1:
                b_bonus = min((0.1 - pct_b) * 5, 0.3)
                entry_score = min(1.0, entry_score + b_bonus)
                logger.info(f"Bônus para LONG na banda inferior (%B={pct_b:.2f}): +{b_bonus:.2f}")

            # Bônus para SHORT próximo da banda superior
            elif trade_direction == "SHORT" and pct_b > 0.9:
                b_bonus = min((pct_b - 0.9) * 5, 0.3)
                entry_score = min(1.0, entry_score + b_bonus)
                logger.info(f"Bônus para SHORT na banda superior (%B={pct_b:.2f}): +{b_bonus:.2f}")

        return entry_score

    def _adjust_score_by_mtf(self, entry_score: float, mtf_alignment: float, trade_direction: str) -> float:
        """
        Em range, MTF neutro é mais favorável.
        """
        # Em range, é melhor ter MTF neutro (próximo de 0.5)
        mtf_neutrality = 1.0 - abs(mtf_alignment - 0.5) * 2
        if mtf_neutrality > 0.7:  # Relativamente neutro
            mtf_bonus = mtf_neutrality * 0.2
            entry_score = min(1.0, entry_score + mtf_bonus)
            logger.info(f"Bônus para MTF neutro em mercado de range: +{mtf_bonus:.2f}")

        return entry_score


class TrendEntryEvaluator(EntryEvaluator):
    """
    Avaliador de entradas especializado para condições de mercado em tendência.
    """

    def __init__(self, config: StrategyConfig, trend_direction: str = "UPTREND"):
        """
        Inicializa o avaliador de entradas para mercados em tendência.

        Args:
            config: Configuração da estratégia
            trend_direction: Direção da tendência ("UPTREND" ou "DOWNTREND")
        """
        super().__init__(config)
        self.trend_direction = trend_direction

    def _adjust_score_by_trend(self, entry_score: float, trend_direction: str, trade_direction: str) -> float:
        """
        Em mercados em tendência, o alinhamento com a tendência é crucial.
        */
        """
        # Em tendência, premiar significativamente trades alinhados
        # e penalizar fortemente trades contra-tendência
        if (trade_direction == "LONG" and trend_direction == "UPTREND") or \
                (trade_direction == "SHORT" and trend_direction == "DOWNTREND"):
            entry_score = min(1.0, entry_score * 1.3)  # Bônus maior
            logger.info(f"Bônus significativo para trade na direção da tendência ({trend_direction})")
        elif (trade_direction == "LONG" and trend_direction == "DOWNTREND") or \
                (trade_direction == "SHORT" and trend_direction == "UPTREND"):
            entry_score = entry_score * 0.6  # Penalidade maior
            logger.info(f"Penalidade significativa para trade contra a tendência ({trend_direction})")

        return entry_score

    def _evaluate_technical_indicators(self, df: pd.DataFrame, entry_score: float, trade_direction: str) -> float:
        """
        Em tendência, indicadores de momentum são mais relevantes.
        """
        # Verificar ADX para força da tendência - crucial em mercados de tendência
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]

            # Tendência forte promove confiança em setups alinhados
            if adx > 25:
                preferred_direction = "LONG" if self.trend_direction == "UPTREND" else "SHORT"
                if trade_direction == preferred_direction:
                    adx_bonus = min((adx - 25) / 25, 0.3)
                    entry_score = min(1.0, entry_score + adx_bonus)
                    logger.info(f"Bônus para trade alinhado em tendência forte (ADX={adx:.1f}): +{adx_bonus:.2f}")

        # Verificar se estamos em pullback/rally para entradas favoráveis
        if 'rsi' in df.columns and self.trend_direction == "UPTREND" and trade_direction == "LONG":
            rsi = df['rsi'].iloc[-1]
            # Pullback em tendência de alta (RSI abaixo de 50 mas não extremo)
            if 30 < rsi < 50:
                pullback_bonus = 0.2
                entry_score = min(1.0, entry_score + pullback_bonus)
                logger.info(f"Bônus para LONG em pullback de tendência de alta (RSI={rsi:.1f}): +{pullback_bonus:.2f}")

        elif 'rsi' in df.columns and self.trend_direction == "DOWNTREND" and trade_direction == "SHORT":
            rsi = df['rsi'].iloc[-1]
            # Rally em tendência de baixa (RSI acima de 50 mas não extremo)
            if 50 < rsi < 70:
                rally_bonus = 0.2
                entry_score = min(1.0, entry_score + rally_bonus)
                logger.info(f"Bônus para SHORT em rally de tendência de baixa (RSI={rsi:.1f}): +{rally_bonus:.2f}")

        return entry_score


class EntryEvaluatorFactory:
    """
    Factory para criar avaliadores de entrada apropriados para diferentes condições de mercado.
    """

    @staticmethod
    def create_evaluator(market_condition: str, config: StrategyConfig) -> EntryEvaluator:
        """
        Cria um avaliador de entrada adequado para a condição de mercado especificada.

        Args:
            market_condition: Condição de mercado ("RANGE", "UPTREND", "DOWNTREND", "HIGH_VOLATILITY")
            config: Configuração da estratégia

        Returns:
            EntryEvaluator apropriado para a condição de mercado
        """
        if market_condition == "RANGE":
            return RangeEntryEvaluator(config)
        elif market_condition == "UPTREND":
            return TrendEntryEvaluator(config, trend_direction="UPTREND")
        elif market_condition == "DOWNTREND":
            return TrendEntryEvaluator(config, trend_direction="DOWNTREND")
        elif market_condition == "HIGH_VOLATILITY" or market_condition == "LOW_VOLATILITY":
            return VolatilityEntryEvaluator(config)
        else:
            # Condição de mercado desconhecida, usar avaliador padrão
            return EntryEvaluator(config)
