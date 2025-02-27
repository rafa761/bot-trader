# services/risk_reward_manager.py
from core.config import settings
from core.logger import logger


class RiskRewardManager:
    """
    Classe responsável pela gestão da relação risk/reward nas operações.
    Implementa estratégias avançadas para otimizar a razão entre recompensa e risco.
    """

    def __init__(self, min_rr_ratio: float = None, atr_multiplier: float = None):
        """
        Inicializa o gerenciador de risk/reward.

        Args:
            min_rr_ratio: Razão mínima aceitável entre recompensa e risco
            atr_multiplier: Multiplicador do ATR para definição de stop loss
        """
        self.min_rr_ratio = min_rr_ratio if min_rr_ratio is not None else settings.MIN_RR_RATIO
        self.atr_multiplier = atr_multiplier if atr_multiplier is not None else settings.ATR_MULTIPLIER

        logger.info(
            f"RiskRewardManager inicializado (min_rr_ratio={self.min_rr_ratio}, atr_multiplier={self.atr_multiplier})")


    def calculate_dynamic_sl(self, current_price: float, atr_value: float) -> float:
        """
        Calcula o stop loss dinâmico baseado no ATR.

        Args:
            current_price: Preço atual do ativo
            atr_value: Valor atual do ATR

        Returns:
            Valor percentual do stop loss
        """
        if atr_value <= 0:
            logger.warning("ATR <= 0, usando valor default para cálculo de SL")
            # Usar 1% como default se ATR for inválido
            return 1.0

        # Cálculo percentual baseado em ATR
        sl_pct = (atr_value / current_price) * 100 * self.atr_multiplier

        # Limites de segurança para o stop loss
        min_sl = 0.5  # Mínimo de 0.5%
        max_sl = 3.0  # Máximo de 3%

        sl_pct = max(min_sl, min(sl_pct, max_sl))

        return sl_pct

    def adjust_tp_for_min_rr(self, tp_pct: float, sl_pct: float) -> float:
        """
        Ajusta o take profit para garantir a razão mínima de risk/reward.

        Args:
            tp_pct: Take profit percentual original
            sl_pct: Stop loss percentual

        Returns:
            Take profit percentual ajustado
        """
        min_tp = sl_pct * self.min_rr_ratio

        # Se o TP original for menor que o mínimo para a razão R:R desejada,
        # ajusta para o mínimo
        if tp_pct < min_tp:
            logger.debug(f"Ajustando TP: {tp_pct:.2f}% -> {min_tp:.2f}% para garantir R:R mínimo")
            return min_tp

        return tp_pct

    def adjust_tp_based_on_volatility(self, tp_pct: float, sl_pct: float,
                                      volatility_factor: float) -> float:
        """
        Ajusta o take profit baseado na volatilidade atual do mercado.

        Args:
            tp_pct: Take profit percentual original
            sl_pct: Stop loss percentual
            volatility_factor: Fator de volatilidade (>1 = maior volatilidade)

        Returns:
            Take profit percentual ajustado
        """
        # Em mercados mais voláteis, podemos ser mais ambiciosos com o TP
        adjusted_tp = tp_pct * volatility_factor

        # Garantir a razão mínima de R:R
        min_tp = sl_pct * self.min_rr_ratio
        adjusted_tp = max(adjusted_tp, min_tp)

        return adjusted_tp

    def evaluate_trade_quality(self, tp_pct: float, sl_pct: float, trend_strength: float = 0.5) -> float:
        """
        Avalia a qualidade de um potencial trade baseado em múltiplos fatores.

        Args:
            tp_pct: Take profit percentual
            sl_pct: Stop loss percentual
            trend_strength: Força da tendência (0-1)

        Returns:
            Pontuação de qualidade entre 0-1 (maior é melhor)
        """
        # Verificar se os valores são válidos
        if sl_pct <= 0 or tp_pct <= 0:
            logger.warning(f"Valores inválidos para avaliação R:R: TP={tp_pct}, SL={sl_pct}")
            return 0.0

        # Calcular razão R:R
        rr_ratio = tp_pct / sl_pct

        # Log para debug
        logger.debug(f"Avaliando trade: TP={tp_pct:.2f}%, SL={sl_pct:.2f}%, R:R={rr_ratio:.2f}")

        # Pontuação base da razão R:R (0-0.6)
        if rr_ratio < 1.0:
            rr_score = 0
        elif rr_ratio < self.min_rr_ratio:
            rr_score = 0.3 * (rr_ratio / self.min_rr_ratio)
        elif rr_ratio < 2.0:
            rr_score = 0.3 + 0.1 * (rr_ratio - self.min_rr_ratio) / 0.5
        elif rr_ratio < 3.0:
            rr_score = 0.4 + 0.1 * (rr_ratio - 2.0)
        else:
            rr_score = 0.5 + 0.1 * min(rr_ratio - 3.0, 1.0)  # Máximo de 0.6

        # Pontuação de TP (0-0.2) - recompensa movimentos maiores
        tp_score = min(0.2, tp_pct / 10)  # Máximo em 10% de movimento

        # Pontuação de SL (0-0.1) - penaliza stops muito grandes
        sl_score = 0.1 * max(0, 1 - (sl_pct / 2))  # 0% para SL de 2% ou mais

        # Pontuação de tendência (0-0.1) - recompensa trades alinhados com tendência forte
        trend_score = 0.1 * trend_strength

        # Calcular pontuação total
        total_score = rr_score + tp_score + sl_score + trend_score

        # Log detalhado
        logger.debug(
            f"Scores detalhados - R:R: {rr_score:.2f}, TP: {tp_score:.2f}, "
            f"SL: {sl_score:.2f}, Tendência: {trend_score:.2f}, Total: {total_score:.2f}"
        )

        return min(total_score, 1.0)  # Garantir que não exceda 1.0

    def should_take_trade(self, tp_pct: float, sl_pct: float,
                          trend_strength: float = 0.5,
                          quality_threshold: float = 0.6) -> bool:
        """
        Determina se um trade deve ser tomado baseado na qualidade.

        Args:
            tp_pct: Take profit percentual
            sl_pct: Stop loss percentual
            trend_strength: Força da tendência (0-1)
            quality_threshold: Limiar de qualidade para aceitação do trade

        Returns:
            True se o trade deve ser tomado, False caso contrário
        """
        # Verifica se a razão R:R mínima é atendida
        if tp_pct / sl_pct < self.min_rr_ratio:
            return False

        # Avalia qualidade geral do trade
        quality = self.evaluate_trade_quality(tp_pct, sl_pct, trend_strength)

        return quality >= quality_threshold
