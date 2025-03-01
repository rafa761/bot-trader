# services/market_analyzers.py

from core.config import settings
from core.logger import logger
from services.base.schemas import TradingSignal


class MarketTrendAnalyzer:
    """
    Analisador de tendências de mercado e ajustador de parâmetros.

    Esta classe é responsável por analisar condições de mercado e ajustar
    parâmetros de trading com base nessas condições, seguindo estratégias específicas.
    """

    def __init__(self):
        """Inicializa o analisador de mercado com valores padrão."""
        # Valores padrão para parâmetros de trading
        self.default_entry_threshold = settings.ENTRY_THRESHOLD_DEFAULT
        self.default_tp_adjustment = 1.0
        self.default_sl_adjustment = 1.0

    def adjust_parameters(
            self, trend_direction: str, trend_strength: str, trade_direction: str,
            mtf_trend: str = None
    ) -> tuple[float, float, float]:
        """
        Ajusta os parâmetros de trading baseado na tendência atual e análise multi-timeframe.

        Args:
            trend_direction: Direção da tendência ("UPTREND", "DOWNTREND", "NEUTRAL")
            trend_strength: Força da tendência ("STRONG_TREND", "WEAK_TREND")
            trade_direction: Direção do trade ("LONG", "SHORT")
            mtf_trend: Tendência multi-timeframe (opcional)

        Returns:
            tuple: (entry_threshold, tp_adjustment_factor, sl_adjustment_factor)
        """
        # Valores padrão
        entry_threshold = self.default_entry_threshold
        tp_adjustment = 1.0  # Valor neutro, sem ajuste
        sl_adjustment = 1.0  # Valor neutro, sem ajuste

        # Verificar se a tendência é forte
        is_strong_trend = trend_strength == "STRONG_TREND"

        # Ajustar com base na tendência e direção do trade
        if trend_direction == "UPTREND":
            if trade_direction == "LONG":
                # Trade a favor da tendência de alta
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_ALIGNED
                if is_strong_trend:
                    # Em tendência forte, podemos ser mais agressivos com TP (aumentar) e menos com SL (reduzir)
                    tp_adjustment = 1.2  # Aumenta TP em 20%
                    sl_adjustment = 0.9  # Reduz SL em 10%
                logger.info(f"LONG alinhado com tendência de ALTA: menos seletivo, TP mais agressivo")
            else:  # SHORT
                # Trade contra a tendência de alta
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_AGAINST
                if is_strong_trend:
                    # Em tendência forte de alta, ser mais conservador com trades SHORT
                    tp_adjustment = 0.8  # Reduz TP em 20% (mais conservador)
                    sl_adjustment = 0.7  # Reduz SL em 30% (mais próximo, protege mais)
                logger.info(f"SHORT contra tendência de ALTA: mais seletivo, alvos reduzidos")

        elif trend_direction == "DOWNTREND":
            if trade_direction == "SHORT":
                # Trade a favor da tendência de baixa
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_ALIGNED
                if is_strong_trend:
                    # Em tendência forte, podemos ser mais agressivos com TP e menos com SL
                    tp_adjustment = 1.2  # Aumenta TP em 20%
                    sl_adjustment = 0.9  # Reduz SL em 10%
                logger.info(f"SHORT alinhado com tendência de BAIXA: menos seletivo, TP mais agressivo")
            else:  # LONG
                # Trade contra a tendência de baixa
                entry_threshold = settings.ENTRY_THRESHOLD_TREND_AGAINST
                if is_strong_trend:
                    # Em tendência forte de baixa, ser mais conservador com trades LONG
                    tp_adjustment = 0.8  # Reduz TP em 20% (mais conservador)
                    sl_adjustment = 0.7  # Reduz SL em 30% (mais próximo, protege mais)
                logger.info(f"LONG contra tendência de BAIXA: mais seletivo, alvos reduzidos")

        else:  # NEUTRAL
            # Em mercado sem tendência clara, usar configurações específicas para mercado em range
            entry_threshold = settings.ENTRY_THRESHOLD_RANGE
            tp_adjustment = settings.TP_ADJUSTMENT_RANGE
            sl_adjustment = settings.SL_ADJUSTMENT_RANGE
            logger.info(f"Mercado NEUTRAL: ajustando parâmetros para operação em range")

        # Ajustar baseado na análise multi-timeframe
        if mtf_trend:
            # Analisar tendência multi-timeframe
            is_strong_mtf_up = mtf_trend in ["STRONG_UPTREND", "MODERATE_UPTREND"]
            is_strong_mtf_down = mtf_trend in ["STRONG_DOWNTREND", "MODERATE_DOWNTREND"]

            if is_strong_mtf_up:
                if trade_direction == "LONG":
                    # LONG em tendência de alta forte multi-timeframe
                    entry_threshold *= 0.8  # Reduzir threshold (mais fácil entrar)
                    tp_adjustment *= 1.3  # Aumentar TP em mais 30%
                    logger.info(f"LONG alinhado com tendência MTF FORTE de ALTA: threshold reduzido, TP mais agressivo")
                else:  # SHORT
                    # SHORT contra tendência de alta forte multi-timeframe
                    entry_threshold *= 1.4  # Aumentar threshold (mais difícil entrar)
                    tp_adjustment *= 0.7  # Reduzir TP drasticamente
                    sl_adjustment *= 0.6  # Reduzir SL drasticamente (mais próximo)
                    logger.info(f"SHORT contra tendência MTF FORTE de ALTA: threshold muito aumentado, alvos reduzidos")

            elif is_strong_mtf_down:
                if trade_direction == "SHORT":
                    # SHORT em tendência de baixa forte multi-timeframe
                    entry_threshold *= 0.8  # Reduzir threshold (mais fácil entrar)
                    tp_adjustment *= 1.3  # Aumentar TP em mais 30%
                    logger.info(
                        f"SHORT alinhado com tendência MTF FORTE de BAIXA: threshold reduzido, TP mais agressivo")
                else:  # LONG
                    # LONG contra tendência de baixa forte multi-timeframe
                    entry_threshold *= 1.4  # Aumentar threshold (mais difícil entrar)
                    tp_adjustment *= 0.7  # Reduzir TP drasticamente
                    sl_adjustment *= 0.6  # Reduzir SL drasticamente (mais próximo)
                    logger.info(f"LONG contra tendência MTF FORTE de BAIXA: threshold muito aumentado, alvos reduzidos")

        return entry_threshold, tp_adjustment, sl_adjustment

    def adjust_signal_parameters(
            self, signal: TradingSignal, tp_adjustment: float, sl_adjustment: float
    ) -> TradingSignal:
        """
        Ajusta os parâmetros do sinal com base nos fatores de ajuste.

        Args:
            signal: Sinal de trading a ser ajustado
            tp_adjustment: Fator de ajuste para take profit
            sl_adjustment: Fator de ajuste para stop loss

        Returns:
            TradingSignal: Sinal ajustado
        """
        # Clone o sinal para não modificar o original
        # (na verdade, como TradingSignal é um Pydantic model, vamos criar uma cópia direta)

        # Ajustar TP/SL baseado na tendência e recalcular os preços
        if signal.direction == "LONG":
            # Para LONG: ajuste de TP é sobre o percentual (mantém o preço acima do atual)
            new_tp_pct = abs(signal.predicted_tp_pct) * tp_adjustment
            # Para LONG: ajuste de SL é sobre o percentual (mantém o preço abaixo do atual)
            new_sl_pct = abs(signal.predicted_sl_pct) * sl_adjustment

            # Recalcula os fatores
            tp_factor = 1 + (new_tp_pct / 100)
            sl_factor = 1 - (new_sl_pct / 100)

            # Atualiza o sinal
            signal.tp_factor = tp_factor
            signal.sl_factor = sl_factor
            signal.predicted_tp_pct = new_tp_pct
            signal.predicted_sl_pct = new_sl_pct
        else:  # SHORT
            # Para SHORT: ajuste de TP é sobre o percentual (mantém o preço abaixo do atual)
            new_tp_pct = abs(signal.predicted_tp_pct) * tp_adjustment
            # Para SHORT: ajuste de SL é sobre o percentual (mantém o preço acima do atual)
            new_sl_pct = abs(signal.predicted_sl_pct) * sl_adjustment

            # Recalcula os fatores
            tp_factor = 1 - (new_tp_pct / 100)
            sl_factor = 1 + (new_sl_pct / 100)

            # Atualiza o sinal
            signal.tp_factor = tp_factor
            signal.sl_factor = sl_factor
            signal.predicted_tp_pct = -new_tp_pct  # Negativo para SHORT
            signal.predicted_sl_pct = new_sl_pct

        # Recalcular os preços de TP e SL
        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Verificar se os preços de TP e SL são lógicos
        if signal.direction == "LONG":
            # Para LONG: TP deve ser maior que o preço atual, SL deve ser menor
            if signal.tp_price <= signal.current_price:
                logger.warning(
                    f"TP inválido para LONG: {signal.tp_price} <= {signal.current_price}. Ajustando.")
                signal.tp_price = signal.current_price * 1.02  # Ajuste mínimo de 2%
                signal.predicted_tp_pct = 2.0

            if signal.sl_price >= signal.current_price:
                logger.warning(
                    f"SL inválido para LONG: {signal.sl_price} >= {signal.current_price}. Ajustando.")
                signal.sl_price = signal.current_price * 0.995  # Ajuste mínimo de 0.5%
                signal.predicted_sl_pct = 0.5
        else:  # SHORT
            # Para SHORT: TP deve ser menor que o preço atual, SL deve ser maior
            if signal.tp_price >= signal.current_price:
                logger.warning(
                    f"TP inválido para SHORT: {signal.tp_price} >= {signal.current_price}. Ajustando.")
                signal.tp_price = signal.current_price * 0.98  # Ajuste mínimo de 2%
                signal.predicted_tp_pct = -2.0

            if signal.sl_price <= signal.current_price:
                logger.warning(
                    f"SL inválido para SHORT: {signal.sl_price} <= {signal.current_price}. Ajustando.")
                signal.sl_price = signal.current_price * 1.005  # Ajuste mínimo de 0.5%
                signal.predicted_sl_pct = 0.5

        # Atualizar a razão R:R após todos os ajustes
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

        # Log de debug para garantir que os preços estão corretos
        logger.info(
            f"Preços ajustados: Atual={signal.current_price:.2f}, "
            f"TP={signal.tp_price:.2f} ({signal.predicted_tp_pct:.2f}%), "
            f"SL={signal.sl_price:.2f} ({signal.predicted_sl_pct:.2f}%), "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal

    def log_technical_analysis(
            self, direction: str, trend_direction: str, entry_score: float, threshold: float,
            mtf_trend: str = None
    ) -> None:
        """
        Registra informações de análise técnica no log.

        Args:
            direction: Direção do trade ("LONG", "SHORT")
            trend_direction: Direção da tendência
            entry_score: Pontuação de qualidade da entrada
            threshold: Limiar para entrada
            mtf_trend: Tendência multi-timeframe (opcional)
        """
        log_msg = (
            f"Análise Técnica: "
            f"Direção={direction}, "
            f"Tendência={trend_direction}, "
            f"Score de Entrada={entry_score:.2f}, "
            f"Threshold Ajustado={threshold:.2f}"
        )

        if mtf_trend:
            log_msg += f", MTF Trend={mtf_trend}"

        logger.info(log_msg)
