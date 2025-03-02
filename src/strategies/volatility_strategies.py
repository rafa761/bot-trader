# strategies/volatility_strategies.py

from typing import Optional

import pandas as pd

from core.config import settings
from core.logger import logger
from services.base.schemas import TradingSignal
from strategies.base import BaseStrategy, StrategyConfig


class HighVolatilityStrategy(BaseStrategy):
    """
    Estratégia para mercados com alta volatilidade.
    Adaptada para capitalizar em movimentos amplos enquanto
    gerencia riscos em um ambiente de alta volatilidade.
    """

    def __init__(self):
        """Inicializa a estratégia com configuração otimizada para alta volatilidade."""
        config = StrategyConfig(
            name="High Volatility Strategy",
            description="Estratégia otimizada para mercados com alta volatilidade",
            min_rr_ratio=2.0,  # Exigir R:R maior em mercado volátil
            entry_threshold=0.75,  # Mais rigoroso na entrada
            tp_adjustment=1.5,  # Aumentar TP para capturar movimentos maiores
            sl_adjustment=1.3,  # SL mais largo para evitar stops prematuros
            entry_aggressiveness=0.7,  # Menos agressivo nas entradas
            max_sl_percent=2.5,  # Permitir SL maior em mercado volátil
            min_tp_percent=1.0,  # Exigir TP maior
            required_indicators=["adx", "atr", "atr_pct", "boll_width"]
        )
        super().__init__(config)

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia de alta volatilidade deve ser ativada.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar ATR percentual
        high_atr = False
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            high_atr = atr_pct > settings.VOLATILITY_HIGH_THRESHOLD
        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            high_atr = atr_pct > settings.VOLATILITY_HIGH_THRESHOLD

        # Verificar a largura das Bollinger Bands
        wide_bands = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            wide_bands = boll_width > (avg_width * 1.5)

        # Verificar tendência forte (alta volatilidade em tendência)
        strong_trend = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            strong_trend = adx_value > 30

        # Verificar aumento recente na volatilidade
        increasing_volatility = False
        if 'atr' in df.columns and len(df) > 5:
            recent_atr = df['atr'].iloc[-1]
            prev_atr = df['atr'].iloc[-5]
            increasing_volatility = recent_atr > (prev_atr * 1.3)  # 30% aumento

        # Ativar se pelo menos dois indicadores confirmarem alta volatilidade
        confirmations = sum([high_atr, wide_bands, strong_trend, increasing_volatility])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de ALTA VOLATILIDADE ativada: "
                f"ATR_alto={high_atr}, Bandas_largas={wide_bands}, "
                f"Tendência_forte={strong_trend} (ADX={df['adx'].iloc[-1]:.1f}), "
                f"Volatilidade_crescente={increasing_volatility}"
            )

        return should_activate

    def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com alta volatilidade.
        Em mercados voláteis, buscamos entradas na direção da tendência em pullbacks.
        """
        # Em mercados de alta volatilidade, queremos esperar pullbacks para entrar
        # na direção da tendência principal, com stops mais largos

        # Identificar a tendência principal
        trend = self.calculate_trend_direction(df)

        # Verificar condições específicas para entradas seguras em volatilidade alta

        # Para entradas LONG em tendência de alta
        long_conditions = []
        if trend == "UPTREND":
            # Verificar pullback nos osciladores
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                long_conditions.append(rsi < 45)  # RSI caiu um pouco, mas não demais

            # Verificar Stochastic
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                long_conditions.append(stoch_k < 40 and stoch_d < 40)  # Stoch em região baixa

            # Verificar suporte em Bollinger Band
            if 'close' in df.columns and 'boll_lband' in df.columns:
                close = df['close'].iloc[-1]
                lower_band = df['boll_lband'].iloc[-1]
                long_conditions.append(close < (lower_band * 1.01))  # Perto da banda inferior

            if len(long_conditions) >= 2:
                logger.info(
                    f"Condições para LONG em volatilidade alta: "
                    f"Tendência={trend}, "
                    f"RSI={df['rsi'].iloc[-1]:.1f}, "
                    f"Stoch_K={df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 'N/A'}, "
                    f"Próximo a BB inferior={close < (lower_band * 1.01) if 'boll_lband' in df.columns else 'N/A'}"
                )

        # Para entradas SHORT em tendência de baixa
        short_conditions = []
        if trend == "DOWNTREND":
            # Verificar pullback nos osciladores
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                short_conditions.append(rsi > 55)  # RSI subiu um pouco, mas não demais

            # Verificar Stochastic
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                stoch_k = df['stoch_k'].iloc[-1]
                stoch_d = df['stoch_d'].iloc[-1]
                short_conditions.append(stoch_k > 60 and stoch_d > 60)  # Stoch em região alta

            # Verificar resistência em Bollinger Band
            if 'close' in df.columns and 'boll_hband' in df.columns:
                close = df['close'].iloc[-1]
                upper_band = df['boll_hband'].iloc[-1]
                short_conditions.append(close > (upper_band * 0.99))  # Perto da banda superior

            if len(short_conditions) >= 2:
                logger.info(
                    f"Condições para SHORT em volatilidade alta: "
                    f"Tendência={trend}, "
                    f"RSI={df['rsi'].iloc[-1]:.1f}, "
                    f"Stoch_K={df['stoch_k'].iloc[-1] if 'stoch_k' in df.columns else 'N/A'}, "
                    f"Próximo a BB superior={close > (upper_band * 0.99) if 'boll_hband' in df.columns else 'N/A'}"
                )

        # Retornar None para usar o gerador de sinais existente
        return None

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados com alta volatilidade.
        Aumenta alvos de TP e aumenta SL para acomodar a volatilidade.
        """
        # Obter o ATR para calcular ajustes mais precisos
        atr_value = None
        if 'atr' in df.columns:
            atr_value = df['atr'].iloc[-1]

        # Calcular fatores de ajuste baseados no ATR
        if atr_value is not None:
            atr_percent = (atr_value / signal.current_price) * 100

            # Ajustar fatores baseado na volatilidade real
            vol_tp_factor = min(1.0 + (atr_percent / 2.0), 2.0)  # Limite em 2x
            vol_sl_factor = min(1.0 + (atr_percent / 3.0), 1.75)  # Limite em 1.75x

            logger.info(
                f"Fatores de ajuste baseados em ATR: "
                f"ATR={atr_percent:.2f}%, TP_factor={vol_tp_factor:.2f}, SL_factor={vol_sl_factor:.2f}"
            )

            # Combinar com os fatores da configuração
            tp_factor = self.config.tp_adjustment * vol_tp_factor
            sl_factor = self.config.sl_adjustment * vol_sl_factor
        else:
            # Se não tivermos ATR, usar os valores da configuração
            tp_factor = self.config.tp_adjustment
            sl_factor = self.config.sl_adjustment

        # Aplicar ajustes
        original_tp = signal.predicted_tp_pct
        original_sl = signal.predicted_sl_pct

        if signal.direction == "LONG":
            # Para LONG, aumentar TP e SL
            signal.predicted_tp_pct = original_tp * tp_factor
            signal.predicted_sl_pct = original_sl * sl_factor

            # Recalcular fatores
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:
            # Para SHORT, aumentar TP (que é negativo) e SL
            signal.predicted_tp_pct = original_tp * tp_factor
            signal.predicted_sl_pct = original_sl * sl_factor

            # Recalcular fatores
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        # Recalcular preços
        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Atualizar razão R:R
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

        logger.info(
            f"Sinal ajustado para volatilidade alta: "
            f"TP={original_tp:.2f}% → {signal.predicted_tp_pct:.2f}%, "
            f"SL={original_sl:.2f}% → {signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal


class LowVolatilityStrategy(BaseStrategy):
    """
    Estratégia para mercados com baixa volatilidade.
    Adaptada para tempos de calmaria no mercado, buscando oportunidades
    em pequenos movimentos com risco controlado.
    """

    def __init__(self):
        """Inicializa a estratégia com configuração otimizada para baixa volatilidade."""
        config = StrategyConfig(
            name="Low Volatility Strategy",
            description="Estratégia otimizada para mercados com baixa volatilidade",
            min_rr_ratio=1.3,  # R:R menor em mercados calmos
            entry_threshold=0.65,  # Moderadamente rigoroso na entrada
            tp_adjustment=0.7,  # Reduzir TP para targets mais realistas
            sl_adjustment=0.6,  # Reduzir SL para compensar o TP menor
            entry_aggressiveness=1.1,  # Um pouco mais agressivo nas entradas
            max_sl_percent=0.8,
            min_tp_percent=0.3,
            required_indicators=["adx", "atr", "atr_pct", "boll_width"]
        )
        super().__init__(config)

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia de baixa volatilidade deve ser ativada.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar ATR percentual
        low_atr = False
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            low_atr = atr_pct < settings.VOLATILITY_LOW_THRESHOLD
        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            low_atr = atr_pct < settings.VOLATILITY_LOW_THRESHOLD

        # Verificar a largura das Bollinger Bands
        narrow_bands = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            narrow_bands = boll_width < (avg_width * 0.7)

        # Verificar ausência de tendência forte (baixa volatilidade)
        weak_trend = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            weak_trend = adx_value < 20

        # Verificar diminuição recente na volatilidade
        decreasing_volatility = False
        if 'atr' in df.columns and len(df) > 5:
            recent_atr = df['atr'].iloc[-1]
            prev_atr = df['atr'].iloc[-5]
            decreasing_volatility = recent_atr < (prev_atr * 0.8)  # 20% redução

        # Ativar se pelo menos dois indicadores confirmarem baixa volatilidade
        confirmations = sum([low_atr, narrow_bands, weak_trend, decreasing_volatility])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de BAIXA VOLATILIDADE ativada: "
                f"ATR_baixo={low_atr} ({atr_pct:.2f}% < {settings.VOLATILITY_LOW_THRESHOLD}%), "
                f"Bandas_estreitas={narrow_bands}, "
                f"Tendência_fraca={weak_trend} (ADX={df['adx'].iloc[-1]:.1f}), "
                f"Volatilidade_decrescente={decreasing_volatility}"
            )

        return should_activate

    def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com baixa volatilidade.
        Em mercados calmos, buscamos pequenos movimentos e breakouts.
        """
        # Em baixa volatilidade, buscamos:
        # 1. Breakouts potenciais após compressão de volatilidade
        # 2. Pequenos movimentos entre níveis técnicos

        # Verificar compressão de volatilidade (possível setup para breakout)
        volatility_squeeze = False
        if 'boll_width' in df.columns and 'keltner_width' in df.columns:
            bb_width = df['boll_width'].iloc[-1]
            keltner_width = df['keltner_width'].iloc[-1] if 'keltner_width' in df.columns else float('inf')
            volatility_squeeze = bb_width < keltner_width

            # Verificar também o indicador TTM Squeeze se disponível
            if 'ttm_squeeze' in df.columns:
                ttm_squeeze = df['ttm_squeeze'].iloc[-1]
                ttm_squeeze_positive = ttm_squeeze > 0

                if volatility_squeeze and ttm_squeeze_positive:
                    logger.info(
                        f"Compressão de volatilidade detectada com momentum positivo: "
                        f"BB_width={bb_width:.4f}, TTM_squeeze={ttm_squeeze:.4f}"
                    )

        # Verificar proximidade a níveis técnicos
        near_level = False
        price_level = 0

        # Verificar níveis de pivot
        if 'pivot' in df.columns and 'pivot_r1' in df.columns and 'pivot_s1' in df.columns:
            pivot = df['pivot'].iloc[-1]
            r1 = df['pivot_r1'].iloc[-1]
            s1 = df['pivot_s1'].iloc[-1]

            # Calcular distâncias percentuais
            dist_to_pivot = abs(current_price - pivot) / current_price * 100
            dist_to_r1 = abs(current_price - r1) / current_price * 100
            dist_to_s1 = abs(current_price - s1) / current_price * 100

            # Verificar se estamos próximos de algum nível
            if dist_to_pivot < 0.2:
                near_level = True
                price_level = pivot
                logger.info(f"Preço próximo ao pivot: {dist_to_pivot:.2f}%")
            elif dist_to_r1 < 0.2:
                near_level = True
                price_level = r1
                logger.info(f"Preço próximo a R1: {dist_to_r1:.2f}%")
            elif dist_to_s1 < 0.2:
                near_level = True
                price_level = s1
                logger.info(f"Preço próximo a S1: {dist_to_s1:.2f}%")

        # Buscar reversões em níveis técnicos ou breakouts em compressão
        conditions_met = volatility_squeeze or near_level

        if conditions_met:
            logger.info(
                f"Condições para trade em baixa volatilidade: "
                f"Compressão={volatility_squeeze}, "
                f"Próximo a nível={near_level} (nível={price_level:.2f})"
            )

        # Retornar None para usar o gerador de sinais existente
        return None

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados com baixa volatilidade.
        Reduz TP e SL para se adaptar a movimentos menores.
        """
        # Obter o ATR para calcular ajustes mais precisos
        atr_value = None
        if 'atr' in df.columns:
            atr_value = df['atr'].iloc[-1]

        # Calcular fatores de ajuste baseados no ATR
        if atr_value is not None:
            atr_percent = (atr_value / signal.current_price) * 100

            # Em baixa volatilidade, usamos ATR como base para TP e SL
            # para evitar targets irrealisticamente grandes

            # Calcular R:R razoável baseado no ATR
            reasonable_tp_pct = atr_percent * 2.0  # 2x ATR para TP
            reasonable_sl_pct = atr_percent * 1.0  # 1x ATR para SL

            # Decidir qual usar: o valor previsto ajustado ou o baseado em ATR
            if signal.direction == "LONG":
                if abs(signal.predicted_tp_pct * self.config.tp_adjustment) > reasonable_tp_pct:
                    # Se o TP previsto for muito grande, usar o baseado em ATR
                    signal.predicted_tp_pct = reasonable_tp_pct
                    logger.info(f"TP ajustado para valor baseado em ATR: {reasonable_tp_pct:.2f}%")
                else:
                    # Senão, aplicar o ajuste da configuração
                    signal.predicted_tp_pct = signal.predicted_tp_pct * self.config.tp_adjustment

                if abs(signal.predicted_sl_pct * self.config.sl_adjustment) > reasonable_sl_pct:
                    # Se o SL previsto for muito grande, usar o baseado em ATR
                    signal.predicted_sl_pct = reasonable_sl_pct
                    logger.info(f"SL ajustado para valor baseado em ATR: {reasonable_sl_pct:.2f}%")
                else:
                    # Senão, aplicar o ajuste da configuração
                    signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment
            else:  # SHORT
                if abs(signal.predicted_tp_pct * self.config.tp_adjustment) > reasonable_tp_pct:
                    # Se o TP previsto for muito grande, usar o baseado em ATR
                    signal.predicted_tp_pct = -reasonable_tp_pct  # Negativo para SHORT
                    logger.info(f"TP ajustado para valor baseado em ATR: {-reasonable_tp_pct:.2f}%")
                else:
                    # Senão, aplicar o ajuste da configuração
                    signal.predicted_tp_pct = signal.predicted_tp_pct * self.config.tp_adjustment

                if abs(signal.predicted_sl_pct * self.config.sl_adjustment) > reasonable_sl_pct:
                    # Se o SL previsto for muito grande, usar o baseado em ATR
                    signal.predicted_sl_pct = reasonable_sl_pct
                    logger.info(f"SL ajustado para valor baseado em ATR: {reasonable_sl_pct:.2f}%")
                else:
                    # Senão, aplicar o ajuste da configuração
                    signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment
        else:
            # Se não tivermos ATR, usar os valores da configuração
            signal.predicted_tp_pct = signal.predicted_tp_pct * self.config.tp_adjustment
            signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment

        # Recalcular fatores e preços
        if signal.direction == "LONG":
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:  # SHORT
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Atualizar razão R:R
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

        logger.info(
            f"Sinal ajustado para baixa volatilidade: "
            f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal
