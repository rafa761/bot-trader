# strategies/range_strategy.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from services.prediction.interfaces import ITpSlPredictionService
from services.prediction.tpsl_prediction import TpSlPredictionService
from strategies.base.model import BaseStrategy, StrategyConfig
from strategies.patterns.entry_evaluator import EntryEvaluatorFactory
from strategies.patterns.pattern_analyzer import PatternAnalyzerFactory


class RangeStrategy(BaseStrategy):
    """
    Estratégia avançada para mercados em consolidação (range).

    Foca em capturar movimentos de reversão nos extremos do range com
    alta probabilidade, utilizando múltiplas confirmações e análise de
    rejeição nos limites do range.
    """

    def __init__(self):
        """ Inicializa a estratégia com configuração otimizada para mercados em range. """
        config = StrategyConfig(
            name="Range Strategy",
            description="Estratégia otimizada para mercados laterais (em range)",
            min_rr_ratio=1.1,
            entry_threshold=0.50,
            tp_adjustment=0.85,
            sl_adjustment=0.8,
            entry_aggressiveness=1.0,
            max_sl_percent=1.0,
            min_tp_percent=0.3,
            required_indicators=[
                "adx", "boll_width", "rsi", "boll_lband", "boll_hband",
                "boll_pct_b", "stoch_k", "stoch_d", "atr", "ema_short",
                "ema_long", "volume", "macd", "macd_histogram"
            ]
        )
        super().__init__(config)
        self.prediction_service: ITpSlPredictionService = TpSlPredictionService()

        # Inicializar analisadores de padrões
        pattern_factory = PatternAnalyzerFactory()
        self.momentum_analyzer = pattern_factory.create_momentum_analyzer()
        self.volatility_analyzer = pattern_factory.create_volatility_analyzer()

        # Inicializar avaliador de entradas
        self.entry_evaluator = EntryEvaluatorFactory.create_evaluator("RANGE", config)

        # Armazenar informações do range atual
        self.range_high = None
        self.range_low = None
        self.range_width_pct = None
        self.range_midpoint = None

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.

        Implementa verificações mais robustas para detectar mercados em range,
        analisando ADX, Bollinger Bands, volatilidade e comportamento de preço.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar ADX para identificar ausência de tendência
        adx_low = False
        adx_value = df['adx'].iloc[-1]
        adx_low = adx_value < 20  # ADX baixo indica ausência de tendência

        # Verificar variação do ADX nos últimos períodos
        adx_stable = False
        if len(df) > 10:
            adx_std = df['adx'].iloc[-10:].std()
            adx_stable = adx_std < 2.0  # ADX com pouca variação indica estabilidade

        # Verificar EMAs próximas (indicando ausência de tendência)
        emas_flat = False
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            ema_diff_pct = abs(ema_short - ema_long) / ema_long * 100
            emas_flat = ema_diff_pct < 0.3  # EMAs muito próximas

            # Verificar inclinação das EMAs
            if len(df) > 5:
                ema_slope = 0
                for i in range(1, 5):
                    if df['ema_short'].iloc[-i] > df['ema_short'].iloc[-i - 1]:
                        ema_slope += 1
                    elif df['ema_short'].iloc[-i] < df['ema_short'].iloc[-i - 1]:
                        ema_slope -= 1

                # Se o ema_slope está próximo de zero, a EMA não tem direção clara
                emas_flat = emas_flat and abs(ema_slope) <= 2

        # Verificar tendência multi-timeframe
        mtf_neutral = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_neutral = 'NEUTRAL' in mtf_data['consolidated_trend']

        # Verificar se o preço está contido em um canal horizontal
        price_channel = False
        range_pct = 0.0
        if len(df) > 20:
            # Calcular range dos últimos 20 períodos
            high_range = df['high'].iloc[-20:].max() - df['high'].iloc[-20:].min()
            low_range = df['low'].iloc[-20:].max() - df['low'].iloc[-20:].min()
            close_range = df['close'].iloc[-20:].max() - df['close'].iloc[-20:].min()

            # Se o range é menor que X% do preço atual, consideramos um canal
            avg_price = df['close'].iloc[-1]
            range_pct = (close_range / avg_price) * 100
            price_channel = range_pct < 2.0  # Range menor que 2% indica consolidação

            # Armazenar as informações do range para uso posterior
            if price_channel:
                self.range_high = df['high'].iloc[-20:].max()
                self.range_low = df['low'].iloc[-20:].min()
                self.range_width_pct = range_pct
                self.range_midpoint = (self.range_high + self.range_low) / 2

                logger.info(
                    f"Range identificado: High={self.range_high:.2f}, Low={self.range_low:.2f}, "
                    f"Width={range_pct:.2f}%, Midpoint={self.range_midpoint:.2f}"
                )

        # Verificar largura do Bollinger Band (estreita em consolidação)
        bb_narrow = False
        bb_width = 0.0
        if 'boll_width' in df.columns:
            bb_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            bb_narrow = bb_width < (avg_width * 0.8)  # Bandas mais estreitas que a média

            # Verificar se a largura da banda está estabilizando ou diminuindo
            if len(df) > 5:
                recent_widths = df['boll_width'].iloc[-5:].values
                width_trend = 0
                for i in range(1, len(recent_widths)):
                    if recent_widths[i] < recent_widths[i - 1]:
                        width_trend -= 1
                    else:
                        width_trend += 1

                # Se width_trend é negativo, as bandas estão se estreitando
                bb_narrow = bb_narrow or width_trend < 0

        # Verificar volatilidade baixa usando ATR
        low_volatility = False
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            low_volatility = atr_pct < 0.6
        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            low_volatility = atr_pct < 0.6

        # Verificar volume reduzido (comum em consolidações)
        low_volume = False
        if 'volume' in df.columns and len(df) > 14:
            current_volume = df['volume'].iloc[-5:].mean()
            past_volume = df['volume'].iloc[-14:-5].mean()
            low_volume = current_volume < past_volume * 0.9  # Volume atual < 90% do volume passado

        # Ativar se pelo menos três indicadores confirmarem (aumento do rigor)
        confirmations = sum(
            [
                adx_low, adx_stable, emas_flat, mtf_neutral,
                price_channel, bb_narrow, low_volatility, low_volume
            ]
        )
        should_activate = confirmations >= 3

        if should_activate:
            logger.info(
                f"Estratégia de RANGE ativada: ADX_baixo={adx_low} ({adx_value:.1f}), "
                f"ADX_estável={adx_stable}, EMAs_flat={emas_flat}, MTF_neutral={mtf_neutral}, "
                f"Canal de Preço={price_channel} ({range_pct:.2f}% width), "
                f"BB_estreito={bb_narrow} ({bb_width:.4f}), Volatilidade_baixa={low_volatility}, "
                f"Volume_reduzido={low_volume}"
            )

        return should_activate

    def _identify_range_extremes(self, df: pd.DataFrame) -> tuple[float, float]:
        """
        Identifica os limites superior e inferior do range atual.

        Utiliza múltiplos métodos para determinar os limites com maior precisão,
        incluindo máximos/mínimos recentes, Bollinger Bands e níveis de volume.

        Returns:
            tuple: (limite_superior, limite_inferior) do range
        """
        # Usar os extremos pré-calculados se disponíveis
        if self.range_high is not None and self.range_low is not None:
            return self.range_high, self.range_low

        # Método 1: Usar máximos e mínimos recentes
        if len(df) >= 20:
            high_max = df['high'].iloc[-20:].max()
            low_min = df['low'].iloc[-20:].min()
        else:
            high_max = df['high'].max()
            low_min = df['low'].min()

        # Método 2: Usar Bollinger Bands se disponíveis
        bb_high, bb_low = None, None
        if 'boll_hband' in df.columns and 'boll_lband' in df.columns:
            bb_high = df['boll_hband'].iloc[-1]
            bb_low = df['boll_lband'].iloc[-1]

            # Ajustar com base na força do range
            if 'boll_width' in df.columns:
                width = df['boll_width'].iloc[-1]
                # Se as bandas estão muito estreitas, expandir um pouco
                if width < 0.015:
                    bb_high = bb_high * 1.001
                    bb_low = bb_low * 0.999

        # Método 3: Usar níveis de pivot se disponíveis
        pivot_high, pivot_low = None, None
        if 'pivot_r1' in df.columns and 'pivot_s1' in df.columns:
            pivot_high = df['pivot_r1'].iloc[-1]
            pivot_low = df['pivot_s1'].iloc[-1]

        # Combinar os diferentes métodos para obter limites mais precisos
        upper_levels = [level for level in [high_max, bb_high, pivot_high] if level is not None]
        lower_levels = [level for level in [low_min, bb_low, pivot_low] if level is not None]

        upper_bound = sum(upper_levels) / len(upper_levels) if upper_levels else high_max
        lower_bound = sum(lower_levels) / len(lower_levels) if lower_levels else low_min

        # Armazenar para uso posterior
        self.range_high = upper_bound
        self.range_low = lower_bound
        self.range_width_pct = ((upper_bound - lower_bound) / lower_bound) * 100
        self.range_midpoint = (upper_bound + lower_bound) / 2

        logger.info(
            f"Range calculado: High={upper_bound:.2f}, Low={lower_bound:.2f}, "
            f"Width={self.range_width_pct:.2f}%, Midpoint={self.range_midpoint:.2f}"
        )

        return upper_bound, lower_bound

    def _check_upper_extreme_rejection(self, df: pd.DataFrame, upper_bound: float) -> tuple[bool, float]:
        """
        Verifica se há uma rejeição no extremo superior do range (resistência).

        Returns:
            tuple: (rejeição_detectada, força_da_rejeição de 0 a 1)
        """
        rejection_detected = False
        rejection_strength = 0.0

        # Precisamos do candle atual e anterior
        if len(df) < 2:
            return False, 0.0

        # 1. Verificar proximidade ao limite superior
        current_price = df['close'].iloc[-1]
        previous_high = df['high'].iloc[-2]
        current_high = df['high'].iloc[-1]

        # Distância do preço atual ao limite superior (em percentual)
        distance_to_upper = (upper_bound - current_price) / current_price * 100

        # Verificar se tocamos ou nos aproximamos muito do limite superior
        near_upper = distance_to_upper < 0.3 or previous_high >= upper_bound * 0.998

        if not near_upper:
            return False, 0.0

        # 2. Verificar padrão de rejeição

        # a) Verifica se temos uma vela de alta seguida de uma vela de baixa
        bullish_candle = df['close'].iloc[-2] > df['open'].iloc[-2]
        bearish_candle = df['close'].iloc[-1] < df['open'].iloc[-1]

        pattern1 = bullish_candle and bearish_candle

        # b) Ou se temos uma vela com sombra superior longa
        upper_wick = current_high - max(df['open'].iloc[-1], df['close'].iloc[-1])
        body_size = abs(df['open'].iloc[-1] - df['close'].iloc[-1])

        long_upper_wick = upper_wick > body_size * 1.5

        pattern2 = long_upper_wick and bearish_candle

        # c) Ou se o preço subiu e depois voltou (failed breakout)
        touched_upper = current_high >= upper_bound * 0.998
        closed_below = current_price < upper_bound * 0.995

        pattern3 = touched_upper and closed_below

        # 3. Verificar volume (alto volume na rejeição é mais significativo)
        volume_signal = 0.0
        if 'volume' in df.columns:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:].mean()

            if current_volume > avg_volume * 1.2:  # Volume 20% acima da média
                volume_signal = min((current_volume / avg_volume - 1), 1.0)

        # Combinar todos os sinais para calcular a força da rejeição
        if pattern1 or pattern2 or pattern3:
            rejection_detected = True

            # Base strength
            rejection_strength = 0.6

            # Add to strength based on specific patterns
            if pattern1:
                rejection_strength += 0.1
                logger.info(f"Padrão de rejeição superior 1 detectado: vela alta seguida de vela baixa")
            if pattern2:
                rejection_strength += 0.2
                logger.info(f"Padrão de rejeição superior 2 detectado: vela com sombra superior longa")
            if pattern3:
                rejection_strength += 0.3
                logger.info(f"Padrão de rejeição superior 3 detectado: falha de breakout acima do range")

            # Add volume component
            rejection_strength += volume_signal * 0.2

            # Normalize to 0-1
            rejection_strength = min(rejection_strength, 1.0)

            logger.info(
                f"Rejeição superior detectada próximo a {upper_bound:.2f} com força {rejection_strength:.2f}. "
                f"Distância: {distance_to_upper:.2f}%, Volume: {volume_signal:.2f}"
            )

        return rejection_detected, rejection_strength

    def _check_lower_extreme_rejection(self, df: pd.DataFrame, lower_bound: float) -> tuple[bool, float]:
        """
        Verifica se há uma rejeição no extremo inferior do range (suporte).

        Returns:
            tuple: (rejeição_detectada, força_da_rejeição de 0 a 1)
        """
        rejection_detected = False
        rejection_strength = 0.0

        # Precisamos do candle atual e anterior
        if len(df) < 2:
            return False, 0.0

        # 1. Verificar proximidade ao limite inferior
        current_price = df['close'].iloc[-1]
        previous_low = df['low'].iloc[-2]
        current_low = df['low'].iloc[-1]

        # Distância do preço atual ao limite inferior (em percentual)
        distance_to_lower = (current_price - lower_bound) / current_price * 100

        # Verificar se tocamos ou nos aproximamos muito do limite inferior
        near_lower = distance_to_lower < 0.3 or previous_low <= lower_bound * 1.002

        if not near_lower:
            return False, 0.0

        # 2. Verificar padrão de rejeição

        # a) Verifica se temos uma vela de baixa seguida de uma vela de alta
        bearish_candle = df['close'].iloc[-2] < df['open'].iloc[-2]
        bullish_candle = df['close'].iloc[-1] > df['open'].iloc[-1]

        pattern1 = bearish_candle and bullish_candle

        # b) Ou se temos uma vela com sombra inferior longa
        lower_wick = min(df['open'].iloc[-1], df['close'].iloc[-1]) - current_low
        body_size = abs(df['open'].iloc[-1] - df['close'].iloc[-1])

        long_lower_wick = lower_wick > body_size * 1.5

        pattern2 = long_lower_wick and bullish_candle

        # c) Ou se o preço desceu e depois subiu (failed breakdown)
        touched_lower = current_low <= lower_bound * 1.002
        closed_above = current_price > lower_bound * 1.005

        pattern3 = touched_lower and closed_above

        # 3. Verificar volume (alto volume na rejeição é mais significativo)
        volume_signal = 0.0
        if 'volume' in df.columns:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:].mean()

            if current_volume > avg_volume * 1.2:  # Volume 20% acima da média
                volume_signal = min((current_volume / avg_volume - 1), 1.0)

        # Combinar todos os sinais para calcular a força da rejeição
        if pattern1 or pattern2 or pattern3:
            rejection_detected = True

            # Base strength
            rejection_strength = 0.6

            # Add to strength based on specific patterns
            if pattern1:
                rejection_strength += 0.1
                logger.info(f"Padrão de rejeição inferior 1 detectado: vela baixa seguida de vela alta")
            if pattern2:
                rejection_strength += 0.2
                logger.info(f"Padrão de rejeição inferior 2 detectado: vela com sombra inferior longa")
            if pattern3:
                rejection_strength += 0.3
                logger.info(f"Padrão de rejeição inferior 3 detectado: falha de breakdown abaixo do range")

            # Add volume component
            rejection_strength += volume_signal * 0.2

            # Normalize to 0-1
            rejection_strength = min(rejection_strength, 1.0)

            logger.info(
                f"Rejeição inferior detectada próximo a {lower_bound:.2f} com força {rejection_strength:.2f}. "
                f"Distância: {distance_to_lower:.2f}%, Volume: {volume_signal:.2f}"
            )

        return rejection_detected, rejection_strength



    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em range.
        Busca oportunidades de compra no suporte e venda na resistência.

        Implementa análise avançada de condições de entrada nos extremos do range,
        combinando múltiplas confirmações para maior probabilidade de sucesso.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # 1. Identificar os limites do range
        upper_bound, lower_bound = self._identify_range_extremes(df)
        range_width = upper_bound - lower_bound

        # Se o range for muito estreito, não vale a pena operar
        if (range_width / current_price) * 100 < 0.8:
            logger.info(
                f"Range muito estreito ({(range_width / current_price) * 100:.2f}%), não compensatório para trade"
                )
            return None

        # 2. Verificar rejeição no extremo superior (para SHORT)
        upper_rejection, upper_strength = self._check_upper_extreme_rejection(df, upper_bound)

        # 3. Verificar rejeição no extremo inferior (para LONG)
        lower_rejection, lower_strength = self._check_lower_extreme_rejection(df, lower_bound)

        # 4. Verificar condições de sobrecompra/sobrevenda
        overbought, oversold, oscillator_strength = self.momentum_analyzer.check_overbought_oversold(df)

        # 5. Verificar compressão de volatilidade (squeeze)
        compression, compression_strength = self.volatility_analyzer.detect_range_compression(df)

        # 6. Decidir direção do sinal com base nas análises anteriores
        signal_direction = None
        entry_score = 0.0

        # Condições para SHORT (vender na resistência)
        short_conditions = [
            upper_rejection,
            overbought,
            compression and df['close'].iloc[-1] > self.range_midpoint,  # Compressão na parte superior do range
            current_price > (upper_bound - (range_width * 0.10))  # Preço próximo ao topo do range
        ]

        short_scores = [
            upper_strength * 0.7 if upper_rejection else 0,
            oscillator_strength * 0.6 if overbought else 0,
            compression_strength * 0.5 if compression and df['close'].iloc[-1] > self.range_midpoint else 0,
            min(
                ((current_price - (upper_bound - range_width * 0.10)) / (range_width * 0.10)), 1.0
            ) * 0.4 if current_price > (upper_bound - (range_width * 0.10)) else 0
        ]

        # Condições para LONG (comprar no suporte)
        long_conditions = [
            lower_rejection,
            oversold,
            compression and df['close'].iloc[-1] < self.range_midpoint,  # Compressão na parte inferior do range
            current_price < (lower_bound + (range_width * 0.10))  # Preço próximo ao fundo do range
        ]

        long_scores = [
            lower_strength * 0.7 if lower_rejection else 0,
            oscillator_strength * 0.6 if oversold else 0,
            compression_strength * 0.5 if compression and df['close'].iloc[-1] < self.range_midpoint else 0,
            min(
                ((lower_bound + range_width * 0.10) - current_price) / (range_width * 0.10), 1.0
            ) * 0.4 if current_price < (lower_bound + (range_width * 0.10)) else 0
        ]

        # Calcular score para cada direção - quanto mais condições atendidas e mais fortes, melhor
        short_score = sum(short_scores) / max(1, (sum(1 for c in short_conditions if c) * 0.7))
        long_score = sum(long_scores) / max(1, (sum(1 for c in long_conditions if c) * 0.7))

        # Logar as condições para análise
        logger.info(
            f"Score SHORT: {short_score:.2f} (condições: {sum(1 for c in short_conditions if c)}/4) - "
            f"Rejeição superior={upper_rejection} ({upper_strength:.2f}), "
            f"Sobrecompra={overbought} ({oscillator_strength:.2f}), "
            f"Compressão superior={compression and df['close'].iloc[-1] > self.range_midpoint} ({compression_strength:.2f}), "
            f"Próximo ao topo={current_price > (upper_bound - (range_width * 0.10))}"
        )

        # Percentual mínimo de condições e score mínimo
        min_conditions_pct = 0.5  # Pelo menos 50% das condições
        min_score = 0.5

        # Verificar qual direção tem melhores condições e score
        short_conditions_met = sum(1 for c in short_conditions if c) / len(short_conditions)
        long_conditions_met = sum(1 for c in long_conditions if c) / len(long_conditions)

        if short_conditions_met >= min_conditions_pct and short_score >= min_score and short_score > long_score:
            signal_direction = "SHORT"
            entry_score = short_score
            logger.info(f"Condições para SHORT em range atendidas com score {short_score:.2f}")
        elif long_conditions_met >= min_conditions_pct and long_score >= min_score and long_score > short_score:
            signal_direction = "LONG"
            entry_score = long_score
            logger.info(f"Condições para LONG em range atendidas com score {long_score:.2f}")
        else:
            logger.info(f"Condições insuficientes para gerar sinal neste ciclo")
            return None

        # Se chegamos aqui, temos uma direção definida para o sinal
        if signal_direction:
            # Usar o serviço de previsão para obter TP/SL
            prediction = self.prediction_service.predict_tp_sl(df, current_price, signal_direction)
            if prediction is None:
                return None

            predicted_tp_pct, predicted_sl_pct = prediction

            # Limitar TP ao range em mercados laterais
            if signal_direction == "LONG":
                # Calcular o máximo TP possível (até próximo da resistência)
                max_tp_pct = ((upper_bound * 0.995) - current_price) / current_price * 100

                # Se o TP previsto for maior que o máximo possível, ajustar
                if predicted_tp_pct > max_tp_pct:
                    logger.info(
                        f"Ajustando TP para respeitar o limite superior do range: {predicted_tp_pct:.2f}% -> {max_tp_pct:.2f}%"
                    )
                    predicted_tp_pct = max_tp_pct
            else:  # SHORT
                # Calcular o máximo TP possível (até próximo do suporte)
                max_tp_pct = (current_price - (lower_bound * 1.005)) / current_price * 100

                # Se o TP previsto (em módulo) for maior que o máximo possível, ajustar
                if abs(predicted_tp_pct) > max_tp_pct:
                    logger.info(
                        f"Ajustando TP para respeitar o limite inferior do range: {predicted_tp_pct:.2f}% -> {-max_tp_pct:.2f}%"
                    )
                    predicted_tp_pct = -max_tp_pct

        # Avaliar a qualidade da entrada
        should_enter, final_entry_score = self.entry_evaluator.evaluate_entry_quality(
            df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct,
            entry_threshold=self.config.entry_threshold
        )

        # Combinar o score da avaliação de qualidade com o score de condições
        final_entry_score = 0.6 * final_entry_score + 0.4 * entry_score

        if not should_enter:
            logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {final_entry_score:.2f})")
            return None

        # TP em múltiplos níveis para trade em range
        # Em mercados laterais, é recomendável ter 2 níveis de TP
        tp_levels = []

        first_tp_pct = predicted_tp_pct * 0.40  # 40% do caminho
        second_tp_pct = predicted_tp_pct * 0.70  # 70% do caminho
        third_tp_pct = predicted_tp_pct  # 100% do caminho
        tp_levels = [first_tp_pct, second_tp_pct, third_tp_pct]
        tp_percents = [50, 30, 20]

        # Configurar side e position_side para a Binance
        if signal_direction == "LONG":
            side: Literal["SELL", "BUY"] = "BUY"
            position_side: Literal["LONG", "SHORT"] = "LONG"
            tp_factor = 1 + (predicted_tp_pct / 100)
            sl_factor = 1 - (predicted_sl_pct / 100)
        else:  # SHORT
            side: Literal["SELL", "BUY"] = "SELL"
            position_side: Literal["LONG", "SHORT"] = "SHORT"
            tp_factor = 1 - (abs(predicted_tp_pct) / 100)
            sl_factor = 1 + (predicted_sl_pct / 100)

        # Calcular preços TP/SL
        tp_price = current_price * tp_factor
        sl_price = current_price * sl_factor

        # Obter ATR para ajustes de quantidade
        atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

        # Calcular a razão Risco:Recompensa
        rr_ratio = abs(predicted_tp_pct / predicted_sl_pct)

        # Determinar tendência e força
        market_trend = "NEUTRAL"  # Em range, a tendência é neutra
        market_strength = "WEAK_TREND" if df['adx'].iloc[-1] < 20 else "MODERATE_TREND"

        # Gerar ID único para o sinal
        signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

        # Criar o sinal
        signal = TradingSignal(
            id=signal_id,
            direction=signal_direction,
            side=side,
            position_side=position_side,
            predicted_tp_pct=predicted_tp_pct,
            predicted_sl_pct=predicted_sl_pct,
            tp_price=tp_price,
            sl_price=sl_price,
            current_price=current_price,
            tp_factor=tp_factor,
            sl_factor=sl_factor,
            atr_value=atr_value,
            entry_score=final_entry_score,
            rr_ratio=rr_ratio,
            market_trend=market_trend,
            market_strength=market_strength,
            timestamp=datetime.now(),
            # Campos adicionais para gerenciamento de risco avançado
            mtf_details={
                'tp_levels': tp_levels,
                'tp_percents': tp_percents,
                'range_high': upper_bound,
                'range_low': lower_bound,
                'range_width_pct': (range_width / lower_bound) * 100
            } if 'mtf_details' in TradingSignal.__annotations__ else None
        )

        return signal


    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em range.

        Implementa limites inteligentes para TP e SL com base nos limites do range,
        e configura take profits parciais para maximizar captura em mercados laterais.
        """
        # 1. Identificar os limites do range (se ainda não tivermos)
        if self.range_high is None or self.range_low is None:
            self.range_high, self.range_low = self._identify_range_extremes(df)

        range_high = self.range_high
        range_low = self.range_low

        # 2. Limitar TP para não ultrapassar os limites do range (margem de segurança)
        # Para LONG, o TP não deve exceder a resistência
        if signal.direction == "LONG":
            # Deixar uma margem de 0.5% abaixo da resistência
            reasonable_tp = range_high * 0.995

            # Se o TP previsto for além do razoável, ajustar
            if signal.tp_price > reasonable_tp:
                new_tp_pct = ((reasonable_tp / signal.current_price) - 1) * 100

                # Logar a mudança
                logger.info(
                    f"Ajustando TP para respeitar resistência: "
                    f"{signal.predicted_tp_pct:.2f}% ({signal.tp_price:.2f}) -> "
                    f"{new_tp_pct:.2f}% ({reasonable_tp:.2f})"
                )

                # Atualizar o signal
                signal.predicted_tp_pct = new_tp_pct
                signal.tp_price = reasonable_tp
                signal.tp_factor = reasonable_tp / signal.current_price

        # Para SHORT, o TP não deve exceder o suporte
        elif signal.direction == "SHORT":
            # Deixar uma margem de 0.5% acima do suporte
            reasonable_tp = range_low * 1.005

            # Se o TP previsto for além do razoável, ajustar
            if signal.tp_price < reasonable_tp:
                new_tp_pct = ((signal.current_price / reasonable_tp) - 1) * 100

                # Logar a mudança
                logger.info(
                    f"Ajustando TP para respeitar suporte: "
                    f"{signal.predicted_tp_pct:.2f}% ({signal.tp_price:.2f}) -> "
                    f"{-new_tp_pct:.2f}% ({reasonable_tp:.2f})"
                )

                # Atualizar o signal (TP para short é negativo)
                signal.predicted_tp_pct = -new_tp_pct
                signal.tp_price = reasonable_tp
                signal.tp_factor = reasonable_tp / signal.current_price

        # 3. Ajustar TP/SL usando os fatores de configuração
        original_tp = signal.predicted_tp_pct
        original_sl = signal.predicted_sl_pct

        # Aplicar ajustes gerais da estratégia
        signal.predicted_tp_pct = original_tp * self.config.tp_adjustment
        signal.predicted_sl_pct = original_sl * self.config.sl_adjustment

        # 4. Recalcular preços e fatores
        if signal.direction == "LONG":
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:  # SHORT
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # 5. Definir take profits parciais
        # Em mercados laterais, é melhor ter saídas parciais antes do extremo
        tp_levels = []
        tp_percents = []

        # Calcular com base na largura do range
        range_width_pct = ((range_high - range_low) / range_low) * 100

        # Se o range for amplo, pode valer a pena ter 3 saídas
        if range_width_pct > 3.0:
            tp_levels = [
                signal.predicted_tp_pct * 0.4,  # 40% do movimento
                signal.predicted_tp_pct * 0.7,  # 70% do movimento
                signal.predicted_tp_pct  # movimento completo
            ]
            tp_percents = [40, 40, 20]  # 40%, 40%, 20%
        else:
            # Range mais estreito, apenas 2 saídas
            tp_levels = [
                signal.predicted_tp_pct * 0.6,  # 60% do movimento
                signal.predicted_tp_pct  # movimento completo
            ]
            tp_percents = [60, 40]  # 60%, 40%

        # 6. Adicionar dados de TP parcial se a estrutura do sinal suportar
        if hasattr(signal, 'mtf_details') and signal.mtf_details is not None:
            signal.mtf_details['tp_levels'] = tp_levels
            signal.mtf_details['tp_percents'] = tp_percents
            signal.mtf_details['range_high'] = range_high
            signal.mtf_details['range_low'] = range_low
            signal.mtf_details['range_width_pct'] = range_width_pct

        # 7. Atualizar a razão R:R
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

        logger.info(
            f"Sinal ajustado para mercado em range: "
            f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}, TP Níveis: {[f'{tp:.2f}%' for tp in tp_levels]}"
        )

        return signal
