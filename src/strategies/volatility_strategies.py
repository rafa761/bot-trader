# strategies/volatility_strategies.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.config import settings
from core.logger import logger
from services.base.schemas import TradingSignal
from services.prediction.interfaces import ITpSlPredictionService
from services.prediction.tpsl_prediction import TpSlPredictionService
from strategies.base.model import BaseStrategy, StrategyConfig
from strategies.patterns.entry_evaluator import EntryEvaluatorFactory
from strategies.patterns.pattern_analyzer import PatternAnalyzerFactory


class HighVolatilityStrategy(BaseStrategy):
    """
    Estratégia avançada para mercados com alta volatilidade.

    Capitaliza em movimentos amplos e explosivos enquanto implementa proteções
    adicionais para gerenciar o risco elevado. Utiliza análise de momentum,
    breakouts e volume para entradas precisas e gerenciamento dinâmico de posição.
    """

    def __init__(self):
        """ Inicializa a estratégia com configuração otimizada para alta volatilidade. """
        config = StrategyConfig(
            name="High Volatility Strategy",
            description="Estratégia otimizada para mercados com alta volatilidade",
            min_rr_ratio=1.4,
            entry_threshold=0.55,
            tp_adjustment=1.3,
            sl_adjustment=1.4,
            entry_aggressiveness=0.9,
            max_sl_percent=2.5,
            min_tp_percent=1.0,
            required_indicators=[
                "adx", "atr", "atr_pct", "boll_width", "boll_hband", "boll_lband",
                "rsi", "stoch_k", "stoch_d", "macd", "macd_histogram", "volume",
                "ema_short", "ema_long", "di_plus", "di_minus", "obv"
            ]
        )
        super().__init__(config)
        self.prediction_service: ITpSlPredictionService = TpSlPredictionService()

        # Inicializar analisadores de padrões
        pattern_factory = PatternAnalyzerFactory()
        self.momentum_analyzer = pattern_factory.create_momentum_analyzer()
        self.volatility_analyzer = pattern_factory.create_volatility_analyzer()

        # Inicializar avaliador de entradas
        self.entry_evaluator = EntryEvaluatorFactory.create_evaluator("HIGH_VOLATILITY", config)

        # Armazenar cache de informações sobre a volatilidade
        self.volatility_data = {
            'atr_pct': None,
            'recent_max_vol': None,
            'volatility_onset': False,
            'vol_increasing': False,
            'vol_extremity': 0.0
        }

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia de alta volatilidade deve ser ativada.

        Implementa verificações robustas de múltiplas métricas de volatilidade
        para identificar precisamente mercados de alta volatilidade.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # 1. Verificar ATR percentual - métrica primária de volatilidade
        high_atr = False
        atr_pct = 0.0
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            high_atr = atr_pct > settings.VOLATILITY_HIGH_THRESHOLD

            # Armazenar o ATR% para uso futuro
            self.volatility_data['atr_pct'] = atr_pct

        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            high_atr = atr_pct > settings.VOLATILITY_HIGH_THRESHOLD

            # Armazenar o ATR% para uso futuro
            self.volatility_data['atr_pct'] = atr_pct

        # 2. Verificar tendência crescente de ATR (volatilidade aumentando)
        increasing_volatility = False
        if 'atr' in df.columns and len(df) > 5:
            recent_atr = df['atr'].iloc[-1]
            prev_atr = df['atr'].iloc[-5]

            # Volatilidade crescente se ATR aumentou pelo menos 30%
            increasing_volatility = recent_atr > (prev_atr * 1.3)
            self.volatility_data['vol_increasing'] = increasing_volatility

            # Calcular a extremidade da volatilidade (quão recente é o pico)
            if len(df) > 10:
                max_atr = df['atr'].iloc[-10:].max()
                self.volatility_data['recent_max_vol'] = max_atr

                # Se estamos a menos de 20% do máximo recente, estamos num pico
                vol_extremity = recent_atr / max_atr
                self.volatility_data['vol_extremity'] = vol_extremity

                # Volatilidade "onset" = aumento recente e significativo (mais relevante para entradas)
                if increasing_volatility and vol_extremity > 0.9:
                    self.volatility_data['volatility_onset'] = True
                    logger.info(f"Início recente de alta volatilidade detectado (extremidade: {vol_extremity:.2f})")

        # 3. Verificar a largura das Bollinger Bands
        wide_bands = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            wide_bands = boll_width > (avg_width * 1.5)

            # Verificar também se as bandas estão se expandindo
            if len(df) > 5:
                width_5_ago = df['boll_width'].iloc[-5]
                bands_expanding = boll_width > width_5_ago * 1.2  # 20% mais largas

                if bands_expanding:
                    logger.info(f"Bandas de Bollinger em expansão: {boll_width:.4f} vs {width_5_ago:.4f}")

        # 4. Verificar tendência forte (alta volatilidade em tendência)
        strong_trend = False
        trend_direction = "NEUTRAL"
        if 'adx' in df.columns and 'di_plus' in df.columns and 'di_minus' in df.columns:
            adx_value = df['adx'].iloc[-1]
            di_plus = df['di_plus'].iloc[-1]
            di_minus = df['di_minus'].iloc[-1]

            strong_trend = adx_value > 30

            # Determinar direção da tendência
            if di_plus > di_minus:
                trend_direction = "UPTREND"
            elif di_minus > di_plus:
                trend_direction = "DOWNTREND"

            if strong_trend:
                logger.info(f"Tendência forte detectada: ADX={adx_value:.1f}, Direção={trend_direction}")

        # 5. Verificar movimentos rápidos no preço (candles longos)
        recent_large_moves = False
        if len(df) > 5:
            avg_candle_size = 0
            for i in range(1, 6):  # Últimos 5 candles
                candle_size = abs(df['close'].iloc[-i] - df['open'].iloc[-i]) / df['close'].iloc[-i] * 100
                avg_candle_size += candle_size

            avg_candle_size /= 5
            recent_large_moves = avg_candle_size > 0.8  # Média de candles > 0.8%

            if recent_large_moves:
                logger.info(f"Movimentos recentes significativos: tamanho médio de candle = {avg_candle_size:.2f}%")

        # 6. Verificar aumento de volume
        volume_surge = False
        if 'volume' in df.columns and len(df) > 5:
            recent_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:-1].mean()  # Média de 9 períodos anteriores

            volume_surge = recent_volume > (avg_volume * 1.5)  # Volume 50% acima da média

            if volume_surge:
                logger.info(f"Aumento de volume detectado: {recent_volume:.0f} vs média {avg_volume:.0f}")

        # 7. Verificar alinhamento multi-timeframe
        mtf_volatile = False
        if mtf_data and 'tf_details' in mtf_data:
            # Contar quantos timeframes mostram alta volatilidade
            volatile_timeframes = 0
            total_timeframes = 0

            for tf, details in mtf_data['tf_details'].items():
                total_timeframes += 1
                if 'volatility' in details and details['volatility'] == 'HIGH':
                    volatile_timeframes += 1

            # Se pelo menos 50% dos timeframes mostram alta volatilidade
            if total_timeframes > 0 and (volatile_timeframes / total_timeframes) >= 0.5:
                mtf_volatile = True
                logger.info(f"Volatilidade alta em múltiplos timeframes: {volatile_timeframes}/{total_timeframes}")

        # Ativar se pelo menos três indicadores confirmarem alta volatilidade
        confirmations = sum(
            [
                high_atr, increasing_volatility, wide_bands,
                strong_trend, recent_large_moves, volume_surge, mtf_volatile
            ]
        )
        should_activate = confirmations >= 3

        if should_activate:
            logger.info(
                f"Estratégia de ALTA VOLATILIDADE ativada: "
                f"ATR_alto={high_atr} ({atr_pct:.2f}%), "
                f"Volatilidade_crescente={increasing_volatility}, "
                f"Bandas_largas={wide_bands}, "
                f"Tendência_forte={strong_trend} (ADX={df['adx'].iloc[-1]:.1f}, {trend_direction}), "
                f"Grandes_movimentos={recent_large_moves}, "
                f"Aumento_volume={volume_surge}, "
                f"MTF_volátil={mtf_volatile}"
            )

        return should_activate


    def _detect_strong_momentum(self, df: pd.DataFrame) -> tuple[bool, float, str]:
        """
        Detecta momentum forte, comum em mercados voláteis.

        Momentum forte é caracterizado por movimento consistente em uma direção,
        confirmado por múltiplos indicadores.

        Returns:
            tuple: (momentum_forte, força_do_momentum de 0 a 1, direção do momentum)
        """
        if len(df) < 10:  # Precisamos de histórico suficiente
            return False, 0.0, "NONE"

        strong_momentum = False
        momentum_strength = 0.0
        momentum_direction = "NONE"

        # 1. Verificar ADX e indicadores direcionais
        adx_signal = 0.0
        if 'adx' in df.columns and 'di_plus' in df.columns and 'di_minus' in df.columns:
            adx = df['adx'].iloc[-1]
            di_plus = df['di_plus'].iloc[-1]
            di_minus = df['di_minus'].iloc[-1]

            # ADX > 25 indica tendência
            if adx > 25:
                # Normalizar força do ADX (25-50 -> 0-1)
                adx_strength = min((adx - 25) / 25, 1.0)

                # Direcional Indexes determinam a direção
                if di_plus > di_minus:
                    momentum_direction = "UP"
                    # A diferença entre +DI e -DI indica a força
                    di_diff = (di_plus - di_minus) / 100
                    # Combinar ADX e DI para uma melhor pontuação
                    momentum_strength = adx_strength * 0.7 + di_diff * 0.3

                    logger.info(
                        f"Momentum de alta detectado: ADX={adx:.1f}, +DI={di_plus:.1f}, -DI={di_minus:.1f} "
                        f"(força: {momentum_strength:.2f})"
                    )
                    strong_momentum = True

                elif di_minus > di_plus:
                    momentum_direction = "DOWN"
                    # A diferença entre -DI e +DI indica a força
                    di_diff = (di_minus - di_plus) / 100
                    # Combinar ADX e DI para uma melhor pontuação
                    momentum_strength = adx_signal * 0.7 + di_diff * 0.3

                    logger.info(
                        f"Momentum de baixa detectado: ADX={adx:.1f}, +DI={di_plus:.1f}, -DI={di_minus:.1f} "
                        f"(força: {momentum_strength:.2f})"
                    )
                    strong_momentum = True

        # 2. Verificar MACD para confirmar momentum
        if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_hist = df['macd_histogram'].iloc[-1]
            prev_hist = df['macd_histogram'].iloc[-2] if len(df) > 2 else 0

            # MACD acima da linha de sinal e histograma crescente = momentum de alta
            if macd > macd_signal and macd_hist > 0 and macd_hist > prev_hist:
                # Se já temos um sinal de momentum, verificar se confirma a direção
                if strong_momentum and momentum_direction == "UP":
                    # Adicionar um bônus ao momentum
                    momentum_strength = min(momentum_strength + 0.15, 1.0)
                    logger.info(f"Momentum de alta confirmado por MACD")
                # Senão, criar novo sinal
                elif not strong_momentum:
                    momentum_direction = "UP"
                    momentum_strength = 0.5
                    strong_momentum = True
                    logger.info(f"Momentum de alta detectado pelo MACD")

            # MACD abaixo da linha de sinal e histograma decrescente = momentum de baixa
            elif macd < macd_signal and macd_hist < 0 and macd_hist < prev_hist:
                # Se já temos um sinal de momentum, verificar se confirma a direção
                if strong_momentum and momentum_direction == "DOWN":
                    # Adicionar um bônus ao momentum
                    momentum_strength = min(momentum_strength + 0.15, 1.0)
                    logger.info(f"Momentum de baixa confirmado por MACD")
                # Senão, criar novo sinal
                elif not strong_momentum:
                    momentum_direction = "DOWN"
                    momentum_strength = 0.5
                    strong_momentum = True
                    logger.info(f"Momentum de baixa detectado pelo MACD")

        # 3. Verificar sequência de candles na mesma direção
        if len(df) >= 5:
            bullish_candles = 0
            bearish_candles = 0

            for i in range(1, 6):  # Últimos 5 candles
                if df['close'].iloc[-i] > df['open'].iloc[-i]:
                    bullish_candles += 1
                elif df['close'].iloc[-i] < df['open'].iloc[-i]:
                    bearish_candles += 1

            # Sequência significativa na mesma direção (4+ de 5 candles)
            if bullish_candles >= 4:
                # Se já temos um sinal de momentum, verificar se confirma a direção
                if strong_momentum and momentum_direction == "UP":
                    # Adicionar um bônus ao momentum
                    candle_bonus = (bullish_candles - 3) * 0.1
                    momentum_strength = min(momentum_strength + candle_bonus, 1.0)
                    logger.info(f"Momentum de alta confirmado por {bullish_candles}/5 velas de alta")
                # Senão, criar novo sinal
                elif not strong_momentum:
                    momentum_direction = "UP"
                    momentum_strength = 0.4 + (bullish_candles - 3) * 0.1
                    strong_momentum = True
                    logger.info(f"Momentum de alta detectado por {bullish_candles}/5 velas de alta")

            elif bearish_candles >= 4:
                # Se já temos um sinal de momentum, verificar se confirma a direção
                if strong_momentum and momentum_direction == "DOWN":
                    # Adicionar um bônus ao momentum
                    candle_bonus = (bearish_candles - 3) * 0.1
                    momentum_strength = min(momentum_strength + candle_bonus, 1.0)
                    logger.info(f"Momentum de baixa confirmado por {bearish_candles}/5 velas de baixa")
                # Senão, criar novo sinal
                elif not strong_momentum:
                    momentum_direction = "DOWN"
                    momentum_strength = 0.4 + (bearish_candles - 3) * 0.1
                    strong_momentum = True
                    logger.info(f"Momentum de baixa detectado por {bearish_candles}/5 velas de baixa")

        # 4. Verificar OBV (On-Balance Volume) para confirmar momentum por volume
        if strong_momentum and 'obv' in df.columns and len(df) > 10:
            current_obv = df['obv'].iloc[-1]
            obv_5_ago = df['obv'].iloc[-5]

            # OBV crescente confirma alta, decrescente confirma baixa
            if momentum_direction == "UP" and current_obv > obv_5_ago * 1.01:  # +1%
                obv_boost = min((current_obv / obv_5_ago - 1.0) * 2, 0.2)
                momentum_strength = min(momentum_strength + obv_boost, 1.0)
                logger.info(f"Momentum de alta confirmado por OBV crescente: +{obv_boost:.2f} força")

            elif momentum_direction == "DOWN" and current_obv < obv_5_ago * 0.99:  # -1%
                obv_boost = min((1.0 - current_obv / obv_5_ago) * 2, 0.2)
                momentum_strength = min(momentum_strength + obv_boost, 1.0)
                logger.info(f"Momentum de baixa confirmado por OBV decrescente: +{obv_boost:.2f} força")

        return strong_momentum, momentum_strength, momentum_direction


    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com alta volatilidade.

        Em mercados voláteis, busca entradas de alta probabilidade em:
        1. Breakouts confirmados por volume
        2. Momentum forte na direção da tendência primária
        3. Reversões após movimentos extremos

        Implementa análise avançada de condições de entrada com ponderação.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # 1. Verificar volatilidade extrema (desistir para evitar falsos sinais)
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            if atr_pct > 3.0:  # Volatilidade extremamente alta
                logger.warning(f"Volatilidade extrema detectada (ATR% = {atr_pct:.2f}%). Evitando entradas.")
                return None

        # 2. Detectar breakouts
        breakout_detected, breakout_strength, breakout_direction = self.volatility_analyzer.detect_breakout(
            df, current_price
        )

        # 3. Detectar momentum forte
        strong_momentum, momentum_strength, momentum_direction = self._detect_strong_momentum(df)

        # 4. Detectar potencial de reversão
        reversal_potential, reversal_strength, reversal_direction = self.momentum_analyzer.detect_reversal_potential(df)

        # 5. Verificar alinhamento multi-timeframe
        mtf_aligned = False
        mtf_direction = "NONE"
        mtf_strength = 0.0

        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']

            # Determinar direção e força com base na tendência MTF
            if 'UPTREND' in mtf_trend:
                mtf_aligned = True
                mtf_direction = "UP"
                mtf_strength = 0.6  # Valor base

                # Aumentar força baseado na confiança, se disponível
                if 'confidence' in mtf_data:
                    mtf_strength = min(mtf_data['confidence'] / 100, 1.0) * 0.8

                logger.info(f"Alinhamento MTF para UP: {mtf_trend} (força: {mtf_strength:.2f})")

            elif 'DOWNTREND' in mtf_trend:
                mtf_aligned = True
                mtf_direction = "DOWN"
                mtf_strength = 0.6  # Valor base

                # Aumentar força baseado na confiança, se disponível
                if 'confidence' in mtf_data:
                    mtf_strength = min(mtf_data['confidence'] / 100, 1.0) * 0.8

                logger.info(f"Alinhamento MTF para DOWN: {mtf_trend} (força: {mtf_strength:.2f})")

        # 6. Determinar direção do sinal com base nas análises
        signal_direction = None
        signal_strength = 0.0

        # Condições para LONG
        long_conditions = {
            'breakout': breakout_detected and breakout_direction == "UP",
            'momentum': strong_momentum and momentum_direction == "UP",
            'reversal': reversal_potential and reversal_direction == "UP",
            'mtf_aligned': mtf_aligned and mtf_direction == "UP"
        }

        long_scores = {
            'breakout': breakout_strength * 0.9 if long_conditions['breakout'] else 0,
            'momentum': momentum_strength * 0.8 if long_conditions['momentum'] else 0,
            'reversal': reversal_strength * 0.7 if long_conditions['reversal'] else 0,
            'mtf_aligned': mtf_strength * 0.6 if long_conditions['mtf_aligned'] else 0
        }

        # Condições para SHORT
        short_conditions = {
            'breakout': breakout_detected and breakout_direction == "DOWN",
            'momentum': strong_momentum and momentum_direction == "DOWN",
            'reversal': reversal_potential and reversal_direction == "DOWN",
            'mtf_aligned': mtf_aligned and mtf_direction == "DOWN"
        }

        short_scores = {
            'breakout': breakout_strength * 0.9 if short_conditions['breakout'] else 0,
            'momentum': momentum_strength * 0.8 if short_conditions['momentum'] else 0,
            'reversal': reversal_strength * 0.7 if short_conditions['reversal'] else 0,
            'mtf_aligned': mtf_strength * 0.6 if short_conditions['mtf_aligned'] else 0
        }

        # Calcular pontuação para cada direção
        long_score = sum(long_scores.values())
        short_score = sum(short_scores.values())

        # Número mínimo de condições (pelo menos 2 das 4 condições)
        min_conditions = 2
        min_score = 0.5

        # Contar condições atendidas
        long_conditions_met = sum(1 for c in long_conditions.values() if c)
        short_conditions_met = sum(1 for c in short_conditions.values() if c)

        # Log de condições
        logger.info(
            f"Score LONG: {long_score:.2f} (condições: {long_conditions_met}/{len(long_conditions)}) - "
            f"Score SHORT: {short_score:.2f} (condições: {short_conditions_met}/{len(long_conditions)}) - "
            f"Breakout={long_conditions['breakout']} ({long_scores['breakout']:.2f}), "
            f"Momentum={long_conditions['momentum']} ({long_scores['momentum']:.2f}), "
            f"Reversão={long_conditions['reversal']} ({long_scores['reversal']:.2f}), "
            f"MTF={long_conditions['mtf_aligned']} ({long_scores['mtf_aligned']:.2f})"
        )

        # Determinar sinal de entrada com base em scores e condições
        if long_conditions_met >= min_conditions and long_score >= min_score and long_score > short_score:
            signal_direction: Literal["LONG", "SHORT"] = "LONG"
            signal_strength = long_score
            logger.info(f"Sinal LONG gerado em alta volatilidade com força {signal_strength:.2f}")

        elif short_conditions_met >= min_conditions and short_score >= min_score and short_score > long_score:
            signal_direction: Literal["LONG", "SHORT"] = "SHORT"
            signal_strength = short_score
            logger.info(f"Sinal SHORT gerado em alta volatilidade com força {signal_strength:.2f}")

        # Se não temos direção clara, não gerar sinal
        if not signal_direction:
            logger.info(f"Condições insuficientes para gerar sinal em alta volatilidade")
            return None

        # Usar o serviço de previsão para obter TP/SL
        prediction = self.prediction_service.predict_tp_sl(df, current_price, signal_direction)
        if prediction is None:
            return None

        predicted_tp_pct, predicted_sl_pct = prediction

        # Em mercados voláteis, podemos ser mais agressivos no TP
        # mas precisamos de SL mais largos para evitar stops prematuros

        # Ajustar TP e SL com base no ATR
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            # Para SL, garantir que ele seja pelo menos 1.5x o ATR
            min_sl_pct = atr_pct * 1.5
            if predicted_sl_pct < min_sl_pct:
                logger.info(
                    f"Ajustando SL para melhor proteção em mercado volátil: {predicted_sl_pct:.2f}% -> {min_sl_pct:.2f}%"
                )
                predicted_sl_pct = min_sl_pct

            # Para TP, garantir que seja pelo menos 2x o ATR para compensar o risco
            min_tp_pct = atr_pct * 2.5
            if abs(predicted_tp_pct) < min_tp_pct:
                adjusted_tp = min_tp_pct if signal_direction == "LONG" else -min_tp_pct
                logger.info(
                    f"Ajustando TP para melhor recompensa em mercado volátil: {predicted_tp_pct:.2f}% -> {adjusted_tp:.2f}%"
                )
                predicted_tp_pct = adjusted_tp

        # Avaliar a qualidade da entrada
        should_enter, entry_score = self.entry_evaluator.evaluate_entry_quality(
            df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct,
            entry_threshold=self.config.entry_threshold
        )

        # Combinar o score de condições com o score de qualidade da entrada
        entry_score = 0.5 * entry_score + 0.5 * signal_strength

        if not should_enter:
            logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
            return None

        # TP em múltiplos níveis para mercados voláteis (distribução 30%, 30%, 40%)
        tp_levels = []

        first_tp_pct = predicted_tp_pct * 0.25  # 25% do caminho
        second_tp_pct = predicted_tp_pct * 0.50  # 50% do caminho
        third_tp_pct = predicted_tp_pct * 0.75  # 75% do caminho
        fourth_tp_pct = predicted_tp_pct  # 100% do caminho
        tp_levels = [first_tp_pct, second_tp_pct, third_tp_pct, fourth_tp_pct]
        tp_percents = [30, 30, 30, 10]  # Saída progressiva com menor exposição ao final

        # Configurar parâmetros para o sinal
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
        trend_direction = "UPTREND" if signal_direction == "LONG" else "DOWNTREND"
        market_trend = trend_direction
        market_strength = "HIGH_VOLATILITY"

        # Gerar ID único para o sinal
        signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

        # Criar o sinal com dados adicionais de volatilidade e gerenciamento de risco
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
            entry_score=entry_score,
            rr_ratio=rr_ratio,
            market_trend=market_trend,
            market_strength=market_strength,
            timestamp=datetime.now(),
            # Campos adicionais para gerenciamento de risco avançado
            mtf_trend=market_trend,
            mtf_confidence=mtf_strength if mtf_strength else None,
            mtf_alignment=signal_strength if signal_strength else None,
            mtf_details={
                'entry_conditions_score': signal_strength,
                'breakout_strength': breakout_strength if breakout_detected else 0,
                'momentum_strength': momentum_strength if strong_momentum else 0,
                'reversal_strength': reversal_strength if reversal_potential else 0,
                'tp_levels': tp_levels,
                'tp_percents': tp_percents,
                'volatility_data': self.volatility_data
            } if 'mtf_details' in TradingSignal.__annotations__ else None
        )

        return signal


    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados com alta volatilidade.

        Implementa gerenciamento de risco robusto para alta volatilidade:
        - Trailing stops adaptativos
        - Take profits em múltiplos níveis
        - SL mais largos para evitar stops prematuros
        - TP mais agressivos para maximizar captura em momentos explosivos
        """
        # Obter o ATR para calcular ajustes mais precisos
        atr_value = None
        atr_pct = None

        if 'atr' in df.columns:
            atr_value = df['atr'].iloc[-1]
            atr_pct = (atr_value / signal.current_price) * 100

        elif 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            atr_value = (atr_pct / 100) * signal.current_price

        # Aplicar ajustes baseados no nível de volatilidade
        if atr_pct is not None:
            # Ajustes dinâmicos baseados no nível de volatilidade
            tp_volatility_factor = 1.0
            sl_volatility_factor = 1.0

            # Volatilidade muito alta - ser mais conservador
            if atr_pct > 2.5:
                tp_volatility_factor = 1.2  # Aumentar TP para maior captura
                sl_volatility_factor = 1.4  # SL consideravelmente mais largo
                logger.info(f"Volatilidade extrema ({atr_pct:.2f}%): ajustes maiores em TP/SL")

            # Volatilidade alta - ajustes moderados
            elif atr_pct > 1.5:
                tp_volatility_factor = 1.1
                sl_volatility_factor = 1.3
                logger.info(f"Volatilidade alta ({atr_pct:.2f}%): ajustes moderados em TP/SL")

            # Volatilidade moderada - ajustes leves
            else:
                tp_volatility_factor = 1.0
                sl_volatility_factor = 1.2
                logger.info(f"Volatilidade moderada ({atr_pct:.2f}%): ajustes leves em TP/SL")

            # Aplicar os fatores de ajuste de volatilidade aos fatores base da configuração
            tp_adj_factor = self.config.tp_adjustment * tp_volatility_factor
            sl_adj_factor = self.config.sl_adjustment * sl_volatility_factor

        else:
            # Se não temos ATR, usar os valores padrão da configuração
            tp_adj_factor = self.config.tp_adjustment
            sl_adj_factor = self.config.sl_adjustment

        # Armazenar os valores originais para log
        original_tp = signal.predicted_tp_pct
        original_sl = signal.predicted_sl_pct

        # Aplicar os ajustes
        signal.predicted_tp_pct = abs(original_tp) * tp_adj_factor
        if signal.direction == "SHORT":
            signal.predicted_tp_pct = -signal.predicted_tp_pct

        signal.predicted_sl_pct = original_sl * sl_adj_factor

        # Definir níveis de TP para estratégia de saída parcial
        tp_levels = []
        tp_percents = []

        # Em mercados muito voláteis, distribuição 30-30-40 para capturar movimento grande
        tp_levels = [
            signal.predicted_tp_pct * 0.4,
            signal.predicted_tp_pct * 0.7,
            signal.predicted_tp_pct
        ]
        tp_percents = [30, 30, 40]

        # Recalcular preços com base nos percentuais ajustados
        if signal.direction == "LONG":
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:  # SHORT
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Verificar se o R:R ainda é aceitável após ajustes
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)
        if signal.rr_ratio < self.config.min_rr_ratio * 0.9:
            # Se o R:R ficou muito ruim, ajustar TP para cima para manter R:R
            target_rr = self.config.min_rr_ratio * 1.1  # Adicionar margem
            new_tp_pct = abs(signal.predicted_sl_pct * target_rr)

            if signal.direction == "LONG":
                signal.predicted_tp_pct = new_tp_pct
                signal.tp_factor = 1 + (new_tp_pct / 100)
            else:  # SHORT
                signal.predicted_tp_pct = -new_tp_pct
                signal.tp_factor = 1 - (new_tp_pct / 100)

            signal.tp_price = signal.current_price * signal.tp_factor
            signal.rr_ratio = target_rr

        logger.info(
            f"Ajustando TP para manter R:R mínimo: {original_tp:.2f}% -> {signal.predicted_tp_pct:.2f}%, "
            f"novo R:R = {signal.rr_ratio:.2f}"
        )

        # Adicionar configurações de TP parcial e trailing stop
        if hasattr(signal, 'mtf_details') and signal.mtf_details is not None:
            signal.mtf_details['tp_levels'] = tp_levels
            signal.mtf_details['tp_percents'] = tp_percents

            # Configurações de trailing stop adaptativo
            signal.mtf_details['trailing_stop'] = {
                'activation_pct': 1.0,  # Ativar a 1% de lucro
                'distance_pct': min(atr_pct * 0.7, 0.8) if atr_pct else 0.5,  # Distância baseada em ATR
                'dynamic_adjustment': True  # Permitir ajustes enquanto o trade avança
            }

            # Dados de volatilidade para referência
            if atr_pct:
                signal.mtf_details['volatility_atr_pct'] = atr_pct

            # Instruções específicas para trades de alta volatilidade
            signal.mtf_details['volatile_market_instructions'] = "Use posição menor. Monitore de perto."

        logger.info(
            f"Sinal ajustado para alta volatilidade: "
            f"TP={original_tp:.2f}% -> {signal.predicted_tp_pct:.2f}%, "
            f"SL={original_sl:.2f}% -> {signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal


class LowVolatilityStrategy(BaseStrategy):
    """
    Estratégia avançada para mercados com baixa volatilidade.

    Foca em capturar pequenos movimentos com alto grau de precisão,
    operando eficientemente em ambientes com movimentos reduzidos.
    Utiliza mean reversion e compressão-expansão de volatilidade para entradas.
    """

    def __init__(self):
        """ Inicializa a estratégia com configuração otimizada para baixa volatilidade. """
        config = StrategyConfig(
            name="Low Volatility Strategy",
            description="Estratégia otimizada para mercados com baixa volatilidade",
            min_rr_ratio=1.1,
            entry_threshold=0.50,
            tp_adjustment=0.8,
            sl_adjustment=0.7,
            entry_aggressiveness=1.2,
            max_sl_percent=0.7,
            min_tp_percent=0.25,
            required_indicators=[
                "adx", "atr", "atr_pct", "boll_width", "boll_hband", "boll_lband",
                "boll_pct_b", "rsi", "stoch_k", "stoch_d", "macd", "macd_histogram",
                "ema_short", "ema_long", "volume", "obv"
            ]
        )
        super().__init__(config)
        self.prediction_service: ITpSlPredictionService = TpSlPredictionService()

        # Inicializar analisadores de padrões
        pattern_factory = PatternAnalyzerFactory()
        self.momentum_analyzer = pattern_factory.create_momentum_analyzer()
        self.volatility_analyzer = pattern_factory.create_volatility_analyzer()
        self.trend_analyzer = pattern_factory.create_trend_analyzer()

        # Inicializar avaliador de entradas
        self.entry_evaluator = EntryEvaluatorFactory.create_evaluator("LOW_VOLATILITY", config)

        # Armazenar cache de informações sobre a volatilidade
        self.volatility_data = {
            'atr_pct': None,
            'compression_detected': False,
            'mean_reversion_zone': False,
            'tight_range_width_pct': None
        }

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia de baixa volatilidade deve ser ativada.

        Implementa verificações robustas para identificar mercados com volatilidade
        reduzida e oportunidades em movimentos controlados.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # 1. Verificar ATR percentual - métrica primária de volatilidade
        low_atr = False
        atr_pct = 0.0
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            low_atr = atr_pct < settings.VOLATILITY_LOW_THRESHOLD

            # Armazenar o ATR% para uso futuro
            self.volatility_data['atr_pct'] = atr_pct

        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            low_atr = atr_pct < settings.VOLATILITY_LOW_THRESHOLD

            # Armazenar o ATR% para uso futuro
            self.volatility_data['atr_pct'] = atr_pct

        # 2. Verificar a largura das Bollinger Bands
        narrow_bands = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            narrow_bands = boll_width < (avg_width * 0.7)

            # Verificar também se as bandas estão se estreitando
            if len(df) > 5:
                width_5_ago = df['boll_width'].iloc[-5]
                bands_narrowing = boll_width < width_5_ago * 0.9  # 10% mais estreitas

                if bands_narrowing:
                    self.volatility_data['compression_detected'] = True
                    logger.info(f"Compressão de Bollinger Bands: {boll_width:.4f} vs {width_5_ago:.4f}")

        # 3. Verificar ausência de tendência forte (ADX baixo)
        weak_trend = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            weak_trend = adx_value < 20

            # Verificar também a estabilidade do ADX
            if len(df) > 5:
                adx_std = df['adx'].iloc[-5:].std()
                adx_stable = adx_std < 2.0

                if adx_stable and weak_trend:
                    logger.info(f"ADX baixo e estável: {adx_value:.1f} (desvio: {adx_std:.2f})")

        # 4. Verificar EMAs próximas (indicando ausência de tendência)
        emas_flat = False
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            ema_diff_pct = abs(ema_short - ema_long) / ema_long * 100
            emas_flat = ema_diff_pct < 0.3  # EMAs muito próximas

        # 5. Verificar range de preço reduzido
        tight_range = False
        if len(df) >= 10:
            # Calcular amplitude do range recente (últimos 10 períodos)
            high_max = df['high'].iloc[-10:].max()
            low_min = df['low'].iloc[-10:].min()

            range_width = high_max - low_min
            range_width_pct = (range_width / low_min) * 100

            tight_range = range_width_pct < 1.0  # Range menor que 1%

            if tight_range:
                self.volatility_data['tight_range_width_pct'] = range_width_pct
                logger.info(f"Range estreito detectado: {range_width_pct:.2f}%")

        # 6. Verificar volume reduzido
        low_volume = False
        if 'volume' in df.columns and len(df) > 10:
            recent_volume = df['volume'].iloc[-3:].mean()
            past_volume = df['volume'].iloc[-13:-3].mean()

            low_volume = recent_volume < past_volume * 0.7  # Volume recente 30% menor

            if low_volume:
                logger.info(f"Volume reduzido: {recent_volume:.0f} vs {past_volume:.0f}")

        # 7. Verificar alinhamento multi-timeframe
        mtf_calm = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']
            mtf_calm = 'NEUTRAL' in mtf_trend

            if mtf_calm:
                logger.info(f"Tendência MTF neutra: {mtf_trend}")

        # 8. Verificar a presença em zonas de mean-reversion
        mean_reversion_zone = False
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]

            # Valores próximos de 0 ou 1 são oportunidades de mean-reversion
            if pct_b < 0.15 or pct_b > 0.85:
                mean_reversion_zone = True
                self.volatility_data['mean_reversion_zone'] = True
                logger.info(f"Zona de Mean Reversion: %B={pct_b:.2f}")

        # Ativar se pelo menos três indicadores confirmarem baixa volatilidade
        confirmations = sum(
            [
                low_atr, narrow_bands, weak_trend, emas_flat,
                tight_range, low_volume, mtf_calm, mean_reversion_zone
            ]
        )
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de BAIXA VOLATILIDADE ativada: "
                f"ATR_baixo={low_atr} ({atr_pct:.2f}%), "
                f"Bandas_estreitas={narrow_bands}, "
                f"Tendência_fraca={weak_trend} (ADX={df['adx'].iloc[-1]:.1f}), "
                f"EMAs_flat={emas_flat}, "
                f"Range_estreito={tight_range}, "
                f"Volume_baixo={low_volume}, "
                f"MTF_calmo={mtf_calm}, "
                f"Mean_reversion={mean_reversion_zone}"
            )

        return should_activate



    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com baixa volatilidade.

        Em mercados calmos, busca entradas precisas em:
        1. Desvios extremos para mean-reversion
        2. Breakouts após compressão de volatilidade (squeeze)

        Implementa análise adaptada para movimentos menores com maior precisão.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # 1. Detectar desvios da média para mean-reversion
        deviation_detected, deviation_strength, reversion_direction = self.volatility_analyzer.detect_deviation_from_mean(
            df, current_price
        )

        # 2. Detectar breakouts após squeeze (compressão de volatilidade)
        squeeze_breakout, breakout_strength, breakout_direction = self.volatility_analyzer.detect_squeeze_breakout(df)

        # 3. Verificar alinhamento multi-timeframe
        mtf_neutral = False
        mtf_direction = "NONE"
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']

            # Em baixa volatilidade, um MTF neutral é favorável
            if 'NEUTRAL' in mtf_trend:
                mtf_neutral = True
                logger.info(f"MTF neutro favorável para estratégia de baixa volatilidade: {mtf_trend}")
            # Se não for neutral, verificar direção
            elif 'UPTREND' in mtf_trend:
                mtf_direction = "UP"
                logger.info(f"MTF em tendência de alta: {mtf_trend}")
            elif 'DOWNTREND' in mtf_trend:
                mtf_direction = "DOWN"
                logger.info(f"MTF em tendência de baixa: {mtf_trend}")

        # 4. Determinar direção do sinal com base nas análises
        signal_direction = None
        signal_strength = 0.0

        # Condições para LONG
        long_conditions = {
            'mean_reversion': deviation_detected and reversion_direction == "UP",
            'squeeze_breakout': squeeze_breakout and breakout_direction == "UP",
            'mtf_aligned': mtf_direction == "UP"
        }

        long_scores = {
            'mean_reversion': deviation_strength * 0.8 if long_conditions['mean_reversion'] else 0,
            'squeeze_breakout': breakout_strength * 0.7 if long_conditions['squeeze_breakout'] else 0,
            'mtf_aligned': 0.4 if long_conditions['mtf_aligned'] else 0
        }

        # Condições para SHORT
        short_conditions = {
            'mean_reversion': deviation_detected and reversion_direction == "DOWN",
            'squeeze_breakout': squeeze_breakout and breakout_direction == "DOWN",
            'mtf_aligned': mtf_direction == "DOWN"
        }

        short_scores = {
            'mean_reversion': deviation_strength * 0.8 if short_conditions['mean_reversion'] else 0,
            'squeeze_breakout': breakout_strength * 0.7 if short_conditions['squeeze_breakout'] else 0,
            'mtf_aligned': 0.4 if short_conditions['mtf_aligned'] else 0
        }

        # Bônus para MTF neutro (favorável em baixa volatilidade)
        if mtf_neutral:
            mean_reversion_bonus = 0.15
            # Adicionar o bônus se temos mean reversion
            if long_conditions['mean_reversion']:
                long_scores['mean_reversion'] = min(long_scores['mean_reversion'] + mean_reversion_bonus, 1.0)
                logger.info(f"Bônus de MTF neutro aplicado para LONG mean-reversion: +{mean_reversion_bonus:.2f}")
            if short_conditions['mean_reversion']:
                short_scores['mean_reversion'] = min(short_scores['mean_reversion'] + mean_reversion_bonus, 1.0)
                logger.info(f"Bônus de MTF neutro aplicado para SHORT mean-reversion: +{mean_reversion_bonus:.2f}")

        # Calcular pontuação para cada direção
        long_score = sum(long_scores.values())
        short_score = sum(short_scores.values())

        # Número mínimo de condições (pelo menos 1 das 3 condições com força suficiente)
        # Em baixa volatilidade, uma condição forte pode ser suficiente
        min_conditions = 1
        min_score = 0.6  # Pontuação mínima para gerar sinal

        # Contar condições atendidas com pontuação significativa
        long_conditions_met = sum(1 for k, v in long_scores.items() if v > 0.4)
        short_conditions_met = sum(1 for k, v in short_scores.items() if v > 0.4)

        # Log de condições
        logger.info(
            f"Score LONG: {long_score:.2f} (condições significativas: {long_conditions_met}/{len(long_conditions)}) - "
            f"Mean Reversion={long_conditions['mean_reversion']} ({long_scores['mean_reversion']:.2f}), "
            f"Squeeze Breakout={long_conditions['squeeze_breakout']} ({long_scores['squeeze_breakout']:.2f}), "
            f"MTF={long_conditions['mtf_aligned']} ({long_scores['mtf_aligned']:.2f})"
        )

        logger.info(
            f"Score SHORT: {short_score:.2f} (condições significativas: {short_conditions_met}/{len(short_conditions)}) - "
            f"Mean Reversion={short_conditions['mean_reversion']} ({short_scores['mean_reversion']:.2f}), "
            f"Squeeze Breakout={short_conditions['squeeze_breakout']} ({short_scores['squeeze_breakout']:.2f}), "
            f"MTF={short_conditions['mtf_aligned']} ({short_scores['mtf_aligned']:.2f})"
        )

        # Determinar sinal de entrada com base em scores e condições
        if long_conditions_met >= min_conditions and long_score >= min_score and long_score > short_score:
            signal_direction: Literal["LONG", "SHORT"] = "LONG"
            signal_strength = long_score
            logger.info(f"Sinal LONG gerado em baixa volatilidade com força {signal_strength:.2f}")

        elif short_conditions_met >= min_conditions and short_score >= min_score and short_score > long_score:
            signal_direction: Literal["LONG", "SHORT"] = "SHORT"
            signal_strength = short_score
            logger.info(f"Sinal SHORT gerado em baixa volatilidade com força {signal_strength:.2f}")

        # Se não temos direção clara, não gerar sinal
        if not signal_direction:
            logger.info(f"Condições insuficientes para gerar sinal em baixa volatilidade")
            return None

        # Usar o serviço de previsão para obter TP/SL
        prediction = self.prediction_service.predict_tp_sl(df, current_price, signal_direction)
        if prediction is None:
            return None

        predicted_tp_pct, predicted_sl_pct = prediction

        # Em mercados de baixa volatilidade, TP e SL devem ser menores
        # para se adequar à amplitude reduzida de movimentos

        # Ajustar TP e SL com base no ATR
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            # Limitar TP a um múltiplo razoável do ATR
            max_tp_pct = atr_pct * 3.0  # No máximo 3x o ATR
            if abs(predicted_tp_pct) > max_tp_pct:
                adjusted_tp = max_tp_pct if signal_direction == "LONG" else -max_tp_pct
                logger.info(f"Limitando TP para um valor realista: {predicted_tp_pct:.2f}% -> {adjusted_tp:.2f}%")
                predicted_tp_pct = adjusted_tp

            # Limitar SL a um múltiplo razoável do ATR
            max_sl_pct = atr_pct * 1.5  # No máximo 1.5x o ATR
            if predicted_sl_pct > max_sl_pct:
                logger.info(f"Limitando SL para um valor realista: {predicted_sl_pct:.2f}% -> {max_sl_pct:.2f}%")
                predicted_sl_pct = max_sl_pct

        # Avaliar a qualidade da entrada
        should_enter, entry_score = self.entry_evaluator.evaluate_entry_quality(
            df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct,
            entry_threshold=self.config.entry_threshold
        )

        # Combinar o score de condições com o score de qualidade da entrada
        entry_score = 0.5 * entry_score + 0.5 * signal_strength

        if not should_enter:
            logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
            return None

        # TP em múltiplos níveis para mercados de baixa volatilidade
        # Em baixa vol, é melhor ter 2 níveis com saída maior no primeiro TP
        tp_levels = []

        first_tp_pct = predicted_tp_pct * 0.50  # 50% do caminho
        second_tp_pct = predicted_tp_pct  # 100% do caminho
        tp_levels = [first_tp_pct, second_tp_pct]
        tp_percents = [70, 30]  # Saída ainda mais agressiva no início para volatilidade baixa

        # Configurar parâmetros para o sinal
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
        market_trend = "NEUTRAL"  # Em baixa volatilidade geralmente é neutro
        market_strength = "LOW_VOLATILITY"

        # Gerar ID único para o sinal
        signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

        # Criar o sinal com dados adicionais
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
            entry_score=entry_score,
            rr_ratio=rr_ratio,
            market_trend=market_trend,
            market_strength=market_strength,
            timestamp=datetime.now(),
            # Campos adicionais para gerenciamento de risco avançado
            mtf_trend=market_trend,
            mtf_confidence=mtf_neutral * 0.7 if mtf_neutral else None,  # Neutro é bom em baixa vol
            mtf_alignment=signal_strength if signal_strength else None,
            mtf_details={
                'entry_conditions_score': signal_strength,
                'mean_reversion_strength': deviation_strength if deviation_detected else 0,
                'squeeze_breakout_strength': breakout_strength if squeeze_breakout else 0,
                'tp_levels': tp_levels,
                'tp_percents': tp_percents,
                'volatility_data': self.volatility_data
            } if 'mtf_details' in TradingSignal.__annotations__ else None
        )

        return signal


    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados com baixa volatilidade.

        Implementa ajustes para maximizar a eficiência em ambientes calmos:
        - Targets de TP mais conservadores
        - SL mais apertados para compensar o TP menor
        - Saída parcial antecipada para capturar lucros em movimento limitado
        """
        # Obter o ATR para calcular ajustes mais precisos
        atr_value = None
        atr_pct = None

        if 'atr' in df.columns:
            atr_value = df['atr'].iloc[-1]
            atr_pct = (atr_value / signal.current_price) * 100

        elif 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            atr_value = (atr_pct / 100) * signal.current_price

        # Aplicar ajustes baseados no nível de volatilidade
        if atr_pct is not None:
            # Em baixa volatilidade, ser conservador com TP/SL
            tp_volatility_factor = 0.7  # TP menor para metas realistas
            sl_volatility_factor = 0.6  # SL mais apertado para equilibrar o R:R

            logger.info(f"Ajustes conservadores para baixa volatilidade (ATR: {atr_pct:.2f}%)")

            # Aplicar os fatores de ajuste de volatilidade aos fatores base da configuração
            tp_adj_factor = self.config.tp_adjustment * tp_volatility_factor
            sl_adj_factor = self.config.sl_adjustment * sl_volatility_factor

        else:
            # Se não temos ATR, usar os valores padrão da configuração
            tp_adj_factor = self.config.tp_adjustment
            sl_adj_factor = self.config.sl_adjustment

        # Armazenar os valores originais para log
        original_tp = signal.predicted_tp_pct
        original_sl = signal.predicted_sl_pct

        # Aplicar os ajustes
        signal.predicted_tp_pct = abs(original_tp) * tp_adj_factor
        if signal.direction == "SHORT":
            signal.predicted_tp_pct = -signal.predicted_tp_pct

        signal.predicted_sl_pct = original_sl * sl_adj_factor

        # Definir níveis de TP para estratégia de saída parcial
        # Em baixa volatilidade, sair com mais volume no primeiro TP
        tp_levels = [
            signal.predicted_tp_pct * 0.6,  # 60% do caminho
            signal.predicted_tp_pct  # TP completo
        ]
        tp_percents = [60, 40]  # 60% no primeiro TP, 40% no segundo

        # Recalcular preços com base nos percentuais ajustados
        if signal.direction == "LONG":
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:  # SHORT
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Verificar se o R:R ainda é aceitável após ajustes
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)
        if signal.rr_ratio < self.config.min_rr_ratio * 0.9:
            # Se o R:R ficou muito ruim, ajustar SL para baixo para manter R:R
            target_rr = self.config.min_rr_ratio * 1.1  # Adicionar margem
            new_sl_pct = abs(signal.predicted_tp_pct / target_rr)

            signal.predicted_sl_pct = new_sl_pct
            signal.sl_factor = 1 - (new_sl_pct / 100) if signal.direction == "LONG" else 1 + (new_sl_pct / 100)
            signal.sl_price = signal.current_price * signal.sl_factor
            signal.rr_ratio = target_rr

            logger.info(
                f"Ajustando SL para manter R:R mínimo: {original_sl:.2f}% -> {signal.predicted_sl_pct:.2f}%, "
                f"novo R:R = {signal.rr_ratio:.2f}"
            )

        # Adicionar configurações de TP parcial e gerenciamento de risco
        if hasattr(signal, 'mtf_details') and signal.mtf_details is not None:
            signal.mtf_details['tp_levels'] = tp_levels
            signal.mtf_details['tp_percents'] = tp_percents

            # Ajustes adicionais para baixa volatilidade
            signal.mtf_details['low_volatility_adjustments'] = {
                'trailing_stop': False,  # Sem trailing stop em baixa volatilidade
                'quick_exit': True,  # Saída rápida se a oportunidade diminui
                'position_size_boost': 1.3  # Tamanho da posição 30% maior que normal
            }

            # Dados de volatilidade para referência
            if atr_pct:
                signal.mtf_details['volatility_atr_pct'] = atr_pct

            # Instruções específicas para trades de baixa volatilidade
            signal.mtf_details['low_volatility_instructions'] = ("Use posição 30% maior. "
                                                                 "TP mais conservador. "
                                                                 "Sem trailing stop.")

        logger.info(
            f"Sinal ajustado para baixa volatilidade: "
            f"TP={original_tp:.2f}% -> {signal.predicted_tp_pct:.2f}%, "
            f"SL={original_sl:.2f}% -> {signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal
