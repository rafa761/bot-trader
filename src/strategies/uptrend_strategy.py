# strategies/uptrend_strategy.py

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


class UptrendStrategy(BaseStrategy):
    """
    Estratégia avançada para mercados em tendência de alta.

    Foca em capturar continuações da tendência de alta com entradas
    em pullbacks (correções) usando múltiplos indicadores e confirmações.
    Implementa análise de volume, padrões de candlestick e gerenciamento
    de risco dinâmico para performance otimizada.
    """

    def __init__(self):
        """ Inicializa a estratégia com configuração otimizada para mercados em alta. """
        config = StrategyConfig(
            name="Uptrend Strategy",
            description="Estratégia otimizada para mercados em tendência de alta",
            min_rr_ratio=1.8,
            entry_threshold=0.55,
            tp_adjustment=1.2,
            sl_adjustment=0.85,
            entry_aggressiveness=1.1,
            max_sl_percent=1.5,
            min_tp_percent=0.7,
            required_indicators=[
                "ema_short", "ema_long", "adx", "rsi",
                "stoch_k", "stoch_d", "atr", "vwap",
                "volume", "macd", "macd_signal",
                "macd_histogram",
            ]
        )
        super().__init__(config)
        self.prediction_service: ITpSlPredictionService = TpSlPredictionService()

        # Inicializar analisadores de padrões
        pattern_factory = PatternAnalyzerFactory()
        self.trend_analyzer = pattern_factory.create_trend_analyzer()
        self.momentum_analyzer = pattern_factory.create_momentum_analyzer()

        # Inicializar avaliador de entradas
        self.entry_evaluator = EntryEvaluatorFactory.create_evaluator("UPTREND", config)

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.

        Implementa verificações mais robustas para tendência de alta, incluindo
        EMAs, ADX, análise multi-timeframe e volume.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Adicionar filtro de RSI para evitar entradas em condições extremas
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if rsi > 70:  # Condição sobrecomprada
                logger.warning(f"RSI em condição sobrecomprada ({rsi:.1f}) - evitando ativação da estratégia de alta")
                return False

        # Adicionar filtro de %B para evitar entradas em condições extremas
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]
            if pct_b > 0.9:  # Preço muito próximo da banda superior
                logger.warning(f"Bollinger %B muito alto ({pct_b:.2f}) - evitando ativação da estratégia de alta")
                return False

        # Verificar EMAs no timeframe atual
        ema_short = df['ema_short'].iloc[-1]
        ema_long = df['ema_long'].iloc[-1]
        ema_uptrend = ema_short > ema_long

        # Verificar inclinação das EMAs
        ema_slope_up = False
        if len(df) > 5:
            ema_short_5_bars_ago = df['ema_short'].iloc[-5]
            ema_long_5_bars_ago = df['ema_long'].iloc[-5]
            ema_slope_up = (ema_short > ema_short_5_bars_ago) and (ema_long > ema_long_5_bars_ago)

        # Verificar tendência multi-timeframe
        mtf_uptrend = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_uptrend = 'UPTREND' in mtf_data['consolidated_trend']

        # Verificar força da tendência (ADX)
        adx_strong = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            adx_strong = adx_value > 25

        # Verificar volume (adicionado)
        volume_confirming = False
        if 'volume' in df.columns and len(df) > 10:
            avg_volume = df['volume'].iloc[-10:].mean()
            current_volume = df['volume'].iloc[-1]
            # Volume maior nos movimentos de alta é sinal de tendência forte
            if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Vela de alta
                volume_confirming = current_volume > avg_volume * 1.1

        # Verificar MACD (adicionado)
        macd_bullish = False
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_bullish = macd > 0 and macd > macd_signal

        # Calcular todas as confirmações da tendência de alta
        # Adicionamos mais condições para uma análise mais robusta
        confirmations = sum(
            [
                ema_uptrend,  # EMAs básicas
                ema_slope_up,  # Inclinação das EMAs (novo)
                mtf_uptrend,  # Confirmação multitimeframe
                adx_strong,  # Força da tendência
                volume_confirming,  # Confirmação de volume (novo)
                macd_bullish  # Confirmação de MACD (novo)
            ]
        )

        should_activate = confirmations >= 4

        if should_activate:
            logger.info(
                f"Estratégia de UPTREND ativada: EMA={ema_uptrend}, EMAs slope={ema_slope_up}, "
                f"MTF={mtf_uptrend}, ADX={adx_strong} (valor={df['adx'].iloc[-1]:.1f}), "
                f"Volume={volume_confirming}, MACD={macd_bullish}"
            )

        return should_activate

    def _find_valleys(self, values: np.ndarray) -> list[int]:
        """
        Encontra os índices dos vales (mínimos locais) em uma série de valores.

        Args:
            values: Array de valores numéricos

        Returns:
            list: Índices dos vales encontrados
        """
        valleys = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                valleys.append(i)
        return valleys

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em alta.
        Busca oportunidades de compra em pullbacks (correções) da tendência de alta.

        Implementa análise avançada de condições de entrada, combinando múltiplos
        indicadores com ponderação de importância.
        """
        # 1. Detectar pullback (correção de baixa) na tendência de alta
        in_pullback, pullback_strength = self.trend_analyzer.detect_pullback(df)

        # Verificar se há padrão de invalidação para LONG
        invalidation_pattern = self.trend_analyzer.is_invalidation_pattern(df, "LONG")
        if invalidation_pattern:
            logger.warning("Padrão de invalidação para LONG detectado - sinal rejeitado")
            return None

        # 2. Verificar suporte e possível bounce
        near_support, support_strength = self.trend_analyzer.detect_support(df, current_price)

        # 3. Verificar Stochastic para confirmar sobrevenda
        stoch_oversold, stoch_strength = self.momentum_analyzer.check_stochastic_oversold(df)

        # 4. Verificar divergência bullish
        divergence, divergence_strength = self.momentum_analyzer.check_bullish_divergence(df)

        # 5. Verificar tendência forte com ADX
        strong_trend = False
        trend_strength = 0.0
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            strong_trend = adx_value > 25
            trend_strength = min((adx_value - 20) / 20, 1.0)  # Normalizar 0-1

            if strong_trend:
                logger.info(f"Tendência forte detectada: ADX={adx_value:.1f} (força: {trend_strength:.1f})")

        # 6. Verificar alinhamento multi-timeframe
        mtf_aligned = False
        mtf_strength = 0.0
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']
            mtf_aligned = 'UPTREND' in mtf_trend

            # Obter score de confiança se disponível
            if 'confidence' in mtf_data:
                mtf_strength = mtf_data['confidence'] / 100

            if mtf_aligned:
                logger.info(f"Alinhamento multi-timeframe favorável: {mtf_trend} (força: {mtf_strength:.1f})")

        # 7. Cálculo ponderado do score de condições de entrada
        # Cada condição tem um peso específico de acordo com sua importância
        entry_conditions_score = 0.0
        weights_sum = 0.0

        # Pesos para cada condição (pullback é menos importante que bounce em suporte)
        condition_weights = {
            'pullback': 0.15,
            'support': 0.25,
            'stochastic': 0.15,
            'divergence': 0.15,
            'strong_trend': 0.15,
            'mtf_alignment': 0.15
        }

        # Adicionar score ponderado para cada condição se presente
        if in_pullback:
            entry_conditions_score += pullback_strength * condition_weights['pullback']
            weights_sum += condition_weights['pullback']

        if near_support:
            entry_conditions_score += support_strength * condition_weights['support']
            weights_sum += condition_weights['support']

        if stoch_oversold:
            entry_conditions_score += stoch_strength * condition_weights['stochastic']
            weights_sum += condition_weights['stochastic']

        if divergence:
            entry_conditions_score += divergence_strength * condition_weights['divergence']
            weights_sum += condition_weights['divergence']

        if strong_trend:
            entry_conditions_score += trend_strength * condition_weights['strong_trend']
            weights_sum += condition_weights['strong_trend']

            if mtf_aligned:
                entry_conditions_score += mtf_strength * condition_weights['mtf_alignment']
                weights_sum += condition_weights['mtf_alignment']

        # Normalizar o score para levar em conta apenas as condições presentes
        if weights_sum > 0:
            entry_conditions_score = entry_conditions_score / weights_sum

        # Número mínimo de condições e score mínimo para gerar sinal
        min_conditions = 3
        min_score = 0.6

        conditions_count = sum(
            [in_pullback, near_support, stoch_oversold,
             divergence, strong_trend, mtf_aligned]
        )

        logger.info(
            f"Condições para LONG: {conditions_count}/{min_conditions} atendidas, Score: {entry_conditions_score:.2f} - "
            f"Pullback={in_pullback} ({pullback_strength:.1f}), Suporte={near_support} ({support_strength:.1f}), "
            f"Stoch_Oversold={stoch_oversold} ({stoch_strength:.1f}), Divergência={divergence} ({divergence_strength:.1f}), "
            f"Strong_Trend={strong_trend} ({trend_strength:.1f}), MTF_Aligned={mtf_aligned} ({mtf_strength:.1f})"
        )

        if conditions_count >= min_conditions and entry_conditions_score < min_score:
            logger.warning(
                f"Score insuficiente para gerar sinal: {entry_conditions_score:.2f} < mínimo requerido {min_score:.2f}"
            )

        # Decidir se geramos sinal baseado no número de condições e no score
        generate_signal = conditions_count >= min_conditions and entry_conditions_score >= min_score

        if generate_signal:
            logger.info(
                f"Condições favoráveis para LONG em tendência de alta. "
                f"Score: {entry_conditions_score:.2f} com {conditions_count} condições."
            )

            # Usar o serviço de previsão para obter TP/SL
            prediction = self.prediction_service.predict_tp_sl(df, current_price, "LONG")
            if prediction is None:
                return None

            predicted_tp_pct, predicted_sl_pct = prediction

            # TP em múltiplos níveis para trade em tendência
            tp_levels = []

            first_tp_pct = predicted_tp_pct * 0.25  # 25% do caminho
            second_tp_pct = predicted_tp_pct * 0.50  # 50% do caminho
            third_tp_pct = predicted_tp_pct  # 100% do caminho
            tp_levels = [first_tp_pct, second_tp_pct, third_tp_pct]
            tp_percents = [40, 40, 20]

            # Avaliar a qualidade da entrada com base no cenário atual
            should_enter, entry_score = self.entry_evaluator.evaluate_entry_quality(
                df, current_price, "LONG", predicted_tp_pct, predicted_sl_pct,
                mtf_alignment=mtf_strength
            )

            # Adicionar o score de confluência de condições
            entry_score = 0.5 * entry_score + 0.5 * entry_conditions_score

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Vamos forçar operações LONG em tendência de alta
            side: Literal["SELL", "BUY"] = "BUY"
            position_side: Literal["LONG", "SHORT"] = "LONG"

            # Calcular fatores com base nos percentuais previstos
            tp_factor = 1 + (predicted_tp_pct / 100)
            sl_factor = 1 - (predicted_sl_pct / 100)

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # ATR para ajustes dinâmicos de position sizing e trailing stop
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Calcular a razão Risco:Recompensa
            rr_ratio = abs(predicted_tp_pct / predicted_sl_pct)

            # Determinar tendência e força
            market_trend = "UPTREND"  # Já sabemos que estamos em uptrend
            market_strength = "STRONG_TREND" if df['adx'].iloc[-1] > 25 else "WEAK_TREND"

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Criar o sinal
            signal = TradingSignal(
                id=signal_id,
                direction="LONG",
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
                mtf_trend="UPTREND",
                mtf_confidence=mtf_strength if mtf_strength else None,
                mtf_alignment=mtf_strength if mtf_strength else None,
                mtf_details={
                    'entry_conditions_score': entry_conditions_score,
                    'pullback_strength': pullback_strength if in_pullback else 0,
                    'support_strength': support_strength if near_support else 0,
                    'tp_levels': tp_levels,
                    'tp_percents': tp_percents
                } if 'mtf_details' in TradingSignal.__annotations__ else None
            )

            return signal

        return None

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em alta.

        Em tendência de alta, favorecemos sinais LONG e ajustamos TP/SL
        de forma dinâmica com base na volatilidade e força da tendência.
        """
        # Ajustar TP/SL com base na volatilidade atual (ATR)
        if 'atr' in df.columns and 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            # Base dos ajustes
            tp_adj_factor = self.config.tp_adjustment
            sl_adj_factor = self.config.sl_adjustment

            # Ajustes dinâmicos com base na volatilidade
            if atr_pct > 1.5:  # Alta volatilidade
                # TP mais amplo e SL mais largo em alta volatilidade
                tp_adj_factor = tp_adj_factor * 1.2
                sl_adj_factor = sl_adj_factor * 1.3
                logger.info(f"Ajustando para alta volatilidade (ATR={atr_pct:.2f}%)")
            elif atr_pct < 0.5:  # Baixa volatilidade
                # TP e SL mais próximos em baixa volatilidade
                tp_adj_factor = tp_adj_factor * 0.9
                sl_adj_factor = sl_adj_factor * 0.8
                logger.info(f"Ajustando para baixa volatilidade (ATR={atr_pct:.2f}%)")
        else:
            # Usar os ajustes padrão da configuração
            tp_adj_factor = self.config.tp_adjustment
            sl_adj_factor = self.config.sl_adjustment

        if signal.direction != "LONG":
            # Aumentar o threshold para trades contra a tendência (SHORT em uptrend)
            logger.info(f"Sinal SHORT em tendência de alta: exigindo maior qualidade")
            # Reduzir a pontuação para dificultar a entrada
            if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                signal.entry_score = signal.entry_score * 0.8
        else:  # Para sinais LONG em tendência de alta
            # Ajustar TP para ser mais ambicioso
            signal.predicted_tp_pct = signal.predicted_tp_pct * tp_adj_factor

            # Ajustar SL para ser mais apertado
            signal.predicted_sl_pct = signal.predicted_sl_pct * sl_adj_factor

            # Recalcular preços e fatores
            if signal.direction == "LONG":
                signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
            else:  # SHORT
                signal.tp_factor = 1 - (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

            signal.tp_price = signal.current_price * signal.tp_factor
            signal.sl_price = signal.current_price * signal.sl_factor

            # Verificar a força da tendência para ajustes adicionais
            # Se tendência forte, podemos ser ainda mais agressivos
            if 'adx' in df.columns and df['adx'].iloc[-1] > 30:
                # Tendência forte - podemos aumentar TP e apertar mais o SL
                logger.info("Tendência forte detectada - otimizando trade")

                # Calcular razão R:R atualizada
                signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)
            else:
                # Tendência moderada - usar valores padrão
                signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

            logger.info(
                f"Sinal LONG ajustado para mercado em alta: "
                f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
                f"R:R={signal.rr_ratio:.2f}"
            )

        return signal
