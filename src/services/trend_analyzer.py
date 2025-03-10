# services/trend_analyzer.py
import asyncio
from enum import Enum
from typing import Any

import pandas as pd

from core.logger import logger
from services.binance.binance_client import BinanceClient


class TimeFrame(str, Enum):
    """Enumeração dos timeframes suportados para análise."""
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


class TrendStrength(str, Enum):
    """Enumeração da força e direção das tendências."""
    STRONG_UP = "STRONG_UPTREND"
    MODERATE_UP = "MODERATE_UPTREND"
    WEAK_UP = "WEAK_UPTREND"
    NEUTRAL = "NEUTRAL"
    WEAK_DOWN = "WEAK_DOWNTREND"
    MODERATE_DOWN = "MODERATE_DOWNTREND"
    STRONG_DOWN = "STRONG_DOWNTREND"


class TrendAnalyzer:
    """
    Classe responsável por analisar e identificar tendências no mercado
    usando diferentes métodos e timeframes.
    """

    @staticmethod
    def ema_trend(df: pd.DataFrame, short_period: int = 9, long_period: int = 21) -> str:
        """
        Identifica a tendência com base no cruzamento de EMAs.

        Args:
            df: DataFrame com dados históricos incluindo 'close'
            short_period: Período para EMA curta
            long_period: Período para EMA longa

        Returns:
            str: "UPTREND", "DOWNTREND" ou "NEUTRAL"
        """
        if len(df) < long_period or 'close' not in df.columns:
            return "NEUTRAL"

        # Verificar se já temos EMAs existentes com nome similar
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            # Usar EMAs existentes
            last_short_ema = df['ema_short'].iloc[-1]
            last_long_ema = df['ema_long'].iloc[-1]
        else:
            # Calcular EMAs temporariamente
            short_ema = df['close'].ewm(span=short_period, adjust=False).mean()
            long_ema = df['close'].ewm(span=long_period, adjust=False).mean()
            last_short_ema = short_ema.iloc[-1]
            last_long_ema = long_ema.iloc[-1]

        # Determinar tendência
        if last_short_ema > last_long_ema:
            return "UPTREND"
        elif last_short_ema < last_long_ema:
            return "DOWNTREND"
        else:
            return "NEUTRAL"

    @staticmethod
    def adx_trend(df: pd.DataFrame, period: int = 14, threshold: int = 25) -> str:
        """
        Identifica a força da tendência usando ADX.

        Args:
            df: DataFrame com dados históricos incluindo 'adx'
            period: Período para ADX
            threshold: Limiar para considerar uma tendência forte

        Returns:
            str: "STRONG_TREND" ou "WEAK_TREND"
        """
        if 'adx' not in df.columns or len(df) < period:
            return "WEAK_TREND"

        last_adx = df['adx'].iloc[-1]

        if last_adx > threshold:
            return "STRONG_TREND"
        else:
            return "WEAK_TREND"

    @staticmethod
    async def analyze_trend_strength(multi_tf_analyzer: 'MultiTimeFrameTrendAnalyzer') -> dict[str, Any]:
        """
        Analisa a força da tendência usando múltiplos timeframes.

        Args:
            multi_tf_analyzer: Instância configurada do MultiTimeFrameTrendAnalyzer

        Returns:
            Dicionário com detalhes da análise multi-timeframe
        """
        trend, confidence, details = await multi_tf_analyzer.analyze_multi_timeframe_trend()

        # Criar um resumo mais condensado para uso no bot
        summary = {
            "trend": trend,
            "confidence": confidence,
            "score": details["trend_score"],
            "timeframes": {
                tf: details["tf_summary"][tf]["strength"]
                for tf in details["tf_summary"]
            },
            "agreement": details["tf_agreement"]
        }

        return summary

    @staticmethod
    async def evaluate_trade_with_mtf(
            multi_tf_analyzer: 'MultiTimeFrameTrendAnalyzer',
            trade_direction: str
    ) -> dict[str, Any]:
        """
        Avalia um trade considerando a análise multi-timeframe.

        Args:
            multi_tf_analyzer: Instância configurada do MultiTimeFrameTrendAnalyzer
            trade_direction: Direção do trade ('LONG' ou 'SHORT')

        Returns:
            Dicionário com avaliação do trade
        """
        alignment_score, confidence = await multi_tf_analyzer.get_trend_alignment(trade_direction)

        # Calcular qualidade do trade baseada no alinhamento e confiança
        quality_score = alignment_score * 0.7 + confidence * 0.3

        # Categorizar alinhamento
        if alignment_score >= 0.8:
            alignment_category = "EXCELLENT"
        elif alignment_score >= 0.6:
            alignment_category = "GOOD"
        elif alignment_score >= 0.4:
            alignment_category = "MODERATE"
        elif alignment_score >= 0.3:
            alignment_category = "POOR"
        else:
            alignment_category = "VERY_POOR"

        evaluation = {
            "direction": trade_direction,
            "alignment_score": alignment_score,
            "confidence": confidence,
            "quality_score": quality_score,
            "alignment_category": alignment_category,
            "favorable": alignment_score >= 0.5,  # Se o trade está favorável
            "strong_signal": quality_score >= 0.7  # Se o sinal é forte
        }

        return evaluation


class MultiTimeFrameTrendAnalyzer:
    """
    Classe responsável por analisar tendências em múltiplos timeframes.
    Utiliza análises de tendência em vários períodos (15m, 1h, 4h, 1d)
    para fornecer uma visão mais holística do mercado.
    """

    def __init__(self, binance_client: BinanceClient, symbol: str):
        """
        Inicializa o analisador multi-timeframe.

        Args:
            binance_client: Cliente Binance para obtenção de dados
            symbol: Par de trading (ex.: "BTCUSDT")
        """
        self.client = binance_client
        self.symbol = symbol
        self.timeframes = [
            TimeFrame.MINUTE_15,
            TimeFrame.HOUR_1,
            TimeFrame.HOUR_4,
            TimeFrame.DAY_1
        ]

        # Definir pesos para cada timeframe
        self.tf_weights = {
            TimeFrame.MINUTE_15: 0.65,  # Mais relevante
            TimeFrame.HOUR_1: 0.20,  # Relevante, mas menos que 15m
            TimeFrame.HOUR_4: 0.10,  # Contexto secundário
            TimeFrame.DAY_1: 0.05  # Apenas contexto de longo prazo
        }

        # Cache de dados para cada timeframe
        self.data_cache: dict[TimeFrame, tuple[pd.DataFrame, float]] = {}

    async def get_data_for_timeframe(self, timeframe: TimeFrame, limit: int = 100) -> pd.DataFrame:
        """
        Obtém dados históricos para um timeframe específico.

        Args:
            timeframe: Timeframe a ser obtido
            limit: Número de candles a serem obtidos

        Returns:
            DataFrame com dados históricos do timeframe especificado
        """
        try:
            # Verificar se o cliente está inicializado
            if not self.client.is_client_initialized():
                await self.client.initialize()

            # Obter dados
            klines = await self.client.client.futures_klines(
                symbol=self.symbol,
                interval=timeframe.value,
                limit=limit
            )

            # Criar DataFrame
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])

            # Converter tipos
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                float)

            # Configurar índice
            df.set_index("timestamp", inplace=True)

            # Calcular indicadores técnicos para análise de tendência
            df = self._add_trend_indicators(df)

            # Atualizar cache
            self.data_cache[timeframe] = (df, pd.Timestamp.now().timestamp())

            return df

        except Exception as e:
            logger.error(f"Erro ao obter dados para timeframe {timeframe.value}: {e}")
            return pd.DataFrame()

    @staticmethod
    def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores de tendência ao DataFrame.

        Args:
            df: DataFrame com dados históricos

        Returns:
            DataFrame com indicadores adicionados
        """
        # Adicionar EMAs
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()

        # Adicionar ADX
        try:
            # Calcular +DI e -DI
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().mul(-1)
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            # Condições para +DM e -DM
            plus_dm = plus_dm.copy()
            minus_dm = minus_dm.copy()

            plus_dm[(plus_dm <= minus_dm) | (plus_dm <= 0)] = 0
            minus_dm[(minus_dm <= plus_dm) | (minus_dm <= 0)] = 0

            # Calcular TR (True Range)
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift())
            tr3 = abs(df['low'] - df['close'].shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Calcular +DI, -DI e ADX
            period = 14
            smoothed_plus_dm = plus_dm.ewm(span=period, adjust=False).mean()
            smoothed_minus_dm = minus_dm.ewm(span=period, adjust=False).mean()
            smoothed_tr = tr.ewm(span=period, adjust=False).mean()

            plus_di = 100 * smoothed_plus_dm / smoothed_tr
            minus_di = 100 * smoothed_minus_dm / smoothed_tr

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.ewm(span=period, adjust=False).mean()

            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            df['adx'] = adx
        except Exception as e:
            # Fallback se o cálculo de ADX falhar
            logger.warning(f"Erro ao calcular ADX: {e}")
            df['adx'] = 0
            df['plus_di'] = 0
            df['minus_di'] = 0

        return df

    async def get_all_timeframe_data(self, force_refresh: bool = False) -> dict[TimeFrame, pd.DataFrame]:
        """
        Obtém dados para todos os timeframes configurados.

        Args:
            force_refresh: Se True, força a atualização do cache.

        Returns:
            Dicionário com dataframes para cada timeframe
        """
        now = pd.Timestamp.now().timestamp()
        tasks = []

        for tf in self.timeframes:
            # Verificar se precisamos atualizar os dados do cache
            need_refresh = force_refresh or tf not in self.data_cache

            if not need_refresh:
                # Verificar idade do cache
                _, timestamp = self.data_cache[tf]
                cache_age_seconds = now - timestamp

                # Configurar tempo de expiração do cache com base no timeframe
                if tf == TimeFrame.MINUTE_15 and cache_age_seconds > 300:  # 5 minutos
                    need_refresh = True
                elif tf == TimeFrame.HOUR_1 and cache_age_seconds > 1800:  # 30 minutos
                    need_refresh = True
                elif tf == TimeFrame.HOUR_4 and cache_age_seconds > 7200:  # 2 horas
                    need_refresh = True
                elif tf == TimeFrame.DAY_1 and cache_age_seconds > 21600:  # 6 horas
                    need_refresh = True

            if need_refresh:
                # Obter dados frescos
                tasks.append(self.get_data_for_timeframe(tf))

        # Esperar por todas as requisições assíncronas, se houver
        if tasks:
            await asyncio.gather(*tasks)

        # Construir o dicionário de resultados
        return {tf: self.data_cache[tf][0] if tf in self.data_cache else pd.DataFrame() for tf in self.timeframes}

    async def analyze_multi_timeframe_trend(self) -> tuple[str, float, dict[str, Any]]:
        """
        Analisa a tendência em múltiplos timeframes e retorna uma avaliação consolidada.

        Returns:
            Tupla contendo (tendência consolidada, score de confiança, detalhes por timeframe)
        """
        # Obter dados para todos os timeframes
        tf_data = await self.get_all_timeframe_data()

        # Analisar tendência para cada timeframe
        trend_scores = {}
        trend_strengths = {}
        trend_details = {}

        for tf, df in tf_data.items():
            if df.empty:
                trend_scores[tf.value] = 0
                trend_strengths[tf.value] = "NEUTRAL"
                continue

            # Análise de tendência baseada em EMAs
            ema_trend = TrendAnalyzer.ema_trend(df)

            # Análise de força da tendência baseada em ADX
            adx_strength = TrendAnalyzer.adx_trend(df)

            # Calcular score de tendência (-1 a +1)
            trend_score = 0

            if ema_trend == "UPTREND":
                trend_score = 1.0 if adx_strength == "STRONG_TREND" else 0.5
            elif ema_trend == "DOWNTREND":
                trend_score = -1.0 if adx_strength == "STRONG_TREND" else -0.5

            logger.info(f"TF {tf.value}: EMA={ema_trend}, ADX={adx_strength}, Score={trend_score}")

            # Calcular direção DMI
            dmi_direction = "NEUTRAL"
            if 'plus_di' in df.columns and 'minus_di' in df.columns:
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]

                if plus_di > minus_di:
                    dmi_direction = "UPTREND"
                    # Refinar score baseado na diferença entre +DI e -DI
                    diff_di = plus_di - minus_di
                    if diff_di > 10:
                        trend_score = max(trend_score, 0.8)  # Reforça tendência de alta
                elif minus_di > plus_di:
                    dmi_direction = "DOWNTREND"
                    # Refinar score baseado na diferença entre -DI e +DI
                    diff_di = minus_di - plus_di
                    if diff_di > 10:
                        trend_score = min(trend_score, -0.8)  # Reforça tendência de baixa

            # Armazenar resultados
            trend_scores[tf.value] = trend_score

            # Categorizar força da tendência
            if trend_score >= 0.8:
                trend_strengths[tf.value] = TrendStrength.STRONG_UP.value
            elif trend_score >= 0.5:
                trend_strengths[tf.value] = TrendStrength.MODERATE_UP.value
            elif trend_score > 0:
                trend_strengths[tf.value] = TrendStrength.WEAK_UP.value
            elif trend_score == 0:
                trend_strengths[tf.value] = TrendStrength.NEUTRAL.value
            elif trend_score > -0.5:
                trend_strengths[tf.value] = TrendStrength.WEAK_DOWN.value
            elif trend_score > -0.8:
                trend_strengths[tf.value] = TrendStrength.MODERATE_DOWN.value
            else:
                trend_strengths[tf.value] = TrendStrength.STRONG_DOWN.value

            # Coletar detalhes adicionais
            trend_details[tf.value] = {
                "ema_trend": ema_trend,
                "adx_strength": adx_strength,
                "adx_value": df['adx'].iloc[-1] if 'adx' in df.columns else None,
                "dmi_direction": dmi_direction,
                "plus_di": df['plus_di'].iloc[-1] if 'plus_di' in df.columns else None,
                "minus_di": df['minus_di'].iloc[-1] if 'minus_di' in df.columns else None,
                "score": trend_score
            }

        # Calcular score consolidado ponderado
        weighted_score = 0
        total_weight = 0

        for tf in self.timeframes:
            tf_value = tf.value
            if tf_value in trend_scores:
                weight = self.tf_weights[tf]
                weighted_score += trend_scores[tf_value] * weight
                total_weight += weight

        # Normalizar o score
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0

        # Determinar tendência consolidada baseada no score final
        if final_score >= 0.6:
            consolidated_trend = TrendStrength.STRONG_UP.value
        elif final_score >= 0.3:
            consolidated_trend = TrendStrength.MODERATE_UP.value
        elif final_score > 0.05:
            consolidated_trend = TrendStrength.WEAK_UP.value
        elif final_score >= -0.05:
            consolidated_trend = TrendStrength.NEUTRAL.value
        elif final_score >= -0.3:
            consolidated_trend = TrendStrength.WEAK_DOWN.value
        elif final_score >= -0.6:
            consolidated_trend = TrendStrength.MODERATE_DOWN.value
        else:
            consolidated_trend = TrendStrength.STRONG_DOWN.value

        # Calcular nível de concordância entre timeframes
        trend_directions = [1 if trend_scores[tf.value] > 0 else (-1 if trend_scores[tf.value] < 0 else 0)
                            for tf in self.timeframes if tf.value in trend_scores]

        # Verificar se todos os timeframes concordam na direção
        agreement_score = abs(sum(trend_directions)) / len(trend_directions) if trend_directions else 0

        # Calcular confiança baseada no score final e concordância entre timeframes
        confidence = (abs(final_score) * 0.85 + agreement_score * 0.15) * 100

        # Resumo por timeframe para facilitar leitura
        summary = {
            tf.value: {
                "strength": trend_strengths.get(tf.value, "NEUTRAL"),
                "score": trend_scores.get(tf.value, 0)
            } for tf in self.timeframes
        }

        details = {
            "consolidated_trend": consolidated_trend,
            "trend_score": final_score,
            "confidence": confidence,
            "tf_summary": summary,
            "tf_details": trend_details,
            "tf_agreement": agreement_score
        }

        # Log de análise multi-timeframe
        logger.info(
            f"Análise multi-timeframe: {consolidated_trend}, score: {final_score:.2f}, confiança: {confidence:.2f}%")
        logger.info(f"Tendências por timeframe: {summary}")

        return consolidated_trend, confidence, details

    async def get_trend_alignment(self, trade_direction: str) -> tuple[float, float]:
        """
        Verifica o alinhamento da direção do trade com a tendência multi-timeframe.

        Args:
            trade_direction: Direção pretendida do trade ('LONG' ou 'SHORT')

        Returns:
            Tupla contendo (score de alinhamento, confiança)
        """
        trend, confidence, details = await self.analyze_multi_timeframe_trend()
        trend_score = details['trend_score']

        # Analisar apenas o timeframe de 15m para low-latency trading
        tf_15m_score = 0
        if '15m' in details['tf_summary']:
            tf_15m_data = details['tf_summary']['15m']
            if tf_15m_data['strength'] == 'STRONG_UPTREND' or tf_15m_data['strength'] == 'MODERATE_UPTREND':
                tf_15m_score = 0.8  # Forte tendência de alta em 15m
            elif tf_15m_data['strength'] == 'WEAK_UPTREND':
                tf_15m_score = 0.6  # Tendência fraca de alta em 15m
            elif tf_15m_data['strength'] == 'STRONG_DOWNTREND' or tf_15m_data['strength'] == 'MODERATE_DOWNTREND':
                tf_15m_score = -0.8  # Forte tendência de baixa em 15m
            elif tf_15m_data['strength'] == 'WEAK_DOWNTREND':
                tf_15m_score = -0.6  # Tendência fraca de baixa em 15m
            else:
                tf_15m_score = 0  # Neutro

        # Calcular alinhamento com mais peso para o timeframe de 15m
        combined_score = trend_score * 0.5 + tf_15m_score * 0.5

        # Calcular alinhamento (-1 a 1, onde 1 é perfeitamente alinhado)
        if trade_direction == "LONG":
            alignment = combined_score  # Já está na escala correta
        else:  # SHORT
            alignment = -combined_score  # Inverter para SHORT

        # Normalizar para 0-1 com um ligeiro boost para permitir mais trades
        alignment_score = ((alignment + 1) / 2) * 1.1
        # Limitar a 1.0 caso o boost exceda
        alignment_score = min(alignment_score, 1.0)

        return alignment_score, confidence / 100  # confidence em 0-1
