# strategies\downtrend_strategy.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from services.prediction.interfaces import ITpSlPredictionService
from services.prediction.tpsl_prediction import TpSlPredictionService
from strategies.base.model import BaseStrategy, StrategyConfig


class DowntrendStrategy(BaseStrategy):
    """
    Estratégia para mercados em tendência de baixa.
    Foca em capturar continuações da tendência de baixa com entradas em rallies.
    """

    def __init__(self):
        """ Inicializa a estratégia com configuração otimizada para mercados em baixa. """
        config = StrategyConfig(
            name="Downtrend Strategy",
            description="Estratégia otimizada para mercados em tendência de baixa",
            min_rr_ratio=1.2,
            entry_threshold=0.45,
            tp_adjustment=1.1,
            sl_adjustment=0.85,
            entry_aggressiveness=1.3,
            max_sl_percent=1.8,
            min_tp_percent=0.5,
            required_indicators=[
                "ema_short", "ema_long", "adx", "rsi",
                "stoch_k", "stoch_d", "atr", "vwap",
                "volume", "macd", "macd_histogram",
            ]
        )
        super().__init__(config)
        self.prediction_service: ITpSlPredictionService = TpSlPredictionService()

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.

        Implementa verificações mais robustas para tendência de baixa, incluindo
        EMAs, ADX, análise multi-timeframe e volume.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar EMAs no timeframe atual
        ema_short = df['ema_short'].iloc[-1]
        ema_long = df['ema_long'].iloc[-1]
        ema_downtrend = ema_short < ema_long

        # Verificar inclinação das EMAs (adicionado)
        ema_slope_down = False
        if len(df) > 5:
            ema_short_5_bars_ago = df['ema_short'].iloc[-5]
            ema_long_5_bars_ago = df['ema_long'].iloc[-5]
            ema_slope_down = (ema_short < ema_short_5_bars_ago) and (ema_long < ema_long_5_bars_ago)

        # Verificar tendência multi-timeframe
        mtf_downtrend = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_downtrend = 'DOWNTREND' in mtf_data['consolidated_trend']

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
            # Volume maior nos movimentos de baixa é sinal de tendência forte
            if df['close'].iloc[-1] < df['open'].iloc[-1]:  # Vela de baixa
                volume_confirming = current_volume > avg_volume * 1.1

        # Verificar MACD (adicionado)
        macd_bearish = False
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = df['macd'].iloc[-1]
            macd_signal = df['macd_signal'].iloc[-1]
            macd_bearish = macd < 0 and macd < macd_signal

        # Calcular todas as confirmações da tendência de baixa
        # Adicionamos mais condições para uma análise mais robusta
        confirmations = sum(
            [
                ema_downtrend,  # EMAs básicas
                ema_slope_down,  # Inclinação das EMAs (novo)
                mtf_downtrend,  # Confirmação multitimeframe
                adx_strong,  # Força da tendência
                volume_confirming,  # Confirmação de volume (novo)
                macd_bearish  # Confirmação de MACD (novo)
            ]
        )

        # Ativar se tivermos confirmação em pelo menos 3 dos 6 indicadores
        # Aumentamos o número mínimo de confirmações para maior robustez
        should_activate = confirmations >= 3

        if should_activate:
            logger.info(
                f"Estratégia de DOWNTREND ativada: EMA={ema_downtrend}, EMAs slope={ema_slope_down}, "
                f"MTF={mtf_downtrend}, ADX={adx_strong} (valor={df['adx'].iloc[-1]:.1f}), "
                f"Volume={volume_confirming}, MACD={macd_bearish}"
            )

        return should_activate

    def _detect_rally(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detecta rallies de correção em uma tendência de baixa.

        Um rally é uma correção temporária de alta em uma tendência de baixa,
        geralmente acompanhada por aumento de RSI e volume reduzido.

        Returns:
            tuple: (rally_detectado, força_do_rally de 0 a 1)
        """
        rally_strength = 0.0

        # Verificar RSI para rally
        rsi_rally = False
        rsi_strength = 0.0
        if 'rsi' in df.columns and len(df) > 3:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            prev2_rsi = df['rsi'].iloc[-3]

            # Rally = RSI subindo acima de 50-60 em tendência de baixa
            if rsi > 50 and rsi > prev_rsi and prev_rsi > prev2_rsi:
                rsi_rally = True
                # Calcular força do rally baseado em quão alto está o RSI
                rsi_strength = min((rsi - 50) / 20, 1.0)  # Normalizar entre 0-1
                logger.info(f"Rally de RSI detectado: RSI={rsi:.1f} (força: {rsi_strength:.2f})")

        # Verificar volume baixo durante rally (sinal de fraqueza do rally)
        volume_signal = 0.0
        if 'volume' in df.columns and len(df) > 5:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-5:].mean()

            if current_volume < avg_volume * 0.8:  # Volume baixo no rally
                volume_signal = 0.3  # Sinal positivo para entrada
                logger.info(f"Volume baixo durante rally: {current_volume:.0f} vs média {avg_volume:.0f}")

        # Verificar velas de alta consecutivas (sinal de rally)
        candle_signal = 0.0
        if len(df) > 3:
            if (df['close'].iloc[-1] > df['open'].iloc[-1] and
                    df['close'].iloc[-2] > df['open'].iloc[-2]):
                candle_signal = 0.2
                logger.info("Padrão de velas de alta consecutivas detectado")

        # Calcular a força total do rally
        if rsi_rally:
            rally_strength = rsi_strength + volume_signal + candle_signal
            rally_strength = min(rally_strength, 1.0)  # Normalizar para máximo de 1.0

            return True, rally_strength

        return False, 0.0

    def _detect_resistance(self, df: pd.DataFrame, current_price: float) -> tuple[bool, float]:
        """
        Detecta proximidade a níveis de resistência e possíveis rejeições.

        Verifica múltiplos tipos de resistência: EMA longa, VWAP, níveis de pivot,
        e máximos recentes.

        Returns:
            tuple: (próximo_da_resistência, força_da_resistência de 0 a 1)
        """
        resistance_signals = []

        # Verificar resistência na EMA longa
        if 'ema_long' in df.columns:
            ema_long = df['ema_long'].iloc[-1]
            price_to_ema = (ema_long - current_price) / current_price * 100

            # Resistência = preço próximo da EMA longa por baixo
            if -0.2 < price_to_ema < 0.5:  # Preço próximo ou tocando EMA por baixo
                strength = 0.8 if price_to_ema > 0 else 0.5  # Mais forte se cruzou por cima
                resistance_signals.append(strength)
                logger.info(f"Preço próximo da resistência em EMA longa: {price_to_ema:.2f}% (força: {strength:.1f})")

        # Verificar VWAP como resistência (adicionado)
        if 'vwap' in df.columns:
            vwap = df['vwap'].iloc[-1]
            price_to_vwap = (vwap - current_price) / current_price * 100

            if -0.2 < price_to_vwap < 0.5:  # Preço próximo ou tocando VWAP por baixo
                strength = 0.7 if price_to_vwap > 0 else 0.4
                resistance_signals.append(strength)
                logger.info(f"Preço próximo da resistência em VWAP: {price_to_vwap:.2f}% (força: {strength:.1f})")

        # Verificar níveis de pivot
        if all(col in df.columns for col in ['pivot', 'pivot_r1']):
            pivot = df['pivot'].iloc[-1]
            r1 = df['pivot_r1'].iloc[-1]

            price_to_pivot = (pivot - current_price) / current_price * 100
            price_to_r1 = (r1 - current_price) / current_price * 100

            if -0.2 < price_to_pivot < 0.5:
                resistance_signals.append(0.6)
                logger.info(f"Preço próximo do nível de pivot: {price_to_pivot:.2f}%")

            if -0.2 < price_to_r1 < 0.5:
                resistance_signals.append(0.7)
                logger.info(f"Preço próximo do nível R1: {price_to_r1:.2f}%")

        # Verificar rejeição na resistência baseado no padrão de velas
        if len(df) > 2 and all(col in df.columns for col in ['high', 'close', 'open']):
            current_high = df['high'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_open = df['open'].iloc[-1]

            # Range das últimas 20 barras
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            # Verificar padrão de rejeição (sombra longa no topo)
            if (current_high > current_close * 1.002 and
                    (current_high - current_close) > (recent_range * 0.08) and
                    current_close < current_open):  # Fechou abaixo da abertura

                strength = min((current_high - current_close) / (recent_range * 0.08), 1.0)
                resistance_signals.append(0.8 * strength)
                logger.info(
                    f"Rejeição na resistência detectada: "
                    f"High={current_high:.2f}, Close={current_close:.2f}, "
                    f"Sombra superior={(current_high - current_close):.2f} (força: {strength:.1f})"
                )

        # Combinar sinais de resistência
        if resistance_signals:
            max_resistance = max(resistance_signals)
            return True, max_resistance

        return False, 0.0

    def _check_stochastic_overbought(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se o Stochastic está em condição de sobrecompra.

        Returns:
            tuple: (sobrecompra, força_do_sinal de 0 a 1)
        """
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]

            # Zona de sobrecompra
            if stoch_k > 75 and stoch_d > 75:
                # Calcular a força do sinal baseado em quão alto está o stoch
                strength = min((stoch_k - 75) / 25, 1.0)
                logger.info(f"Stochastic em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {strength:.1f})")
                return True, strength

            if stoch_k > 70 and stoch_d > 70:
                strength = min((stoch_k - 70) / 30, 0.7)
                logger.info(f"Stochastic próximo de sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f}")
                return True, strength

        return False, 0.0

    def _check_bearish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se existe divergência de baixa entre preço e osciladores.

        Uma divergência bearish ocorre quando o preço faz máximos mais altos,
        mas o oscilador (RSI ou MACD) faz máximos mais baixos.

        Returns:
            tuple: (divergência_detectada, força_da_divergência de 0 a 1)
        """
        divergence_detected = False
        divergence_strength = 0.0

        # Precisamos de pelo menos 10 barras para detectar divergência
        if len(df) < 10:
            return False, 0.0

        # Verificar divergência no RSI
        if 'rsi' in df.columns and 'high' in df.columns:
            # Encontrar os últimos dois máximos de preço
            highs = df['high'].iloc[-10:].values
            price_peaks_idx = self._find_peaks(highs)

            if len(price_peaks_idx) >= 2:
                peak1_idx, peak2_idx = price_peaks_idx[-2], price_peaks_idx[-1]

                # Verificar se o preço está fazendo máximos mais altos
                if highs[peak2_idx] > highs[peak1_idx]:
                    # Verificar se o RSI está fazendo máximos mais baixos
                    rsi_values = df['rsi'].iloc[-10:].values
                    if rsi_values[peak2_idx] < rsi_values[peak1_idx]:
                        divergence_detected = True
                        # Força baseada na diferença dos RSI
                        diff = rsi_values[peak1_idx] - rsi_values[peak2_idx]
                        divergence_strength = min(diff / 10, 1.0)
                        logger.info(
                            f"Divergência RSI detectada: "
                            f"Preço {highs[peak1_idx]:.2f}->{highs[peak2_idx]:.2f}, "
                            f"RSI {rsi_values[peak1_idx]:.1f}->{rsi_values[peak2_idx]:.1f} "
                            f"(força: {divergence_strength:.1f})"
                        )

        # Verificar divergência no MACD
        if not divergence_detected and 'macd_histogram' in df.columns and 'high' in df.columns:
            # Usar os mesmos picos de preço
            highs = df['high'].iloc[-10:].values
            price_peaks_idx = self._find_peaks(highs)

            if len(price_peaks_idx) >= 2:
                peak1_idx, peak2_idx = price_peaks_idx[-2], price_peaks_idx[-1]

                # Verificar se o preço está fazendo máximos mais altos
                if highs[peak2_idx] > highs[peak1_idx]:
                    # Verificar se o MACD está fazendo máximos mais baixos
                    macd_values = df['macd_histogram'].iloc[-10:].values
                    if macd_values[peak2_idx] < macd_values[peak1_idx]:
                        divergence_detected = True
                        # Força baseada na diferença dos MACD
                        diff = macd_values[peak1_idx] - macd_values[peak2_idx]
                        divergence_strength = min(diff / 0.0005, 1.0)
                        logger.info(
                            f"Divergência MACD detectada: "
                            f"Preço {highs[peak1_idx]:.2f}->{highs[peak2_idx]:.2f}, "
                            f"MACD hist {macd_values[peak1_idx]:.6f}->{macd_values[peak2_idx]:.6f} "
                            f"(força: {divergence_strength:.1f})"
                        )

        return divergence_detected, divergence_strength

    def _find_peaks(self, values: np.ndarray) -> list[int]:
        """
        Encontra os índices dos picos em uma série de valores.

        Args:
            values: Array de valores numéricos

        Returns:
            list: Índices dos picos encontrados
        """
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em baixa.
        Busca oportunidades de venda em rallies (correções de alta) da tendência de baixa.

        Implementa análise avançada de condições de entrada, combinando múltiplos
        indicadores com ponderação de importância.
        """
        # 1. Detectar rally (correção de alta) na tendência de baixa
        in_rally, rally_strength = self._detect_rally(df)

        # 2. Verificar resistência e possível rejeição
        near_resistance, resistance_strength = self._detect_resistance(df, current_price)

        # 3. Verificar Stochastic para confirmar sobrecompra
        stoch_overbought, stoch_strength = self._check_stochastic_overbought(df)

        # 4. Verificar divergência bearish
        divergence, divergence_strength = self._check_bearish_divergence(df)

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
            mtf_aligned = 'DOWNTREND' in mtf_trend

            # Obter score de confiança se disponível
            if 'confidence' in mtf_data:
                mtf_strength = mtf_data['confidence'] / 100

            if mtf_aligned:
                logger.info(f"Alinhamento multi-timeframe favorável: {mtf_trend} (força: {mtf_strength:.1f})")

        # 7. Cálculo ponderado do score de condições de entrada
        # Cada condição tem um peso específico de acordo com sua importância
        entry_conditions_score = 0.0
        weights_sum = 0.0

        # Pesos para cada condição (rally é menos importante que rejeição em resistência)
        condition_weights = {
            'rally': 0.15,
            'resistance': 0.25,
            'stochastic': 0.15,
            'divergence': 0.15,
            'strong_trend': 0.15,
            'mtf_alignment': 0.15
        }

        # Adicionar score ponderado para cada condição se presente
        if in_rally:
            entry_conditions_score += rally_strength * condition_weights['rally']
            weights_sum += condition_weights['rally']

        if near_resistance:
            entry_conditions_score += resistance_strength * condition_weights['resistance']
            weights_sum += condition_weights['resistance']

        if stoch_overbought:
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
        min_conditions = 2
        min_score = 0.5

        conditions_count = sum(
            [in_rally, near_resistance, stoch_overbought,
             divergence, strong_trend, mtf_aligned]
        )

        logger.info(
            f"Condições para SHORT: {conditions_count}/{min_conditions} atendidas, Score: {entry_conditions_score:.2f} - "
            f"Rally={in_rally} ({rally_strength:.1f}), Resistência={near_resistance} ({resistance_strength:.1f}), "
            f"Stoch_Overbought={stoch_overbought} ({stoch_strength:.1f}), Divergência={divergence} ({divergence_strength:.1f}), "
            f"Strong_Trend={strong_trend} ({trend_strength:.1f}), MTF_Aligned={mtf_aligned} ({mtf_strength:.1f})"
        )

        # Decidir se geramos sinal baseado no número de condições e no score
        generate_signal = conditions_count >= min_conditions and entry_conditions_score >= min_score

        if generate_signal:
            logger.info(
                f"Condições favoráveis para SHORT em tendência de baixa. "
                f"Score: {entry_conditions_score:.2f} com {conditions_count} condições."
            )

            # Usar o serviço de previsão para obter TP/SL
            prediction = self.prediction_service.predict_tp_sl(df, current_price, "SHORT")
            if prediction is None:
                return None

            predicted_tp_pct, predicted_sl_pct = prediction

            # TP em múltiplos níveis para gerenciamento de risco aprimorado
            # Em tendência de baixa forte, usar 3 níveis de TP
            tp_levels = []

            # Primeira parte = 1/3 do movimento
            first_tp_pct = predicted_tp_pct * 0.33
            # Segunda parte = 2/3 do movimento
            second_tp_pct = predicted_tp_pct * 0.66
            # Terceira parte = movimento completo
            third_tp_pct = predicted_tp_pct

            tp_levels = [first_tp_pct, second_tp_pct, third_tp_pct]
            tp_percents = [33, 33, 34]  # Porcentagem da posição para cada nível

            # Avaliar a qualidade da entrada com base no cenário atual
            should_enter, entry_score = self.evaluate_entry_quality(
                df, current_price, "SHORT", predicted_tp_pct, predicted_sl_pct,
                mtf_alignment=mtf_strength
            )

            # Adicionar o score de confluência de condições
            entry_score = 0.5 * entry_score + 0.5 * entry_conditions_score

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Vamos forçar operações SHORT em tendência de baixa
            side: Literal["SELL", "BUY"] = "SELL"
            position_side: Literal["LONG", "SHORT"] = "SHORT"

            # Calcular fatores com base nos percentuais previstos
            tp_factor = 1 - abs(predicted_tp_pct) / 100
            sl_factor = 1 + abs(predicted_sl_pct) / 100

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # ATR para ajustes dinâmicos de position sizing e trailing stop
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Calcular a razão Risco:Recompensa
            rr_ratio = abs(predicted_tp_pct / predicted_sl_pct)

            # Determinar tendência e força
            market_trend = "DOWNTREND"  # Já sabemos que estamos em downtrend
            market_strength = "STRONG_TREND" if df['adx'].iloc[-1] > 25 else "WEAK_TREND"

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Criar o sinal
            signal = TradingSignal(
                id=signal_id,
                direction="SHORT",
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
                mtf_trend="DOWNTREND",
                mtf_confidence=mtf_strength * 100 if mtf_strength else None,
                mtf_alignment=mtf_strength if mtf_strength else None,
                mtf_details={
                    'entry_conditions_score': entry_conditions_score,
                    'rally_strength': rally_strength if in_rally else 0,
                    'resistance_strength': resistance_strength if near_resistance else 0,
                    'tp_levels': tp_levels,
                    'tp_percents': tp_percents
                } if 'mtf_details' in TradingSignal.__annotations__ else None
            )

            return signal

        return None

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
        Avalia a qualidade da entrada para DowntrendStrategy.

        Implementa avaliação avançada com múltiplos critérios de qualidade
        e ajustes para o cenário de tendência de baixa.
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

        # Verificações adicionais baseadas em indicadores
        if trade_direction == "SHORT" and self.calculate_trend_direction(df) == "DOWNTREND":
            # Bônus para trades SHORT em tendência de baixa
            entry_score = min(1.0, entry_score * 1.2)
        elif trade_direction == "LONG" and self.calculate_trend_direction(df) == "DOWNTREND":
            # Penalidade para trades LONG em tendência de baixa
            entry_score = entry_score * 0.7

        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if trade_direction == "SHORT" and rsi > 55:
                # Bônus para trades SHORT quando RSI está alto (sobrecompra)
                entry_score = min(1.0, entry_score * 1.2)
            elif trade_direction == "SHORT" and rsi < 30:
                # Penalidade para trades SHORT quando RSI está baixo (sobrevenda)
                entry_score = entry_score * 0.8

        # Verificar ADX para força da tendência
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            if adx > 30:  # Tendência forte
                if trade_direction == "SHORT":
                    # Bônus para SHORT em tendência forte de baixa
                    entry_score = min(1.0, entry_score * 1.15)
            elif adx < 20:  # Tendência fraca
                # Penalidade para qualquer trade em tendência fraca
                entry_score = entry_score * 0.9

        # Verificar condição de volatilidade
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]

            if atr_pct > 1.5:  # Alta volatilidade
                # Em alta volatilidade, ser mais cauteloso
                entry_score = entry_score * 0.9
            elif atr_pct < 0.5:  # Baixa volatilidade
                # Em baixa volatilidade, ser menos rígido
                entry_score = min(1.0, entry_score * 1.1)

        # Bônus para alinhamento multi-timeframe forte
        if mtf_alignment is not None and mtf_alignment > 0.5:
            # Aumentar score com base no alinhamento MTF
            entry_score = min(1.0, entry_score * (1.0 + (mtf_alignment - 0.5)))

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em baixa.

        Em tendência de baixa, favorecemos sinais SHORT e ajustamos TP/SL
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

        if signal.direction != "SHORT":
            # Aumentar o threshold para trades contra a tendência (LONG em downtrend)
            logger.info(f"Sinal LONG em tendência de baixa: exigindo maior qualidade")
            # Reduzir a pontuação para dificultar a entrada
            if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                signal.entry_score = signal.entry_score * 0.8
        else:  # Para sinais SHORT em tendência de baixa
            # Ajustar TP para ser mais ambicioso
            signal.predicted_tp_pct = abs(signal.predicted_tp_pct) * tp_adj_factor
            if signal.predicted_tp_pct > 0:  # Garantir que é negativo para SHORT
                signal.predicted_tp_pct = -signal.predicted_tp_pct

            # Ajustar SL para ser mais apertado
            signal.predicted_sl_pct = signal.predicted_sl_pct * sl_adj_factor

            # Recalcular preços e fatores
            if signal.direction == "LONG":
                signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
            else:  # SHORT
                signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
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
                f"Sinal SHORT ajustado para mercado em baixa: "
                f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
                f"R:R={signal.rr_ratio:.2f}"
            )

        return signal
