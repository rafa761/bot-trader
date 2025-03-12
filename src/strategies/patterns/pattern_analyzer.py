# strategies/patterns/pattern_analyzer.py

import numpy as np
import pandas as pd

from core.logger import logger


class TechnicalPatternAnalyzer:
    """
    Classe base para análise de padrões técnicos.

    Contém métodos auxiliares compartilhados entre analisadores específicos
    e pode ser estendida para análise de padrões específicos.
    """

    @staticmethod
    def find_peaks(values: np.ndarray) -> list[int]:
        """
        Encontra os índices dos picos em uma série de valores.

        Args:
            values: Array de valores numéricos

        Returns:
            list[int]: Índices dos picos encontrados
        """
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                peaks.append(i)
        return peaks

    @staticmethod
    def find_valleys(values: np.ndarray) -> list[int]:
        """
        Encontra os índices dos vales (mínimos locais) em uma série de valores.

        Args:
            values: Array de valores numéricos

        Returns:
            list[int]: Índices dos vales encontrados
        """
        valleys = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                valleys.append(i)
        return valleys

    @staticmethod
    def calculate_trend_direction(df: pd.DataFrame) -> str:
        """
        Calcula a direção atual da tendência com base nas médias móveis.

        Args:
            df: DataFrame com os dados históricos

        Returns:
            str: "UPTREND", "DOWNTREND" ou "NEUTRAL"
        """
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]

            if ema_short > ema_long:
                return "UPTREND"
            elif ema_short < ema_long:
                return "DOWNTREND"

        return "NEUTRAL"

    @staticmethod
    def calculate_volatility_level(df: pd.DataFrame) -> float:
        """
        Calcula o nível de volatilidade atual com base no ATR percentual.

        Args:
            df: DataFrame com os dados históricos

        Returns:
            float: Nível de volatilidade (0.0 a 1.0)
        """
        if 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100

            # Normalizar para uma escala de 0 a 1
            # Considerando que ATR acima de 3% é volatilidade extrema (1.0)
            # e ATR abaixo de 0.3% é volatilidade mínima (0.0)
            volatility = (atr_pct - 0.3) / (3.0 - 0.3)
            return max(0.0, min(1.0, volatility))

        return 0.5  # Valor padrão médio se ATR não estiver disponível


class MomentumPatternAnalyzer(TechnicalPatternAnalyzer):
    """
    Analisador de padrões baseados em momentum de mercado.

    Detecta padrões como reversões em pontos extremos, divergências,
    e condições de sobrecompra/sobrevenda.
    """

    def check_overbought_oversold(self, df: pd.DataFrame) -> tuple[bool, bool, float]:
        """
        Verifica se o mercado está em condição de sobrecompra ou sobrevenda
        usando múltiplos osciladores.

        Returns:
            tuple[bool, bool, float]: (em_sobrecompra, em_sobrevenda, força_do_sinal)
        """
        overbought = False
        oversold = False
        signal_strength = 0.0
        signals = []

        # 1. Verificar RSI
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            rsi_prev = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Sobrecompra e sobrevenda no RSI + direção de mudança
            if rsi > 70:
                overbought_strength = min((rsi - 70) / 30, 1.0)
                # Maior força se o RSI está começando a cair (possível reversão)
                if rsi < rsi_prev:
                    overbought_strength *= 1.2

                signals.append(('overbought', overbought_strength))
                logger.info(f"RSI em sobrecompra: {rsi:.1f} (força: {overbought_strength:.2f})")

            elif rsi < 30:
                oversold_strength = min((30 - rsi) / 30, 1.0)
                # Maior força se o RSI está começando a subir (possível reversão)
                if rsi > rsi_prev:
                    oversold_strength *= 1.2

                signals.append(('oversold', oversold_strength))
                logger.info(f"RSI em sobrevenda: {rsi:.1f} (força: {oversold_strength:.2f})")

        # 2. Verificar Stochastic
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            stoch_k_prev = df['stoch_k'].iloc[-2] if len(df) > 2 else 50

            # Sobrecompra e sobrevenda no Stochastic + direção de mudança
            if stoch_k > 80 and stoch_d > 80:
                overbought_strength = min((stoch_k - 80) / 20, 1.0) * 0.8  # Peso menor que RSI
                # Maior força se o Stoch está começando a cair (possível reversão)
                if stoch_k < stoch_k_prev:
                    overbought_strength *= 1.2

                signals.append(('overbought', overbought_strength))
                logger.info(
                    f"Stochastic em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {overbought_strength:.2f})"
                )

            elif stoch_k < 20 and stoch_d < 20:
                oversold_strength = min((20 - stoch_k) / 20, 1.0) * 0.8  # Peso menor que RSI
                # Maior força se o Stoch está começando a subir (possível reversão)
                if stoch_k > stoch_k_prev:
                    oversold_strength *= 1.2

                signals.append(('oversold', oversold_strength))
                logger.info(
                    f"Stochastic em sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {oversold_strength:.2f})"
                )

        # 3. Verificar Bollinger %B
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]
            pct_b_prev = df['boll_pct_b'].iloc[-2] if len(df) > 2 else 0.5

            # %B próximo de 1 = sobrecompra, %B próximo de 0 = sobrevenda
            if pct_b > 0.95:
                overbought_strength = min((pct_b - 0.95) * 20, 1.0) * 0.7  # Peso menor ainda
                # Maior força se %B está começando a cair (possível reversão)
                if pct_b < pct_b_prev:
                    overbought_strength *= 1.2

                signals.append(('overbought', overbought_strength))
                logger.info(f"Bollinger %B em sobrecompra: {pct_b:.2f} (força: {overbought_strength:.2f})")

            elif pct_b < 0.05:
                oversold_strength = min((0.05 - pct_b) * 20, 1.0) * 0.7  # Peso menor ainda
                # Maior força se %B está começando a subir (possível reversão)
                if pct_b > pct_b_prev:
                    oversold_strength *= 1.2

                signals.append(('oversold', oversold_strength))
                logger.info(f"Bollinger %B em sobrevenda: {pct_b:.2f} (força: {oversold_strength:.2f})")

        # 4. Verificar MACD histograma para confirmar momentum
        if 'macd_histogram' in df.columns and len(df) > 2:
            hist = df['macd_histogram'].iloc[-1]
            hist_prev = df['macd_histogram'].iloc[-2]

            # Se o histograma está diminuindo numa região positiva = perda de momentum de alta
            if hist > 0 and hist < hist_prev:
                overbought_boost = 0.1
                signals.append(('overbought', overbought_boost))
                logger.info(f"MACD histograma diminuindo em região positiva: {hist:.6f} < {hist_prev:.6f}")

            # Se o histograma está aumentando numa região negativa = perda de momentum de baixa
            elif hist < 0 and hist > hist_prev:
                oversold_boost = 0.1
                signals.append(('oversold', oversold_boost))
                logger.info(f"MACD histograma aumentando em região negativa: {hist:.6f} > {hist_prev:.6f}")

        # Processar os sinais acumulados
        if signals:
            # Separe os sinais de sobrecompra e sobrevenda
            overbought_signals = [strength for signal_type, strength in signals if signal_type == 'overbought']
            oversold_signals = [strength for signal_type, strength in signals if signal_type == 'oversold']

            # Determinar se estamos em sobrecompra ou sobrevenda com base no tipo dominante
            if overbought_signals and len(overbought_signals) > len(oversold_signals):
                overbought = True
                # Use o sinal mais forte como base e adicione pequenos bônus para sinais adicionais
                signal_strength = max(overbought_signals) + 0.05 * (len(overbought_signals) - 1)
                signal_strength = min(signal_strength, 1.0)

            elif oversold_signals and len(oversold_signals) > len(overbought_signals):
                oversold = True
                # Use o sinal mais forte como base e adicione pequenos bônus para sinais adicionais
                signal_strength = max(oversold_signals) + 0.05 * (len(oversold_signals) - 1)
                signal_strength = min(signal_strength, 1.0)

        return overbought, oversold, signal_strength

    def check_stochastic_oversold(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se o Stochastic está em condição de sobrevenda.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (sobrevenda, força_do_sinal de 0 a 1)
        """
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]

            # Zona de sobrevenda
            if stoch_k < 25 and stoch_d < 25:
                # Calcular a força do sinal baseado em quão baixo está o stoch
                strength = min((25 - stoch_k) / 25, 1.0)
                logger.info(f"Stochastic em sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {strength:.1f})")
                return True, strength

            if stoch_k < 30 and stoch_d < 30:
                strength = min((30 - stoch_k) / 30, 0.7)
                logger.info(f"Stochastic próximo de sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f}")
                return True, strength

        return False, 0.0

    def check_stochastic_overbought(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se o Stochastic está em condição de sobrecompra.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (sobrecompra, força_do_sinal de 0 a 1)
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

    def check_bullish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se existe divergência de alta entre preço e osciladores.

        Uma divergência bullish ocorre quando o preço faz mínimos mais baixos,
        mas o oscilador (RSI ou MACD) faz mínimos mais altos.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (divergência_detectada, força_da_divergência de 0 a 1)
        """
        divergence_detected = False
        divergence_strength = 0.0

        # Precisamos de pelo menos 10 barras para detectar divergência
        if len(df) < 10:
            return False, 0.0

        # Verificar divergência no RSI
        if 'rsi' in df.columns and 'low' in df.columns:
            # Encontrar os últimos dois mínimos de preço
            lows = df['low'].iloc[-10:].values
            price_valleys_idx = self.find_valleys(lows)

            if len(price_valleys_idx) >= 2:
                valley1_idx, valley2_idx = price_valleys_idx[-2], price_valleys_idx[-1]

                # Verificar se o preço está fazendo mínimos mais baixos
                if lows[valley2_idx] < lows[valley1_idx]:
                    # Verificar se o RSI está fazendo mínimos mais altos
                    rsi_values = df['rsi'].iloc[-10:].values
                    if rsi_values[valley2_idx] > rsi_values[valley1_idx]:
                        divergence_detected = True
                        # Força baseada na diferença dos RSI
                        diff = rsi_values[valley2_idx] - rsi_values[valley1_idx]
                        divergence_strength = min(diff / 10, 1.0)
                        logger.info(
                            f"Divergência RSI detectada: "
                            f"Preço {lows[valley1_idx]:.2f}->{lows[valley2_idx]:.2f}, "
                            f"RSI {rsi_values[valley1_idx]:.1f}->{rsi_values[valley2_idx]:.1f} "
                            f"(força: {divergence_strength:.1f})"
                        )

        # Verificar divergência no MACD
        if not divergence_detected and 'macd_histogram' in df.columns and 'low' in df.columns:
            # Usar os mesmos vales de preço
            lows = df['low'].iloc[-10:].values
            price_valleys_idx = self.find_valleys(lows)

            if len(price_valleys_idx) >= 2:
                valley1_idx, valley2_idx = price_valleys_idx[-2], price_valleys_idx[-1]

                # Verificar se o preço está fazendo mínimos mais baixos
                if lows[valley2_idx] < lows[valley1_idx]:
                    # Verificar se o MACD está fazendo mínimos mais altos
                    macd_values = df['macd_histogram'].iloc[-10:].values
                    if macd_values[valley2_idx] > macd_values[valley1_idx]:
                        divergence_detected = True
                        # Força baseada na diferença dos MACD
                        diff = macd_values[valley2_idx] - macd_values[valley1_idx]
                        divergence_strength = min(diff / 0.0005, 1.0)
                        logger.info(
                            f"Divergência MACD detectada: "
                            f"Preço {lows[valley1_idx]:.2f}->{lows[valley2_idx]:.2f}, "
                            f"MACD hist {macd_values[valley1_idx]:.6f}->{macd_values[valley2_idx]:.6f} "
                            f"(força: {divergence_strength:.1f})"
                        )

        return divergence_detected, divergence_strength

    def check_bearish_divergence(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se existe divergência de baixa entre preço e osciladores.

        Uma divergência bearish ocorre quando o preço faz máximos mais altos,
        mas o oscilador (RSI ou MACD) faz máximos mais baixos.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (divergência_detectada, força_da_divergência de 0 a 1)
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
            price_peaks_idx = self.find_peaks(highs)

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
            price_peaks_idx = self.find_peaks(highs)

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

    def detect_reversal_potential(self, df: pd.DataFrame) -> tuple[bool, float, str]:
        """
        Detecta potencial de reversão em movimento extremo, comum após volatilidade alta.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float, str]: (potencial_de_reversão, força_do_sinal de 0 a 1, direção da reversão)
        """
        if len(df) < 10:  # Precisamos de histórico suficiente
            return False, 0.0, "NONE"

        reversal_potential = False
        reversal_strength = 0.0
        reversal_direction = "NONE"

        # 1. Verificar RSI em níveis extremos
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # RSI em sobrevenda e começando a subir
            if rsi < 30 and rsi > prev_rsi:
                reversal_potential = True
                reversal_direction = "UP"
                # Força baseada em quão extremo está o RSI
                rsi_strength = min((30 - rsi) / 20, 1.0)
                reversal_strength = rsi_strength * 0.7

                logger.info(f"Potencial de reversão de ALTA: RSI={rsi:.1f} (força: {reversal_strength:.2f})")

            # RSI em sobrecompra e começando a cair
            elif rsi > 70 and rsi < prev_rsi:
                reversal_potential = True
                reversal_direction = "DOWN"
                # Força baseada em quão extremo está o RSI
                rsi_strength = min((rsi - 70) / 20, 1.0)
                reversal_strength = rsi_strength * 0.7

                logger.info(f"Potencial de reversão de BAIXA: RSI={rsi:.1f} (força: {reversal_strength:.2f})")

        # 2. Verificar padrões de velas de reversão
        if len(df) >= 3:
            # Obter dados dos últimos candles
            c0, o0 = df['close'].iloc[-1], df['open'].iloc[-1]
            c1, o1 = df['close'].iloc[-2], df['open'].iloc[-2]
            c2, o2 = df['close'].iloc[-3], df['open'].iloc[-3]
            h0, l0 = df['high'].iloc[-1], df['low'].iloc[-1]
            h1, l1 = df['high'].iloc[-2], df['low'].iloc[-2]

            # Potencial de reversão de BAIXA
            if (c2 > o2 and c1 > o1  # Duas velas de alta
                    and c0 < o0  # Seguidas por uma vela de baixa
                    and c0 < c1  # Fechamento abaixo do anterior
                    and abs(c0 - o0) > abs(c1 - o1) * 1.2):  # Vela de baixa mais forte

                if not reversal_potential or (reversal_potential and reversal_direction == "DOWN"):
                    # Se ainda não temos sinal ou confirma direção existente
                    reversal_potential = True
                    reversal_direction = "DOWN"
                    candle_strength = min(abs(c0 - o0) / (h1 - l1), 1.0)

                    if not reversal_strength:
                        reversal_strength = candle_strength * 0.6
                    else:
                        # Adicionar à força existente
                        reversal_strength = min(reversal_strength + candle_strength * 0.3, 1.0)

                    logger.info(
                        f"Padrão de reversão de BAIXA: velas de alta seguidas por vela de baixa forte (força: {candle_strength:.2f})"
                    )

            # Potencial de reversão de ALTA
            elif (c2 < o2 and c1 < o1  # Duas velas de baixa
                  and c0 > o0  # Seguidas por uma vela de alta
                  and c0 > c1  # Fechamento acima do anterior
                  and abs(c0 - o0) > abs(c1 - o1) * 1.2):  # Vela de alta mais forte

                if not reversal_potential or (reversal_potential and reversal_direction == "UP"):
                    # Se ainda não temos sinal ou confirma direção existente
                    reversal_potential = True
                    reversal_direction = "UP"
                    candle_strength = min(abs(c0 - o0) / (h1 - l1), 1.0)

                    if not reversal_strength:
                        reversal_strength = candle_strength * 0.6
                    else:
                        # Adicionar à força existente
                        reversal_strength = min(reversal_strength + candle_strength * 0.3, 1.0)

                    logger.info(
                        f"Padrão de reversão de ALTA: velas de baixa seguidas por vela de alta forte (força: {candle_strength:.2f})"
                    )

        # 3. Verificar divergências entre preço e osciladores
        # Divergência bullish: preço faz mínimos mais baixos, mas oscilador faz mínimos mais altos
        if 'rsi' in df.columns and len(df) >= 10:
            # Encontrar mínimos e máximos locais de preço
            lows = df['low'].iloc[-10:].values
            highs = df['high'].iloc[-10:].values
            low_indices = [i for i in range(1, len(lows) - 1) if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]]
            high_indices = [i for i in range(1, len(highs) - 1) if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]]

            # Encontrar mínimos e máximos correspondentes no RSI
            rsi_values = df['rsi'].iloc[-10:].values

            # Verificar divergência bullish (mínimos)
            if len(low_indices) >= 2:
                idx1, idx2 = low_indices[-2], low_indices[-1]
                # Se preço fez mínimo mais baixo
                if lows[idx2] < lows[idx1]:
                    # Mas RSI fez mínimo mais alto (divergência bullish)
                    if rsi_values[idx2] > rsi_values[idx1]:
                        # Divergência significativa
                        if rsi_values[idx2] - rsi_values[idx1] > 2.0:
                            if not reversal_potential or (reversal_potential and reversal_direction == "UP"):
                                reversal_potential = True
                                reversal_direction = "UP"
                                div_strength = min((rsi_values[idx2] - rsi_values[idx1]) / 10, 0.8)

                                if not reversal_strength:
                                    reversal_strength = div_strength
                                else:
                                    # Adicionar à força existente
                                    reversal_strength = min(reversal_strength + div_strength * 0.5, 1.0)

                                logger.info(
                                    f"Divergência bullish: Preço {lows[idx1]:.2f}->{lows[idx2]:.2f}, "
                                    f"RSI {rsi_values[idx1]:.1f}->{rsi_values[idx2]:.1f} (força: {div_strength:.2f})"
                                )

            # Verificar divergência bearish (máximos)
            if len(high_indices) >= 2:
                idx1, idx2 = high_indices[-2], high_indices[-1]
                # Se preço fez máximo mais alto
                if highs[idx2] > highs[idx1]:
                    # Mas RSI fez máximo mais baixo (divergência bearish)
                    if rsi_values[idx2] < rsi_values[idx1]:
                        # Divergência significativa
                        if rsi_values[idx1] - rsi_values[idx2] > 2.0:
                            if not reversal_potential or (reversal_potential and reversal_direction == "DOWN"):
                                reversal_potential = True
                                reversal_direction = "DOWN"
                                div_strength = min((rsi_values[idx1] - rsi_values[idx2]) / 10, 0.8)

                                if not reversal_strength:
                                    reversal_strength = div_strength
                                else:
                                    # Adicionar à força existente
                                    reversal_strength = min(reversal_strength + div_strength * 0.5, 1.0)

                                logger.info(
                                    f"Divergência bearish: Preço {highs[idx1]:.2f}->{highs[idx2]:.2f}, "
                                    f"RSI {rsi_values[idx1]:.1f}->{rsi_values[idx2]:.1f} (força: {div_strength:.2f})"
                                )

        return reversal_potential, reversal_strength, reversal_direction


class TrendPatternAnalyzer(TechnicalPatternAnalyzer):
    """
    Analisador de padrões relacionados a tendências de mercado.

    Detecta padrões como pullbacks em tendência de alta, rallies em tendência de baixa,
    e movimentos de continuação de tendência.
    """

    def detect_pullback(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detecta pullbacks (correções) em uma tendência de alta.

        Um pullback é uma correção temporária de baixa em uma tendência de alta,
        geralmente acompanhada por diminuição de RSI e volume reduzido.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (pullback_detectado, força_do_pullback de 0 a 1)
        """
        pullback_strength = 0.0

        # Verificar RSI para pullback
        rsi_pullback = False
        rsi_strength = 0.0
        if 'rsi' in df.columns and len(df) > 3:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2]
            prev2_rsi = df['rsi'].iloc[-3]

            # Pullback = RSI caindo abaixo de 50 em tendência de alta
            if rsi < 50 and rsi < prev_rsi and prev_rsi < prev2_rsi:
                rsi_pullback = True
                # Calcular força do pullback baseado em quão baixo está o RSI
                rsi_strength = min((50 - rsi) / 20, 1.0)  # Normalizar entre 0-1
                logger.info(f"Pullback de RSI detectado: RSI={rsi:.1f} (força: {rsi_strength:.2f})")

        # Verificar volume baixo durante pullback (sinal de fraqueza do pullback)
        volume_signal = 0.0
        if 'volume' in df.columns and len(df) > 5:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-5:].mean()

            if current_volume < avg_volume * 0.8:  # Volume baixo no pullback
                volume_signal = 0.3  # Sinal positivo para entrada
                logger.info(f"Volume baixo durante pullback: {current_volume:.0f} vs média {avg_volume:.0f}")

        # Verificar velas de baixa consecutivas (sinal de pullback)
        candle_signal = 0.0
        if len(df) > 3:
            if (df['close'].iloc[-1] < df['open'].iloc[-1] and
                    df['close'].iloc[-2] < df['open'].iloc[-2]):
                candle_signal = 0.2
                logger.info("Padrão de velas de baixa consecutivas detectado")

        # Calcular a força total do pullback
        if rsi_pullback:
            pullback_strength = rsi_strength + volume_signal + candle_signal
            pullback_strength = min(pullback_strength, 1.0)  # Normalizar para máximo de 1.0

            return True, pullback_strength

        return False, 0.0

    def detect_rally(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Detecta rallies de correção em uma tendência de baixa.

        Um rally é uma correção temporária de alta em uma tendência de baixa,
        geralmente acompanhada por aumento de RSI e volume reduzido.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (rally_detectado, força_do_rally de 0 a 1)
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

    def detect_support(self, df: pd.DataFrame, current_price: float) -> tuple[bool, float]:
        """
        Detecta proximidade a níveis de suporte e possíveis bounces.

        Verifica múltiplos tipos de suporte: EMA longa, VWAP, níveis de pivot,
        e mínimos recentes.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual

        Returns:
            tuple[bool, float]: (próximo_do_suporte, força_do_suporte de 0 a 1)
        """
        support_signals = []

        # Verificar suporte na EMA longa
        if 'ema_long' in df.columns:
            ema_long = df['ema_long'].iloc[-1]
            price_to_ema = (current_price - ema_long) / current_price * 100

            # Suporte = preço próximo da EMA longa por cima
            if -0.2 < price_to_ema < 0.5:  # Preço próximo ou tocando EMA por cima
                strength = 0.8 if price_to_ema > 0 else 0.5  # Mais forte se ainda está acima
                support_signals.append(strength)
                logger.info(f"Preço próximo do suporte em EMA longa: {price_to_ema:.2f}% (força: {strength:.1f})")

        # Verificar VWAP como suporte
        if 'vwap' in df.columns:
            vwap = df['vwap'].iloc[-1]
            price_to_vwap = (current_price - vwap) / current_price * 100

            if -0.2 < price_to_vwap < 0.5:  # Preço próximo ou tocando VWAP por cima
                strength = 0.7 if price_to_vwap > 0 else 0.4
                support_signals.append(strength)
                logger.info(f"Preço próximo do suporte em VWAP: {price_to_vwap:.2f}% (força: {strength:.1f})")

        # Verificar níveis de pivot
        if all(col in df.columns for col in ['pivot', 'pivot_s1']):
            pivot = df['pivot'].iloc[-1]
            s1 = df['pivot_s1'].iloc[-1]

            price_to_pivot = (current_price - pivot) / current_price * 100
            price_to_s1 = (current_price - s1) / current_price * 100

            if -0.2 < price_to_pivot < 0.5:
                support_signals.append(0.6)
                logger.info(f"Preço próximo do nível de pivot: {price_to_pivot:.2f}%")

            if -0.2 < price_to_s1 < 0.5:
                support_signals.append(0.7)
                logger.info(f"Preço próximo do nível S1: {price_to_s1:.2f}%")

        # Verificar bounce no suporte baseado no padrão de velas
        if len(df) > 2 and all(col in df.columns for col in ['low', 'close', 'open']):
            current_low = df['low'].iloc[-1]
            current_close = df['close'].iloc[-1]
            current_open = df['open'].iloc[-1]

            # Range das últimas 20 barras
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            # Verificar padrão de bounce (sombra longa na parte inferior)
            if (current_low < current_close * 0.998 and
                    (current_close - current_low) > (recent_range * 0.08) and
                    current_close > current_open):  # Fechou acima da abertura

                strength = min((current_close - current_low) / (recent_range * 0.08), 1.0)
                support_signals.append(0.8 * strength)
                logger.info(
                    f"Bounce no suporte detectado: "
                    f"Low={current_low:.2f}, Close={current_close:.2f}, "
                    f"Sombra inferior={(current_close - current_low):.2f} (força: {strength:.1f})"
                )

        # Combinar sinais de suporte
        if support_signals:
            max_support = max(support_signals)
            return True, max_support

        return False, 0.0

    def detect_resistance(self, df: pd.DataFrame, current_price: float) -> tuple[bool, float]:
        """
        Detecta proximidade a níveis de resistência e possíveis rejeições.

        Verifica múltiplos tipos de resistência: EMA longa, VWAP, níveis de pivot,
        e máximos recentes.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual

        Returns:
            tuple[bool, float]: (próximo_da_resistência, força_da_resistência de 0 a 1)
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

        # Verificar VWAP como resistência
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

    def is_invalidation_pattern(self, df: pd.DataFrame, signal_direction: str) -> bool:
        """
        Verifica se existe padrão de invalidação para a direção do sinal.

        Args:
            df: DataFrame com dados históricos
            signal_direction: "LONG" ou "SHORT"

        Returns:
            bool: True se um padrão de invalidação for detectado
        """
        if len(df) < 3:
            return False

        if signal_direction == "SHORT":
            # Verifica padrão de reversão de baixa (hammer, engulfing bullish, etc)
            c0, o0 = df['close'].iloc[-1], df['open'].iloc[-1]
            c1, o1 = df['close'].iloc[-2], df['open'].iloc[-2]
            h0, l0 = df['high'].iloc[-1], df['low'].iloc[-1]

            # Hammer/Doji com sombra inferior longa
            lower_wick = min(o0, c0) - l0
            body_size = abs(c0 - o0)
            if (lower_wick > body_size * 2) and (c0 > o0):
                logger.warning("Padrão de invalidação SHORT: Hammer/Doji detectado")
                return True

            # Bullish engulfing
            if (c1 < o1) and (c0 > o0) and (c0 > o1) and (o0 < c1):
                logger.warning("Padrão de invalidação SHORT: Engulfing Bullish detectado")
                return True

            # Verificar RSI sobrevendido
            if 'rsi' in df.columns and df['rsi'].iloc[-1] < 30:
                logger.warning(f"Padrão de invalidação SHORT: RSI sobrevendido ({df['rsi'].iloc[-1]:.1f})")
                return True

            # Verificar %B muito baixo
            if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[-1] < 0.1:
                logger.warning(f"Padrão de invalidação SHORT: %B extremamente baixo ({df['boll_pct_b'].iloc[-1]:.2f})")
                return True

        elif signal_direction == "LONG":
            # Verifica padrão de reversão de alta (shooting star, engulfing bearish, etc)
            c0, o0 = df['close'].iloc[-1], df['open'].iloc[-1]
            c1, o1 = df['close'].iloc[-2], df['open'].iloc[-2]
            h0, l0 = df['high'].iloc[-1], df['low'].iloc[-1]

            # Shooting star/Doji com sombra superior longa
            upper_wick = h0 - max(o0, c0)
            body_size = abs(c0 - o0)
            if (upper_wick > body_size * 2) and (c0 < o0):
                logger.warning("Padrão de invalidação LONG: Shooting Star/Doji detectado")
                return True

            # Bearish engulfing
            if (c1 > o1) and (c0 < o0) and (c0 < o1) and (o0 > c1):
                logger.warning("Padrão de invalidação LONG: Engulfing Bearish detectado")
                return True

            # Verificar RSI sobrecomprado
            if 'rsi' in df.columns and df['rsi'].iloc[-1] > 70:
                logger.warning(f"Padrão de invalidação LONG: RSI sobrecomprado ({df['rsi'].iloc[-1]:.1f})")
                return True

            # Verificar %B muito alto
            if 'boll_pct_b' in df.columns and df['boll_pct_b'].iloc[-1] > 0.9:
                logger.warning(f"Padrão de invalidação LONG: %B extremamente alto ({df['boll_pct_b'].iloc[-1]:.2f})")
                return True

        return False


class VolatilityPatternAnalyzer(TechnicalPatternAnalyzer):
    """
    Analisador de padrões relacionados à volatilidade do mercado.

    Detecta padrões como breakouts, squeezes (compressão seguida de expansão),
    e desvios significativos da média.
    """

    def detect_breakout(self, df: pd.DataFrame, current_price: float) -> tuple[bool, float, str]:
        """
        Detecta breakouts de níveis importantes, comum em mercados voláteis.

        Um breakout é um movimento que rompe um nível de suporte/resistência
        ou padrão de consolidação, geralmente com aumento de volume.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual

        Returns:
            tuple[bool, float, str]: (breakout_detectado, força_do_breakout de 0 a 1, direção do breakout)
        """
        if len(df) < 20:  # Precisamos de histórico suficiente
            return False, 0.0, "NONE"

        breakout_detected = False
        breakout_strength = 0.0
        breakout_direction = "NONE"

        # 1. Verificar breakout das Bollinger Bands
        if 'boll_hband' in df.columns and 'boll_lband' in df.columns:
            upper_band = df['boll_hband'].iloc[-1]
            lower_band = df['boll_lband'].iloc[-1]

            # Checar preços anteriores para ver se estávamos contidos nas bandas
            prev_contained = True
            for i in range(2, 6):  # Verificar 4 períodos anteriores
                if i < len(df):
                    if df['close'].iloc[-i] > df['boll_hband'].iloc[-i] or df['close'].iloc[-i] < df['boll_lband'].iloc[
                        -i]:
                        prev_contained = False
                        break

            # Se estávamos contidos e agora rompemos, é um breakout
            if prev_contained:
                # Breakout para cima
                if current_price > upper_band:
                    breakout_detected = True
                    breakout_direction = "UP"
                    # Força baseada na distância do breakout da banda
                    breakout_strength = min((current_price - upper_band) / upper_band * 100, 2.0) / 2.0
                    logger.info(
                        f"Breakout de alta detectado: {current_price:.2f} > {upper_band:.2f} (força: {breakout_strength:.2f})"
                    )

                # Breakout para baixo
                elif current_price < lower_band:
                    breakout_detected = True
                    breakout_direction = "DOWN"
                    # Força baseada na distância do breakout da banda
                    breakout_strength = min((lower_band - current_price) / lower_band * 100, 2.0) / 2.0
                    logger.info(
                        f"Breakout de baixa detectado: {current_price:.2f} < {lower_band:.2f} (força: {breakout_strength:.2f})"
                    )

        # 2. Verificar breakout de máximos/mínimos recentes
        if not breakout_detected:
            # Encontrar máximos e mínimos recentes (últimos 20 períodos)
            recent_high = df['high'].iloc[-20:-1].max()
            recent_low = df['low'].iloc[-20:-1].min()

            # Breakout de máximos recentes
            if current_price > recent_high * 1.005:  # 0.5% acima do máximo recente
                days_since_high = 0
                for i in range(2, 20):
                    if i < len(df) and df['high'].iloc[-i] == recent_high:
                        days_since_high = i - 1
                        break

                # Quanto mais tempo desde o último máximo, mais significativo o breakout
                if days_since_high >= 5:
                    breakout_detected = True
                    breakout_direction = "UP"
                    # Força baseada na distância percentual e no tempo
                    pct_breakout = (current_price - recent_high) / recent_high * 100
                    time_factor = min(days_since_high / 10, 1.0)
                    breakout_strength = min(pct_breakout / 1.0, 1.0) * 0.6 + time_factor * 0.4

                    logger.info(
                        f"Breakout de máximos recentes: {current_price:.2f} > {recent_high:.2f} "
                        f"({pct_breakout:.2f}%, {days_since_high} períodos atrás, força: {breakout_strength:.2f})"
                    )

            # Breakout de mínimos recentes
            elif current_price < recent_low * 0.995:  # 0.5% abaixo do mínimo recente
                days_since_low = 0
                for i in range(2, 20):
                    if i < len(df) and df['low'].iloc[-i] == recent_low:
                        days_since_low = i - 1
                        break

                # Quanto mais tempo desde o último mínimo, mais significativo o breakout
                if days_since_low >= 5:
                    breakout_detected = True
                    breakout_direction = "DOWN"
                    # Força baseada na distância percentual e no tempo
                    pct_breakout = (recent_low - current_price) / recent_low * 100
                    time_factor = min(days_since_low / 10, 1.0)
                    breakout_strength = min(pct_breakout / 1.0, 1.0) * 0.6 + time_factor * 0.4

                    logger.info(
                        f"Breakout de mínimos recentes: {current_price:.2f} < {recent_low:.2f} "
                        f"({pct_breakout:.2f}%, {days_since_low} períodos atrás, força: {breakout_strength:.2f})"
                    )

        # 3. Verificar confirmação por volume
        if breakout_detected and 'volume' in df.columns:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:-1].mean()

            # Volume acima da média confirma breakout
            if current_volume > avg_volume * 1.3:  # 30% acima da média
                vol_boost = min((current_volume / avg_volume - 1.0) * 0.5, 0.3)
                breakout_strength = min(breakout_strength + vol_boost, 1.0)
                logger.info(f"Breakout confirmado por volume: +{vol_boost:.2f} na força")
            # Volume baixo reduz a confiança
            elif current_volume < avg_volume * 0.7:  # 30% abaixo da média
                vol_penalty = min((1.0 - current_volume / avg_volume) * 0.5, 0.3)
                breakout_strength = max(breakout_strength - vol_penalty, 0.0)
                logger.info(f"Breakout com volume baixo: -{vol_penalty:.2f} na força")

        return breakout_detected, breakout_strength, breakout_direction

    def detect_squeeze_breakout(self, df: pd.DataFrame) -> tuple[bool, float, str]:
        """
        Detecta breakouts após compressão de volatilidade, oportunidade em
        mercados que passam de baixa para média/alta volatilidade.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float, str]: (breakout_detectado, força_do_breakout de 0 a 1, direção do breakout)
        """
        if len(df) < 12:  # Precisamos de histórico suficiente
            return False, 0.0, "NONE"

        squeeze_breakout = False
        breakout_strength = 0.0
        breakout_direction = "NONE"

        # 1. Verificar se houve compressão recente nas Bollinger Bands
        if 'boll_width' in df.columns:
            current_width = df['boll_width'].iloc[-1]
            prev_width = df['boll_width'].iloc[-2] if len(df) > 2 else current_width

            # Verificar histórico recente para detectar compressão seguida de expansão
            compression_period = False
            if len(df) > 10:
                # Calcular média e desvio padrão das últimas 10 barras
                avg_width = df['boll_width'].iloc[-8:].mean()
                std_width = df['boll_width'].iloc[-8:].std()

                # Verificar se houve compressão significativa seguida de expansão
                min_width = df['boll_width'].iloc[-8:].min()

                # Compressão = pelo menos 1 barra com largura < (média - 1 desvio padrão)
                compression_threshold = avg_width - std_width
                compression_period = min_width < compression_threshold

                # Expansão atual = largura atual > largura anterior
                current_expanding = current_width > prev_width * 1.1  # 10% maior

                if compression_period and current_expanding:
                    # Determinar direção do breakout baseado na vela atual
                    if len(df) > 1:
                        if df['close'].iloc[-1] > df['open'].iloc[-1]:
                            squeeze_breakout = True
                            breakout_direction = "UP"
                            # Força baseada em quanto a largura expandiu
                            expansion_ratio = current_width / min_width
                            breakout_strength = min((expansion_ratio - 1.0) * 0.5, 1.0)
                            logger.info(
                                f"Breakout de squeeze para CIMA: expansão={expansion_ratio:.2f}x "
                                f"(força: {breakout_strength:.2f})"
                            )

                        elif df['close'].iloc[-1] < df['open'].iloc[-1]:
                            squeeze_breakout = True
                            breakout_direction = "DOWN"
                            # Força baseada em quanto a largura expandiu
                            expansion_ratio = current_width / min_width
                            breakout_strength = min((expansion_ratio - 1.0) * 0.5, 1.0)
                            logger.info(
                                f"Breakout de squeeze para BAIXO: expansão={expansion_ratio:.2f}x "
                                f"(força: {breakout_strength:.2f})"
                            )

        # 2. Verificar confirmação pelo MACD
        if squeeze_breakout and 'macd_histogram' in df.columns and len(df) > 3:
            hist = df['macd_histogram'].iloc[-1]
            prev_hist = df['macd_histogram'].iloc[-2]

            # MACD histograma crescendo confirma breakout para cima
            if breakout_direction == "UP" and hist > 0 and hist > prev_hist:
                # Adicionar força baseada na magnitude do histograma
                macd_strength = min(abs(hist / 0.0005), 0.3)
                breakout_strength = min(breakout_strength + macd_strength, 1.0)
                logger.info(f"MACD confirma breakout para cima: {hist:.6f} > {prev_hist:.6f}")

            # MACD histograma decrescendo confirma breakout para baixo
            elif breakout_direction == "DOWN" and hist < 0 and hist < prev_hist:
                # Adicionar força baseada na magnitude do histograma
                macd_strength = min(abs(hist / 0.0005), 0.3)
                breakout_strength = min(breakout_strength + macd_strength, 1.0)
                logger.info(f"MACD confirma breakout para baixo: {hist:.6f} < {prev_hist:.6f}")

        # 3. Verificar confirmação por volume
        if squeeze_breakout and 'volume' in df.columns and len(df) > 10:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].iloc[-10:-1].mean()

            # Volume acima da média confirma breakout
            if current_volume > avg_volume * 1.3:  # 30% acima da média
                vol_strength = min((current_volume / avg_volume - 1.0) * 0.5, 0.3)
                breakout_strength = min(breakout_strength + vol_strength, 1.0)
                logger.info(
                    f"Volume confirma breakout: {current_volume:.0f} vs {avg_volume:.0f} (força: +{vol_strength:.2f})"
                )

        return squeeze_breakout, breakout_strength, breakout_direction

    def detect_deviation_from_mean(self, df: pd.DataFrame, current_price: float) -> tuple[bool, float, str]:
        """
        Detecta desvios significativos da média, oportunidades em baixa volatilidade.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual

        Returns:
            tuple[bool, float, str]: (desvio_detectado, força_do_desvio de 0 a 1, direção da reversão esperada)
        """
        if len(df) < 10:  # Precisamos de histórico suficiente
            return False, 0.0, "NONE"

        deviation_detected = False
        deviation_strength = 0.0
        mean_reversion_direction = "NONE"

        # 1. Verificar desvio das Bollinger Bands (principal indicador)
        if 'boll_pct_b' in df.columns:
            pct_b = df['boll_pct_b'].iloc[-1]

            # Valores extremos indicam oportunidade de mean-reversion
            if pct_b < 0.1:  # Extremo inferior
                deviation_detected = True
                mean_reversion_direction = "UP"  # Esperamos reversão para cima
                # Força baseada em quão extremo é o desvio
                deviation_strength = min((0.1 - pct_b) * 10, 1.0)
                logger.info(f"Desvio extremo inferior: %B={pct_b:.2f} (força: {deviation_strength:.2f})")

            elif pct_b > 0.9:  # Extremo superior
                deviation_detected = True
                mean_reversion_direction = "DOWN"  # Esperamos reversão para baixo
                # Força baseada em quão extremo é o desvio
                deviation_strength = min((pct_b - 0.9) * 10, 1.0)
                logger.info(f"Desvio extremo superior: %B={pct_b:.2f} (força: {deviation_strength:.2f})")

        # 2. Verificar desvio da média móvel (EMA)
        if 'ema_short' in df.columns:
            ema = df['ema_short'].iloc[-1]

            # Calcular desvio percentual
            price_deviation = (current_price - ema) / ema * 100

            # Desvio significativo em mercado de baixa volatilidade (>0.5%)
            if abs(price_deviation) > 0.5:
                ema_deviation_strength = min(abs(price_deviation) / 1.0, 1.0)

                # Se já temos um sinal de desvio, verificar a concordância
                if deviation_detected:
                    if (price_deviation < 0 and mean_reversion_direction == "UP") or \
                            (price_deviation > 0 and mean_reversion_direction == "DOWN"):
                        # Adicionar ao sinal existente se confirma a direção
                        deviation_strength = (deviation_strength * 0.7) + (ema_deviation_strength * 0.3)
                        logger.info(f"Desvio da EMA confirma direção: {price_deviation:.2f}%")
                # Caso contrário, criar novo sinal
                else:
                    deviation_detected = True
                    deviation_strength = ema_deviation_strength * 0.6  # Peso menor que Bollinger
                    mean_reversion_direction = "UP" if price_deviation < 0 else "DOWN"
                    logger.info(
                        f"Desvio significativo da EMA: {price_deviation:.2f}% (força: {ema_deviation_strength:.2f})"
                    )

        # 3. Verificar RSI em extremos
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]

            # RSI em sobrevenda = oportunidade de alta
            if rsi < 30:
                rsi_strength = min((30 - rsi) / 20, 1.0)

                # Se já temos um sinal, verificar concordância
                if deviation_detected:
                    if mean_reversion_direction == "UP":
                        # Adicionar ao sinal existente
                        deviation_strength = (deviation_strength * 0.7) + (rsi_strength * 0.3)
                        logger.info(f"RSI em sobrevenda confirma reversão para cima: {rsi:.1f}")
                # Caso contrário, criar novo sinal
                else:
                    deviation_detected = True
                    deviation_strength = rsi_strength * 0.6
                    mean_reversion_direction = "UP"
                    logger.info(f"RSI em sobrevenda: {rsi:.1f} (força: {rsi_strength:.2f})")

            # RSI em sobrecompra = oportunidade de baixa
            elif rsi > 70:
                rsi_strength = min((rsi - 70) / 20, 1.0)

                # Se já temos um sinal, verificar concordância
                if deviation_detected:
                    if mean_reversion_direction == "DOWN":
                        # Adicionar ao sinal existente
                        deviation_strength = (deviation_strength * 0.7) + (rsi_strength * 0.3)
                        logger.info(f"RSI em sobrecompra confirma reversão para baixo: {rsi:.1f}")
                # Caso contrário, criar novo sinal
                else:
                    deviation_detected = True
                    deviation_strength = rsi_strength * 0.6
                    mean_reversion_direction = "DOWN"
                    logger.info(f"RSI em sobrecompra: {rsi:.1f} (força: {rsi_strength:.2f})")

        # 4. Verificar Stochastic em extremos
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]

            # Stochastic em sobrevenda = oportunidade de alta
            if stoch_k < 20 and stoch_d < 20:
                stoch_strength = min((20 - stoch_k) / 20, 1.0) * 0.8  # Peso menor que RSI

                # Se já temos um sinal, verificar concordância
                if deviation_detected:
                    if mean_reversion_direction == "UP":
                        # Adicionar ao sinal existente
                        deviation_strength = (deviation_strength * 0.8) + (stoch_strength * 0.2)
                        logger.info(
                            f"Stochastic em sobrevenda confirma reversão para cima: K={stoch_k:.1f}, D={stoch_d:.1f}"
                        )
                # Caso contrário, criar novo sinal
                else:
                    deviation_detected = True
                    deviation_strength = stoch_strength * 0.5
                    mean_reversion_direction = "UP"
                    logger.info(
                        f"Stochastic em sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {stoch_strength:.2f})"
                    )

            # Stochastic em sobrecompra = oportunidade de baixa
            elif stoch_k > 80 and stoch_d > 80:
                stoch_strength = min((stoch_k - 80) / 20, 1.0) * 0.8  # Peso menor que RSI

                # Se já temos um sinal, verificar concordância
                if deviation_detected:
                    if mean_reversion_direction == "DOWN":
                        # Adicionar ao sinal existente
                        deviation_strength = (deviation_strength * 0.8) + (stoch_strength * 0.2)
                        logger.info(
                            f"Stochastic em sobrecompra confirma reversão para baixo: K={stoch_k:.1f}, D={stoch_d:.1f}"
                        )
                # Caso contrário, criar novo sinal
                else:
                    deviation_detected = True
                    deviation_strength = stoch_strength * 0.5
                    mean_reversion_direction = "DOWN"
                    logger.info(
                        f"Stochastic em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f} (força: {stoch_strength:.2f})"
                    )

        return deviation_detected, deviation_strength, mean_reversion_direction

    def detect_range_compression(self, df: pd.DataFrame) -> tuple[bool, float]:
        """
        Verifica se o mercado está em compressão de volatilidade (squeeze),
        o que pode indicar um movimento explosivo iminente.

        Args:
            df: DataFrame com dados históricos

        Returns:
            tuple[bool, float]: (em_compressão, força_da_compressão de 0 a 1)
        """
        compression_detected = False
        compression_strength = 0.0

        if len(df) < 20:  # Precisamos de histórico para detectar compressão
            return False, 0.0

        # 1. Verificar compressão das Bollinger Bands
        if 'boll_width' in df.columns:
            current_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].iloc[-20:].mean()

            # Bandas mais estreitas que X% da média = compressão
            width_ratio = current_width / avg_width

            if width_ratio < 0.8:
                # Quanto menor o width_ratio, maior a compressão
                bb_compression = min((0.8 - width_ratio) * 5, 1.0)
                logger.info(f"Compressão nas Bollinger Bands: {width_ratio:.2f} vs média (força: {bb_compression:.2f})")

                # Verificar também se as bandas estão se estreitando ou já expandindo
                width_direction = 0
                for i in range(1, min(5, len(df))):
                    if df['boll_width'].iloc[-i] < df['boll_width'].iloc[-i - 1]:
                        width_direction -= 1  # Estreitando
                    else:
                        width_direction += 1  # Expandindo

                # Se as bandas começaram a expandir após compressão, isso é um sinal de movimento iminente
                if width_direction > 0:
                    bb_compression *= 1.2
                    logger.info(f"Bollinger Bands começando a expandir após compressão")

                compression_detected = True
                compression_strength = bb_compression

        # 2. Verificar redução do ATR (outra medida de compressão de volatilidade)
        if 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].iloc[-20:].mean()

            # ATR atual menor que X% da média = compressão
            atr_ratio = current_atr / avg_atr

            if atr_ratio < 0.8:
                # Quanto menor o atr_ratio, maior a compressão
                atr_compression = min((0.8 - atr_ratio) * 5, 1.0)
                logger.info(f"Compressão no ATR: {atr_ratio:.2f} vs média (força: {atr_compression:.2f})")

                if not compression_detected:
                    compression_detected = True
                    compression_strength = atr_compression
                else:
                    # Se já detectamos compressão nas BBands, combinar os sinais
                    compression_strength = 0.7 * compression_strength + 0.3 * atr_compression

        # 3. Verificar volume reduzido (comum antes de um movimento explosivo)
        if 'volume' in df.columns:
            current_volume = df['volume'].iloc[-3:].mean()  # Média dos últimos 3 períodos
            avg_volume = df['volume'].iloc[-20:].mean()

            # Volume atual menor que X% da média = possível compressão
            volume_ratio = current_volume / avg_volume

            if volume_ratio < 0.7:
                # Quanto menor o volume_ratio, maior a possibilidade de movimento iminente
                vol_compression = min((0.7 - volume_ratio) * 3, 1.0)
                logger.info(f"Volume reduzido: {volume_ratio:.2f} vs média (força: {vol_compression:.2f})")

                if not compression_detected:
                    compression_detected = True
                    compression_strength = vol_compression * 0.7  # Menor peso para volume sozinho
                else:
                    # Se já detectamos compressão, adicionar componente de volume
                    compression_strength = compression_strength * 0.8 + vol_compression * 0.2

        return compression_detected, min(compression_strength, 1.0)


class PatternAnalyzerFactory:
    """
    Factory para criação de analisadores de padrões.

    Permite obter diferentes tipos de analisadores de padrões a partir de um único ponto.
    """

    @staticmethod
    def create_momentum_analyzer() -> MomentumPatternAnalyzer:
        """Cria um analisador de padrões de momentum."""
        return MomentumPatternAnalyzer()

    @staticmethod
    def create_trend_analyzer() -> TrendPatternAnalyzer:
        """Cria um analisador de padrões de tendência."""
        return TrendPatternAnalyzer()

    @staticmethod
    def create_volatility_analyzer() -> VolatilityPatternAnalyzer:
        """Cria um analisador de padrões de volatilidade."""
        return VolatilityPatternAnalyzer()
