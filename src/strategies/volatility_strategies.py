# strategies/volatility_strategies.py
from datetime import datetime

import numpy as np
import pandas as pd

from core.config import settings
from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.schemas import TradingSignal
from strategies.base.model import BaseStrategy, StrategyConfig


class HighVolatilityStrategy(BaseStrategy):
    """
    Estratégia para mercados com alta volatilidade.
    Adaptada para capitalizar em movimentos amplos enquanto
    gerencia riscos em um ambiente de alta volatilidade.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
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
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.sequence_length = 24
        self.preprocessor = None
        self.preprocessor_fitted = False

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

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com alta volatilidade.
        Em mercados voláteis, buscamos entradas na direção da tendência principal em breakouts.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # Identificar a tendência principal
        trend = self.calculate_trend_direction(df)
        logger.info(f"Tendência identificada: {trend}")

        # Calcular o nível atual de volatilidade
        volatility_level = self.calculate_volatility_level(df)
        logger.info(f"Nível de volatilidade: {volatility_level:.2f}")

        # Verificar se estamos em volatilidade extrema (desistir em condições extremas)
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            if atr_pct > 3.0:  # Volatilidade extremamente alta
                logger.warning(f"Volatilidade extrema detectada (ATR% = {atr_pct:.2f}%). Evitando entradas.")
                return None

        # Em alta volatilidade, buscamos:
        # 1. Breakouts de níveis importantes
        # 2. Continuidade de tendência após consolidação
        # 3. Reversões após movimentos extremos (sobrecompra/sobrevenda)

        # Verificar breakouts
        breakout_detected = False
        breakout_direction = None

        if 'boll_hband' in df.columns and 'boll_lband' in df.columns:
            upper_band = df['boll_hband'].iloc[-1]
            lower_band = df['boll_lband'].iloc[-1]
            prev_close = df['close'].iloc[-2] if len(df) > 2 else 0

            # Breakout para cima
            if prev_close < upper_band and current_price > upper_band:
                breakout_detected = True
                breakout_direction = "LONG"
                logger.info(f"Breakout de ALTA detectado: {current_price:.2f} > {upper_band:.2f}")

            # Breakout para baixo
            elif prev_close > lower_band and current_price < lower_band:
                breakout_detected = True
                breakout_direction = "SHORT"
                logger.info(f"Breakout de BAIXA detectado: {current_price:.2f} < {lower_band:.2f}")
            else:
                logger.info(
                    f"Sem breakout: Preço={current_price:.2f}, Anterior={prev_close:.2f}, "
                    f"Upper={upper_band:.2f}, Lower={lower_band:.2f}"
                )

        # Verificar reversão após movimento extremo
        reversal_potential = False
        reversal_direction = None

        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Potencial de reversão de baixa (sobrevenda)
            if rsi < 30 and rsi > prev_rsi:
                reversal_potential = True
                reversal_direction = "LONG"
                logger.info(f"Potencial de reversão de ALTA: RSI em sobrevenda ({rsi:.1f}) e subindo")

            # Potencial de reversão de alta (sobrecompra)
            elif rsi > 70 and rsi < prev_rsi:
                reversal_potential = True
                reversal_direction = "SHORT"
                logger.info(f"Potencial de reversão de BAIXA: RSI em sobrecompra ({rsi:.1f}) e caindo")
            else:
                logger.info(f"Sem potencial de reversão: RSI={rsi:.1f}, Anterior={prev_rsi:.1f}")

        # Verificar impulsividade (momentum forte)
        strong_momentum = False
        momentum_direction = None

        if 'adx' in df.columns and 'di_plus' in df.columns and 'di_minus' in df.columns:
            adx = df['adx'].iloc[-1]
            di_plus = df['di_plus'].iloc[-1]
            di_minus = df['di_minus'].iloc[-1]

            if adx > 30:  # ADX forte indica tendência robusta
                if di_plus > di_minus:
                    strong_momentum = True
                    momentum_direction = "LONG"
                    logger.info(f"Momentum forte de ALTA: ADX={adx:.1f}, +DI={di_plus:.1f}, -DI={di_minus:.1f}")
                elif di_minus > di_plus:
                    strong_momentum = True
                    momentum_direction = "SHORT"
                    logger.info(f"Momentum forte de BAIXA: ADX={adx:.1f}, +DI={di_plus:.1f}, -DI={di_minus:.1f}")
            else:
                logger.info(f"Sem momentum forte: ADX={adx:.1f}, +DI={di_plus:.1f}, -DI={di_minus:.1f}")

        # Verificar alinhamento multi-timeframe
        mtf_aligned = False
        mtf_trend_direction = None

        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']

            if 'UPTREND' in mtf_trend:
                mtf_aligned = True
                mtf_trend_direction = "LONG"
                logger.info(f"Tendência MTF favorável para LONG: {mtf_trend}")
            elif 'DOWNTREND' in mtf_trend:
                mtf_aligned = True
                mtf_trend_direction = "SHORT"
                logger.info(f"Tendência MTF favorável para SHORT: {mtf_trend}")
            else:
                logger.info(f"Sem alinhamento MTF claro: {mtf_trend}")

        # Condições para LONG
        long_conditions = 0
        if breakout_detected and breakout_direction == "LONG":
            long_conditions += 1
        if reversal_potential and reversal_direction == "LONG":
            long_conditions += 1
        if strong_momentum and momentum_direction == "LONG":
            long_conditions += 1
        if mtf_aligned and mtf_trend_direction == "LONG":
            long_conditions += 1

        # Condições para SHORT
        short_conditions = 0
        if breakout_detected and breakout_direction == "SHORT":
            short_conditions += 1
        if reversal_potential and reversal_direction == "SHORT":
            short_conditions += 1
        if strong_momentum and momentum_direction == "SHORT":
            short_conditions += 1
        if mtf_aligned and mtf_trend_direction == "SHORT":
            short_conditions += 1

        # Log das condições
        logger.info(
            f"Condições ALTA VOLATILIDADE - LONG: {long_conditions}/2, SHORT: {short_conditions}/2 - "
            f"Breakout={breakout_detected} ({breakout_direction}), "
            f"Reversão={reversal_potential} ({reversal_direction}), "
            f"Momentum={strong_momentum} ({momentum_direction}), "
            f"MTF={mtf_aligned} ({mtf_trend_direction})"
        )

        # Determinar direção do sinal
        signal_direction = None

        # Decidir direção baseada no maior número de condições (mínimo 2)
        if long_conditions >= 2 and long_conditions >= short_conditions:
            signal_direction = "LONG"
            logger.info(f"Sinal LONG gerado em alta volatilidade com {long_conditions} condições favoráveis")
        elif short_conditions >= 2 and short_conditions > long_conditions:
            signal_direction = "SHORT"
            logger.info(f"Sinal SHORT gerado em alta volatilidade com {short_conditions} condições favoráveis")
        else:
            logger.info(f"Condições insuficientes para sinal: LONG={long_conditions}, SHORT={short_conditions}")
            return None

        # Se nenhuma direção for determinada, não gerar sinal
        if signal_direction is None:
            return None

        # Gerar previsões usando os modelos LSTM
        try:
            X_seq = self._prepare_sequence(df)
            if X_seq is None:
                return None

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

            # Ajustar direção do TP para SHORT
            if signal_direction == "SHORT" and predicted_tp_pct > 0:
                predicted_tp_pct = -predicted_tp_pct

            # Garantir valores positivos para SL
            predicted_sl_pct = abs(predicted_sl_pct)

            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Validar previsões - evitar valores absurdos ou muito pequenos
            if abs(predicted_tp_pct) > 20:
                predicted_tp_pct = 20.0 if signal_direction == "LONG" else -20.0

            if predicted_sl_pct > 10:
                predicted_sl_pct = 10.0

            # Em alta volatilidade, ajustar TP e SL para serem mais conservadores
            if predicted_sl_pct < 0.5:
                # Calcular o SL dinâmico baseado em ATR
                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
                if atr_value:
                    # SL maior em alta volatilidade
                    predicted_sl_pct = (atr_value / current_price) * 100 * 1.8
                    logger.info(f"SL ajustado para alta volatilidade: {predicted_sl_pct:.2f}%")

            # Aumentar TP em alta volatilidade para capturar mais do movimento
            predicted_tp_pct = predicted_tp_pct * 1.3  # 30% maior

            # Avaliar a qualidade da entrada
            should_enter, entry_score = self.evaluate_entry_quality(
                df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct
            )

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Configurar parâmetros para o sinal
            if signal_direction == "LONG":
                side = "BUY"
                position_side = "LONG"
                tp_factor = 1 + (predicted_tp_pct / 100)
                sl_factor = 1 - (predicted_sl_pct / 100)
            else:  # SHORT
                side = "SELL"
                position_side = "SHORT"
                tp_factor = 1 - (abs(predicted_tp_pct) / 100)
                sl_factor = 1 + (predicted_sl_pct / 100)

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Obter ATR para ajustes de quantidade
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Determinar tendência e força
            market_trend = trend
            market_strength = "HIGH_VOLATILITY"

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
                rr_ratio=abs(predicted_tp_pct / predicted_sl_pct),
                market_trend=market_trend,
                market_strength=market_strength,
                timestamp=datetime.now()
            )

            return signal

        except Exception as e:
            logger.error(f"Erro na geração de sinal: {e}", exc_info=True)
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
        Avalia a qualidade da entrada em condições de alta volatilidade.

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

            # Em alta volatilidade, exigir R:R maior
            should_enter = rr_ratio >= self.config.min_rr_ratio

            # Pontuação básica baseada em R:R
            entry_score = min(1.0, rr_ratio / 3.0)  # Pontuação de 0 a 1
        else:
            # Valores padrão se tp e sl não forem fornecidos
            should_enter = False
            entry_score = 0.0

        # Em alta volatilidade, dar preferência para operações na direção da tendência
        trend_direction = self.calculate_trend_direction(df)
        if (trade_direction == "LONG" and trend_direction == "UPTREND") or \
                (trade_direction == "SHORT" and trend_direction == "DOWNTREND"):
            # Bônus para trades na direção da tendência
            entry_score = min(1.0, entry_score * 1.3)
        else:
            # Penalidade para trades contra a tendência em volatilidade alta
            entry_score = entry_score * 0.7

        # Verificar RSI para evitar operações em extremos (risco de reversões rápidas)
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if (trade_direction == "LONG" and rsi < 20) or (trade_direction == "SHORT" and rsi > 80):
                # Bônus para reversões em níveis extremos
                entry_score = min(1.0, entry_score * 1.2)
            elif (trade_direction == "LONG" and rsi > 70) or (trade_direction == "SHORT" and rsi < 30):
                # Penalidade para entradas na direção de sobrecompra/sobrevenda
                entry_score = entry_score * 0.6

        # Verificar Bollinger Bands para identificar extremos
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02

            # Se as bandas estão muito largas, exigir mais qualidade
            if boll_width > (avg_width * 2):
                entry_score = entry_score * 0.8
                logger.info(f"Bandas muito largas (BB width={boll_width:.4f}): reduzindo score")

        # Usar alinhamento multi-timeframe para refinar a decisão
        if mtf_alignment is not None:
            # Em alta volatilidade, o alinhamento multitimeframe é crucial
            entry_score = entry_score * (0.5 + mtf_alignment * 0.5)

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM.

        Args:
            df: DataFrame com dados históricos

        Returns:
            np.ndarray: Sequência formatada para o modelo LSTM ou None se houver erro
        """
        try:
            from core.constants import FEATURE_COLUMNS
            from repositories.data_preprocessor import DataPreprocessor

            # Verificar se temos dados suficientes
            if len(df) < self.sequence_length:
                return None

            # Inicializar preprocessador se necessário
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor(
                    feature_columns=FEATURE_COLUMNS,
                    outlier_method='iqr',
                    scaling_method='robust'
                )
                self.preprocessor.fit(df)

            # Preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df, sequence_length=self.sequence_length
            )

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência: {e}", exc_info=True)
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

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
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
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.sequence_length = 24
        self.preprocessor = None
        self.preprocessor_fitted = False

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

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados com baixa volatilidade.
        Em mercados calmos, buscamos pequenos movimentos e breakouts após compressão.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # Em baixa volatilidade, buscamos:
        # 1. Compressão seguida de breakout (squeeze)
        # 2. Níveis técnicos importantes (suporte/resistência)
        # 3. Desvios significativos da média (mean reversion)

        # Verificar compressão de volatilidade (squeeze)
        squeeze_detected = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02

            # Bandas estão se estreitando?
            boll_widths = df['boll_width'].iloc[-5:].values
            narrowing = len(boll_widths) > 1 and boll_widths[0] > boll_widths[-1]

            # Squeeze = Bandas muito estreitas E se estreitando
            squeeze_detected = boll_width < (avg_width * 0.7) and narrowing

            if squeeze_detected:
                logger.info(f"Compressão de volatilidade detectada: BB width={boll_width:.4f} (avg={avg_width:.4f})")
            else:
                logger.info(
                    f"Sem compressão: BB width={boll_width:.4f} (avg={avg_width:.4f}), "
                    f"Estreitando={narrowing}"
                )

        # Verificar desvio da média móvel
        mean_deviation = False
        deviation_direction = None
        if 'ema_short' in df.columns:
            ema = df['ema_short'].iloc[-1]
            # Calcular desvio percentual
            deviation_pct = (current_price - ema) / ema * 100

            # Significativo se acima de 0.3% em baixa volatilidade
            if abs(deviation_pct) > 0.3:
                mean_deviation = True
                deviation_direction = "LONG" if deviation_pct < 0 else "SHORT"
                logger.info(f"Desvio significativo da média: {deviation_pct:.2f}% ({deviation_direction})")
            else:
                logger.debug(f"Desvio da média não significativo: {deviation_pct:.2f}%")

        # Verificar níveis técnicos importantes
        near_level = False
        level_direction = None

        # Testar níveis de pivot
        if 'pivot' in df.columns and 'pivot_s1' in df.columns and 'pivot_r1' in df.columns:
            pivot = df['pivot'].iloc[-1]
            r1 = df['pivot_r1'].iloc[-1]
            s1 = df['pivot_s1'].iloc[-1]

            # Calcular distâncias percentuais
            dist_to_pivot = abs(current_price - pivot) / current_price * 100
            dist_to_r1 = abs(current_price - r1) / current_price * 100
            dist_to_s1 = abs(current_price - s1) / current_price * 100

            if dist_to_pivot < 0.15:
                near_level = True
                # Determinar direção baseada na história recente
                if len(df) > 3 and df['close'].iloc[-3] < pivot:
                    level_direction = "SHORT"
                else:
                    level_direction = "LONG"
                logger.info(f"Preço próximo ao pivot: {dist_to_pivot:.2f}%")
            elif dist_to_r1 < 0.15:
                near_level = True
                level_direction = "SHORT"  # Prováveis reversões em R1
                logger.info(f"Preço próximo a R1: {dist_to_r1:.2f}%")
            elif dist_to_s1 < 0.15:
                near_level = True
                level_direction = "LONG"  # Prováveis reversões em S1
                logger.info(f"Preço próximo a S1: {dist_to_s1:.2f}%")
            else:
                logger.debug(
                    f"Preço longe de níveis importantes: "
                    f"Pivot={dist_to_pivot:.2f}%, R1={dist_to_r1:.2f}%, S1={dist_to_s1:.2f}%"
                )

        # Verificar condições de RSI para mean reversion
        rsi_signal = False
        rsi_direction = None
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Em baixa volatilidade, valores extremos de RSI são boas oportunidades
            if rsi < 30 and rsi > prev_rsi:  # Oversold e começando a subir
                rsi_signal = True
                rsi_direction = "LONG"
                logger.info(f"RSI em sobrevenda e subindo: {rsi:.1f} > {prev_rsi:.1f}")
            elif rsi > 70 and rsi < prev_rsi:  # Overbought e começando a cair
                rsi_signal = True
                rsi_direction = "SHORT"
                logger.info(f"RSI em sobrecompra e caindo: {rsi:.1f} < {prev_rsi:.1f}")
            else:
                logger.debug(f"RSI sem sinal claro: {rsi:.1f}")

        # Verificar alinhamento multi-timeframe
        mtf_aligned = False
        mtf_direction = None
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']

            if 'NEUTRAL' in mtf_trend:
                # Em baixa volatilidade, tendência neutra favorece mean reversion
                mtf_aligned = True
                # A direção depende de outros indicadores
                mtf_direction = deviation_direction or rsi_direction
                logger.info(f"Tendência MTF neutra favorável para mean reversion: {mtf_trend}")
            elif 'UPTREND' in mtf_trend:
                mtf_aligned = True
                mtf_direction = "LONG"
                logger.info(f"Tendência MTF favorável para LONG: {mtf_trend}")
            elif 'DOWNTREND' in mtf_trend:
                mtf_aligned = True
                mtf_direction = "SHORT"
                logger.info(f"Tendência MTF favorável para SHORT: {mtf_trend}")
            else:
                logger.debug(f"Tendência MTF não fornece direção clara: {mtf_trend}")

        # Condições para LONG
        long_conditions = 0
        if squeeze_detected and df['close'].iloc[-1] > df['open'].iloc[-1]:  # Squeeze + vela de alta
            long_conditions += 1
        if mean_deviation and deviation_direction == "LONG":
            long_conditions += 1
        if near_level and level_direction == "LONG":
            long_conditions += 1
        if rsi_signal and rsi_direction == "LONG":
            long_conditions += 1
        if mtf_aligned and mtf_direction == "LONG":
            long_conditions += 1

        # Condições para SHORT
        short_conditions = 0
        if squeeze_detected and df['close'].iloc[-1] < df['open'].iloc[-1]:  # Squeeze + vela de baixa
            short_conditions += 1
        if mean_deviation and deviation_direction == "SHORT":
            short_conditions += 1
        if near_level and level_direction == "SHORT":
            short_conditions += 1
        if rsi_signal and rsi_direction == "SHORT":
            short_conditions += 1
        if mtf_aligned and mtf_direction == "SHORT":
            short_conditions += 1

        # Log das condições
        logger.info(
            f"Condições BAIXA VOLATILIDADE - LONG: {long_conditions}/2, SHORT: {short_conditions}/2 - "
            f"Squeeze={squeeze_detected}, Desvio da média={mean_deviation} ({deviation_direction}), "
            f"Nível técnico={near_level} ({level_direction}), RSI={rsi_signal} ({rsi_direction}), "
            f"MTF={mtf_aligned} ({mtf_direction})"
        )

        # Determinar direção do sinal
        signal_direction = None

        # Prioridade 1: Squeeze seguido de breakout
        if squeeze_detected:
            if long_conditions >= 2:
                signal_direction = "LONG"
                logger.info(f"LONG baseado em squeeze com {long_conditions} condições favoráveis")
            elif short_conditions >= 2:
                signal_direction = "SHORT"
                logger.info(f"SHORT baseado em squeeze com {short_conditions} condições favoráveis")

        # Prioridade 2: Outras condições
        elif long_conditions >= 2 and long_conditions >= short_conditions:
            signal_direction = "LONG"
            logger.info(f"LONG baseado em {long_conditions} condições favoráveis em baixa volatilidade")
        elif short_conditions >= 2 and short_conditions > long_conditions:
            signal_direction = "SHORT"
            logger.info(f"SHORT baseado em {short_conditions} condições favoráveis em baixa volatilidade")
        else:
            logger.debug(f"Condições insuficientes para sinal: LONG={long_conditions}, SHORT={short_conditions}")
            return None

        # Se nenhuma direção for determinada, não gerar sinal
        if signal_direction is None:
            return None

        # Gerar previsões usando os modelos LSTM
        try:
            X_seq = self._prepare_sequence(df)
            if X_seq is None:
                return None

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

            # Ajustar direção do TP para SHORT
            if signal_direction == "SHORT" and predicted_tp_pct > 0:
                predicted_tp_pct = -predicted_tp_pct

            # Garantir valores positivos para SL
            predicted_sl_pct = abs(predicted_sl_pct)

            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Validar previsões - evitar valores absurdos
            if abs(predicted_tp_pct) > 20:
                predicted_tp_pct = 20.0 if signal_direction == "LONG" else -20.0

            if predicted_sl_pct > 10:
                predicted_sl_pct = 10.0

            # Em baixa volatilidade, limitar TP e SL
            # Obter ATR
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
            if atr_value:
                atr_pct = (atr_value / current_price) * 100

                # TP não deve ser mais do que 3x ATR em baixa volatilidade
                max_tp_pct = atr_pct * 3.0
                if abs(predicted_tp_pct) > max_tp_pct:
                    predicted_tp_pct = max_tp_pct if signal_direction == "LONG" else -max_tp_pct
                    logger.info(f"TP limitado para 3x ATR: {predicted_tp_pct:.2f}%")

                # SL não deve ser mais do que 1.5x ATR em baixa volatilidade
                max_sl_pct = atr_pct * 1.5
                if predicted_sl_pct > max_sl_pct:
                    predicted_sl_pct = max_sl_pct
                    logger.info(f"SL limitado para 1.5x ATR: {predicted_sl_pct:.2f}%")

            # Avaliar a qualidade da entrada
            should_enter, entry_score = self.evaluate_entry_quality(
                df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct
            )

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Configurar parâmetros para o sinal
            if signal_direction == "LONG":
                side = "BUY"
                position_side = "LONG"
                tp_factor = 1 + (predicted_tp_pct / 100)
                sl_factor = 1 - (predicted_sl_pct / 100)
            else:  # SHORT
                side = "SELL"
                position_side = "SHORT"
                tp_factor = 1 - (abs(predicted_tp_pct) / 100)
                sl_factor = 1 + (predicted_sl_pct / 100)

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Determinar tendência e força
            market_trend = "NEUTRAL"  # Em baixa volatilidade geralmente não há tendência forte
            market_strength = "LOW_VOLATILITY"

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
                rr_ratio=abs(predicted_tp_pct / predicted_sl_pct),
                market_trend=market_trend,
                market_strength=market_strength,
                timestamp=datetime.now()
            )

            return signal

        except Exception as e:
            logger.error(f"Erro na geração de sinal: {e}", exc_info=True)
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
        Avalia a qualidade da entrada em condições de baixa volatilidade.

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

        # Em baixa volatilidade, verificar proximidade a níveis técnicos
        levels_bonus = 0.0

        # Verificar proximidade a suportes/resistências
        if 'pivot' in df.columns and 'pivot_s1' in df.columns and 'pivot_r1' in df.columns:
            pivot = df['pivot'].iloc[-1]
            r1 = df['pivot_r1'].iloc[-1]
            s1 = df['pivot_s1'].iloc[-1]

            # Calcular distâncias percentuais
            dist_to_pivot = abs(current_price - pivot) / current_price * 100
            dist_to_r1 = abs(current_price - r1) / current_price * 100
            dist_to_s1 = abs(current_price - s1) / current_price * 100

            if dist_to_pivot < 0.15 or dist_to_r1 < 0.15 or dist_to_s1 < 0.15:
                levels_bonus = 0.2  # Bônus significativo para entradas próximas a níveis

        # Verificar compressão de volatilidade
        squeeze_bonus = 0.0
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02

            if boll_width < (avg_width * 0.7):
                squeeze_bonus = 0.15  # Bônus para volatilidade comprimida (potencial breakout)

        # Aplicar bônus ao score
        entry_score = min(1.0, entry_score + levels_bonus + squeeze_bonus)

        # Verificar desvio da média
        if 'ema_short' in df.columns:
            ema = df['ema_short'].iloc[-1]
            deviation_pct = (current_price - ema) / ema * 100

            # Verificar se o trade vai na direção do retorno à média
            if (trade_direction == "LONG" and deviation_pct < -0.3) or \
                    (trade_direction == "SHORT" and deviation_pct > 0.3):
                # Bônus para mean reversion em baixa volatilidade
                entry_score = min(1.0, entry_score * 1.2)

        # Em baixa volatilidade, ser mais cauteloso com o MTF alignment
        if mtf_alignment is not None:
            # Aplicar o MTF com menos peso (mercados de baixa vol tendem a ser mais isolados)
            entry_score = entry_score * (0.8 + mtf_alignment * 0.2)

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def _prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM.

        Args:
            df: DataFrame com dados históricos

        Returns:
            np.ndarray: Sequência formatada para o modelo LSTM ou None se houver erro
        """
        try:
            from core.constants import FEATURE_COLUMNS
            from repositories.data_preprocessor import DataPreprocessor

            # Verificar se temos dados suficientes
            if len(df) < self.sequence_length:
                return None

            # Inicializar preprocessador se necessário
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor(
                    feature_columns=FEATURE_COLUMNS,
                    outlier_method='iqr',
                    scaling_method='robust'
                )
                self.preprocessor.fit(df)

            # Preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df, sequence_length=self.sequence_length
            )

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência: {e}", exc_info=True)
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
