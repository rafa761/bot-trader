# strategies/range_strategy.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.schemas import TradingSignal
from strategies.base.model import BaseStrategy, StrategyConfig


class RangeStrategy(BaseStrategy):
    """
    Estratégia para mercados em consolidação (range).
    Foca em capturar movimentos de reversão dos extremos do range.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """Inicializa a estratégia com configuração otimizada para mercados em range."""
        config = StrategyConfig(
            name="Range Strategy",
            description="Estratégia otimizada para mercados laterais (em range)",
            min_rr_ratio=1.3,  # R:R menor em mercado de range
            entry_threshold=0.7,  # Mais rigoroso na entrada em range
            tp_adjustment=0.8,  # TP menor pois os movimentos são limitados
            sl_adjustment=0.9,  # SL ligeiramente mais apertado
            entry_aggressiveness=0.8,  # Menos agressivo nas entradas
            max_sl_percent=1.2,
            min_tp_percent=0.4,
            required_indicators=["adx", "boll_width", "rsi", "boll_lband", "boll_hband"]
        )
        super().__init__(config)
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.sequence_length = 24
        self.preprocessor = None
        self.preprocessor_fitted = False

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar ADX para identificar ausência de tendência
        adx_low = False
        adx_value = df['adx'].iloc[-1]
        adx_low = adx_value < 20  # ADX baixo indica ausência de tendência

        # Verificar EMAs próximas (indicando ausência de tendência)
        emas_flat = False
        if 'ema_short' in df.columns and 'ema_long' in df.columns:
            ema_short = df['ema_short'].iloc[-1]
            ema_long = df['ema_long'].iloc[-1]
            ema_diff_pct = abs(ema_short - ema_long) / ema_long * 100
            emas_flat = ema_diff_pct < 0.3  # EMAs muito próximas

        # Verificar tendência multi-timeframe
        mtf_neutral = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_neutral = 'NEUTRAL' in mtf_data['consolidated_trend']

        # Verificar largura do Bollinger Band (estreita em consolidação)
        bb_narrow = False
        bb_width = df['boll_width'].iloc[-1]
        bb_narrow = bb_width < 0.03  # Bandas estreitas

        # Verificar volatilidade baixa usando ATR
        low_volatility = False
        if 'atr_pct' in df.columns:
            atr_pct = df['atr_pct'].iloc[-1]
            low_volatility = atr_pct < 0.5
        elif 'atr' in df.columns and 'close' in df.columns:
            atr = df['atr'].iloc[-1]
            close = df['close'].iloc[-1]
            atr_pct = (atr / close) * 100
            low_volatility = atr_pct < 0.5

        # Ativar se pelo menos dois indicadores confirmarem
        confirmations = sum([adx_low, emas_flat, mtf_neutral, bb_narrow, low_volatility])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de RANGE ativada: ADX_baixo={adx_low} ({adx_value:.1f}), "
                f"EMAs_flat={emas_flat}, MTF_neutral={mtf_neutral}, "
                f"BB_estreito={bb_narrow} ({bb_width:.4f}), Volatilidade_baixa={low_volatility}"
            )

        return should_activate

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em range.
        Busca oportunidades de compra no suporte e venda na resistência.
        """
        # Em mercados laterais, queremos comprar no suporte e vender na resistência

        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return None

        # Verificar se estamos próximos do suporte ou resistência nas Bollinger Bands
        lower_band = df['boll_lband'].iloc[-1]
        upper_band = df['boll_hband'].iloc[-1]

        # Calcular proximidade com as bandas
        dist_to_lower = (current_price - lower_band) / current_price * 100
        dist_to_upper = (upper_band - current_price) / current_price * 100

        near_support = dist_to_lower < 0.3  # Preço próximo da banda inferior
        near_resistance = dist_to_upper < 0.3  # Preço próximo da banda superior

        if near_support:
            logger.info(f"Preço próximo do suporte (Bollinger Lower): {dist_to_lower:.2f}%")
        else:
            logger.info(f"Preço não está próximo do suporte: {dist_to_lower:.2f}%")

        if near_resistance:
            logger.info(f"Preço próximo da resistência (Bollinger Upper): {dist_to_upper:.2f}%")
        else:
            logger.info(f"Preço não está próximo da resistência: {dist_to_upper:.2f}%")

        # Verificar condições adicionais usando RSI
        rsi = df['rsi'].iloc[-1]
        oversold = rsi < 30  # RSI em condição de sobrevenda
        overbought = rsi > 70  # RSI em condição de sobrecompra

        if oversold:
            logger.info(f"RSI em sobrevenda: {rsi:.1f}")
        else:
            logger.info(f"RSI não está em sobrevenda: {rsi:.1f}")

        if overbought:
            logger.info(f"RSI em sobrecompra: {rsi:.1f}")
        else:
            logger.info(f"RSI não está em sobrecompra: {rsi:.1f}")

        # Verificar padrões de reversão em velas japonesas
        reversal_pattern = False
        if len(df) >= 3 and 'open' in df.columns and 'close' in df.columns:
            # Verificar padrão de reversão simples (vela longa + vela de reversão)
            c0, o0 = df['close'].iloc[-1], df['open'].iloc[-1]
            c1, o1 = df['close'].iloc[-2], df['open'].iloc[-2]
            c2, o2 = df['close'].iloc[-3], df['open'].iloc[-3]

            # Exemplo: após duas velas de baixa, vela de alta forte
            if c2 < o2 and c1 < o1 and c0 > o0 and c0 > c1 * 1.005:
                reversal_pattern = True
                logger.info("Padrão de reversão de baixa para alta detectado")

            # Exemplo: após duas velas de alta, vela de baixa forte
            elif c2 > o2 and c1 > o1 and c0 < o0 and c0 < c1 * 0.995:
                reversal_pattern = True
                logger.info("Padrão de reversão de alta para baixa detectado")
            else:
                logger.info(f"Sem padrão de reversão: c0={c0:.2f}, o0={o0:.2f}, "
                            f"c1={c1:.2f}, o1={o1:.2f}, c2={c2:.2f}, o2={o2:.2f}"
                            )

        # Verificar compressão de volatilidade (squeeze)
        volatility_compression = False
        if 'boll_width' in df.columns:
            boll_width = df['boll_width'].iloc[-1]
            avg_width = df['boll_width'].rolling(20).mean().iloc[-1] if len(df) >= 20 else 0.02
            volatility_compression = boll_width < (avg_width * 0.8)

            if volatility_compression:
                logger.info(f"Compressão de volatilidade detectada: {boll_width:.4f} vs {avg_width:.4f}")
            else:
                logger.info(f"Sem compressão de volatilidade: {boll_width:.4f} vs {avg_width:.4f}")

        # Verificar alinhamento multi-timeframe (neutro é favorável em range)
        mtf_neutral = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']
            mtf_neutral = 'NEUTRAL' in mtf_trend
            if mtf_neutral:
                logger.info(f"Tendência MTF neutra favorável para range: {mtf_trend}")
            else:
                logger.info(f"Tendência MTF não é neutra: {mtf_trend}")

        # Determinar direção do sinal
        signal_direction = None

        conditions_long = 0
        conditions_short = 0

        # Condições para LONG
        if near_support or oversold:
            conditions_long += 1
        if reversal_pattern and c0 > o0:  # Vela de alta
            conditions_long += 1
        if volatility_compression and df['close'].iloc[-1] > df['open'].iloc[-1]:
            conditions_long += 1
        if mtf_neutral:
            conditions_long += 0.5  # Condição parcial

        # Condições para SHORT
        if near_resistance or overbought:
            conditions_short += 1
        if reversal_pattern and c0 < o0:  # Vela de baixa
            conditions_short += 1
        if volatility_compression and df['close'].iloc[-1] < df['open'].iloc[-1]:
            conditions_short += 1
        if mtf_neutral:
            conditions_short += 0.5  # Condição parcial

        logger.info(
            f"Condições RANGE - LONG: {conditions_long}/2, SHORT: {conditions_short}/2 - "
            f"Suporte={near_support}, Resistência={near_resistance}, "
            f"Oversold={oversold}, Overbought={overbought}, "
            f"Reversão={reversal_pattern}, Compressão={volatility_compression}, "
            f"MTF_Neutro={mtf_neutral}"
        )

        # Selecionar a direção com mais condições favoráveis
        if conditions_long >= 2 and conditions_long > conditions_short:
            signal_direction: Literal["LONG", "SHORT"] = "LONG"
            logger.info(
                f"Condições para LONG em range: Suporte={near_support}, "
                f"Oversold={oversold}, Reversão={reversal_pattern}, "
                f"Compressão={volatility_compression}, MTF_Neutro={mtf_neutral}"
            )
        elif conditions_short >= 2 and conditions_short > conditions_long:
            signal_direction: Literal["LONG", "SHORT"] = "SHORT"
            logger.info(
                f"Condições para SHORT em range: Resistência={near_resistance}, "
                f"Overbought={overbought}, Reversão={reversal_pattern}, "
                f"Compressão={volatility_compression}, MTF_Neutro={mtf_neutral}"
            )
        else:
            # Sem condições suficientes para gerar sinal
            logger.info("Condições insuficientes para gerar sinal em mercado em range")
            return None

        # Gerar previsões usando os modelos LSTM
        try:
            X_seq = self._prepare_sequence(df)
            if X_seq is None:
                return None

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

            # Garantir direção correta para TP em SHORT
            if signal_direction == "SHORT" and predicted_tp_pct > 0:
                predicted_tp_pct = -predicted_tp_pct

            # Garantir valores positivos para SL
            predicted_sl_pct = abs(predicted_sl_pct)

            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Validar previsões
            if abs(predicted_tp_pct) > 20:
                predicted_tp_pct = 20.0 if predicted_tp_pct > 0 else -20.0

            if predicted_sl_pct > 10:
                predicted_sl_pct = 10.0

            # Ajustar SL dinamicamente se for muito pequeno
            if predicted_sl_pct < 0.5:
                # Calcular o SL dinâmico baseado em ATR
                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
                if atr_value:
                    predicted_sl_pct = (atr_value / current_price) * 100 * 1.5

            # Avaliar a qualidade da entrada
            should_enter, entry_score = self.evaluate_entry_quality(
                df, current_price, signal_direction, predicted_tp_pct, predicted_sl_pct
            )

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

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

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Obter ATR para ajustes de quantidade
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Determinar tendência e força
            market_trend = "NEUTRAL"  # Em range, a tendência é neutra
            market_strength = "WEAK_TREND" if df['adx'].iloc[-1] < 20 else "MODERATE_TREND"

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
        Avalia a qualidade da entrada para RangeStrategy.

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

            # Pontuação básica baseada em RR
            entry_score = min(1.0, rr_ratio / 3.0)  # Pontuação de 0 a 1
        else:
            # Valores padrão se tp e sl não forem fornecidos
            should_enter = False
            entry_score = 0.0

        # Verificar proximidade às bandas de Bollinger
        if 'boll_lband' in df.columns and 'boll_hband' in df.columns:
            lower_band = df['boll_lband'].iloc[-1]
            upper_band = df['boll_hband'].iloc[-1]

            # Calcular proximidade com as bandas
            dist_to_lower = (current_price - lower_band) / current_price * 100
            dist_to_upper = (upper_band - current_price) / current_price * 100

            if trade_direction == "LONG" and dist_to_lower < 0.3:
                # Bônus para LONG perto do suporte
                entry_score = min(1.0, entry_score * 1.3)
            elif trade_direction == "SHORT" and dist_to_upper < 0.3:
                # Bônus para SHORT perto da resistência
                entry_score = min(1.0, entry_score * 1.3)

        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if trade_direction == "LONG" and rsi < 30:
                # Bônus para trades LONG quando RSI está baixo (sobrevenda)
                entry_score = min(1.0, entry_score * 1.2)
            elif trade_direction == "SHORT" and rsi > 70:
                # Bônus para trades SHORT quando RSI está alto (sobrecompra)
                entry_score = min(1.0, entry_score * 1.2)

        # Em range, o alinhamento MTF pode ser menos importante
        if mtf_alignment is not None:
            # Apenas pequeno bônus para alinhamento forte
            entry_score = min(1.0, entry_score * (1.0 + (mtf_alignment - 0.5) * 0.5))

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
        Ajusta um sinal para mercados em range.
        Reduz TP e ajusta SL para otimizar operações em range.
        """
        # Estimar os limites do range para definir alvos
        range_high = 0
        range_low = 0

        # Usando Bollinger Bands para estimar o range
        range_high = df['boll_hband'].iloc[-1]
        range_low = df['boll_lband'].iloc[-1]

        # Ajustar TP para não ultrapassar os limites do range
        if signal.direction == "LONG":
            # Para LONG, TP não deve exceder o topo do range
            max_tp_price = range_high
            if signal.tp_price > max_tp_price:
                new_tp_factor = max_tp_price / signal.current_price
                new_tp_pct = (new_tp_factor - 1) * 100

                # Atualizar sinal
                signal.predicted_tp_pct = new_tp_pct
                signal.tp_factor = new_tp_factor
                signal.tp_price = max_tp_price

                logger.info(
                    f"TP ajustado para respeitar o topo do range: {new_tp_pct:.2f}% "
                    f"(preço: {max_tp_price:.2f})"
                )
        else:  # SHORT
            # Para SHORT, TP não deve exceder o fundo do range
            min_tp_price = range_low
            if signal.tp_price < min_tp_price:
                new_tp_factor = min_tp_price / signal.current_price
                new_tp_pct = (1 - new_tp_factor) * 100

                # Atualizar sinal
                signal.predicted_tp_pct = new_tp_pct
                signal.tp_factor = new_tp_factor
                signal.tp_price = min_tp_price

                logger.info(
                    f"TP ajustado para respeitar o fundo do range: {new_tp_pct:.2f}% "
                    f"(preço: {min_tp_price:.2f})"
                )

        # Aplicar ajustes gerais da estratégia
        signal.predicted_tp_pct = signal.predicted_tp_pct * self.config.tp_adjustment
        signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment

        # Recalcular preços com base nos percentuais ajustados
        if signal.direction == "LONG":
            signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
            signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
        else:  # SHORT
            signal.tp_factor = 1 - (abs(signal.predicted_tp_pct) / 100)
            signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

        signal.tp_price = signal.current_price * signal.tp_factor
        signal.sl_price = signal.current_price * signal.sl_factor

        # Atualizar a razão R:R
        signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

        logger.info(
            f"Sinal ajustado para mercado em range: "
            f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
            f"R:R={signal.rr_ratio:.2f}"
        )

        return signal
