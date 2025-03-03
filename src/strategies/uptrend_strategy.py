# strategies/uptrend_strategy.py

from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from core.logger import logger
from models.lstm.model import LSTMModel
from services.base.schemas import TradingSignal
from services.prediction.interfaces import IPredictionService
from services.prediction.tpsl_prediction import TpSlPredictionService
from strategies.base.model import BaseStrategy, StrategyConfig


class UptrendStrategy(BaseStrategy):
    """
    Estratégia para mercados em tendência de alta.
    Foca em capturar continuações de tendência com entradas em pullbacks.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """Inicializa a estratégia com configuração otimizada para mercados em alta."""
        config = StrategyConfig(
            name="Uptrend Strategy",
            description="Estratégia otimizada para mercados em tendência de alta",
            min_rr_ratio=1.8,  # Exigir R:R maior em tendência de alta
            entry_threshold=0.55,  # Menos rigoroso na entrada por estar a favor da tendência
            tp_adjustment=1.2,  # Aumentar TP para capturar mais do movimento
            sl_adjustment=0.8,  # Stops mais apertados por estar a favor da tendência
            entry_aggressiveness=1.2,  # Mais agressivo nas entradas
            max_sl_percent=1.8,
            min_tp_percent=0.7,
            required_indicators=["ema_short", "ema_long", "adx", "rsi"]
        )
        super().__init__(config)
        self.prediction_service: IPredictionService = TpSlPredictionService(tp_model, sl_model)

    def should_activate(self, df: pd.DataFrame, mtf_data: dict) -> bool:
        """
        Determina se a estratégia deve ser ativada com base nas condições de mercado.
        """
        # Verificar se temos todos os indicadores necessários
        if not self.verify_indicators(df):
            return False

        # Verificar EMAs no timeframe atual
        ema_short = df['ema_short'].iloc[-1]
        ema_long = df['ema_long'].iloc[-1]
        ema_uptrend = ema_short > ema_long

        # Verificar tendência multi-timeframe
        mtf_uptrend = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_uptrend = 'UPTREND' in mtf_data['consolidated_trend']

        # Verificar força da tendência (ADX)
        adx_strong = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            adx_strong = adx_value > 25

        # Ativar se tivermos confirmação em pelo menos 2 dos 3 indicadores
        confirmations = sum([ema_uptrend, mtf_uptrend, adx_strong])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de UPTREND ativada: EMA={ema_uptrend}, MTF={mtf_uptrend}, "
                f"ADX={adx_strong} (valor={df['adx'].iloc[-1]:.1f})"
            )

        return should_activate

    async def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em alta.
        Busca oportunidades de compra em pullbacks (correções) da tendência.
        """
        # Na estratégia de tendência de alta, queremos entrar principalmente em LONGs
        # E queremos encontrar pontos de entrada em correções (pullbacks)

        # Verificar se estamos em um pullback usando RSI
        in_pullback = False
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Pullback = RSI caindo abaixo de 40 em tendência de alta
            if rsi < 40 and rsi < prev_rsi:
                in_pullback = True
                logger.info(f"Pullback detectado: RSI={rsi:.1f} (anterior: {prev_rsi:.1f})")
            # Adicionar log para entender por que não está detectando pullback
            else:
                logger.info(
                    f"Sem pullback: RSI={rsi:.1f} (anterior: {prev_rsi:.1f}), condições: {rsi < 40} e {rsi < prev_rsi}"
                )

        # Verificar suporte na média móvel
        near_support = False
        if 'ema_long' in df.columns:
            ema_long = df['ema_long'].iloc[-1]
            price_to_ema = abs(current_price - ema_long) / current_price * 100
            near_support = price_to_ema < 0.5  # Preço próximo da EMA longa

            if near_support:
                logger.info(f"Preço próximo do suporte em EMA longa: {price_to_ema:.2f}%")

        # Verificar Stochastic para confirmar oversold
        stoch_oversold = False
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            stoch_oversold = stoch_k < 20 and stoch_d < 20

            if stoch_oversold:
                logger.info(f"Stochastic em sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f}")
            # Adicionar log para entender valores do estocástico
            else:
                logger.info(f"Stochastic não está em sobrevenda: K={stoch_k:.1f}, D={stoch_d:.1f}")

        # Verificar bounce em suporte
        bounce_from_support = False
        if len(df) > 2 and 'low' in df.columns:
            current_low = df['low'].iloc[-1]
            prev_low = df['low'].iloc[-2]

            # Range das últimas 20 barras
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            # Verificar se o preço tocou no suporte e subiu
            if abs(current_low - prev_low) < (recent_range * 0.05) and current_price > current_low * 1.002:
                bounce_from_support = True
                logger.info(f"Bounce detectado em suporte: Preço atual={current_price}, Low={current_low}")
            # Adicionar log para entender por que não está detectando bounce
            else:
                logger.info(
                    f"Sem bounce: diff_lows={abs(current_low - prev_low):.2f}, "
                    f"threshold={recent_range * 0.05:.2f}, "
                    f"price_cond={current_price > current_low * 1.002}"
                )

        # Verificar tendência forte
        strong_trend = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            strong_trend = adx_value > 25
            if strong_trend:
                logger.info(f"Tendência forte detectada: ADX={adx_value:.1f}")

        # Verificar alinhamento multi-timeframe
        mtf_aligned = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_trend = mtf_data['consolidated_trend']
            mtf_aligned = 'UPTREND' in mtf_trend
            if mtf_aligned:
                logger.info(f"Alinhamento multi-timeframe favorável: {mtf_trend}")

        # Adicionar ao conjunto de condições a tendência forte e alinhamento MTF
        # Isso aumenta as chances de um sinal ser gerado
        conditions_met = sum(
            [in_pullback, near_support, stoch_oversold, bounce_from_support, strong_trend, mtf_aligned]
        )

        # Número mínimo de condições para geração de sinais
        min_conditions = 2

        logger.info(
            f"Condições para LONG: {conditions_met}/{min_conditions} atendidas - "
            f"Pullback={in_pullback}, EMA_Suporte={near_support}, "
            f"Stoch_Oversold={stoch_oversold}, Bounce={bounce_from_support}, "
            f"Strong_Trend={strong_trend}, MTF_Aligned={mtf_aligned}"
        )

        if conditions_met >= min_conditions:
            logger.info(
                f"Condições LONG em tendência de alta: Pullback={in_pullback}, "
                f"Suporte={near_support}, Stoch_Oversold={stoch_oversold}, "
                f"Bounce={bounce_from_support}, Strong_Trend={strong_trend}, MTF_Aligned={mtf_aligned}"
            )

            prediction = self.prediction_service.predict_tp_sl(
                df, current_price, "LONG"
            )  # Sempre LONG para uptrend
            if prediction is None:
                return None

            predicted_tp_pct, predicted_sl_pct = prediction

            # Vamos forçar operações LONG em tendência de alta
            side: Literal["SELL", "BUY"] = "BUY"
            position_side: Literal["LONG", "SHORT"] = "LONG"
            tp_factor = 1 + max(abs(predicted_tp_pct) / 100, 0.02)
            sl_factor = 1 - max(abs(predicted_sl_pct) / 100, 0.005)

            # Calcular preços TP/SL
            tp_price = current_price * tp_factor
            sl_price = current_price * sl_factor

            # Avaliar a qualidade da entrada
            should_enter, entry_score = self.evaluate_entry_quality(
                df, current_price, "LONG", predicted_tp_pct, predicted_sl_pct
            )

            if not should_enter:
                logger.info(f"Trade rejeitado pela avaliação de qualidade (score: {entry_score:.2f})")
                return None

            # Gerar ID único para o sinal
            signal_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000, 9999)}"

            # Obter ATR para ajustes de quantidade
            atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None

            # Determinar tendência e força
            market_trend = "UPTREND"  # Já sabemos que estamos em uptrend
            market_strength = "STRONG_TREND" if df['adx'].iloc[-1] > 25 else "WEAK_TREND"

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
                rr_ratio=abs(predicted_tp_pct / predicted_sl_pct),
                market_trend=market_trend,
                market_strength=market_strength,
                timestamp=datetime.now()
            )

            return signal

        return None

    # Método evaluate_entry_quality a ser adicionado à classe UptrendStrategy
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
        Avalia a qualidade da entrada para UptrendStrategy.

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

        # Verificações adicionais baseadas em indicadores
        if trade_direction == "LONG" and self.calculate_trend_direction(df) == "UPTREND":
            # Bônus para trades LONG em tendência de alta
            entry_score = min(1.0, entry_score * 1.2)
        elif trade_direction == "SHORT" and self.calculate_trend_direction(df) == "DOWNTREND":
            # Bônus para trades SHORT em tendência de baixa
            entry_score = min(1.0, entry_score * 1.2)

        # Verificar RSI se disponível
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            if trade_direction == "LONG" and rsi < 40:
                # Bônus para trades LONG quando RSI está baixo (sobrevenda)
                entry_score = min(1.0, entry_score * 1.1)
            elif trade_direction == "SHORT" and rsi > 60:
                # Bônus para trades SHORT quando RSI está alto (sobrecompra)
                entry_score = min(1.0, entry_score * 1.1)

        # Bônus para alinhamento multi-timeframe forte
        if mtf_alignment is not None and mtf_alignment > 0.5:
            entry_score = min(1.0, entry_score * (1.0 + (mtf_alignment - 0.5)))

        # Usar limiar de entrada da configuração se não for fornecido
        if entry_threshold is None:
            entry_threshold = self.config.entry_threshold

        # Decidir se deve entrar baseado na pontuação e no limiar
        should_enter = entry_score >= entry_threshold

        return should_enter, entry_score

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em alta.
        Em tendência de alta, favorecemos sinais LONG e ajustamos TP/SL.
        """
        if signal.direction != "LONG":
            # Aumentar o threshold para trades contra a tendência (SHORT em uptrend)
            logger.info(f"Sinal SHORT em tendência de alta: exigindo maior qualidade")
            # Reduzir a pontuação para dificultar a entrada
            if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                signal.entry_score = signal.entry_score * 0.8

        else:  # Para sinais LONG em tendência de alta
            # Ajustar TP para ser mais ambicioso
            signal.predicted_tp_pct = signal.predicted_tp_pct * self.config.tp_adjustment

            # Ajustar SL para ser mais apertado
            signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment

            # Recalcular preços e fatores
            if signal.direction == "LONG":
                signal.tp_factor = 1 + (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 - (signal.predicted_sl_pct / 100)
            else:  # SHORT
                signal.tp_factor = 1 - (signal.predicted_tp_pct / 100)
                signal.sl_factor = 1 + (signal.predicted_sl_pct / 100)

            signal.tp_price = signal.current_price * signal.tp_factor
            signal.sl_price = signal.current_price * signal.sl_factor

            # Atualizar a razão R:R
            signal.rr_ratio = abs(signal.predicted_tp_pct / signal.predicted_sl_pct)

            logger.info(
                f"Sinal LONG ajustado para mercado em alta: "
                f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
                f"R:R={signal.rr_ratio:.2f}"
            )

        return signal
