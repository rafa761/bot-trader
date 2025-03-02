# strategies/uptrend_strategy.py

import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from strategies.base.model import BaseStrategy, StrategyConfig


class UptrendStrategy(BaseStrategy):
    """
    Estratégia para mercados em tendência de alta.
    Foca em capturar continuações de tendência com entradas em pullbacks.
    """

    def __init__(self):
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

    def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
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

        # Decidir se geramos sinal
        conditions_met = sum([in_pullback, near_support, stoch_oversold, bounce_from_support])

        if conditions_met >= 2:
            logger.info(
                f"Condições LONG em tendência de alta: Pullback={in_pullback}, "
                f"Suporte={near_support}, Stoch_Oversold={stoch_oversold}, "
                f"Bounce={bounce_from_support}"
            )

        # Retornar None para usar o gerador de sinais existente
        # As condições detectadas servem apenas para logging e diagnóstico
        # O gerador real de sinais continuará sendo o LSTM ou outro componente existente
        return None

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
