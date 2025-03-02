# strategies\downtrend_strategy.py

from typing import Optional

import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from strategies.base import BaseStrategy, StrategyConfig


class DowntrendStrategy(BaseStrategy):
    """
    Estratégia para mercados em tendência de baixa.
    Foca em capturar continuações da tendência de baixa com entradas em rallies.
    """

    def __init__(self):
        """Inicializa a estratégia com configuração otimizada para mercados em baixa."""
        config = StrategyConfig(
            name="Downtrend Strategy",
            description="Estratégia otimizada para mercados em tendência de baixa",
            min_rr_ratio=1.8,  # Exigir R:R maior em tendência de baixa
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
        ema_downtrend = ema_short < ema_long

        # Verificar tendência multi-timeframe
        mtf_downtrend = False
        if mtf_data and 'consolidated_trend' in mtf_data:
            mtf_downtrend = 'DOWNTREND' in mtf_data['consolidated_trend']

        # Verificar força da tendência (ADX)
        adx_strong = False
        if 'adx' in df.columns:
            adx_value = df['adx'].iloc[-1]
            adx_strong = adx_value > 25

        # Ativar se tivermos confirmação em pelo menos 2 dos 3 indicadores
        confirmations = sum([ema_downtrend, mtf_downtrend, adx_strong])
        should_activate = confirmations >= 2

        if should_activate:
            logger.info(
                f"Estratégia de DOWNTREND ativada: EMA={ema_downtrend}, MTF={mtf_downtrend}, "
                f"ADX={adx_strong} (valor={df['adx'].iloc[-1]:.1f})"
            )

        return should_activate

    def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em baixa.
        Busca oportunidades de venda em rallies (correções de alta) da tendência de baixa.
        """
        # Na estratégia de tendência de baixa, queremos entrar principalmente em SHORTs
        # E queremos encontrar pontos de entrada em correções para cima (rallies)

        # Verificar se estamos em um rally usando RSI
        in_rally = False
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            prev_rsi = df['rsi'].iloc[-2] if len(df) > 2 else 50

            # Rally = RSI subindo acima de 60 em tendência de baixa
            if rsi > 60 and rsi > prev_rsi:
                in_rally = True
                logger.info(f"Rally detectado: RSI={rsi:.1f} (anterior: {prev_rsi:.1f})")

        # Verificar resistência na média móvel
        near_resistance = False
        if 'ema_long' in df.columns:
            ema_long = df['ema_long'].iloc[-1]
            price_to_ema = abs(current_price - ema_long) / current_price * 100
            near_resistance = price_to_ema < 0.5  # Preço próximo da EMA longa

            if near_resistance:
                logger.info(f"Preço próximo da resistência em EMA longa: {price_to_ema:.2f}%")

        # Verificar Stochastic para confirmar overbought
        stoch_overbought = False
        if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
            stoch_k = df['stoch_k'].iloc[-1]
            stoch_d = df['stoch_d'].iloc[-1]
            stoch_overbought = stoch_k > 80 and stoch_d > 80

            if stoch_overbought:
                logger.info(f"Stochastic em sobrecompra: K={stoch_k:.1f}, D={stoch_d:.1f}")

        # Verificar rejeição em resistência
        rejection_at_resistance = False
        if len(df) > 2 and 'high' in df.columns and 'close' in df.columns:
            current_high = df['high'].iloc[-1]
            current_close = df['close'].iloc[-1]

            # Range das últimas 20 barras
            recent_range = df['high'].iloc[-20:].max() - df['low'].iloc[-20:].min()

            # Verificar se o preço tocou na resistência e foi rejeitado
            if current_high > current_close * 1.002 and (current_high - current_close) > (recent_range * 0.1):
                rejection_at_resistance = True
                logger.info(
                    f"Rejeição detectada em resistência: "
                    f"High={current_high}, Close={current_close}, Diferença={current_high - current_close:.2f}"
                )

        # Decidir se geramos sinal
        conditions_met = sum([in_rally, near_resistance, stoch_overbought, rejection_at_resistance])

        if conditions_met >= 2:
            logger.info(
                f"Condições SHORT em tendência de baixa: Rally={in_rally}, "
                f"Resistência={near_resistance}, Stoch_Overbought={stoch_overbought}, "
                f"Rejeição={rejection_at_resistance}"
            )

        # Retornar None para usar o gerador de sinais existente
        # As condições detectadas servem apenas para logging e diagnóstico
        return None

    def adjust_signal(self, signal: TradingSignal, df: pd.DataFrame, mtf_data: dict) -> TradingSignal:
        """
        Ajusta um sinal para mercados em baixa.
        Em tendência de baixa, favorecemos sinais SHORT e ajustamos TP/SL.
        """
        if signal.direction != "SHORT":
            # Aumentar o threshold para trades contra a tendência (LONG em downtrend)
            logger.info(f"Sinal LONG em tendência de baixa: exigindo maior qualidade")
            # Reduzir a pontuação para dificultar a entrada
            if hasattr(signal, 'entry_score') and signal.entry_score is not None:
                signal.entry_score = signal.entry_score * 0.8

        else:  # Para sinais SHORT em tendência de baixa
            # Ajustar TP para ser mais ambicioso
            signal.predicted_tp_pct = abs(signal.predicted_tp_pct) * self.config.tp_adjustment
            if signal.predicted_tp_pct > 0:  # Garantir que é negativo para SHORT
                signal.predicted_tp_pct = -signal.predicted_tp_pct

            # Ajustar SL para ser mais apertado
            signal.predicted_sl_pct = signal.predicted_sl_pct * self.config.sl_adjustment

            # Recalcular preços e fatores
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
                f"Sinal SHORT ajustado para mercado em baixa: "
                f"TP={signal.predicted_tp_pct:.2f}%, SL={signal.predicted_sl_pct:.2f}%, "
                f"R:R={signal.rr_ratio:.2f}"
            )

        return signal
