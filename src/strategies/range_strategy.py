# strategies/range_strategy.py

from typing import Optional

import pandas as pd

from core.logger import logger
from services.base.schemas import TradingSignal
from strategies.base import BaseStrategy, StrategyConfig


class RangeStrategy(BaseStrategy):
    """
    Estratégia para mercados em consolidação (range).
    Foca em capturar movimentos de reversão dos extremos do range.
    """

    def __init__(self):
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

    def generate_signal(self, df: pd.DataFrame, current_price: float, mtf_data: dict) -> TradingSignal | None:
        """
        Gera um sinal de trading para mercados em range.
        Busca oportunidades de compra no suporte e venda na resistência.
        """
        # Em mercados laterais, queremos comprar no suporte e vender na resistência

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
        if near_resistance:
            logger.info(f"Preço próximo da resistência (Bollinger Upper): {dist_to_upper:.2f}%")

        # Verificar condições adicionais usando RSI
        rsi = df['rsi'].iloc[-1]
        oversold = rsi < 30  # RSI em condição de sobrevenda
        overbought = rsi > 70  # RSI em condição de sobrecompra

        if oversold:
            logger.info(f"RSI em sobrevenda: {rsi:.1f}")
        if overbought:
            logger.info(f"RSI em sobrecompra: {rsi:.1f}")

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

        # Gerar sinais com base nas condições
        signal_direction = None

        if (near_support or oversold) and not (near_resistance or overbought):
            # Condições para LONG
            signal_direction = "LONG"
            logger.info(
                f"Condições para LONG em range: Suporte={near_support}, "
                f"Oversold={oversold}, Reversão={reversal_pattern}"
            )

        elif (near_resistance or overbought) and not (near_support or oversold):
            # Condições para SHORT
            signal_direction = "SHORT"
            logger.info(
                f"Condições para SHORT em range: Resistência={near_resistance}, "
                f"Overbought={overbought}, Reversão={reversal_pattern}"
            )

        # Retornar None para usar o gerador de sinais existente
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
