# services\trading_strategy.py

import math

from core.config import settings
from core.logger import logger
from services.risk_reward_manager import RiskRewardManager


class TradingStrategy:
    """
    Classe responsável pela lógica de decisão (LONG, SHORT ou neutro),
    bem como pelo cálculo de quantidade e ajuste de preços.
    """

    def __init__(self):
        """
        Inicializa a estratégia de trading com um gerenciador de risk/reward.
        """
        self.risk_reward_manager = RiskRewardManager(min_rr_ratio=1.5, atr_multiplier=1.5)

    def decide_direction(self, predicted_tp_pct: float, predicted_sl_pct: float,
                         threshold: float = 0.2) -> str | None:
        """
        Decide se vamos abrir uma posição LONG, SHORT ou permanecer neutro,
        com base nos valores previstos de TP e SL, considerando a relação R:R.

        Args:
            predicted_tp_pct: Previsão de variação percentual para TP
            predicted_sl_pct: Previsão de variação percentual para SL
            threshold: Limiar para decidir se é LONG/SHORT

        Returns:
            "LONG", "SHORT" ou None
        """
        # Verificar se os valores são válidos
        if not isinstance(predicted_tp_pct, (int, float)) or not isinstance(predicted_sl_pct, (int, float)):
            logger.warning(f"Valores de previsão inválidos: TP={predicted_tp_pct}, SL={predicted_sl_pct}")
            return None

        # Assegurar que SL é positivo
        predicted_sl_pct = abs(predicted_sl_pct)

        # Calcular a razão RR para esta previsão
        if predicted_sl_pct <= 0.1:  # Evitar divisão por zero ou SL muito pequeno
            logger.warning(f"SL previsto muito pequeno ou inválido: {predicted_sl_pct}")
            return None

        rr_ratio = abs(predicted_tp_pct / predicted_sl_pct)

        # Verificar se a razão RR é boa o suficiente
        if rr_ratio < self.risk_reward_manager.min_rr_ratio:
            logger.info(f"Razão R:R insuficiente: {rr_ratio:.2f} < {self.risk_reward_manager.min_rr_ratio}")
            return None

        # Decisão de direção baseada no TP previsto
        if predicted_tp_pct > threshold:
            logger.info(
                f"Sinal LONG gerado: TP={predicted_tp_pct:.2f}%, SL={predicted_sl_pct:.2f}%, R:R={rr_ratio:.2f}")
            return "LONG"
        elif predicted_tp_pct < -threshold:
            logger.info(
                f"Sinal SHORT gerado: TP={predicted_tp_pct:.2f}%, SL={predicted_sl_pct:.2f}%, R:R={rr_ratio:.2f}")
            return "SHORT"
        else:
            logger.info(f"Sinal neutro: TP={predicted_tp_pct:.2f}% dentro do threshold ({threshold})")
            return None

    def calculate_trade_quantity(
            self,
            capital: float,
            current_price: float,
            leverage: float,
            risk_per_trade: float,
            atr_value: float = None,
            min_notional: float = 100.0
    ) -> float:
        """
        Calcula a quantidade a ser negociada com ajuste de volatilidade.
        """
        risk_amount = capital * risk_per_trade
        original_risk = risk_amount

        # Ajuste baseado em ATR
        if atr_value is not None:
            atr_percentage = atr_value / current_price * 100

            if atr_percentage > settings.VOLATILITY_HIGH_THRESHOLD:  # Alta volatilidade
                volatility_factor = 0.7
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR alto ({atr_percentage:.2f}%) - "
                    f"Reduzindo risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            elif atr_percentage < settings.VOLATILITY_LOW_THRESHOLD:  # Baixa volatilidade
                volatility_factor = 1.3
                risk_amount *= volatility_factor
                logger.info(
                    f"ATR baixo ({atr_percentage:.2f}%) - "
                    f"Aumentando risco de {original_risk:.2f} para {risk_amount:.2f} "
                    f"({volatility_factor * 100:.0f}% do normal)"
                )
            else:
                logger.info(f"ATR normal ({atr_percentage:.2f}%) - Mantendo risco padrão")

        # Calcular quantidade básica
        base_quantity = (risk_amount / current_price) * leverage

        # Verificar se excede o tamanho máximo permitido
        max_quantity = (capital * settings.MAX_POSITION_SIZE_PCT) / current_price * leverage

        # Usar o menor valor entre a quantidade calculada e o máximo permitido
        quantity = min(base_quantity, max_quantity)

        if quantity < base_quantity:
            logger.info(f"Quantidade ajustada para limite máximo: {quantity:.4f} (era {base_quantity:.4f})")

            # Verificar se atende ao valor mínimo notional da Binance
            notional_value = quantity * current_price
            if notional_value < min_notional:
                # Ajustar para o mínimo requerido com margem de segurança
                min_quantity = (min_notional * 1.05) / current_price
                logger.warning(
                    f"Quantidade calculada ({quantity:.4f} BTC, ${notional_value:.2f}) abaixo do valor mínimo da Binance. "
                    f"Ajustando para {min_quantity:.4f} BTC (${min_quantity * current_price:.2f})"
                )
                quantity = min_quantity

        return quantity

    def adjust_price_to_tick_size(self, price: float, tick_size: float) -> float:
        """
        Arredonda 'price' para baixo (floor) ao múltiplo de tick_size.

        :param price: Preço original
        :param tick_size: Valor de tick size
        :return: Preço arredondado
        """
        return math.floor(price / tick_size) * tick_size

    def format_price_for_tick_size(self, price: float, tick_size: float) -> str:
        """
        Formata 'price' com a quantidade correta de casas decimais
        baseada no tick_size.

        :param price: Valor do preço
        :param tick_size: Tick size do símbolo
        :return: Preço formatado em string
        """
        decimals = 0
        if '.' in str(tick_size):
            decimals = len(str(tick_size).split('.')[-1])
        return f"{price:.{decimals}f}"

    def adjust_quantity_to_step_size(self, qty: float, step_size: float) -> float:
        """
        Arredonda 'qty' para o múltiplo do step_size.

        :param qty: Quantidade original
        :param step_size: Step size do símbolo
        :return: Quantidade arredondada
        """
        return math.floor(qty / step_size) * step_size

    def evaluate_entry_quality(
            self,
            df,
            current_price: float,
            trade_direction: str,
            predicted_tp_pct: float = None,
            predicted_sl_pct: float = None,
            entry_threshold: float = settings.ENTRY_THRESHOLD_DEFAULT
    ) -> tuple[bool, float]:
        """
        Avalia a qualidade da entrada potencial usando múltiplos critérios.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual do ativo (usado para cálculos relativos)
            trade_direction: "LONG" ou "SHORT"
            predicted_tp_pct: Take profit percentual previsto (opcional)
            predicted_sl_pct: Stop loss percentual previsto (opcional)
            entry_threshold: Pontuação mínima para considerar a entrada

        Returns:
            tuple[bool, float]: (Deve entrar, pontuação da entrada)
        """
        # Obter informações de tendência
        from services.trend_analyzer import TrendAnalyzer
        trend = TrendAnalyzer.ema_trend(df)

        # Obter valor de ADX para medir força da tendência
        adx_value = df['adx'].iloc[-1] if 'adx' in df.columns else 25

        # Calcular alinhamento da tendência com a direção do trade (0-1)
        trend_alignment = 0.5  # neutro por padrão
        if trend == "UPTREND" and trade_direction == "LONG":
            trend_alignment = 0.9
        elif trend == "DOWNTREND" and trade_direction == "SHORT":
            trend_alignment = 0.9
        elif trend == "UPTREND" and trade_direction == "SHORT":
            trend_alignment = 0.1
        elif trend == "DOWNTREND" and trade_direction == "LONG":
            trend_alignment = 0.1

        # Se ADX for forte, aumentar o peso do alinhamento com a tendência
        if adx_value > 25:  # ADX > 25 indica tendência forte
            trend_alignment = trend_alignment * (1 + (adx_value - 25) / 50)
            trend_alignment = min(1.0, trend_alignment)  # Limitar a 1.0

        # Utilizar current_price para cálculos relativos à volatilidade
        # Por exemplo, para determinar se o preço atual está próximo a suportes/resistências
        atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
        volatility_factor = 1.0

        if atr_value:
            # Calcular volatilidade relativa (ATR em % do preço)
            relative_volatility = (atr_value / current_price) * 100
            # Ajustar fator de volatilidade
            if relative_volatility > 1.5:  # Alta volatilidade
                volatility_factor = 0.8  # Ser mais cauteloso
            elif relative_volatility < 0.5:  # Baixa volatilidade
                volatility_factor = 1.2  # Ser mais agressivo

        # Se ambos os valores previstos TP e SL foram fornecidos, use-os para avaliação
        if predicted_tp_pct is not None and predicted_sl_pct is not None and predicted_sl_pct > 0:
            # Usar o RiskRewardManager para avaliar a qualidade do trade com os valores previstos
            quality_score = self.risk_reward_manager.evaluate_trade_quality(
                tp_pct=abs(predicted_tp_pct),
                sl_pct=abs(predicted_sl_pct),
                trend_strength=trend_alignment
            ) * volatility_factor  # Aplicar fator de volatilidade
        else:
            # Caso contrário, usar o EntryScorer para avaliação baseada apenas em indicadores técnicos
            from services.entry_scorer import EntryScorer
            quality_score = EntryScorer.calculate_entry_score(
                df=df,
                current_price=current_price,
                trade_direction=trade_direction,
                trend_direction=trend
            ) * volatility_factor  # Aplicar fator de volatilidade

        quality_score = min(quality_score, 1.0)

        # Decidir se deve entrar no trade
        should_enter = quality_score >= entry_threshold

        # Logar avaliação detalhada
        if predicted_tp_pct is not None and predicted_sl_pct is not None:
            logger.info(
                f"Avaliação de Entrada: Direção={trade_direction}, Tendência={trend}, "
                f"ADX={adx_value:.1f}, Alinhamento={trend_alignment:.2f}, "
                f"TP={predicted_tp_pct:.2f}%, SL={predicted_sl_pct:.2f}%, "
                f"R:R={(predicted_tp_pct / predicted_sl_pct if predicted_sl_pct > 0 else 0):.2f}, "
                f"Volatilidade={relative_volatility:.2f}%, Fator={volatility_factor:.1f}, "
                f"Score={quality_score:.2f}, Threshold={entry_threshold:.2f}, "
                f"Decisão={'Entrar' if should_enter else 'Ignorar'}"
            )
        else:
            logger.info(
                f"Avaliação de Entrada: Direção={trade_direction}, Tendência={trend}, "
                f"ADX={adx_value:.1f}, Alinhamento={trend_alignment:.2f}, "
                f"Volatilidade Relativa={relative_volatility:.2f}%, "
                f"Score={quality_score:.2f}, Threshold={entry_threshold:.2f}, "
                f"Decisão={'Entrar' if should_enter else 'Ignorar'}"
            )

        return should_enter, quality_score
