# trading_bot.py

"""
Este módulo coordena o loop principal de trading, integrando o DataHandler,
ModelManager, TradingStrategy e BinanceClient.
Lida com a criação de ordens, registro de trades e todo o fluxo assíncrono.
"""

import asyncio
import sys

import pandas as pd

from binance_client import BinanceClient
from config import config
from data_handler import DataHandler
from logger import logger
from model_manager import ModelManager
from trading_strategy import TradingStrategy


class TradingBot:
    """
    Classe principal do bot de trading, que faz a orquestração entre
    DataHandler, ModelManager, TradingStrategy e BinanceClient.
    """

    def __init__(self):
        """
        Construtor que inicializa as classes auxiliares e
        configura variáveis necessárias.
        """
        self.binance_client = BinanceClient()
        self.data_handler = DataHandler(self.binance_client)
        self.model_manager = ModelManager()
        self.strategy = TradingStrategy()

        # Parâmetros de configuração
        self.symbol = config.SYMBOL
        self.interval = config.INTERVAL
        self.capital = config.CAPITAL
        self.leverage = config.LEVERAGE
        self.risk_per_trade = config.RISK_PER_TRADE

        # Filtros (tick_size, step_size) para formatar ordens
        self.tick_size, self.step_size = 0.0, 0.0

        # Eventual controle de retreinamento
        self.model_initialized = True  # Assume que os modelos já existem
        self.training_interval = 50
        self.cycle_count = 0

        self.last_position = None

        logger.info("classe TradingBot inicializada com sucesso.")

    async def initialize_filters(self) -> None:
        """
        Inicializa os filtros (tickSize e stepSize) e tenta ajustar alavancagem
        no par de trading configurado.
        """
        self.tick_size, self.step_size = self.binance_client.get_symbol_filters(self.symbol)

        if not self.tick_size or not self.step_size:
            logger.error("Não foi possível obter tickSize/stepSize. Encerrando.")
            sys.exit(1)

        logger.info(f"{self.symbol} -> tickSize={self.tick_size}, stepSize={self.step_size}")

        # Define alavancagem
        self.binance_client.set_leverage(self.symbol, self.leverage)

    async def run(self) -> None:
        """
        Método principal de execução do bot, contendo o loop assíncrono de:
          - Atualização de dados
          - Retreinamento online
          - Verificação de posição aberta
          - Colocação de novas ordens
        """
        await self.initialize_filters()

        while True:
            self.cycle_count += 1
            logger.debug("Iniciando ciclo {self.cycle_count}")

            # Passo 1: Obter dados recentes
            logger.info(f"Coletando dados do intervalo {self.interval}")
            new_data = await self.data_handler.get_latest_data(self.symbol, self.interval, limit=2)
            if not new_data.empty:
                if self.data_handler.historical_df.empty:
                    # Carrega 1000 candles iniciais
                    large_df = await self.data_handler.get_latest_data(self.symbol, self.interval, limit=1000)
                    with self.data_handler.data_lock:
                        self.data_handler.historical_df = self.data_handler.add_technical_indicators(large_df)
                else:
                    # Atualiza candle a candle
                    for i in range(len(new_data)):
                        row = new_data.iloc[i]
                        new_row = {
                            "timestamp": row["timestamp"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"]
                        }
                        self.data_handler.update_historical_data(new_row)
                        logger.debug(f"Atualizado candle: {new_row}")

            # Passo 2: Retreinamento periódico (online learning)
            if (self.cycle_count % self.training_interval) == 0:
                df_train = self.data_handler.historical_df.copy()
                if len(df_train) >= 300:
                    logger.info("Treinando modelo")
                    df_train["pct_change_next"] = df_train["close"].pct_change().shift(-1) * 100

                    features = df_train[["sma_short", "sma_long", "rsi", "macd",
                                         "boll_hband", "boll_lband", "atr"]].copy()
                    features.dropna(inplace=True)
                    target_tp = df_train["pct_change_next"].copy()
                    target_sl = df_train["pct_change_next"].copy()

                    min_len = min(len(features), len(target_tp))
                    features = features.iloc[:min_len]
                    target_tp = target_tp.iloc[:min_len]
                    target_sl = target_sl.iloc[:min_len]

                    # Remove última linha pois shift(-1) gera NaN no final
                    features = features.iloc[:-1]
                    target_tp = target_tp.iloc[:-1]
                    target_sl = target_sl.iloc[:-1]

                    self.model_manager.train_models(features, target_tp, target_sl)
                    logger.info("Modelo treinado com sucesso")
                else:
                    logger.info("Ainda não há dados suficientes para treinar.")

            # Passo 3: Verificar se há posição aberta
            open_long = self.binance_client.get_open_position_by_side(self.symbol, "LONG")
            open_short = self.binance_client.get_open_position_by_side(self.symbol, "SHORT")

            # Se não há posição aberta no momento...
            if open_long is None and open_short is None:
                # ...mas antes havia, então foi fechada
                if self.last_position is not None:
                    # Aqui podemos calcular o PnL (lucro/prejuízo).
                    # Maneira simples (com a diferença de preço do candle atual):
                    # - ATENÇÃO: isso não reflete taxas de trading, funding, slippage real etc.
                    exit_price = self.binance_client.get_futures_last_price(self.symbol)
                    entry_price = self.last_position["entry_price"]
                    quantity = self.last_position["quantity"]
                    side = self.last_position["side"]  # LONG ou SHORT

                    # Calcula lucro/preju baseado na direção
                    if side == "LONG":
                        # PnL = (exit_price - entry_price) * quantity
                        pnl = (exit_price - entry_price) * quantity
                    else:  # SHORT
                        pnl = (entry_price - exit_price) * quantity

                    # Loga o fechamento
                    logger.info(
                        f"Posição {side} encerrada. Preço de entrada: {entry_price}, "
                        f"Preço de saída: {exit_price}, Quantidade: {quantity}, "
                        f"Lucro/Prejuízo = {pnl:.2f} USDT"
                    )

                    # Zera a referência à posição
                    self.last_position = None

                logger.info("Nenhuma posição aberta no momento. Aguardando novo sinal.")

            elif open_long is not None or open_short is not None:
                logger.info("Já existe posição aberta. Aguardando fechamento para abrir novo trade.")
            else:
                # Passo 4: Caso não haja posição, checar sinal do modelo
                if self.model_initialized and not self.data_handler.historical_df.empty:
                    current_price = self.binance_client.get_futures_last_price(self.symbol)
                    if current_price <= 0:
                        logger.warning("Falha ao obter last price. Nenhum trade.")
                        await asyncio.sleep(5)
                        continue

                    df_eval = self.data_handler.historical_df.copy()
                    last_row = df_eval.iloc[-1]
                    X_eval = pd.DataFrame([[
                        last_row["sma_short"],
                        last_row["sma_long"],
                        last_row["rsi"],
                        last_row["macd"],
                        last_row["boll_hband"],
                        last_row["boll_lband"],
                        last_row["atr"]
                    ]],
                        columns=["sma_short", "sma_long", "rsi", "macd", "boll_hband", "boll_lband", "atr"])

                    predicted_tp_pct, predicted_sl_pct = self.model_manager.predict_tp_sl(X_eval)
                    logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

                    direction = self.strategy.decide_direction(predicted_tp_pct, threshold=0.2)
                    if direction is None:
                        logger.info("Sinal neutro, não abrir trade.")
                    else:
                        # Define side e position_side
                        if direction == "LONG":
                            side = "BUY"
                            position_side = "LONG"
                            tp_factor = 1 + max(abs(predicted_tp_pct) / 100, 0.02)
                            sl_factor = 1 - max(abs(predicted_sl_pct) / 100, 0.02)
                        else:  # SHORT
                            side = "SELL"
                            position_side = "SHORT"
                            tp_factor = 1 - max(abs(predicted_tp_pct) / 100, 0.02)
                            sl_factor = 1 + max(abs(predicted_sl_pct) / 100, 0.02)

                        tp_price = current_price * tp_factor
                        sl_price = current_price * sl_factor

                        # Calcula quantidade
                        qty = self.strategy.calculate_trade_quantity(
                            capital=self.capital,
                            current_price=current_price,
                            leverage=self.leverage,
                            risk_per_trade=self.risk_per_trade
                        )

                        # Ajusta quantidade
                        qty_adj = self.strategy.adjust_quantity_to_step_size(qty, self.step_size)
                        if qty_adj <= 0:
                            logger.warning("Qty ajustada <= 0. Trade abortado.")
                            await asyncio.sleep(5)
                            continue

                            # Salva informação da posição "aberta" em self.last_position
                            self.last_position = {
                                "side": position_side,  # LONG ou SHORT
                                "entry_price": current_price,
                                "quantity": qty_adj,
                                "symbol": self.symbol
                            }

                        logger.info(
                            f"Sinal gerado: "
                            f"side={side}, "
                            f"position_side={position_side}, "
                            f"predicted_tp={tp_factor}, "
                            f"predicted_sl={sl_factor}, "
                            f"tp_price={tp_price}, "
                            f"sl_price={sl_price}, "
                            f"current_price={current_price}, "
                            f"leverage={self.leverage}, "
                            f"risk_per_trade={self.risk_per_trade}"
                        )
                        logger.info(f"Abrindo {direction} c/ qty={qty_adj}, lastPrice={current_price:.2f}...")

                        order_resp = self.binance_client.place_order_with_retry(
                            symbol=self.symbol,
                            side=side,
                            quantity=qty_adj,
                            position_side=position_side,
                            step_size=self.step_size
                        )
                        if order_resp:
                            logger.info(f"Ordem de abertura executada: {order_resp}")
                            self.place_tp_sl(direction, current_price, tp_price, sl_price)
                        else:
                            logger.info("Não foi possível colocar ordem de abertura.")

            await asyncio.sleep(5)

    def place_tp_sl(
            self,
            direction: str,
            current_price: float,
            tp_price: float,
            sl_price: float
    ) -> None:
        """
        Cria ordens de TAKE_PROFIT e STOP_MARKET de acordo com a direção
        (LONG ou SHORT).

        :param direction: "LONG" ou "SHORT"
        :param current_price: Preço atual do ativo
        :param tp_price: Preço calculado para Take-Profit
        :param sl_price: Preço calculado para Stop-Loss
        """
        position_side = direction
        if direction == "LONG":
            # Ajuste de TP
            if tp_price <= current_price:
                tp_price = current_price + (self.tick_size * 10)
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj <= current_price:
                tp_price_adj = current_price + (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            if sl_price >= current_price:
                sl_price = current_price - (self.tick_size * 10)
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj >= current_price:
                sl_price_adj = current_price - (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "SELL"
        else:  # SHORT
            position_side = "SHORT"
            # Ajuste de TP
            if tp_price >= current_price:
                tp_price = current_price - (self.tick_size * 10)
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj >= current_price:
                tp_price_adj = current_price - (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            if sl_price <= current_price:
                sl_price = current_price + (self.tick_size * 10)
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj <= current_price:
                sl_price_adj = current_price + (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "BUY"

        # Cria ordem TAKE_PROFIT
        try:
            tp_order = self.binance_client.client.futures_create_order(
                symbol=self.symbol,
                side=side_for_tp_sl,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_str,
                closePosition=True,
                positionSide=position_side
            )
            logger.info(f"Ordem TAKE_PROFIT criada: {tp_order}")
        except Exception as e:
            logger.error(f"Erro ao criar TAKE_PROFIT: {e}", exc_info=True)

        # Cria ordem STOP
        try:
            sl_order = self.binance_client.client.futures_create_order(
                symbol=self.symbol,
                side=side_for_tp_sl,
                type="STOP_MARKET",
                stopPrice=sl_str,
                closePosition=True,
                positionSide=position_side
            )
            logger.info(f"Ordem STOP (SL) criada: {sl_order}")
        except Exception as e:
            logger.error(f"Erro ao criar STOP (SL): {e}", exc_info=True)
