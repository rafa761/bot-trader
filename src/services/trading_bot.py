# services/trading_bot.py

import asyncio
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.config import settings
from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_handler import DataHandler
from services.binance_client import BinanceClient
from services.trading_strategy import TradingStrategy


class TradingBot:
    """
    Classe principal do bot de trading, que faz a orquestração entre
    DataHandler, modelos LSTM, TradingStrategy e BinanceClient.
    """

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """
        Construtor que inicializa as classes auxiliares e configura variáveis necessárias.

        Args:
            tp_model: Modelo LSTM pré-treinado para previsão de Take Profit
            sl_model: Modelo LSTM pré-treinado para previsão de Stop Loss
        """
        # O cliente Binance será inicializado assincronamente no método initialize()
        self.binance_client = BinanceClient()
        # O DataHandler será inicializado após o cliente Binance
        self.data_handler = None

        self.tp_model = tp_model
        self.sl_model = sl_model
        self.strategy = TradingStrategy()

        # Filtros (tick_size, step_size) para formatar ordens
        self.tick_size, self.step_size = 0.0, 0.0

        # Controle de verificação dos modelos
        self.models_loaded = tp_model is not None and sl_model is not None
        self.cycle_count = 0

        # Parâmetros LSTM
        self.sequence_length = self.tp_model.config.sequence_length if self.models_loaded else 16

        logger.info("Classe TradingBot inicializada com sucesso.")

    async def initialize(self) -> None:
        """
        Inicializa os componentes assíncronos do bot:
        - Cliente Binance
        - DataHandler
        - Filtros de trading (tick_size, step_size)
        - Alavancagem

        Este método deve ser chamado antes de run().
        """
        logger.info("Iniciando componentes assíncronos do TradingBot...")

        # Inicializa o cliente Binance
        await self.binance_client.initialize()

        # Inicializa o DataHandler com o cliente inicializado
        self.data_handler = DataHandler(self.binance_client)

        # Inicializa filtros e alavancagem
        await self.initialize_filters()

        logger.info("TradingBot inicializado com sucesso.")

    async def initialize_filters(self) -> None:
        """
        Inicializa os filtros (tickSize e stepSize) e tenta ajustar alavancagem
        no par de trading configurado.
        """
        self.tick_size, self.step_size = await self.binance_client.get_symbol_filters(settings.SYMBOL)

        if not self.tick_size or not self.step_size:
            logger.error("Não foi possível obter tickSize/stepSize. Encerrando.")
            sys.exit(1)

        logger.info(f"{settings.SYMBOL} -> tickSize={self.tick_size}, stepSize={self.step_size}")

        # Define alavancagem
        await self.binance_client.set_leverage(settings.SYMBOL, settings.LEVERAGE)

    def _prepare_sequence_for_prediction(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM.

        Args:
            df: DataFrame com os dados históricos

        Returns:
            Optional[np.ndarray]: Sequência formatada para o modelo LSTM ou None se dados insuficientes
        """
        # Verifica se todos os FEATURE_COLUMNS existem no DataFrame
        missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if missing_columns:
            logger.warning(f"Colunas ausentes no DataFrame: {missing_columns}")
            return None

        # Verifica se há dados suficientes
        if len(df) < self.sequence_length:
            logger.warning(f"Dados insuficientes para LSTM. Necessário: {self.sequence_length}, Disponível: {len(df)}")
            return None

        # Verifica se há valores NaN nas features requeridas
        if df[FEATURE_COLUMNS].isna().any().any():
            logger.warning("Existem valores NaN nas features necessárias para o LSTM")
            return None

        try:
            # Pegar as últimas 'sequence_length' entradas para a previsão
            last_sequence = df[FEATURE_COLUMNS].values[-self.sequence_length:]

            # Reformatar para o formato que o LSTM espera [samples, time steps, features]
            x_pred = np.array([last_sequence])

            return x_pred
        except Exception as e:
            logger.error(f"Erro ao preparar sequência para LSTM: {e}", exc_info=True)
            return None

    async def place_tp_sl(
            self,
            direction: str,
            current_price: float,
            tp_price: float,
            sl_price: float
    ) -> tuple[dict | None, dict | None]:
        """
        Cria ordens de TAKE_PROFIT e STOP_MARKET de acordo com a direção
        (LONG ou SHORT).

        Args:
            direction: "LONG" ou "SHORT"
            current_price: Preço atual do ativo
            tp_price: Preço calculado para Take-Profit
            sl_price: Preço calculado para Stop-Loss

        Returns:
            Tuple[Optional[Dict], Optional[Dict]]: Respostas das ordens TP e SL
        """
        position_side = direction
        if direction == "LONG":
            # Ajuste de TP
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj <= current_price:
                tp_price_adj = current_price + (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj >= current_price:
                sl_price_adj = current_price - (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "SELL"
        else:  # SHORT
            position_side = "SHORT"
            # Ajuste de TP
            tp_price_adj = self.strategy.adjust_price_to_tick_size(tp_price, self.tick_size)
            if tp_price_adj >= current_price:
                tp_price_adj = current_price - (self.tick_size * 10)
            tp_str = self.strategy.format_price_for_tick_size(tp_price_adj, self.tick_size)

            # Ajuste de SL
            sl_price_adj = self.strategy.adjust_price_to_tick_size(sl_price, self.tick_size)
            if sl_price_adj <= current_price:
                sl_price_adj = current_price + (self.tick_size * 10)
            sl_str = self.strategy.format_price_for_tick_size(sl_price_adj, self.tick_size)

            side_for_tp_sl = "BUY"

        # Cria ordens TP e SL de forma assíncrona
        return await self.binance_client.place_tp_sl_orders(
            symbol=settings.SYMBOL,
            side=side_for_tp_sl,
            position_side=position_side,
            tp_price=tp_str,
            sl_price=sl_str
        )

    async def run(self) -> None:
        """
        Método principal de execução do bot, contendo o loop assíncrono de:
          - Atualização de dados
          - Verificação de posição aberta
          - Colocação de novas ordens
        """
        try:
            # Inicializa componentes assíncronos
            await self.initialize()

            while True:
                self.cycle_count += 1
                logger.debug(f"Iniciando ciclo {self.cycle_count}")

                # Passo 1: Obter dados recentes
                logger.info(f"Coletando dados do intervalo {settings.INTERVAL}")
                new_data = await self.data_handler.get_latest_data(settings.SYMBOL, settings.INTERVAL, limit=2)
                if not new_data.empty:
                    if self.data_handler.historical_df.empty:
                        # Carrega 1000 candles iniciais
                        large_df = await self.data_handler.get_latest_data(settings.SYMBOL, settings.INTERVAL,
                                                                           limit=1000)

                        if large_df.empty:
                            logger.error("Não foi possível obter dados históricos iniciais")
                            await asyncio.sleep(5)
                            continue

                        # Garantir que temos pelo menos o mínimo de candles necessários para os indicadores
                        if len(large_df) < 100:  # Valor conservador
                            logger.warning(
                                f"Dados históricos insuficientes: obtidos {len(large_df)} candles, necessários pelo menos 100")
                            await asyncio.sleep(5)
                            continue

                        try:
                            with self.data_handler.data_lock:
                                self.data_handler.historical_df = self.data_handler.technical_indicator_adder.add_technical_indicators(
                                    large_df)

                            # Verificar se todos os indicadores foram calculados corretamente
                            missing_indicators = [col for col in FEATURE_COLUMNS if
                                                  col not in self.data_handler.historical_df.columns]
                            if missing_indicators:
                                logger.error(f"Indicadores ausentes após cálculo: {missing_indicators}")
                                await asyncio.sleep(5)
                                continue

                            # Verificar se há valores NaN nos indicadores
                            if self.data_handler.historical_df[FEATURE_COLUMNS].isna().any().any():
                                logger.warning("Existem valores NaN nos indicadores técnicos")
                                # Remove as linhas com NaN para garantir dados limpos
                                self.data_handler.historical_df.dropna(subset=FEATURE_COLUMNS, inplace=True)

                            logger.info(
                                f"Dados históricos iniciais carregados com sucesso: {len(self.data_handler.historical_df)} candles")
                        except Exception as e:
                            logger.error(f"Erro ao processar dados históricos iniciais: {e}", exc_info=True)
                            await asyncio.sleep(5)
                            continue
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

                            # Verifica se já existe este timestamp para evitar duplicatas
                            if self.data_handler.historical_df is not None and not self.data_handler.historical_df.empty:
                                existing_timestamps = self.data_handler.historical_df["timestamp"].astype(str).tolist()
                                if str(row["timestamp"]) in existing_timestamps:
                                    logger.debug(f"Candle com timestamp {row['timestamp']} já existe, ignorando.")
                                    continue

                            # Usa o método original do DataHandler, agora mais seguro
                            self.data_handler.update_historical_data(new_row)
                            logger.debug(f"Atualizado candle: {new_row}")

                # Passo 2: Verificar se há posição aberta
                open_long = await self.binance_client.get_open_position_by_side(settings.SYMBOL, "LONG")
                open_short = await self.binance_client.get_open_position_by_side(settings.SYMBOL, "SHORT")

                if open_long is not None or open_short is not None:
                    logger.info("Já existe posição aberta. Aguardando fechamento para abrir novo trade.")
                else:
                    # Passo 3: Caso não haja posição, checar sinal do modelo
                    if self.models_loaded and not self.data_handler.historical_df.empty:
                        current_price = await self.binance_client.get_futures_last_price(settings.SYMBOL)
                        if current_price <= 0:
                            logger.warning("Falha ao obter last price. Nenhum trade.")
                            await asyncio.sleep(5)
                            continue

                        df_eval = self.data_handler.historical_df.copy()

                        # Verificar se temos dados suficientes
                        if len(df_eval) < self.sequence_length:
                            logger.warning(f"Dados insuficientes para previsão com LSTM. Aguardando mais dados...")
                            await asyncio.sleep(5)
                            continue

                        # Preparar sequência para LSTM
                        X_seq = self._prepare_sequence_for_prediction(df_eval)
                        if X_seq is None:
                            logger.warning("Falha ao preparar dados para LSTM. Aguardando próximo ciclo.")
                            await asyncio.sleep(5)
                            continue

                        try:
                            # Previsões com LSTM
                            predicted_tp_pct = self.tp_model.predict(X_seq)[0][0]
                            predicted_sl_pct = self.sl_model.predict(X_seq)[0][0]
                            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")
                        except Exception as e:
                            logger.error(f"Erro na previsão com LSTM: {e}", exc_info=True)
                            await asyncio.sleep(5)
                            continue

                        direction = self.strategy.decide_direction(predicted_tp_pct, threshold=0.2)
                        if direction is None:
                            logger.info("Sinal neutro, não abrir trade.")
                        else:
                            # Define side e position_side
                            if direction == "LONG":
                                side = "BUY"
                                position_side = "LONG"
                                tp_factor = 1 + max(abs(predicted_tp_pct) / 100, 0.02)
                                sl_factor = 1 - max(abs(predicted_sl_pct) / 100, 0.005)
                            else:  # SHORT
                                side = "SELL"
                                position_side = "SHORT"
                                tp_factor = 1 - max(abs(predicted_tp_pct) / 100, 0.02)
                                sl_factor = 1 + max(abs(predicted_sl_pct) / 100, 0.005)

                            tp_price = current_price * tp_factor
                            sl_price = current_price * sl_factor

                            # Calcula quantidade
                            qty = self.strategy.calculate_trade_quantity(
                                capital=settings.CAPITAL,
                                current_price=current_price,
                                leverage=settings.LEVERAGE,
                                risk_per_trade=settings.RISK_PER_TRADE
                            )

                            # Ajusta quantidade
                            qty_adj = self.strategy.adjust_quantity_to_step_size(qty, self.step_size)
                            if qty_adj <= 0:
                                logger.warning("Qty ajustada <= 0. Trade abortado.")
                                await asyncio.sleep(5)
                                continue

                            logger.info(
                                f"Sinal gerado (LSTM): "
                                f"side={side}, "
                                f"position_side={position_side}, "
                                f"predicted_tp={predicted_tp_pct:.2f}%, "
                                f"predicted_sl={predicted_sl_pct:.2f}%, "
                                f"tp_price={tp_price:.2f}, "
                                f"sl_price={sl_price:.2f}, "
                                f"current_price={current_price:.2f}, "
                                f"leverage={settings.LEVERAGE}, "
                                f"risk_per_trade={settings.RISK_PER_TRADE}"
                            )
                            logger.info(f"Abrindo {direction} c/ qty={qty_adj}, lastPrice={current_price:.2f}...")

                            order_resp = await self.binance_client.place_order_with_retry(
                                symbol=settings.SYMBOL,
                                side=side,
                                quantity=qty_adj,
                                position_side=position_side,
                                step_size=self.step_size
                            )
                            if order_resp:
                                logger.info(f"Ordem de abertura executada: {order_resp}")
                                await self.place_tp_sl(direction, current_price, tp_price, sl_price)
                            else:
                                logger.info("Não foi possível colocar ordem de abertura.")

                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Erro no loop principal do bot: {e}", exc_info=True)
        finally:
            # Garantir que o cliente seja fechado corretamente em qualquer situação
            await self.binance_client.close()
            logger.info("Conexões do bot fechadas corretamente.")
