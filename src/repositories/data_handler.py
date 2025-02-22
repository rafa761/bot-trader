# data_handler.py

"""
Este módulo gerencia a coleta de dados do histórico (klines) e atualização
do DataFrame que armazena as informações de preço. Também inclui
cálculo de indicadores técnicos.
"""

import asyncio
import threading

import pandas as pd
import requests
from binance.exceptions import BinanceAPIException

from core.logger import logger
from models.base import TechnicalIndicatorAdder
from services.binance_client import BinanceClient


class DataHandler:
    """
    Classe responsável por gerenciar a coleta e atualização dos dados
    de mercado, bem como o cálculo de indicadores técnicos.
    """

    def __init__(self, binance_client: BinanceClient):
        """
        Construtor que recebe uma instância de BinanceClient e prepara
        os atributos de controle de dados.

        :param binance_client: Instância do cliente da Binance
        """
        logger.info("Iniciando classe de DataHandler...")
        self.client = binance_client
        self.historical_df = pd.DataFrame()
        self.data_lock = threading.Lock()
        self.technical_indicator_adder = TechnicalIndicatorAdder()

    async def get_latest_data(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Coleta as últimas velas de Futuros para um determinado símbolo e intervalo.

        :param symbol: Par de trading, ex.: "BTCUSDT"
        :param interval: Intervalo, ex.: "15m"
        :param limit: Quantidade de velas a serem obtidas
        :return: DataFrame com colunas [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Coletando {limit} velas de {symbol} (intervalo={interval})")
        attempt, max_attempts = 0, 5

        while attempt < max_attempts:
            try:
                klines = self.client.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
                df = pd.DataFrame(klines, columns=[
                    "timestamp", "open", "high", "low", "close", "volume",
                    "close_time", "quote_asset_volume", "number_of_trades",
                    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                ])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                    float)
                df.dropna(inplace=True)
                df.reset_index(drop=True, inplace=True)
                return df[["timestamp", "open", "high", "low", "close", "volume"]]
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout ao coletar dados, tentativa {attempt + 1}")
                await asyncio.sleep(3)
            except BinanceAPIException as e:
                logger.error(f"Erro API Binance: {e}")
                return pd.DataFrame()
            except Exception as e:
                logger.error(f"Erro ao coletar dados: {e}", exc_info=True)
                return pd.DataFrame()
            attempt += 1

        logger.error("Tentativas esgotadas. Não foi possível coletar dados.")
        return pd.DataFrame()

    def update_historical_data(self, new_row: dict) -> None:
        """
        Atualiza o DataFrame histórico com uma nova linha e recalcula indicadores.

        :param new_row: Dicionário com colunas [timestamp, open, high, low, close, volume].
        """
        try:
            with self.data_lock:
                df = self.historical_df
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
                df.sort_values(by="timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)
                df = self.technical_indicator_adder.add_technical_indicators(df)
                self.historical_df = df
        except Exception as e:
            logger.error(f"Erro ao atualizar histórico: {e}", exc_info=True)
