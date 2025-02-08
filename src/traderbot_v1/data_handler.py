# data_handler.py

import numpy as np
import pandas as pd
import requests
import ta
from binance.exceptions import BinanceAPIException

from binance_client import BinanceClientService
from config import SMA_WINDOW_SHORT, SMA_WINDOW_LONG
from logger import logger


class DataHandler:
    """
    Classe responsável por gerenciar dados históricos e cálculo de indicadores.
    """

    def __init__(self, binance_service: BinanceClientService) -> None:
        """
        Inicializa a instância do DataHandler.

        :param binance_service: Instância de BinanceClientService para interagir com a Binance.
        """
        self.binance_service = binance_service
        self.historical_df: pd.DataFrame = pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame fornecido.

        :param df: DataFrame contendo os dados de preços.
        :return: DataFrame atualizado com os indicadores técnicos.
        """
        logger.info("Calculando indicadores técnicos usando 'ta'")
        try:
            df['sma_short'] = ta.trend.sma_indicator(df['close'], window=SMA_WINDOW_SHORT)
            df['sma_long'] = ta.trend.sma_indicator(df['close'], window=SMA_WINDOW_LONG)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['bollinger_hband'] = ta.volatility.bollinger_hband(df['close'], window=20)
            df['bollinger_lband'] = ta.volatility.bollinger_lband(df['close'], window=20)
            df['atr'] = self.calculate_atr(df, window=14)
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores técnicos: {e}", exc_info=True)
            return df

    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calcula o indicador Average True Range (ATR).

        :param df: DataFrame contendo os dados de preços.
        :param window: Período da média móvel.
        :return: Série com os valores do ATR.
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()

    def get_latest_data(self, symbol: str = 'BTCUSDT', interval: str = '1m', limit: int = 5000) -> pd.DataFrame:
        """
        Coleta dados históricos mais recentes da Binance Futures Testnet.

        :param symbol: Símbolo do ativo a ser consultado.
        :param interval: Intervalo de tempo dos candles.
        :param limit: Quantidade máxima de registros a serem coletados.
        :return: DataFrame contendo os dados coletados.
        """
        logger.info(f"Coletando {limit} dados mais recentes para {symbol} com intervalo {interval}")
        client = self.binance_service.client  # Acessa diretamente o client

        try:
            max_limit_per_call = 1500
            data = []
            endTime = None
            while limit > 0:
                current_limit = min(limit, max_limit_per_call)
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': current_limit
                }
                if endTime:
                    params['endTime'] = endTime
                klines = client.futures_klines(**params)
                logger.info(f"Linhas retornadas nesta chamada: {len(klines)}")
                if not klines:
                    break
                data.extend(klines)
                limit -= current_limit
                endTime = klines[0][0] - 1  # Pega o primeiro kline para buscar dados anteriores

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['high'] = pd.to_numeric(df['high'], errors='coerce')
            df['low'] = pd.to_numeric(df['low'], errors='coerce')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.dropna(inplace=True)
            df.sort_values('timestamp', inplace=True)

            logger.info(f"Dados mais recentes de {symbol} coletados com sucesso")
            self.historical_df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            return self.historical_df

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout ao coletar dados.")
            return pd.DataFrame()
        except BinanceAPIException as e:
            logger.error(f"Erro da API da Binance: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro ao coletar dados: {e}", exc_info=True)
            return pd.DataFrame()

    def update_historical_df(self, new_row: dict[str, float]) -> None:
        """
        Atualiza o DataFrame histórico com um novo candle.

        :param new_row: Dicionário contendo os dados do novo candle.
        """
        temp_df = pd.DataFrame([new_row])
        self.historical_df = pd.concat([self.historical_df, temp_df], ignore_index=True)
        self.historical_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
        # Recalcula indicadores
        self.historical_df = self.add_technical_indicators(self.historical_df)

    def get_current_features(self) -> pd.DataFrame:
        """
        Retorna a última linha de indicadores calculados, se existir.

        :return: DataFrame contendo os últimos valores dos indicadores técnicos.
        """
        if not self.historical_df.empty:
            return self.historical_df.tail(1)[
                ['sma_short', 'sma_long', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband']
            ]
        else:
            return pd.DataFrame()
