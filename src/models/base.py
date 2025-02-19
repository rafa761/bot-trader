from pathlib import Path

import arrow
import pandas as pd
import ta
from binance import Client, BinanceAPIException

from core.constants import TRAIN_DATA_DIR
from core.logger import logger


class DataCollector:
    CACHE_RETENTION_DAYS = 7

    def __init__(self, client: Client):
        self.client = client

    def get_historical_klines(self, symbol: str, interval: str, start_str: str,
                              end_str: str | None = None) -> pd.DataFrame:
        """Obtém dados históricos da Binance e adiciona indicadores técnicos. Usa cache se disponível."""
        try:
            filepath = self.check_existing_data(interval, start_str)
            if filepath:
                logger.info(f"Arquivo encontrado em cache: {filepath}. Carregando dados do arquivo.")
                df = pd.read_csv(filepath, sep=';', encoding='utf-8', parse_dates=['timestamp'], index_col='timestamp')
                return df

            logger.info(f"Coletando dados para {symbol} - Intervalo: {interval} - Início: {start_str}")

            klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
            df = self._process_klines(klines)

            logger.info(f"Coleta concluída: {len(df)} registros.")

            # Adiciona indicadores técnicos
            df = TechnicalIndicatorAdder.add_technical_indicators(df)

            # Salva o CSV atualizado com indicadores técnicos
            self.save_to_csv(df, interval, start_str)

            return df
        except BinanceAPIException as e:
            logger.error(f"Erro ao coletar dados: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return pd.DataFrame()

    def _process_klines(self, klines: list) -> pd.DataFrame:
        """Converte os dados da Binance para DataFrame."""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

    def check_existing_data(self, interval: str, start_str: str) -> Path | None:
        """Verifica se um arquivo cacheado já existe e limpa arquivos antigos."""
        self.clean_old_files()

        iso_date = arrow.utcnow().format("YYYY-MM-DD")
        filename = f"{iso_date}-binance-historical-data-interval-{interval}-start-date-{start_str}.csv"
        filepath = Path(TRAIN_DATA_DIR) / filename

        return filepath if filepath.exists() else None

    def clean_old_files(self) -> None:
        """Remove arquivos de cache mais antigos que CACHE_RETENTION_DAYS."""
        try:
            cache_dir = Path(TRAIN_DATA_DIR)
            retention_date = arrow.utcnow().shift(days=-self.CACHE_RETENTION_DAYS).date()

            for file in cache_dir.glob("*.csv"):
                try:
                    file_date_str = file.stem.split("-")[0]  # Pegando a data corretamente
                    file_date = arrow.get(file_date_str, "YYYY-MM-DD").date()

                    if file_date < retention_date:
                        file.unlink()
                        logger.info(f"Arquivo antigo removido: {file}")
                except Exception as e:
                    logger.warning(f"Erro ao processar arquivo {file}: {e}")
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos antigos: {e}")

    def save_to_csv(self, df: pd.DataFrame, interval: str, start_str: str) -> None:
        """Salva o DataFrame em CSV, incluindo indicadores técnicos."""
        try:
            iso_date = arrow.utcnow().format("YYYY-MM-DD")
            filename = f"{iso_date}-binance-historical-data-interval-{interval}-start-date-{start_str}.csv"
            filepath = Path(TRAIN_DATA_DIR) / filename

            df.to_csv(filepath, sep=';', encoding='utf-8', index=True)

            logger.info(f"Dados salvos em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar CSV: {e}")


class TechnicalIndicatorAdder:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos ao DataFrame."""
        try:
            logger.info("Adicionando indicadores técnicos ao DataFrame.")

            df['sma_short'] = ta.trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['sma_long'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'], close=df['close'],
                window=14
            ).average_true_range()
            df["macd"] = ta.trend.macd_diff(df["close"])
            df["boll_hband"] = ta.volatility.bollinger_hband(df["close"], window=20)
            df["boll_lband"] = ta.volatility.bollinger_lband(df["close"], window=20)

            # Garantia de remoção de NaN
            df.dropna(inplace=True)

            logger.info("Indicadores técnicos adicionados com sucesso.")

            return df
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
            return df


class LabelCreator:
    @staticmethod
    def create_labels(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
        """Cria labels para TP e SL com base no movimento de preço futuro."""
        try:
            logger.info(f"Criando labels para TP e SL com horizon={horizon} períodos.")

            df['future_high'] = df['high'].rolling(window=horizon).max().shift(-horizon)
            df['future_low'] = df['low'].rolling(window=horizon).min().shift(-horizon)

            df['TP_pct'] = ((df['future_high'] - df['close']) / df['close']) * 100
            df['SL_pct'] = ((df['close'] - df['future_low']) / df['close']) * 100

            df.drop(['future_high', 'future_low'], axis=1, inplace=True)

            df.dropna(inplace=True)

            logger.info("Labels para TP e SL criados com sucesso.")

            return df
        except Exception as e:
            logger.error(f"Erro ao criar labels: {e}")
            return df
