# services/binance/binance_data_provider.py

import pandas as pd

from core.config import settings
from core.constants import FEATURE_COLUMNS
from core.logger import logger
from repositories.data_handler import DataHandler
from services.base.services import MarketDataProvider
from services.binance.binance_client import BinanceClient


class BinanceDataProvider(MarketDataProvider):
    """
    Provedor de dados de mercado da Binance.

    Responsável por obter e atualizar dados de mercado da Binance,
    implementando a interface MarketDataProvider para garantir
    substituibilidade seguindo o princípio de Liskov.
    """

    def __init__(self, binance_client: BinanceClient, data_handler: DataHandler):
        """
        Inicializa o provedor de dados.

        Args:
            binance_client: Cliente da Binance para acessar a API
            data_handler: Manipulador de dados para armazenar e processar dados históricos
        """
        self.client = binance_client
        self.data_handler = data_handler
        self.min_candles_required = 100

    async def initialize(self) -> None:
        """
        Inicializa a conexão com a Binance e carrega os dados iniciais.
        """
        if not self.client.is_client_initialized():
            await self.client.initialize()
            logger.info("Provedor de dados Binance inicializado")

    async def get_latest_data(self) -> pd.DataFrame:
        """
        Obtém os dados mais recentes da Binance.

        Returns:
            pd.DataFrame: DataFrame com os dados mais recentes
        """
        # Se não há dados históricos, carrega um conjunto inicial maior
        if self.data_handler.historical_df.empty:
            logger.info(f"Carregando conjunto inicial de dados - {settings.INTERVAL}")
            large_df = await self.data_handler.get_latest_data(
                settings.SYMBOL, settings.INTERVAL, limit=1000
            )

            if large_df.empty:
                logger.error("Não foi possível obter dados históricos iniciais")
                return pd.DataFrame()

            # Verificar se há dados suficientes
            if len(large_df) < self.min_candles_required:
                logger.warning(
                    f"Dados históricos insuficientes: obtidos {len(large_df)} candles, "
                    f"necessários pelo menos {self.min_candles_required}"
                )
                return pd.DataFrame()

            try:
                with self.data_handler.data_lock:
                    self.data_handler.historical_df = (
                        self.data_handler.technical_indicator_adder.add_technical_indicators(large_df)
                    )

                # Verificar integridade dos indicadores
                missing_indicators = [
                    col for col in FEATURE_COLUMNS if col not in self.data_handler.historical_df.columns
                ]
                if missing_indicators:
                    logger.error(f"Indicadores ausentes após cálculo: {missing_indicators}")
                    return pd.DataFrame()

                # Remover linhas com valores NaN
                if self.data_handler.historical_df[FEATURE_COLUMNS].isna().any().any():
                    logger.warning("Existem valores NaN nos indicadores técnicos")
                    self.data_handler.historical_df.dropna(subset=FEATURE_COLUMNS, inplace=True)

                logger.info(
                    f"Dados históricos iniciais carregados: {len(self.data_handler.historical_df)} candles"
                )
                return self.data_handler.historical_df

            except Exception as e:
                logger.error(f"Erro ao processar dados históricos iniciais: {e}", exc_info=True)
                return pd.DataFrame()

        # Caso já tenha dados, apenas atualiza com os novos candles
        else:
            new_data = await self.data_handler.get_latest_data(
                settings.SYMBOL, settings.INTERVAL, limit=2
            )

            if not new_data.empty:
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

                    # Verifica se o timestamp já existe
                    if not self.data_handler.historical_df.empty:
                        existing_timestamps = self.data_handler.historical_df["timestamp"].astype(str).tolist()
                        if str(row["timestamp"]) in existing_timestamps:
                            continue

                    # Atualiza o DataFrame histórico
                    self.data_handler.update_historical_data(new_row)

            return self.data_handler.historical_df

    def get_historical_data(self) -> pd.DataFrame:
        """
        Retorna os dados históricos armazenados.

        Returns:
            pd.DataFrame: DataFrame com dados históricos armazenados
        """
        return self.data_handler.historical_df
