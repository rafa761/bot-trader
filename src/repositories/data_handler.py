# repositories\data_handler.py

import asyncio
import re
import threading
from pathlib import Path

import arrow
import numpy as np
import pandas as pd
import ta
from binance import Client
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel, Field

from core.config import settings
from core.constants import TRAIN_DATA_DIR
from core.logger import logger
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
        # Número mínimo de candles necessários para calcular indicadores técnicos com segurança
        self.min_candles_required = 50

    async def get_latest_data(self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """
        Coleta as últimas velas de Futuros para um determinado símbolo e intervalo de forma assíncrona.

        Args:
            symbol: Par de trading, ex.: "BTCUSDT"
            interval: Intervalo, ex.: "15m"
            limit: Quantidade de velas a serem obtidas

        Returns:
            pd.DataFrame: DataFrame com colunas [timestamp, open, high, low, close, volume]
        """
        logger.info(f"Coletando {limit} velas de {symbol} (intervalo={interval})")
        attempt, max_attempts = 0, 5

        while attempt < max_attempts:
            try:
                klines = await self.client.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
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
            except BinanceAPIException as e:
                logger.error(f"Erro API Binance: {e}")
                # Erros específicos da API Binance geralmente indicam problemas que novas tentativas não resolverão
                # Exemplos: permissão negada, símbolo inválido, etc.
                return pd.DataFrame()
            except TimeoutError:
                logger.warning(f"Timeout ao coletar dados, tentativa {attempt + 1}")
                attempt += 1
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Erro ao coletar dados: {e}", exc_info=True)
                attempt += 1
                await asyncio.sleep(3)

        logger.error("Tentativas esgotadas. Não foi possível coletar dados.")
        return pd.DataFrame()
    def update_historical_data(self, new_row: dict) -> None:
        """
        Atualiza o DataFrame histórico com uma nova linha e recalcula indicadores
        de forma segura, garantindo que existam dados suficientes para os cálculos.

        :param new_row: Dicionário com colunas [timestamp, open, high, low, close, volume].
        """
        try:
            with self.data_lock:
                # Se o DataFrame estiver vazio, apenas adiciona o novo registro
                if self.historical_df.empty:
                    self.historical_df = pd.DataFrame([new_row])
                    logger.debug("Primeiro registro adicionado ao DataFrame histórico.")
                    return

                # Adiciona a nova linha, remove duplicatas e ordena
                df = self.historical_df
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
                df.sort_values(by="timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Só recalcula indicadores se tiver dados suficientes
                if len(df) >= self.min_candles_required:
                    try:
                        df = self.technical_indicator_adder.add_technical_indicators(df)
                    except Exception as e:
                        logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
                        # Mesmo com erro, mantemos os dados brutos
                        self.historical_df = df
                        return
                else:
                    logger.warning(
                        f"Dados insuficientes para cálculo de indicadores técnicos. "
                        f"Necessário: {self.min_candles_required}, Disponível: {len(df)}"
                    )
                    # Mantém os dados brutos sem calcular indicadores
                    self.historical_df = df
                    return

                # Atualiza o DataFrame histórico
                self.historical_df = df

        except Exception as e:
            logger.error(f"Erro ao atualizar histórico: {e}", exc_info=True)


class DataCollectorConfig(BaseModel):
    """Configuração base para coleta de dados"""
    symbol: str = settings.SYMBOL
    interval: str = settings.INTERVAL
    start_str: str = settings.MODEL_DATA_TRAINING_START_DATE
    end_str: str | None = None
    cache_retention_days: int = 7


class DataCollector:
    CACHE_RETENTION_DAYS = 7

    def __init__(self, client: Client, config: DataCollectorConfig | None = None):
        self.client = client
        self.config = config

        if self.config is None:
            self.config = DataCollectorConfig()

    def get_historical_klines(self) -> pd.DataFrame:
        """Obtém dados históricos da Binance e adiciona indicadores técnicos. Usa cache se disponível."""
        try:
            filepath = self.check_existing_data()
            if filepath:
                logger.info(f"Arquivo encontrado em cache: {filepath}. Carregando dados do arquivo.")
                df = pd.read_csv(filepath, sep=';', encoding='utf-8', parse_dates=['timestamp'], index_col='timestamp')
                return df

            logger.info(f"Coletando dados para {self.config.symbol} - "
                        f"Intervalo: {self.config.interval} - "
                        f"Início: {self.config.start_str}")

            klines = self.client.get_historical_klines(
                self.config.symbol,
                self.config.interval,
                self.config.start_str,
                self.config.end_str
            )
            df = self._process_klines(klines)

            logger.info(f"Coleta concluída: {len(df)} registros.")

            # Adiciona indicadores técnicos
            df = TechnicalIndicatorAdder.add_technical_indicators(df)

            # Salva o CSV atualizado com indicadores técnicos
            self.save_to_csv(df)

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

    def check_existing_data(self) -> Path | None:
        """Verifica se um arquivo cacheado já existe e limpa arquivos antigos."""
        self.clean_old_files()

        iso_date = arrow.utcnow().format("YYYY-MM-DD")
        filename = f"{iso_date}-binance-historical-data-interval-{self.config.interval}-start-date-{self.config.start_str}.csv"
        filepath = Path(TRAIN_DATA_DIR) / filename

        return filepath if filepath.exists() else None

    def clean_old_files(self) -> None:
        """Remove arquivos de cache mais antigos que CACHE_RETENTION_DAYS."""
        try:
            cache_dir = Path(TRAIN_DATA_DIR)
            retention_date = arrow.utcnow().shift(days=-self.CACHE_RETENTION_DAYS).date()

            for file in cache_dir.glob("*.csv"):
                try:
                    match = re.search(r"^\d{4}-\d{2}-\d{2}", file.stem)
                    if not match:
                        logger.warning(f"Nome de arquivo inválido, não foi possível fazer exclusão: {file}")
                        continue

                    file_date_str = match.group()
                    file_date = arrow.get(file_date_str, "YYYY-MM-DD").date()

                    if file_date < retention_date:
                        file.unlink()
                        logger.info(f"Arquivo antigo removido: {file}")
                except Exception as e:
                    logger.warning(f"Erro ao processar arquivo {file}: {e}")
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos antigos: {e}")

    def save_to_csv(self, df: pd.DataFrame) -> None:
        """Salva o DataFrame em CSV, incluindo indicadores técnicos."""
        try:
            iso_date = arrow.utcnow().format("YYYY-MM-DD")
            filename = f"{iso_date}-binance-historical-data-interval-{self.config.interval}-start-date-{self.config.start_str}.csv"
            filepath = Path(TRAIN_DATA_DIR) / filename

            df.to_csv(filepath, sep=';', encoding='utf-8', index=True)

            logger.info(f"Dados salvos em {filepath}")
        except Exception as e:
            logger.error(f"Erro ao salvar CSV: {e}")


class TechnicalIndicatorConfig(BaseModel):
    """Configuração para indicadores técnicos"""
    sma_windows: tuple[int, int] = (10, 50)
    bollinger_window: int = 21
    ema_windows: tuple[int, int] = (12, 26)
    rsi_window: int = 14
    stoch_k_window: int = 14
    stoch_d_window: int = 3
    cci_window: int = 20
    volume_macd_windows: tuple[int, int, int] = (26, 12, 9)
    atr_window: int = 14
    keltner_window: int = 20
    vwap_window: int = 14
    adx_window: int = 14
    roc_window: int = 12


class TechnicalIndicatorAdder:
    @classmethod
    def add_technical_indicators(cls, df: pd.DataFrame, config: TechnicalIndicatorConfig | None = None) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame com validação de tamanho mínimo.

        :param df: DataFrame com dados OHLCV
        :param config: Configuração opcional para indicadores técnicos
        :return: DataFrame com indicadores técnicos adicionados
        """
        if config is None:
            config = TechnicalIndicatorConfig()

        # Definir o tamanho mínimo necessário para os indicadores
        min_size = max(
            config.sma_windows[1],
            config.bollinger_window,
            config.ema_windows[1],
            config.rsi_window,
            config.stoch_k_window + config.stoch_d_window,
            config.cci_window,
            config.volume_macd_windows[0],
            config.atr_window,
            config.keltner_window,
            config.vwap_window,
            config.adx_window,
            config.roc_window
        )

        # Verificar se há dados suficientes
        if len(df) < min_size:
            logger.warning(
                f"DataFrame com tamanho insuficiente para calcular indicadores. "
                f"Necessário: {min_size}, Disponível: {len(df)}"
            )
            return df

        try:
            logger.info("Adicionando indicadores técnicos ao DataFrame.")

            ## Indicadores de Tendência
            df['sma_short'] = ta.trend.SMAIndicator(
                close=df['close'],
                window=config.sma_windows[0],
            ).sma_indicator()
            df['sma_long'] = ta.trend.SMAIndicator(
                close=df['close'],
                window=config.sma_windows[1]
            ).sma_indicator()

            # Bollinger
            df["boll_hband"] = ta.volatility.bollinger_hband(
                df["close"],
                window=config.bollinger_window
            )
            df["boll_lband"] = ta.volatility.bollinger_lband(
                df["close"],
                window=config.bollinger_window
            )

            # Exponential Moving Average (EMA)
            # EMA é semelhante à SMA, mas dá mais peso aos preços recentes.
            df['ema_short'] = ta.trend.EMAIndicator(
                close=df['close'],
                window=config.ema_windows[0]
            ).ema_indicator()
            df['ema_long'] = ta.trend.EMAIndicator(
                close=df['close'],
                window=config.ema_windows[1]
            ).ema_indicator()

            # Parabolic SAR
            # Indicador que ajuda a identificar a direção da tendência e possíveis pontos de reversão.
            df['parabolic_sar'] = ta.trend.PSARIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close']
            ).psar()

            ## Indicadores de Momento
            df['rsi'] = ta.momentum.RSIIndicator(
                close=df['close'],
                window=config.rsi_window
            ).rsi()

            # Stochastic Oscillator
            # Mede a posição do preço de fechamento em relação à faixa de preço em um determinado período.
            df['stoch_k'] = ta.momentum.StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.stoch_k_window
            ).stoch()
            df['stoch_d'] = df['stoch_k'].rolling(window=config.stoch_d_window).mean()

            # Commodity Channel Index (CCI)
            # Mede a variação do preço em relação à sua média estatística.
            df['cci'] = ta.trend.CCIIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.cci_window
            ).cci()

            ## Indicadores de Volatilidade
            df["macd"] = ta.trend.macd_diff(df["close"])

            # Volume Weighted MACD
            df['volume_macd'] = ta.trend.MACD(
                df['close'],
                window_slow=config.volume_macd_windows[0],
                window_fast=config.volume_macd_windows[1],
                window_sign=config.volume_macd_windows[2]
            ).macd_diff()

            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'], close=df['close'],
                window=config.atr_window
            ).average_true_range()

            # Bollinger Bands Width
            # Mede a largura das bandas de Bollinger, indicando a volatilidade do mercado.
            df['boll_width'] = df['boll_hband'] - df['boll_lband']

            # Keltner Channels
            # Outro indicador de volatilidade que usa ATR para definir os canais
            df['keltner_hband'] = ta.volatility.KeltnerChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.keltner_window
            ).keltner_channel_hband()
            df['keltner_lband'] = ta.volatility.KeltnerChannel(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.keltner_window
            ).keltner_channel_lband()

            ## Indicadores de Volume
            # On-Balance Volume (OBV)
            # Mede o fluxo de volume positivo e negativo para prever mudanças de preço.
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()

            # Volume Weighted Average Price (VWAP)
            # Média ponderada pelo volume, útil para identificar níveis de suporte e resistência.
            df['vwap'] = ta.volume.VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=config.vwap_window
            ).volume_weighted_average_price()

            ## Outros indicadores
            # Average Directional Index (ADX)
            df['adx'] = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.adx_window
            ).adx()

            # Rate of Change (ROC)
            # Mede a variação percentual do preço em relação ao preço de um período anterior.
            df['roc'] = ta.momentum.ROCIndicator(
                close=df['close'],
                window=config.roc_window
            ).roc()

            # Garantia de remoção de NaN
            df.dropna(inplace=True)

            logger.info("Indicadores técnicos adicionados com sucesso.")

            return df
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}")
            return df


class LabelConfig(BaseModel):
    """Configuração para criação de labels"""
    horizon: int = Field(12, gt=0, description="Horizonte de previsão em períodos")
    min_price_move: float = Field(0.5, description="Movimento mínimo percentual para considerar sinal")


class LabelCreator:
    @classmethod
    def create_labels(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels para TP e SL com base no movimento de preço futuro,
        com ajustes para garantir melhor razão risk/reward.
        """
        try:
            logger.info(f"Criando labels para TP e SL com horizon={settings.MODEL_DATA_PREDICTION_HORIZON} períodos.")

            # 1. Calcular futuros máximos e mínimos
            df['future_high'] = df['high'].rolling(
                window=settings.MODEL_DATA_PREDICTION_HORIZON
            ).max().shift(-settings.MODEL_DATA_PREDICTION_HORIZON)

            df['future_low'] = df['low'].rolling(
                window=settings.MODEL_DATA_PREDICTION_HORIZON
            ).min().shift(-settings.MODEL_DATA_PREDICTION_HORIZON)

            # 2. Calcular ATR para uso no ajuste dinâmico de SL
            atr = ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range()

            # 3. Calcular take profit e stop loss percentuais básicos
            df['raw_tp_pct'] = ((df['future_high'] - df['close']) / df['close']) * 100
            df['raw_sl_pct'] = ((df['close'] - df['future_low']) / df['close']) * 100

            # 4. Calcular stop loss dinâmico baseado em ATR (mais realista)
            df['atr_sl_pct'] = (atr / df['close']) * 100 * 1.5  # 1.5x ATR para SL

            # 5. Usar o menor entre o SL baseado em preço futuro e o SL baseado em ATR
            df['stop_loss_pct'] = df[['raw_sl_pct', 'atr_sl_pct']].min(axis=1)

            # 6. Ajustar o take profit para garantir razão R:R mínima de 1.5:1
            min_rr_ratio = 1.5
            df['take_profit_pct'] = np.maximum(
                df['raw_tp_pct'],  # TP original
                df['stop_loss_pct'] * min_rr_ratio  # TP mínimo para garantir razão R:R
            )

            # 7. Adicionar features de relação entre take profit e stop loss
            df['tp_sl_ratio'] = df['take_profit_pct'] / df['stop_loss_pct']

            # 8. Adicionar sinalizador de qualidade de trade
            df['trade_quality'] = ((df['tp_sl_ratio'] > min_rr_ratio) &
                                   (df['take_profit_pct'] > 0.5) &  # Mínimo de 0.5% de movimento
                                   (df['stop_loss_pct'] < 2.0)).astype(float)  # Máximo de 2% de risco

            # Limpar valores infinitos e NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Remover colunas temporárias
            df.drop(['future_high', 'future_low', 'raw_tp_pct', 'raw_sl_pct', 'atr_sl_pct'],
                    axis=1, inplace=True)

            # Remover linhas com valores NaN
            df.dropna(inplace=True)

            logger.info("Labels para TP e SL criados com sucesso com razão R:R aprimorada.")

            # Verificar estatísticas da razão R:R
            avg_ratio = df['tp_sl_ratio'].mean()
            median_ratio = df['tp_sl_ratio'].median()
            logger.info(f"Razão R:R média: {avg_ratio:.2f}, Mediana: {median_ratio:.2f}")

            # Verificar percentual de trades de qualidade
            quality_pct = df['trade_quality'].mean() * 100
            logger.info(f"Percentual de trades com boa qualidade: {quality_pct:.2f}%")

            return df
        except Exception as e:
            logger.error(f"Erro ao criar labels: {e}")
            return df
