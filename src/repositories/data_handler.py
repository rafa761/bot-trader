# repositories\data_handler.py

import asyncio
import re
import threading
from pathlib import Path

import arrow
import numpy as np
import pandas as pd
from binance import Client
from binance.exceptions import BinanceAPIException
from pydantic import BaseModel, Field
from ta.momentum import ROCIndicator, RSIIndicator, StochRSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, PSARIndicator, SMAIndicator, WMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from core.config import settings
from core.constants import TRAIN_DATA_DIR
from core.logger import logger
from services.binance.binance_client import BinanceClient


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
                df = pd.DataFrame(
                    klines, columns=[
                        "timestamp", "open", "high", "low", "close", "volume",
                        "close_time", "quote_asset_volume", "number_of_trades",
                        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
                    ]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(
                    float
                )
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
                    logger.info("Primeiro registro adicionado ao DataFrame histórico.")
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

    def get_historical_klines(self, add_indicators=True) -> pd.DataFrame:
        """
        Obtém dados históricos da Binance e opcionalmente adiciona indicadores técnicos.

        Args:
            add_indicators: Se True, adiciona indicadores técnicos. Se False, retorna apenas dados OHLCV.

        Returns:
            pd.DataFrame: DataFrame com dados históricos
        """
        try:
            filepath = self.check_existing_data()
            if filepath:
                logger.info(f"Arquivo encontrado em cache: {filepath}. Carregando dados do arquivo.")
                df = pd.read_csv(filepath, sep=';', encoding='utf-8', parse_dates=['timestamp'], index_col='timestamp')
                # Se temos dados de cache mas não queremos indicadores,
                # verificamos se há colunas além de OHLCV e as removemos
                if not add_indicators and len(df.columns) > 5:
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                return df

            logger.info(
                f"Coletando dados para {self.config.symbol} - "
                f"Intervalo: {self.config.interval} - "
                f"Início: {self.config.start_str}"
            )

            klines = self.client.get_historical_klines(
                self.config.symbol,
                self.config.interval,
                self.config.start_str,
                self.config.end_str
            )
            df = self._process_klines(klines)

            logger.info(f"Coleta concluída: {len(df)} registros.")

            # Adiciona indicadores técnicos apenas se solicitado
            if add_indicators:
                df = TechnicalIndicatorAdder.add_technical_indicators(df)
                # Salva o CSV atualizado com indicadores técnicos
                self.save_to_csv(df)
            else:
                # Opcionalmente, podemos salvar os dados brutos com um sufixo diferente
                # self.save_to_csv(df, suffix="_raw")
                pass

            return df
        except BinanceAPIException as e:
            logger.error(f"Erro ao coletar dados: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro inesperado: {e}")
            return pd.DataFrame()

    @staticmethod
    def _process_klines(klines: list) -> pd.DataFrame:
        """Converte os dados da Binance para DataFrame."""
        df = pd.DataFrame(
            klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
        )
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
    """Configuração otimizada para indicadores técnicos em day trading de 15 minutos"""
    # Médias móveis mais curtas e reativas
    ema_windows: tuple[int, int] = (3, 10, 50)

    # Hull Moving Average - excelente para day trading
    hull_window: int = 7

    # Indicadores de volatilidade otimizados
    bollinger_window: int = 14
    bollinger_std: float = 2.0
    atr_window: int = 10

    # Osciladores otimizados para day trading
    rsi_window: int = 5
    stoch_k_window: int = 7
    stoch_d_window: int = 3
    stoch_smooth_k: int = 2

    # MACD otimizado para 15min
    macd_windows: tuple[int, int, int] = (12, 5, 7)  # Ajustado para maior sensibilidade

    vwap_window: int = 14
    adx_window: int = 8

    # Supertrend - crítico para day trading
    supertrend_atr_multiplier: float = 2.0  # Ajustado para ser mais reativo
    supertrend_atr_period: int = 7  # Reduzido para maior sensibilidade

    # Outros parâmetros
    heikin_ashi_enabled: bool = True  # Manter - excelente para clareza em 15min
    pivot_lookback: int = 3  # Reduzido para capturar pivôs mais recentes


class TechnicalIndicatorAdder:
    @classmethod
    def add_technical_indicators(cls, df: pd.DataFrame, config: TechnicalIndicatorConfig | None = None) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos otimizados para day trading de 15 minutos ao DataFrame.

        :param df: DataFrame com dados OHLCV
        :param config: Configuração opcional para indicadores técnicos
        :return: DataFrame com indicadores técnicos adicionados
        """
        if config is None:
            config = TechnicalIndicatorConfig()

        # Definir o tamanho mínimo necessário para os indicadores
        min_size = max(
            config.bollinger_window,
            config.ema_windows[1],
            config.rsi_window,
            config.stoch_k_window + config.stoch_d_window,
            config.macd_windows[0],
            config.atr_window,
            config.vwap_window,
            config.adx_window,
            config.supertrend_atr_period + 5,  # SuperTrend
            config.pivot_lookback * 2  # Pivôs intradiários
        )

        # Verificar se há dados suficientes
        if len(df) < min_size:
            logger.warning(
                f"DataFrame com tamanho insuficiente para calcular indicadores. "
                f"Necessário: {min_size}, Disponível: {len(df)}"
            )
            return df

        try:
            logger.info("Adicionando indicadores técnicos otimizados para day trading ao DataFrame.")

            # Cópias do DataFrame para cálculos específicos
            df_copy = df.copy()

            # Velas Heikin Ashi para análise de momentum mais clara
            if config.heikin_ashi_enabled:
                # Cálculo das velas Heikin Ashi
                df['ha_open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
                df.loc[df.index[0], 'ha_open'] = (df.loc[df.index[0], 'open'] + df.loc[df.index[0], 'close']) / 2

                df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
                df['ha_high'] = df[['high', 'ha_open', 'ha_close']].max(axis=1)
                df['ha_low'] = df[['low', 'ha_open', 'ha_close']].min(axis=1)

            # Exponential Moving Average (EMA) ajustadas para day trading
            df['ema_short'] = EMAIndicator(
                close=df['close'],
                window=config.ema_windows[0]
            ).ema_indicator()

            df['ema_long'] = EMAIndicator(
                close=df['close'],
                window=config.ema_windows[1]
            ).ema_indicator()

            df['ema_50'] = EMAIndicator(
                close=df['close'],
                window=config.ema_windows[2]
            ).ema_indicator()

            # Hull Moving Average (HMA) - Mais rápida que EMA, menos lag
            half_window = config.ema_windows[0] // 2
            sqrt_window = int(np.sqrt(config.ema_windows[0]))

            # Passo 1: Calcular WMA com metade do período
            wma_half = WMAIndicator(
                close=df['close'],
                window=half_window
            ).wma()

            # Passo 2: Calcular WMA com período completo
            wma_full = WMAIndicator(
                close=df['close'],
                window=config.ema_windows[0]
            ).wma()

            # Passo 3: Calcular 2*WMA_half - WMA_full
            df['raw_hma'] = 2 * wma_half - wma_full

            # Passo 4: Calcular WMA final com período sqrt
            df['hma'] = WMAIndicator(
                close=df['raw_hma'],
                window=sqrt_window
            ).wma()

            # Parabolic SAR - Mais sensível para day trading
            df['parabolic_sar'] = PSARIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                step=0.02,  # Mais reativo (padrão é 0.02)
                max_step=0.2  # Limite mais alto (padrão é 0.2)
            ).psar()

            # SuperTrend - Excelente para identificar tendências em day trading
            # Cálculo do ATR para o SuperTrend
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.supertrend_atr_period
            ).average_true_range()

            # Cálculo das bandas do SuperTrend
            upper_band = ((df['high'] + df['low']) / 2) + (config.supertrend_atr_multiplier * atr)
            lower_band = ((df['high'] + df['low']) / 2) - (config.supertrend_atr_multiplier * atr)

            # Inicialização do SuperTrend
            supertrend = pd.Series(0.0, index=df.index)
            direction = pd.Series(1, index=df.index)  # 1 para alta, -1 para baixa

            # Cálculo do SuperTrend
            for i in range(1, len(df)):
                if df['close'].iloc[i] > upper_band.iloc[i - 1]:
                    direction.iloc[i] = 1
                elif df['close'].iloc[i] < lower_band.iloc[i - 1]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i - 1]

                    if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i - 1]:
                        lower_band.iloc[i] = lower_band.iloc[i - 1]
                    if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i - 1]:
                        upper_band.iloc[i] = upper_band.iloc[i - 1]

                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = upper_band.iloc[i]

            df['supertrend'] = supertrend
            df['supertrend_direction'] = direction.astype(float)  # Usar float em vez de int para evitar problemas

            ## Indicadores de Volatilidade
            # Bollinger Bands com ajustes para day trading
            bollinger = BollingerBands(
                close=df['close'],
                window=config.bollinger_window,
                window_dev=config.bollinger_std
            )
            df["boll_hband"] = bollinger.bollinger_hband()
            df["boll_lband"] = bollinger.bollinger_lband()
            df["boll_mavg"] = bollinger.bollinger_mavg()
            df['boll_width'] = (df['boll_hband'] - df['boll_lband']) / df['boll_mavg']
            df['boll_pct_b'] = (df['close'] - df['boll_lband']) / (df['boll_hband'] - df['boll_lband'])

            # ATR para medir volatilidade - crítico para day trading
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.atr_window
            ).average_true_range()

            # ATR percentual (ATR relativo ao preço) - mais útil para decisões em day trading
            df['atr_pct'] = (df['atr'] / df['close']) * 100

            # Squeeze Momentum Indicator
            # Identifica quando o mercado está "comprimido" e prestes a explodir
            df['squeeze'] = ((df['boll_width'] / df['boll_mavg']) < 0.1).astype(float)

            # TTM Squeeze momentum
            df['ttm_squeeze'] = ROCIndicator(
                close=df['close'] - ((df['high'] + df['low']) / 2),
                window=20
            ).roc()

            ## Indicadores de Momento
            # RSI otimizado para day trading (período menor)
            df['rsi'] = RSIIndicator(
                close=df['close'],
                window=config.rsi_window
            ).rsi()

            # RSI Divergence (2 períodos)
            df['rsi_prev'] = df['rsi'].shift(2)
            df['price_prev'] = df['close'].shift(2)
            df['rsi_divergence_bull'] = ((df['close'] < df['price_prev']) & (df['rsi'] > df['rsi_prev'])).astype(float)
            df['rsi_divergence_bear'] = ((df['close'] > df['price_prev']) & (df['rsi'] < df['rsi_prev'])).astype(float)

            # Stochastic Oscillator ajustado para day trading
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.stoch_k_window,
                smooth_window=config.stoch_smooth_k
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = df['stoch_k'].rolling(window=config.stoch_d_window).mean()

            # Stochastic RSI - hibridação útil para day trading
            df['stoch_rsi'] = StochRSIIndicator(
                close=df['close'],
                window=config.rsi_window,
                smooth1=3,
                smooth2=3
            ).stochrsi()

            # MACD otimizado para day trading
            macd = MACD(
                close=df['close'],
                window_slow=config.macd_windows[0],
                window_fast=config.macd_windows[1],
                window_sign=config.macd_windows[2]
            )
            df['macd'] = macd.macd_diff()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            ## Indicadores de Volume e Fluxo de Dinheiro
            # On-Balance Volume (OBV)
            df['obv'] = OnBalanceVolumeIndicator(
                close=df['close'],
                volume=df['volume']
            ).on_balance_volume()

            # Chaikin Money Flow - crucial para day trading
            df['cmf'] = ChaikinMoneyFlowIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=20
            ).chaikin_money_flow()

            # Volume Weighted Average Price (VWAP) - essencial para day trading
            df['vwap'] = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=config.vwap_window
            ).volume_weighted_average_price()

            # VWAP Distance - posição relativa do preço vs VWAP
            df['vwap_distance'] = ((df['close'] - df['vwap']) / df['vwap']) * 100

            ## Indicadores de Tendência e Direção
            # Average Directional Index (ADX) - mais sensível
            adx = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.adx_window
            )
            df['adx'] = adx.adx()
            df['di_plus'] = adx.adx_pos()
            df['di_minus'] = adx.adx_neg()

            # ADX com classificação de força da tendência (como string, não categórica)
            df['trend_strength'] = 'Moderada'  # Valor padrão
            df.loc[df['adx'] <= 15, 'trend_strength'] = 'Ausente'
            df.loc[(df['adx'] > 15) & (df['adx'] <= 25), 'trend_strength'] = 'Fraca'
            df.loc[(df['adx'] > 25) & (df['adx'] <= 35), 'trend_strength'] = 'Moderada'
            df.loc[(df['adx'] > 35) & (df['adx'] <= 50), 'trend_strength'] = 'Forte'
            df.loc[df['adx'] > 50, 'trend_strength'] = 'Extrema'

            # Pivôs intradiários - essenciais para day trading
            df['pivot'] = np.nan
            df['pivot_r1'] = np.nan
            df['pivot_r2'] = np.nan
            df['pivot_s1'] = np.nan
            df['pivot_s2'] = np.nan

            for i in range(config.pivot_lookback, len(df)):
                # Cálculo do pivot point para o último pivô_lookback
                window = df.iloc[i - config.pivot_lookback:i]
                pivot = (window['high'].max() + window['low'].min() + window['close'].iloc[-1]) / 3
                r1 = 2 * pivot - window['low'].min()
                s1 = 2 * pivot - window['high'].max()
                r2 = pivot + (window['high'].max() - window['low'].min())
                s2 = pivot - (window['high'].max() - window['low'].min())

                df.loc[df.index[i], 'pivot'] = pivot
                df.loc[df.index[i], 'pivot_r1'] = r1
                df.loc[df.index[i], 'pivot_r2'] = r2
                df.loc[df.index[i], 'pivot_s1'] = s1
                df.loc[df.index[i], 'pivot_s2'] = s2

            # Detecção de suporte e resistência intradiários
            price_shift = df['close'].shift(1)
            df['pivot_resistance'] = ((df['close'] < df['pivot_r1']) & (price_shift >= df['pivot_r1']) |
                                      (df['close'] < df['pivot_r2']) & (price_shift >= df['pivot_r2'])).astype(float)
            df['pivot_support'] = ((df['close'] > df['pivot_s1']) & (price_shift <= df['pivot_s1']) |
                                   (df['close'] > df['pivot_s2']) & (price_shift <= df['pivot_s2'])).astype(float)

            # Detecção de Fases de Mercado
            # 0: Range, 1: Tendência Alta, -1: Tendência Baixa
            df['market_phase'] = 0.0  # Inicializar como range (lateral)

            # Detectar tendência baseado em vários indicadores
            df.loc[(df['close'] > df['ema_long']) &
                   (df['ema_short'] > df['ema_long']) &
                   (df['adx'] > 20) &
                   (df['di_plus'] > df['di_minus']), 'market_phase'] = 1.0  # Tendência de alta

            df.loc[(df['close'] < df['ema_long']) &
                   (df['ema_short'] < df['ema_long']) &
                   (df['adx'] > 20) &
                   (df['di_minus'] > df['di_plus']), 'market_phase'] = -1.0  # Tendência de baixa

            # Volatilidade classificada (como string, não categórica)
            df['volatility_class'] = 'Média'  # Valor padrão
            df.loc[df['atr_pct'] <= 0.5, 'volatility_class'] = 'Muito Baixa'
            df.loc[(df['atr_pct'] > 0.5) & (df['atr_pct'] <= 1.0), 'volatility_class'] = 'Baixa'
            df.loc[(df['atr_pct'] > 1.0) & (df['atr_pct'] <= 1.5), 'volatility_class'] = 'Média'
            df.loc[(df['atr_pct'] > 1.5) & (df['atr_pct'] <= 2.0), 'volatility_class'] = 'Alta'
            df.loc[df['atr_pct'] > 2.0, 'volatility_class'] = 'Extrema'

            # Classificação de volume (comparado com média móvel de volume)
            df['volume_sma'] = SMAIndicator(close=df['volume'], window=20).sma_indicator()
            df['volume_ratio'] = df['volume'] / df['volume_sma']

            # Classificação como string em vez de categórica
            df['volume_class'] = 'Normal'  # Valor padrão
            df.loc[df['volume_ratio'] <= 0.5, 'volume_class'] = 'Muito Baixo'
            df.loc[(df['volume_ratio'] > 0.5) & (df['volume_ratio'] <= 0.8), 'volume_class'] = 'Baixo'
            df.loc[(df['volume_ratio'] > 0.8) & (df['volume_ratio'] <= 1.2), 'volume_class'] = 'Normal'
            df.loc[(df['volume_ratio'] > 1.2) & (df['volume_ratio'] <= 2.0), 'volume_class'] = 'Alto'
            df.loc[df['volume_ratio'] > 2.0, 'volume_class'] = 'Muito Alto'

            # Limpar valores infinitos e NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Remover colunas temporárias e auxiliares
            df.drop(
                ['raw_hma', 'rsi_prev', 'price_prev'],
                axis=1, errors='ignore', inplace=True
            )

            # Preencher NaN com forward fill e depois com backward fill
            # Usando sintaxe moderna recomendada
            df = df.ffill()
            df = df.bfill()

            # Se houver NaNs restantes, substituir por zeros (apenas para colunas numéricas)
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(0.0)

            # Para colunas de string, preencher com valores vazios
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna('')

            logger.info("Indicadores técnicos otimizados para day trading adicionados com sucesso.")

            return df
        except Exception as e:
            logger.error(f"Erro ao adicionar indicadores técnicos: {e}", exc_info=True)
            return df


class LabelConfig(BaseModel):
    """Configuração otimizada para criação de labels em day trading de 15 minutos"""
    # Horizonte baseado na configuração existente
    base_horizon: int = Field(
        6, gt=0,
        description="Horizonte de previsão base em períodos (do settings)"
    )
    min_horizon: int = Field(
        4, gt=0,
        description="Horizonte mínimo para mercados voláteis"
    )
    max_horizon: int = Field(
        10, gt=0,
        description="Horizonte máximo para mercados lentos"
    )

    # Configurações de movimento de preço
    min_price_move: float = Field(0.35, description="Movimento mínimo percentual para considerar sinal")

    # Take profit e stop loss
    base_rr_ratio: float = Field(
        1.5,
        description="Razão base entre take profit e stop loss (do settings)"
    )
    trend_rr_ratio: float = Field(
        2.0,
        description="Razão R:R otimizada para mercados em tendência"
    )
    range_rr_ratio: float = Field(
        1.2,
        description="Razão R:R otimizada para mercados em range"
    )

    # ATR para stops dinâmicos
    atr_sl_multiplier: float = Field(
        1.5,
        description="Multiplicador do ATR para stop loss (do settings)"
    )

    # Take profit parcial
    tp_levels: list[float] = Field(
        [0.382, 0.618, 1.0],
        description="Níveis de take profit (% do alvo total)"
    )
    partial_tp_weights: list[float] = Field(
        [0.3, 0.4, 0.3],
        description="Pesos para cada nível de TP"
    )

    # Qualificadores de trade
    max_sl_pct: float = Field(1.5, description="Máximo stop loss percentual para day trading")
    min_tp_pct: float = Field(0.5, description="Mínimo take profit percentual para day trading")

    # Ajuste dos limiares de qualidade baseados nas configurações existentes
    quality_threshold: float = Field(
        0.6,
        description="Limiar de qualidade para considerar um trade válido"
    )

    # Flags de condição de mercado
    trend_aligned_bonus: float = Field(
        0.2,
        description="Bônus de qualidade para trades alinhados com a tendência"
    )
    counter_trend_penalty: float = Field(
        0.3,
        description="Penalidade para trades contra a tendência"
    )
    volatility_adjustment: float = Field(
        0.15,
        description="Ajuste para condições de volatilidade"
    )

    class Config:
        """Configuração adicional do modelo Pydantic"""
        arbitrary_types_allowed = True
        validate_assignment = True

    def __init__(self, **data):
        """
        Inicialização com valores ajustados baseados em settings
        """
        # Usar valores explícitos ao invés de funções lambda
        if 'base_horizon' not in data:
            data['base_horizon'] = int(settings.MODEL_DATA_PREDICTION_HORIZON)

        if 'base_horizon' in data and 'min_horizon' not in data:
            data['min_horizon'] = max(4, int(data['base_horizon'] * 0.7))

        if 'base_horizon' in data and 'max_horizon' not in data:
            data['max_horizon'] = max(8, int(data['base_horizon'] * 1.5))

        # Configurações relacionadas ao R:R
        if 'base_rr_ratio' not in data:
            data['base_rr_ratio'] = float(settings.MIN_RR_RATIO)

        if 'base_rr_ratio' in data and 'trend_rr_ratio' not in data:
            data['trend_rr_ratio'] = float(data['base_rr_ratio'] * 1.2)

        if 'base_rr_ratio' in data and 'range_rr_ratio' not in data:
            data['range_rr_ratio'] = max(1.0, float(data['base_rr_ratio'] * 0.9))

        # Configuração do multiplicador ATR
        if 'atr_sl_multiplier' not in data:
            data['atr_sl_multiplier'] = float(settings.ATR_MULTIPLIER)

        # Limiares de qualidade
        if 'quality_threshold' not in data:
            data['quality_threshold'] = float(settings.ENTRY_THRESHOLD_DEFAULT)

        # Chamar o construtor da classe pai
        super().__init__(**data)

        # Verificar e garantir que os valores são dos tipos corretos
        assert isinstance(self.base_horizon, int), f"base_horizon deve ser int, não {type(self.base_horizon)}"
        assert isinstance(self.min_horizon, int), f"min_horizon deve ser int, não {type(self.min_horizon)}"
        assert isinstance(self.max_horizon, int), f"max_horizon deve ser int, não {type(self.max_horizon)}"
        assert isinstance(self.base_rr_ratio, float), f"base_rr_ratio deve ser float, não {type(self.base_rr_ratio)}"


class LabelCreator:
    @classmethod
    def create_labels(cls, df: pd.DataFrame, config: LabelConfig | None = None) -> pd.DataFrame:
        """
        Cria labels otimizados para day trading de 15 minutos, incluindo múltiplos
        níveis de take profit, stop loss dinâmico e qualificadores de trade.

        Args:
            df: DataFrame com dados OHLCV e indicadores técnicos
            config: Configuração opcional para criação de labels

        Returns:
            DataFrame com labels adicionados
        """
        if config is None:
            config = LabelConfig()

        try:
            logger.info(f"Criando labels otimizados para day trading (horizonte base={config.base_horizon})")

            # Cópia do DataFrame para evitar modificações no original
            df_result = df.copy()

            # Usar horizonte fixo para consistência e simplicidade
            horizon = int(config.base_horizon)

            # 1. Calcular máximos e mínimos futuros com horizonte fixo
            df_result['future_high'] = np.nan
            df_result['future_low'] = np.nan
            df_result['used_horizon'] = horizon

            # Calcular máximos e mínimos futuros para cada ponto com verificação adequada de limites
            for i in range(len(df_result) - horizon):
                if i + horizon < len(df_result):
                    df_result.loc[df_result.index[i], 'future_high'] = df_result['high'].iloc[
                                                                       i + 1:i + 1 + horizon].max()
                    df_result.loc[df_result.index[i], 'future_low'] = df_result['low'].iloc[i + 1:i + 1 + horizon].min()

            # 2. Calcular TP/SL percentuais básicos
            df_result['raw_tp_pct'] = ((df_result['future_high'] - df_result['close']) / df_result['close']) * 100
            df_result['raw_sl_pct'] = ((df_result['close'] - df_result['future_low']) / df_result['close']) * 100

            # 3. Obter ATR para referência de volatilidade
            if 'atr' not in df_result.columns:
                atr = AverageTrueRange(
                    high=df_result['high'],
                    low=df_result['low'],
                    close=df_result['close'],
                    window=14
                ).average_true_range()
            else:
                atr = df_result['atr']

            # ATR como percentual do preço para melhor escalabilidade
            atr_pct = (atr / df_result['close']) * 100

            # 4. Definir limites realistas para TP e SL baseados na realidade do mercado
            # - TP máximo: 5%
            # - TP mínimo: ATR% ou 0.25% (o que for maior)
            # - SL máximo: 3%
            # - SL mínimo: 0.5 * ATR% ou 0.15% (o que for maior)

            # Aplicar limites para TP
            max_tp_pct = 5.0  # Máximo 5% para take profit
            min_tp_pct = np.maximum(atr_pct, 0.25)  # No mínimo ATR% ou 0.25%

            df_result['take_profit_pct'] = np.clip(df_result['raw_tp_pct'], min_tp_pct, max_tp_pct)

            # Aplicar limites para SL
            max_sl_pct = 3.0  # Máximo 3% para stop loss
            min_sl_pct = np.maximum(atr_pct * 0.5, 0.15)  # No mínimo metade do ATR% ou 0.15%

            df_result['stop_loss_pct'] = np.clip(df_result['raw_sl_pct'], min_sl_pct, max_sl_pct)

            # 5. Remover outliers estatísticos adicionais
            for col in ['take_profit_pct', 'stop_loss_pct']:
                q1 = df_result[col].quantile(0.25)
                q3 = df_result[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = max(q1 - 1.5 * iqr, 0.15)  # Nunca menor que 0.15%
                upper_bound = min(q3 + 1.5 * iqr, 5.0)  # Nunca maior que 5%

                # Aplicar clipping
                df_result[col] = df_result[col].clip(lower_bound, upper_bound)

            # 6. Calcular razão TP/SL (R:R) com limites razoáveis
            df_result['tp_sl_ratio'] = df_result['take_profit_pct'] / df_result['stop_loss_pct']
            df_result['tp_sl_ratio'] = df_result['tp_sl_ratio'].clip(0.5, 5.0)

            # 7. Gerar sinais de trade baseados em critérios de R:R
            min_rr = config.base_rr_ratio
            df_result['trade_quality'] = np.where(df_result['tp_sl_ratio'] >= min_rr, 0.7, 0.3)
            df_result['trade_signal'] = np.where(df_result['tp_sl_ratio'] >= min_rr, 1, 0)

            # 8. Criar sinais direcionais (long/short) baseados em qual movimento é mais forte
            df_result['long_signal'] = (
                    (df_result['trade_signal'] == 1) &
                    (df_result['raw_tp_pct'] > df_result['raw_sl_pct'])
            ).astype(float)
            df_result['short_signal'] = (
                    (df_result['trade_signal'] == 1) &
                    (df_result['raw_tp_pct'] < df_result['raw_sl_pct'])
            ).astype(float)

            # 9. Adicionar alinhamento com supertrend se disponível
            if 'supertrend_direction' in df_result.columns:
                df_result['aligned_with_supertrend'] = (
                        (df_result['long_signal'] == 1) & (df_result['supertrend_direction'] == 1) |
                        (df_result['short_signal'] == 1) & (df_result['supertrend_direction'] == -1)
                ).astype(float)

            # 10. Limpar valores infinitos, NaN e colunas temporárias
            df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
            df_result.drop(['raw_tp_pct', 'raw_sl_pct'], axis=1, errors='ignore', inplace=True)

            # Preencher NaN com forward fill e depois com backward fill
            df_result = df_result.ffill().bfill()

            # Estatísticas de qualidade
            valid_data = df_result.dropna(subset=['take_profit_pct', 'stop_loss_pct'])
            if not valid_data.empty:
                avg_tp = valid_data['take_profit_pct'].mean()
                avg_sl = valid_data['stop_loss_pct'].mean()
                median_ratio = valid_data['tp_sl_ratio'].median()
                signal_pct = valid_data['trade_signal'].mean() * 100
                logger.info(f"TP médio: {avg_tp:.2f}%, SL médio: {avg_sl:.2f}%, Razão R:R mediana: {median_ratio:.2f}")
                logger.info(f"Percentual de sinais gerados: {signal_pct:.2f}%")

                if 'aligned_with_supertrend' in valid_data.columns:
                    aligned_pct = valid_data['aligned_with_supertrend'].sum() / valid_data[
                        'trade_signal'].sum() * 100 if valid_data['trade_signal'].sum() > 0 else 0
                    logger.info(f"Percentual de sinais alinhados com tendência: {aligned_pct:.2f}%")

            return df_result

        except Exception as e:
            logger.error(f"Erro ao criar labels otimizados: {e}", exc_info=True)
            # Caso falhe, usar a abordagem mais simples
            try:
                logger.warning("Tentando método alternativo de criação de labels")
                return cls.create_labels_simple(df)
            except Exception as backup_error:
                logger.error(f"Erro ao criar labels (método alternativo): {backup_error}")
                return df

    @classmethod
    def create_labels_simple(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Método simplificado de criação de labels quando o método principal falha.
        Cria take_profit_pct e stop_loss_pct com base em máximos e mínimos futuros fixos.

        Args:
            df: DataFrame com dados OHLCV

        Returns:
            DataFrame com labels básicos adicionados
        """
        logger.info(f"Criando labels simples com horizon={settings.MODEL_DATA_PREDICTION_HORIZON} períodos.")

        # Nova cópia para evitar modificações no original
        df_result = df.copy()

        # Usando horizonte fixo do settings
        horizon = settings.MODEL_DATA_PREDICTION_HORIZON

        # Calcular futuros máximos e mínimos
        future_high = []
        future_low = []

        for i in range(len(df_result)):
            if i + horizon < len(df_result):
                future_high.append(df_result['high'].iloc[i + 1:i + 1 + horizon].max())
                future_low.append(df_result['low'].iloc[i + 1:i + 1 + horizon].min())
            else:
                future_high.append(np.nan)
                future_low.append(np.nan)

        df_result['future_high'] = future_high
        df_result['future_low'] = future_low

        # Calcular take profit e stop loss percentuais
        df_result['take_profit_pct'] = ((df_result['future_high'] - df_result['close']) / df_result['close']) * 100
        df_result['stop_loss_pct'] = ((df_result['close'] - df_result['future_low']) / df_result['close']) * 100

        # Calcular razão R:R
        df_result['tp_sl_ratio'] = df_result['take_profit_pct'] / df_result['stop_loss_pct']

        # Remover valores infinitos e NaN
        df_result['raw_tp_pct'] = ((df_result['future_high'] - df_result['close']) / df_result['close']) * 100
        df_result['raw_sl_pct'] = ((df_result['close'] - df_result['future_low']) / df_result['close']) * 100

        # Aplicar limites razoáveis
        df_result['take_profit_pct'] = df_result['raw_tp_pct'].clip(0.25, 5.0)
        df_result['stop_loss_pct'] = df_result['raw_sl_pct'].clip(0.15, 3.0)

        # Calcular razão R:R
        df_result['tp_sl_ratio'] = (df_result['take_profit_pct'] / df_result['stop_loss_pct']).clip(0.5, 5.0)

        # Remover valores infinitos e NaN
        df_result.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_result = df_result.ffill().bfill()

        # Remover colunas temporárias
        df_result.drop(
            ['raw_tp_pct', 'raw_sl_pct', 'future_high', 'future_low'],
            axis=1, errors='ignore', inplace=True
        )

        logger.info("Labels simples criados com sucesso.")

        return df_result
