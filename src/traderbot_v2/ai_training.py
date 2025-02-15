from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import pandas as pd
import ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import config
from logger import logger


# ---------------------------- Classes e Funções Refatoradas ----------------------------

class DataCollector:
    def __init__(self, client: Client):
        self.client = client

    def get_historical_klines(self, symbol: str, interval: str, start_str: str,
                              end_str: Optional[str] = None) -> pd.DataFrame:
        """Obtém dados históricos de candles da Binance."""
        try:
            logger.info(f"Coletando dados históricos para {symbol} com intervalo {interval} desde {start_str}")
            klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            logger.info(f"Coleta de dados concluída: {len(df)} registros coletados.")
            return df
        except BinanceAPIException as e:
            logger.error(f"Erro ao coletar dados históricos: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Erro inesperado ao coletar dados históricos: {e}")
            return pd.DataFrame()


class TechnicalIndicatorAdder:
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona indicadores técnicos ao DataFrame."""
        try:
            logger.info("Adicionando indicadores técnicos ao DataFrame.")
            df['sma_short'] = ta.trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
            df['sma_long'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'],
                                                       window=14).average_true_range()
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


class DataPreprocessor:
    @staticmethod
    def preprocess_data(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[
        Optional[StandardScaler], Optional[StandardScaler], Optional[StandardScaler], Optional[pd.DataFrame]]:
        """Normaliza as features e as labels."""
        try:
            logger.info("Iniciando pré-processamento dos dados.")
            X = df[feature_columns]
            y_tp = df['TP_pct']
            y_sl = df['SL_pct']

            scaler_X = StandardScaler()
            scaler_y_tp = StandardScaler()
            scaler_y_sl = StandardScaler()

            X_scaled = scaler_X.fit_transform(X)
            y_tp_scaled = scaler_y_tp.fit_transform(y_tp.values.reshape(-1, 1)).flatten()
            y_sl_scaled = scaler_y_sl.fit_transform(y_sl.values.reshape(-1, 1)).flatten()

            data_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
            data_scaled['TP_pct'] = y_tp_scaled
            data_scaled['SL_pct'] = y_sl_scaled

            logger.info("Pré-processamento concluído com sucesso.")
            return scaler_X, scaler_y_tp, scaler_y_sl, data_scaled
        except Exception as e:
            logger.error(f"Erro no pré-processamento dos dados: {e}")
            return None, None, None, None


class DataSplitter:
    @staticmethod
    def split_data(data_scaled: pd.DataFrame, feature_columns: List[str], test_size: float = 0.2) -> Tuple[
        Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series], Optional[pd.Series],
        Optional[pd.Series]]:
        """Divide os dados em conjuntos de treino e teste."""
        try:
            logger.info("Dividindo os dados em conjuntos de treino e teste.")
            X = data_scaled[feature_columns]
            y_tp = data_scaled['TP_pct']
            y_sl = data_scaled['SL_pct']

            X_train, X_test, y_tp_train, y_tp_test = train_test_split(X, y_tp, test_size=test_size, shuffle=False)
            _, _, y_sl_train, y_sl_test = train_test_split(X, y_sl, test_size=test_size, shuffle=False)

            logger.info("Divisão dos dados concluída.")
            return X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test
        except Exception as e:
            logger.error(f"Erro ao dividir os dados: {e}")
            return None, None, None, None, None, None


class ModelTrainer:
    def __init__(self, train_data_dir: Path):
        self.train_data_dir = train_data_dir

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str) -> Optional[
        RandomForestRegressor]:
        """Treina um modelo de regressão e salva o modelo treinado."""
        try:
            logger.info(f"Iniciando treinamento do modelo para {model_name}.")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            joblib.dump(model, self.train_data_dir / f'model_{model_name}.pkl')
            logger.info(f"Modelo para {model_name} treinado e salvo como model_{model_name}.pkl.")
            return model
        except Exception as e:
            logger.error(f"Erro ao treinar o modelo para {model_name}: {e}")
            return None

    @staticmethod
    def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> \
    Optional[float]:
        """Avalia o modelo usando o conjunto de teste."""
        try:
            logger.info(f"Avaliação do modelo para {model_name}.")
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Mean Absolute Error para {model_name}: {mae:.2f}%")
            return mae
        except Exception as e:
            logger.error(f"Erro ao avaliar o modelo para {model_name}: {e}")
            return None


# ---------------------------- Fluxo Principal ----------------------------

def main():
    # Configurações
    symbol = config.SYMBOL
    interval = config.INTERVAL
    start_date = '1 Jan, 2020'
    horizon = 12
    feature_columns = ['sma_short', 'sma_long', 'rsi', 'atr', 'volume']
    train_data_dir = Path('train_data')
    train_data_dir.mkdir(parents=True, exist_ok=True)

    # Inicializa o cliente da Binance
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, requests_params={"timeout": 20})

    # Coleta de dados
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines(symbol, interval, start_date)
    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando o script.")
        return

    # Adiciona indicadores técnicos
    df = TechnicalIndicatorAdder.add_technical_indicators(df)

    # Cria labels
    df = LabelCreator.create_labels(df, horizon)

    # Pré-processamento
    scaler_X, scaler_y_tp, scaler_y_sl, data_scaled = DataPreprocessor.preprocess_data(df, feature_columns)
    if data_scaled is None:
        logger.error("Pré-processamento falhou. Encerrando o script.")
        return

    # Divisão dos dados
    X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test = DataSplitter.split_data(data_scaled,
                                                                                            feature_columns)
    if X_train is None:
        logger.error("Divisão dos dados falhou. Encerrando o script.")
        return

    # Treinamento dos modelos
    model_trainer = ModelTrainer(train_data_dir)
    model_tp = model_trainer.train_model(X_train, y_tp_train, 'tp')
    model_sl = model_trainer.train_model(X_train, y_sl_train, 'sl')

    if model_tp is None or model_sl is None:
        logger.error("Treinamento dos modelos falhou. Encerrando o script.")
        return

    # Avaliação dos modelos
    mae_tp = model_trainer.evaluate_model(model_tp, X_test, y_tp_test, 'tp')
    mae_sl = model_trainer.evaluate_model(model_sl, X_test, y_sl_test, 'sl')

    # Salva os scalers
    try:
        joblib.dump(scaler_X, train_data_dir / 'scaler_X.pkl')
        joblib.dump(scaler_y_tp, train_data_dir / 'scaler_y_tp.pkl')
        joblib.dump(scaler_y_sl, train_data_dir / 'scaler_y_sl.pkl')
        logger.info("Scalers salvos como scaler_X.pkl, scaler_y_tp.pkl e scaler_y_sl.pkl.")
    except Exception as e:
        logger.error(f"Erro ao salvar scalers: {e}")

    logger.info("Processo de treinamento concluído com sucesso.")


if __name__ == "__main__":
    main()