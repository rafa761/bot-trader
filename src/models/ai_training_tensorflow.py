from pathlib import Path

import numpy as np
import pandas as pd
import ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.metrics import mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers

from core.config import config
from core.constants import FEATURE_COLUMNS
from core.logger import logger


# ------------------------------------------
# 1) Classes de coleta, indicadores e labels
# ------------------------------------------
class DataCollector:
    def __init__(self, client: Client):
        self.client = client

    def get_historical_klines(self, symbol: str, interval: str, start_str: str,
                              end_str: str | None = None) -> pd.DataFrame:
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
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14
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


# ------------------------------------------
# 2) Manipulação especial para LSTM (janelas)
# ------------------------------------------
class DataSplitterLSTM:
    @staticmethod
    def create_sliding_windows(df: pd.DataFrame,
                               feature_columns: list[str],
                               window_size: int,
                               horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cria janelas de tamanho 'window_size' a partir do DataFrame (para dados temporais),
        de modo que cada X[i] seja uma matriz (window_size x n_features),
        e y_tp[i], y_sl[i] sejam as labels referentes à posição i + window_size.

        Se preferir prever a média ou max/min no horizonte, pode ajustar os índices.
        """
        logger.info("Criando janelas (sliding windows) para dados LSTM...")
        X, y_tp, y_sl = [], [], []
        for i in range(len(df) - window_size - horizon):
            # Janela de features
            window_data = df[feature_columns].iloc[i: i + window_size].values
            X.append(window_data)

            # Usar o valor de TP/SL logo após a janela (ex: i + window_size)
            # ou alguma agregação no horizonte [i+window_size : i+window_size+horizon]
            future_tp = df['TP_pct'].iloc[i + window_size]
            future_sl = df['SL_pct'].iloc[i + window_size]

            y_tp.append(future_tp)
            y_sl.append(future_sl)

        X = np.array(X, dtype=np.float32)
        y_tp = np.array(y_tp, dtype=np.float32)
        y_sl = np.array(y_sl, dtype=np.float32)
        logger.info(f"Total de amostras após criar janelas: {len(X)}")
        return X, y_tp, y_sl

    @staticmethod
    def split_data_timebased(X, y_tp, y_sl, test_size=0.2):
        """
        Divide em treino e teste respeitando a ordem temporal (sem shuffle).
        """
        n = len(X)
        split_index = int((1 - test_size) * n)

        X_train = X[:split_index]
        X_test = X[split_index:]
        y_tp_train = y_tp[:split_index]
        y_tp_test = y_tp[split_index:]
        y_sl_train = y_sl[:split_index]
        y_sl_test = y_sl[split_index:]

        return X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test


# ------------------------------------------
# 3) Classe de Treinamento com LSTM
# ------------------------------------------
class ModelTrainerLSTM:
    def __init__(self, train_data_dir: Path):
        self.train_data_dir = train_data_dir

    def _build_lstm_model(self, window_size: int, n_features: int) -> keras.Model:
        """
        Constrói e compila um modelo LSTM simples para regressão de TP ou SL.
        """
        model = keras.Sequential()
        model.add(layers.Input(shape=(window_size, n_features)))
        model.add(layers.LSTM(64, return_sequences=False))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    def train_lstm_model(self,
                         X_train: np.ndarray,
                         y_train: np.ndarray,
                         X_val: np.ndarray,
                         y_val: np.ndarray,
                         window_size: int,
                         n_features: int,
                         model_name: str,
                         epochs: int = 50,
                         batch_size: int = 32) -> keras.Model | None:
        """
        Treina o modelo LSTM e salva em disco.
        """
        try:
            logger.info(f"Iniciando treinamento do modelo LSTM para {model_name}...")

            model = self._build_lstm_model(window_size, n_features)

            # Callbacks para early stopping e para salvar o melhor checkpoint
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            checkpoint_path = self.train_data_dir / f"model_{model_name}_lstm.h5"
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop, checkpoint],
                verbose=1
            )

            logger.info(f"Modelo LSTM '{model_name}' treinado e salvo em {checkpoint_path}.")
            return model
        except Exception as e:
            logger.error(f"Erro ao treinar o modelo LSTM para {model_name}: {e}")
            return None

    def evaluate_lstm_model(self,
                            model: keras.Model,
                            X_test: np.ndarray,
                            y_test: np.ndarray,
                            model_name: str) -> float:
        """
        Avalia o modelo LSTM usando o conjunto de teste.
        """
        try:
            logger.info(f"Avaliação do modelo LSTM para {model_name}...")
            y_pred = model.predict(X_test).flatten()
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Mean Absolute Error (MAE) para {model_name}: {mae:.4f}")

            # Exemplo de cálculo de "assertividade de direção"
            total_count = len(y_test)
            correct_direction_count = 0
            for real, pred in zip(y_test, y_pred):
                if (real >= 0 and pred >= 0) or (real < 0 and pred < 0):
                    correct_direction_count += 1
            accuracy_direction = (correct_direction_count / total_count) if total_count else 0

            logger.info(f"[{model_name}] Assertividade de direção: {accuracy_direction:.2%}")

            return mae
        except Exception as e:
            logger.error(f"Erro ao avaliar o modelo LSTM para {model_name}: {e}")
            return float('inf')


# ------------------------------------------
# 4) Fluxo Principal
# ------------------------------------------
def main():
    train_data_dir = Path('../train_data')
    train_data_dir.mkdir(parents=True, exist_ok=True)

    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, requests_params={"timeout": 20})

    # 1) Coleta de dados
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines(config.SYMBOL, config.INTERVAL, config.MODEL_DATA_TRAINING_START_DATE)
    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando o script.")
        return

    # 2) Adiciona indicadores técnicos
    df = TechnicalIndicatorAdder.add_technical_indicators(df)

    # 3) Cria labels (TP_pct, SL_pct)
    df = LabelCreator.create_labels(df, config.MODEL_DATA_PREDICTION_HORIZON)
    if df.empty:
        logger.error("Não foi possível criar labels. Encerrando o script.")
        return

    # 4) Cria janelas de sequência para LSTM
    window_size = 30  # Exemplo: usar 30 candles para "olhar para trás"
    horizon = config.MODEL_DATA_PREDICTION_HORIZON  # 12, conforme config
    X, y_tp, y_sl = DataSplitterLSTM.create_sliding_windows(df, FEATURE_COLUMNS, window_size, horizon)
    if len(X) == 0:
        logger.error("Nenhuma amostra foi gerada após criação das janelas. Encerrando o script.")
        return

    # 5) Divide dados em treino e teste (time-based)
    X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test = DataSplitterLSTM.split_data_timebased(
        X, y_tp, y_sl, test_size=0.2
    )
    if len(X_train) == 0:
        logger.error("Divisão dos dados falhou. Encerrando o script.")
        return

    # 6) Treinamento dos modelos LSTM
    model_trainer = ModelTrainerLSTM(train_data_dir)

    # Treinamos dois modelos: um para TP e outro para SL
    model_tp = model_trainer.train_lstm_model(
        X_train, y_tp_train,
        X_test, y_tp_test,  # usando X_test, y_test como "validação" (simplificado)
        window_size=window_size,
        n_features=len(FEATURE_COLUMNS),
        model_name='tp_lstm'
    )

    model_sl = model_trainer.train_lstm_model(
        X_train, y_sl_train,
        X_test, y_sl_test,
        window_size=window_size,
        n_features=len(FEATURE_COLUMNS),
        model_name='sl_lstm'
    )

    # 7) Avaliação
    if model_tp:
        model_trainer.evaluate_lstm_model(model_tp, X_test, y_tp_test, 'tp_lstm')
    if model_sl:
        model_trainer.evaluate_lstm_model(model_sl, X_test, y_sl_test, 'sl_lstm')

    logger.info("Processo de treinamento com LSTM concluído.")


if __name__ == "__main__":
    main()
