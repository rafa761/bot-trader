from pathlib import Path

import joblib
import pandas as pd
import ta
from binance.client import Client
from binance.exceptions import BinanceAPIException
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import config
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger


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
                high=df['high'],
                low=df['low'], close=df['close'],
                window=14
            ).average_true_range()
            df["macd"] = ta.trend.macd_diff(df["close"])
            df["boll_hband"] = ta.volatility.bollinger_hband(df["close"], window=20)
            df["boll_lband"] = ta.volatility.bollinger_lband(df["close"], window=20)

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


class DataSplitter:
    @staticmethod
    def split_data(data: pd.DataFrame,
                   feature_columns: list[str],
                   test_size: float = 0.2) -> tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Divide os dados em conjuntos de treino e teste (shuffle=False para simular dados temporais).
        Retorna X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test.
        """
        try:
            logger.info("Dividindo os dados em conjuntos de treino e teste.")

            X = data[feature_columns]
            y_tp = data['TP_pct']
            y_sl = data['SL_pct']

            X_train, X_test, y_tp_train, y_tp_test = train_test_split(X, y_tp, test_size=test_size, shuffle=False)
            _, _, y_sl_train, y_sl_test = train_test_split(X, y_sl, test_size=test_size, shuffle=False)

            logger.info("Divisão dos dados concluída.")

            return X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test
        except Exception as e:
            logger.error(f"Erro ao dividir os dados: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), pd.Series()


class ModelTrainer:
    def __init__(self, train_data_dir: Path):
        self.train_data_dir = train_data_dir

    @staticmethod
    def _build_pipeline(feature_columns: list[str]) -> Pipeline:
        """
        Constrói um pipeline com um ColumnTransformer para escalonar apenas as colunas de interesse,
        seguido de um RandomForestRegressor.
        """
        # Transformador numérico (StandardScaler) aplicado apenas às colunas de features
        numeric_transformer = Pipeline(
            steps=[('scaler', StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feature_columns)
            ]
        )

        # Montamos o pipeline final com o preprocessor + regressão
        pipeline = Pipeline(
            steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ]
        )
        return pipeline

    def train_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            feature_columns: list[str],
            model_name: str
    ) -> Pipeline | None:
        """Treina um modelo (pipeline) e salva-o em disco."""
        try:
            logger.info(f"Iniciando treinamento do modelo para {model_name}...")

            # Cria o pipeline
            model_pipeline = self._build_pipeline(feature_columns)

            # Treina
            model_pipeline.fit(X_train, y_train)

            # Salva em disco
            output_path = self.train_data_dir / f'model_{model_name}.pkl'
            joblib.dump(model_pipeline, output_path)
            logger.info(f"Modelo '{model_name}' treinado e salvo em {output_path}.")

            return model_pipeline
        except Exception as e:
            logger.error(f"Erro ao treinar o modelo para {model_name}: {e}")
            return None

    @staticmethod
    def evaluate_model(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                       model_name: str) -> float | None:
        """Avalia o modelo (pipeline) usando o conjunto de teste e exibe logs adicionais no modo debug."""
        try:
            logger.info(f"Avaliação do modelo para {model_name}...")
            y_pred = model_pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            logger.info(f"Mean Absolute Error para {model_name}: {mae:.4f}")

            # Mostra algumas previsões vs. valores reais no logger.debug
            # (ajusta 'range(5)' para ver mais ou menos exemplos)
            logger.info(f"[{model_name}] Primeiras previsões vs. valores reais:")
            for idx in range(min(5, len(y_test))):
                real = y_test.iloc[idx]
                pred = y_pred[idx]
                logger.info(f"Idx={y_test.index[idx]} | Prev={pred:.4f}, Real={real:.4f}")

            # Exemplo de cálculo de "assertividade de direção"
            total_count = len(y_test)
            correct_direction_count = 0
            for real, pred in zip(y_test, y_pred):
                if (real >= 0 and pred >= 0) or (real < 0 and pred < 0):
                    correct_direction_count += 1
            accuracy_direction = correct_direction_count / total_count if total_count else 0

            logger.info(f"[{model_name}] Assertividade de direção: {accuracy_direction:.2%}")

            return mae
        except Exception as e:
            logger.error(f"Erro ao avaliar o modelo para {model_name}: {e}")
            return None


# ---------------------------- Fluxo Principal ----------------------------
def main():
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, requests_params={"timeout": 20})

    # 1) Coleta de dados
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines(config.SYMBOL, config.INTERVAL, config.MODEL_DATA_TRAINING_START_DATE)
    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando o script.")
        return

    # 2) Adiciona indicadores técnicos
    df = TechnicalIndicatorAdder.add_technical_indicators(df)

    # 3) Cria labels
    df = LabelCreator.create_labels(df, config.MODEL_DATA_PREDICTION_HORIZON)
    if df.empty:
        logger.error("Não foi possível criar labels. Encerrando o script.")
        return

    # 4) Divide dados em treino e teste (sem escalonamento manual)
    X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test = DataSplitter.split_data(df, FEATURE_COLUMNS)
    if X_train.empty:
        logger.error("Divisão dos dados falhou. Encerrando o script.")
        return

    # 5) Treinamento dos modelos usando Pipeline + ColumnTransformer
    model_trainer = ModelTrainer(TRAINED_MODELS_DIR)

    # Treinamos dois modelos: um para TP e outro para SL
    model_tp = model_trainer.train_model(X_train, y_tp_train, FEATURE_COLUMNS, 'tp')
    model_sl = model_trainer.train_model(X_train, y_sl_train, FEATURE_COLUMNS, 'sl')

    # 6) Avaliação
    if model_tp:
        model_trainer.evaluate_model(model_tp, X_test, y_tp_test, 'tp')
    if model_sl:
        model_trainer.evaluate_model(model_sl, X_test, y_sl_test, 'sl')

    logger.info("Processo de treinamento concluído.")


if __name__ == "__main__":
    main()
