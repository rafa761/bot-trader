from pathlib import Path

import joblib
import pandas as pd
from binance.client import Client
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.config import config
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.base import DataCollector, LabelCreator


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
