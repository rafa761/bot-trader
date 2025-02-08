# model_manager.py

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor  # Exemplo

from logger import logger


class ModelManager:
    """
    Classe responsável pelo carregamento, treinamento e predição dos modelos de Take-Profit e Stop-Loss.
    """

    def __init__(self, tp_model_path: str = 'train_data/model_tp.pkl',
                 sl_model_path: str = 'train_data/model_sl.pkl') -> None:
        """
        Inicializa o gerenciador de modelos carregando os modelos treinados se disponíveis.

        :param tp_model_path: Caminho para o modelo de Take-Profit.
        :param sl_model_path: Caminho para o modelo de Stop-Loss.
        """
        self.tp_model_path: str = tp_model_path
        self.sl_model_path: str = sl_model_path
        self.pipeline_tp: Pipeline | None = None
        self.pipeline_sl: Pipeline | None = None
        self.model_initialized: bool = False

        self.load_models()

    def load_models(self) -> None:
        """
        Tenta carregar os modelos pré-treinados do disco. Caso falhe, define os modelos como None.
        """
        try:
            self.pipeline_tp = joblib.load(self.tp_model_path)
            self.pipeline_sl = joblib.load(self.sl_model_path)
            logger.info("Modelos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar os modelos: {e}", exc_info=True)
            self.pipeline_tp = None
            self.pipeline_sl = None

    def train_models(self, features: pd.DataFrame, target_tp: pd.Series, target_sl: pd.Series) -> None:
        """
        Treina (ou re-treina) os modelos de Take-Profit e Stop-Loss.

        :param features: DataFrame contendo as features de entrada para o modelo.
        :param target_tp: Série contendo os valores alvo para Take-Profit.
        :param target_sl: Série contendo os valores alvo para Stop-Loss.
        """
        if not self.pipeline_tp or not self.pipeline_sl:
            # Inicializa pipelines simples com XGBRegressor, por exemplo
            self.pipeline_tp = Pipeline([
                ('xgb', XGBRegressor())
            ])
            self.pipeline_sl = Pipeline([
                ('xgb', XGBRegressor())
            ])

        # Ajusta os modelos
        self.pipeline_tp.fit(features, target_tp)
        self.pipeline_sl.fit(features, target_sl)
        self.model_initialized = True
        logger.info("Modelos treinados/re-treinados com sucesso.")

    def predict_tp_sl(self, X_current: pd.DataFrame) -> tuple[float | None, float | None]:
        """
        Realiza predições para Take-Profit e Stop-Loss a partir dos modelos carregados.

        :param X_current: DataFrame contendo as features da amostra atual.
        :return: Tupla com valores preditos para Take-Profit e Stop-Loss.
        """
        if not self.pipeline_tp or not self.pipeline_sl:
            logger.error("Modelos não inicializados.")
            return None, None
        try:
            predicted_tp = self.pipeline_tp.predict(X_current)[0]
            predicted_sl = abs(self.pipeline_sl.predict(X_current)[0])  # Garante ser positivo
            return predicted_tp, predicted_sl
        except Exception as e:
            logger.error(f"Erro ao fazer predições: {e}", exc_info=True)
            return None, None
