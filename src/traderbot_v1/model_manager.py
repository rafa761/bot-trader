# model_manager.py

import joblib
import numpy as np
import pandas as pd
from logger import logger
from xgboost import XGBRegressor  # Exemplo
from sklearn.pipeline import Pipeline

class ModelManager:
    """
    Classe responsável pelo carregamento, treinamento e predição dos modelos.
    """

    def __init__(self, tp_model_path='train_data/model_tp.pkl', sl_model_path='train_data/model_sl.pkl'):
        self.tp_model_path = tp_model_path
        self.sl_model_path = sl_model_path
        self.pipeline_tp = None
        self.pipeline_sl = None
        self.model_initialized = False

        self.load_models()

    def load_models(self):
        """
        Tenta carregar os modelos pré-treinados do disco.
        """
        try:
            self.pipeline_tp = joblib.load(self.tp_model_path)
            self.pipeline_sl = joblib.load(self.sl_model_path)
            logger.info("Modelos carregados com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao carregar os modelos: {e}", exc_info=True)
            self.pipeline_tp = None
            self.pipeline_sl = None

    def train_models(self, features: pd.DataFrame, target_tp: pd.Series, target_sl: pd.Series):
        """
        Treina (ou re-treina) os modelos de Take-Profit e Stop-Loss.
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

    def predict_tp_sl(self, X_current: pd.DataFrame):
        """
        Realiza predições para Take-Profit e Stop-Loss a partir dos modelos carregados.
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
