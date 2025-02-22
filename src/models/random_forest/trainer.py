from typing import Optional
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from models.base.trainer import BaseTrainer
from models.random_forest.model import RandomForestModel


class RandomForestTrainer(BaseTrainer):
    """Treinador específico para Random Forest"""

    def __init__(self, model: RandomForestModel):
        self.model = model
        self.metrics: dict | None = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Executa o treinamento do modelo"""
        self.model.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Avalia o modelo com métricas"""
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        direction_accuracy = self._calc_direction_accuracy(y_test, y_pred)

        self.metrics = {
            'mae': mae,
            'direction_accuracy': direction_accuracy
        }
        return self.metrics

    def _calc_direction_accuracy(self, y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calcula a acurácia de direção das previsões"""
        correct = ((y_true >= 0) & (y_pred >= 0)) | ((y_true < 0) & (y_pred < 0))
        return correct.mean()
