from abc import ABC, abstractmethod

import pandas as pd


class BaseTrainer(ABC):
    """Classe abstrata para treinadores de modelos"""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Treina o modelo"""
        pass

    @abstractmethod
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Avalia o modelo"""
        pass
