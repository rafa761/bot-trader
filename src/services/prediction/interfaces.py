# services/prediction/interfaces.py

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class IBasePredictionService(ABC):
    """ Interface base para modelos de previsão. """

    @abstractmethod
    def load_model(self):
        """ Inicializa modelos de previsão. """
        raise NotImplementedError("Subclasses must implement this method.")


class ITpSlPredictionService(IBasePredictionService):
    """Interface para serviços de previsão de TP/SL."""

    @abstractmethod
    def prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """Prepara uma sequência para previsão."""
        pass

    @abstractmethod
    def predict_tp_sl(
            self, df: pd.DataFrame, current_price: float,
            signal_direction: Literal["LONG", "SHORT"]
    ) -> tuple[float, float] | None:
        """Prediz TP/SL e retorna valores percentuais junto com ATR."""
        pass
