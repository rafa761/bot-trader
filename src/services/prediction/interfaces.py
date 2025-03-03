# services/prediction/interfaces.py

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import pandas as pd


class IPredictionService(ABC):
    """Interface para serviços de previsão de TP/SL."""

    @abstractmethod
    def prepare_sequence(self, df: pd.DataFrame, sequence_length: int) -> np.ndarray | None:
        """Prepara uma sequência para previsão."""
        pass

    @abstractmethod
    def predict_tp_sl(
            self, df: pd.DataFrame, current_price: float,
            signal_direction: Literal["LONG", "SHORT"]
    ) -> tuple[float, float, float] | None:
        """Prediz TP/SL e retorna valores percentuais junto com ATR."""
        pass
