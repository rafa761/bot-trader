# services/prediction/lstm_prediction.py

from typing import Literal

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_preprocessor import DataPreprocessor
from services.prediction.interfaces import IPredictionService


class TpSlPredictionService(IPredictionService):
    """Serviço de previsão utilizando modelos LSTM."""

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel):
        """
        Inicializa o serviço de previsão LSTM.

        Args:
            tp_model: Modelo LSTM para previsão de Take Profit
            sl_model: Modelo LSTM para previsão de Stop Loss
        """
        self.tp_model = tp_model
        self.sl_model = sl_model

        # Componente de preprocessador de dados
        self.preprocessor: DataPreprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            outlier_method='iqr',
            scaling_method='robust'
        )

        # Define o tamanho padrao da sequência
        self.default_sequence_length = 24

    def prepare_sequence(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão.

        Args:
            df: DataFrame com dados históricos

        Returns:
            np.ndarray: Sequência formatada ou None se houver erro
        """
        try:
            # Verificar se temos dados suficientes
            if len(df) < self.default_sequence_length:
                return None

            self.preprocessor.fit(df)

            # Preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df, sequence_length=self.default_sequence_length
            )

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência: {e}", exc_info=True)
            return None

    def predict_tp_sl(
            self, df: pd.DataFrame, current_price: float,
            signal_direction: Literal["LONG", "SHORT"]
    ) -> tuple[float, float] | None:
        """
        Realiza previsões de TP e SL usando modelos LSTM.

        Args:
            df: DataFrame com dados históricos
            current_price: Preço atual
            signal_direction: Direção do sinal ("LONG" ou "SHORT")

        Returns:
            tuple: (predicted_tp_pct, predicted_sl_pct, atr_value) ou None se falhar
        """
        try:
            X_seq = self.prepare_sequence(df)
            if X_seq is None:
                return None

            # Previsões com LSTM
            predicted_tp_pct = float(self.tp_model.predict(X_seq)[0][0])
            predicted_sl_pct = float(self.sl_model.predict(X_seq)[0][0])

            # Para SHORT, garantir que o TP seja negativo
            if signal_direction == "SHORT" and predicted_tp_pct > 0:
                predicted_tp_pct = -predicted_tp_pct

            # Garantir valores positivos para SL
            predicted_sl_pct = abs(predicted_sl_pct)

            logger.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")

            # Validar previsões - evitar valores absurdos
            if abs(predicted_tp_pct) > 20:
                predicted_tp_pct = 20.0 if signal_direction == "LONG" else -20.0

            if predicted_sl_pct > 10:
                predicted_sl_pct = 10.0

            # Ajustar SL dinamicamente se for muito pequeno
            atr_value = None
            if predicted_sl_pct < 0.5:
                # Calcular o SL dinâmico baseado em ATR
                atr_value = df['atr'].iloc[-1] if 'atr' in df.columns else None
                if atr_value:
                    predicted_sl_pct = (atr_value / current_price) * 100 * 1.5

            return predicted_tp_pct, predicted_sl_pct

        except Exception as e:
            logger.error(f"Erro ao fazer previsões LSTM: {e}", exc_info=True)
            return None
