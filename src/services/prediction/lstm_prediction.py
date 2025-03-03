# services/prediction/lstm_prediction.py

from typing import Literal

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from repositories.data_preprocessor import DataPreprocessor
from services.prediction.interfaces import IPredictionService


class LSTMPredictionService(IPredictionService):
    """Serviço de previsão utilizando modelos LSTM."""

    def __init__(self, tp_model: LSTMModel, sl_model: LSTMModel, sequence_length: int = 24):
        """
        Inicializa o serviço de previsão LSTM.

        Args:
            tp_model: Modelo LSTM para previsão de Take Profit
            sl_model: Modelo LSTM para previsão de Stop Loss
            sequence_length: Tamanho da sequência para previsão (default=24)
        """
        self.tp_model = tp_model
        self.sl_model = sl_model
        self.sequence_length = sequence_length
        self.preprocessor = None

    def prepare_sequence(self, df: pd.DataFrame, sequence_length: int = None) -> np.ndarray | None:
        """
        Prepara uma sequência para previsão com modelo LSTM.

        Args:
            df: DataFrame com dados históricos
            sequence_length: Tamanho da sequência (opcional, usa o padrão da classe se não especificado)

        Returns:
            np.ndarray: Sequência formatada ou None se houver erro
        """
        try:
            seq_len = sequence_length or self.sequence_length

            # Verificar se temos dados suficientes
            if len(df) < seq_len:
                return None

            # Inicializar preprocessador se necessário
            if self.preprocessor is None:
                self.preprocessor = DataPreprocessor(
                    feature_columns=FEATURE_COLUMNS,
                    outlier_method='iqr',
                    scaling_method='robust'
                )
                self.preprocessor.fit(df)

            # Preparar a sequência
            x_pred = self.preprocessor.prepare_sequence_for_prediction(
                df, sequence_length=seq_len
            )

            return x_pred

        except Exception as e:
            logger.error(f"Erro ao preparar sequência: {e}", exc_info=True)
            return None

    def predict_tp_sl(
            self, df: pd.DataFrame, current_price: float,
            signal_direction: Literal["LONG", "SHORT"]
    ) -> tuple[float, float, float] | None:
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

            # Se ainda não obtivemos o ATR, mas precisamos para outros cálculos
            if atr_value is None and 'atr' in df.columns:
                atr_value = df['atr'].iloc[-1]

            return predicted_tp_pct, predicted_sl_pct, atr_value

        except Exception as e:
            logger.error(f"Erro ao fazer previsões LSTM: {e}", exc_info=True)
            return None
