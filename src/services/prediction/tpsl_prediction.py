# services/prediction/lstm_prediction.py

from typing import Literal

import numpy as np
import pandas as pd

from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig
from repositories.data_preprocessor import DataPreprocessor
from services.prediction.interfaces import ITpSlPredictionService


class TpSlPredictionService(ITpSlPredictionService):
    """Serviço de previsão utilizando modelos LSTM."""

    def __init__(self):
        """ Inicializa o serviço de previsão LSTM. """
        self.tp_model: LSTMModel | None = None
        self.sl_model: LSTMModel | None = None

        # Componente de preprocessador de dados
        self.preprocessor: DataPreprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            outlier_method='iqr',
            scaling_method='robust'
        )

        self.model_loaded: bool = False

        # Define o tamanho padrao da sequência
        self.default_sequence_length = 24

    def load_model(self):
        try:
            # Configurações básicas para carregamento dos modelos
            tp_config = LSTMConfig(
                model_name="lstm_btc_take_profit_pct",
                version="1.1.0",
                description="Modelo LSTM para previsão de take profit do Bitcoin"
            )

            sl_config = LSTMConfig(
                model_name="lstm_btc_stop_loss_pct",
                version="1.1.0",
                description="Modelo LSTM para previsão de stop loss do Bitcoin"
            )

            # Caminhos para os modelos treinados
            tp_model_path = TRAINED_MODELS_DIR / "lstm_btc_take_profit_pct.keras"
            sl_model_path = TRAINED_MODELS_DIR / "lstm_btc_stop_loss_pct.keras"

            # Verificação de existência dos arquivos
            if not tp_model_path.exists():
                logger.error(f"Modelo take profit não encontrado em {tp_model_path}")
                return

            if not sl_model_path.exists():
                logger.error(f"Modelo stop loss não encontrado em {sl_model_path}")
                return

            # Carregamento dos modelos
            self.tp_model = LSTMModel.load(tp_model_path, tp_config)
            self.sl_model = LSTMModel.load(sl_model_path, sl_config)

            logger.info("Modelos LSTM carregados com sucesso!")

        except Exception as e:
            logger.error(f"Erro ao carregar modelos LSTM: {e}", exc_info=True)
            return

        self.model_loaded = True

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
        if not self.model_loaded:
            self.load_model()

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
