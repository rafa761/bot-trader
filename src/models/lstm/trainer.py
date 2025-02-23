import numpy as np
import pandas as pd

from core.logger import logger
from models.base.trainer import BaseTrainer
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMTrainingConfig


class LSTMTrainer(BaseTrainer):
    """Treinador específico para o modelo LSTM"""

    def __init__(self, model: LSTMModel, config: LSTMTrainingConfig):
        self.model = model
        self.config = config
        self.history = None

    def _prepare_sequences(self, data: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        """Prepara as sequências para treinamento do LSTM"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Treina o modelo LSTM"""
        logger.info("Iniciando treinamento do modelo LSTM...")
        try:
            # Preparar callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience
                )
            ]

            # Preparar sequências
            X_sequences, y_sequences = self._prepare_sequences(
                np.column_stack((X_train.values, y_train.values)),
                self.model.config.sequence_length
            )

            # Treinar modelo
            self.history = self.model.model.fit(
                X_sequences,
                y_sequences,
                batch_size=self.model.config.batch_size,
                epochs=self.model.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Treinamento concluído com sucesso")

        except Exception as e:
            logger.error(f"Erro durante treinamento: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Avalia o modelo com dados de teste"""
        logger.info("Avaliando modelo...")
        try:
            X_test_seq, y_test_seq = self._prepare_sequences(
                np.column_stack((X_test.values, y_test.values)),
                self.model.config.sequence_length
            )

            evaluation = self.model.model.evaluate(X_test_seq, y_test_seq)
            metrics = {
                'test_loss': float(evaluation[0]),
                'test_mae': float(evaluation[1])
            }

            if self.history:
                metrics.update({
                    'final_train_loss': float(self.history.history['loss'][-1]),
                    'final_val_loss': float(self.history.history['val_loss'][-1]),
                    'best_val_loss': float(min(self.history.history['val_loss']))
                })

            logger.info(f"Avaliação concluída: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Erro durante avaliação: {e}")
            raise
