# models\lstm\main.py

from pathlib import Path

import numpy as np
import pandas as pd
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from core.logger import logger
from models.base.trainer import BaseTrainer
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMTrainingConfig


class LSTMTrainer(BaseTrainer):
    """
    Treinador específico para o modelo LSTM.

    Responsável por preparar os dados, treinar o modelo e avaliar seu desempenho.
    """

    def __init__(self, model: LSTMModel, config: LSTMTrainingConfig):
        """
        Inicializa o treinador com o modelo e configuração especificados.

        Args:
            model: Instância do modelo LSTM a ser treinado.
            config: Configuração de treinamento com parâmetros como validation_split,
                   early_stopping_patience, etc.
        """
        self.model = model
        self.config = config
        self.history = None

    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara as sequências para treinamento do LSTM.

        Transforma os dados em sequências de tamanho fixo para alimentar o modelo LSTM.

        Args:
            X: Array numpy contendo as features.
            y: Array numpy contendo os alvos.
            sequence_length: Tamanho de cada sequência.

        Returns:
            Tupla contendo dois arrays numpy: X_seq (sequências de features) e y_seq (alvos correspondentes).
        """
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:(i + sequence_length)])
            y_seq.append(y[i + sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, checkpoint_dir: Path = None):
        """
        Treina o modelo LSTM.

        Prepara os dados, configura callbacks e executa o treinamento do modelo.

        Args:
            X_train: DataFrame pandas contendo as features de treinamento.
            y_train: Series pandas contendo os alvos de treinamento.
            checkpoint_dir: Diretório opcional para salvar checkpoints durante o treinamento.

        Raises:
            Exception: Se ocorrer algum erro durante o treinamento.
        """
        logger.info("Iniciando treinamento do modelo LSTM...")
        try:
            # Preparar callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience
                )
            ]

            # Adicionar checkpoint se diretório for fornecido
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{self.model.config.model_name}_checkpoint.h5"
                callbacks.append(
                    ModelCheckpoint(
                        filepath=str(checkpoint_path),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                )

            # Converter para numpy arrays
            X_np = X_train.values
            y_np = y_train.values.reshape(-1, 1)  # Garante que y seja 2D

            # Preparar sequências - agora tratando X e y separadamente
            X_sequences, y_sequences = self._prepare_sequences(
                X_np,
                y_np,
                self.model.config.sequence_length
            )

            # Log de dimensões para debugging
            logger.info(f"Dimensões das sequências de treino: X={X_sequences.shape}, y={y_sequences.shape}")

            # Verificar compatibilidade com o modelo
            expected_shape = self.model.model.input_shape
            if X_sequences.shape[1:] != expected_shape[1:]:
                logger.warning(
                    f"Incompatibilidade de dimensões: modelo espera {expected_shape}, "
                    f"mas os dados são {X_sequences.shape}"
                )
                # Reajustar o número de features se necessário
                if expected_shape[2] < X_sequences.shape[2]:
                    logger.info(f"Reduzindo número de features para {expected_shape[2]}")
                    X_sequences = X_sequences[:, :, :expected_shape[2]]
                elif expected_shape[2] > X_sequences.shape[2]:
                    raise ValueError(
                        f"Modelo espera {expected_shape[2]} features, mas os dados têm apenas {X_sequences.shape[2]}"
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
        """
        Avalia o modelo com dados de teste.

        Prepara os dados de teste em sequências e avalia o desempenho do modelo.

        Args:
            X_test: DataFrame pandas contendo as features de teste.
            y_test: Series pandas contendo os alvos de teste.

        Returns:
            Dicionário contendo métricas de avaliação como test_loss, test_mae, etc.

        Raises:
            Exception: Se ocorrer algum erro durante a avaliação.
        """
        logger.info("Avaliando modelo...")
        try:
            # Converter para numpy arrays
            X_np = X_test.values
            y_np = y_test.values.reshape(-1, 1)  # Garante que y seja 2D

            # Preparar sequências - agora tratando X e y separadamente
            X_test_seq, y_test_seq = self._prepare_sequences(
                X_np,
                y_np,
                self.model.config.sequence_length
            )

            # Verificar compatibilidade com o modelo
            expected_shape = self.model.model.input_shape
            if X_test_seq.shape[1:] != expected_shape[1:]:
                logger.warning(
                    f"Incompatibilidade de dimensões: modelo espera {expected_shape}, "
                    f"mas os dados são {X_test_seq.shape}"
                )
                # Reajustar o número de features se necessário
                if expected_shape[2] < X_test_seq.shape[2]:
                    logger.info(f"Reduzindo número de features para {expected_shape[2]}")
                    X_test_seq = X_test_seq[:, :, :expected_shape[2]]
                elif expected_shape[2] > X_test_seq.shape[2]:
                    raise ValueError(
                        f"Modelo espera {expected_shape[2]} features, mas os dados têm apenas {X_test_seq.shape[2]}"
                    )

            evaluation = self.model.model.evaluate(X_test_seq, y_test_seq)
            metrics = {
                'test_loss': float(evaluation[0]),
                'test_mae': float(evaluation[1])
            }

            # Adicionar métricas do histórico de treinamento
            if self.history:
                metrics.update({
                    'final_train_loss': float(self.history.history['loss'][-1]),
                    'final_val_loss': float(self.history.history['val_loss'][-1]),
                    'best_val_loss': float(min(self.history.history['val_loss']))
                })

                # Calcular métricas adicionais se disponíveis
                if 'mae' in self.history.history:
                    metrics.update({
                        'final_train_mae': float(self.history.history['mae'][-1]),
                        'final_val_mae': float(self.history.history['val_mae'][-1]),
                        'best_val_mae': float(min(self.history.history['val_mae']))
                    })

            logger.info(f"Avaliação concluída: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Erro durante avaliação: {e}")
            raise
