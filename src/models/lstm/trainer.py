# models\lstm\trainer.py

import math
import time

import numpy as np
import pandas as pd
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

from core.constants import TRAINED_MODELS_CHECKPOINTS_DIR
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
        # OTIMIZAÇÃO: Usando stride para reduzir o número de sequências e acelerar o treinamento
        # stride = 2 significa que pegamos a cada 2 pontos de dados, reduzindo pela metade o número de sequências
        stride = 1  # OTIMIZAÇÃO: Ajuste esse valor conforme necessário (1 = todas as sequências, 2 = metade, etc.)

        # OTIMIZAÇÃO: Pré-alocação de memória para melhorar performance
        n_samples = (len(X) - sequence_length) // stride
        # Pré-alocar arrays para melhor performance
        X_seq = np.zeros((n_samples, sequence_length, X.shape[1]), dtype=X.dtype)
        y_seq = np.zeros((n_samples, y.shape[1]), dtype=y.dtype)

        # Preencher arrays
        for i in range(n_samples):
            idx = i * stride
            X_seq[i] = X[idx:idx + sequence_length]
            y_seq[i] = y[idx + sequence_length]

        return X_seq, y_seq

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Treina o modelo LSTM com callbacks otimizados para hardware de alto desempenho.
        """
        logger.info("Iniciando treinamento do modelo LSTM com configuração otimizada...")
        try:
            # Learning rate schedule personalizado
            def lr_scheduler(epoch, lr):
                if epoch < 10:
                    return lr  # Manter taxa inicial para warm-up
                else:
                    # Decay exponencial suave
                    return self.model.config.learning_rate * math.exp(-0.025 * (epoch - 10))

            callbacks = [
                LearningRateScheduler(lr_scheduler, verbose=1),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=self.config.reduce_lr_patience,
                    min_lr=1e-6,  # Definir limite mínimo
                    verbose=1
                )
            ]

            if self.config.use_early_stopping:
                callbacks.append(
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.config.early_stopping_patience,
                        restore_best_weights=True,
                        min_delta=self.config.min_delta,
                        mode='min',
                        verbose=1
                    )
                )

            # Adicionar múltiplos checkpoints aproveitando o SSD rápido
            # Checkpoint para melhor modelo por loss
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(
                        TRAINED_MODELS_CHECKPOINTS_DIR / f"{self.model.config.model_name}_best_val_loss.keras"),
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )
            )

            # Checkpoint para melhor modelo por MAE
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(TRAINED_MODELS_CHECKPOINTS_DIR / f"{self.model.config.model_name}_best_val_mae.keras"),
                    save_best_only=True,
                    monitor='val_mae',
                    mode='min',
                    verbose=1
                )
            )

            # Checkpoint periódico a cada 10 épocas
            callbacks.append(
                ModelCheckpoint(
                    filepath=str(
                        TRAINED_MODELS_CHECKPOINTS_DIR / f"{self.model.config.model_name}_epoch_{{epoch:03d}}.keras"),
                    # Calcular frequência para salvar a cada 10 épocas (número de batches por época * 10)
                    save_freq=int(len(X_train) // self.model.config.batch_size * 10),
                    save_best_only=False,
                    verbose=0
                )
            )

            # Verificar e garantir que todos os dados são numéricos
            # Converter para numpy arrays e forçar tipo numérico
            X_train_numeric = X_train.astype(np.float64)
            y_train_numeric = y_train.astype(np.float64)

            logger.info(f"X_train dtype: {X_train_numeric.dtypes.unique()}")
            logger.info(f"y_train dtype: {y_train_numeric.dtype}")

            # Converter para numpy arrays
            X_np = X_train_numeric.values
            y_np = y_train_numeric.values.reshape(-1, 1)  # Garante que y seja 2D

            # Verificar valores infinitos, NaN, etc.
            if np.isnan(X_np).any():
                logger.warning("Valores NaN encontrados em X_train. Substituindo por zeros.")
                X_np = np.nan_to_num(X_np, nan=0.0)

            if np.isnan(y_np).any():
                logger.warning("Valores NaN encontrados em y_train. Substituindo por zeros.")
                y_np = np.nan_to_num(y_np, nan=0.0)

            if (~np.isfinite(X_np)).any():
                logger.warning("Valores infinitos encontrados em X_train. Substituindo.")
                X_np = np.nan_to_num(X_np, nan=0.0, posinf=1e10, neginf=-1e10)

            if (~np.isfinite(y_np)).any():
                logger.warning("Valores infinitos encontrados em y_train. Substituindo.")
                y_np = np.nan_to_num(y_np, nan=0.0, posinf=1e10, neginf=-1e10)

            # Medir o tempo de preparação das sequências
            start_prep_time = time.time()

            # Preparar sequências
            X_sequences, y_sequences = self._prepare_sequences(
                X_np,
                y_np,
                self.model.config.sequence_length
            )

            prep_time = time.time() - start_prep_time
            logger.info(f"Preparação de sequências concluída em {prep_time:.2f} segundos")

            # Verificar novamente valores numéricos depois de preparar sequências
            if not np.issubdtype(X_sequences.dtype, np.floating):
                logger.warning(f"Forçando conversão de X_sequences para float64. Tipo atual: {X_sequences.dtype}")
                X_sequences = X_sequences.astype(np.float64)

            if not np.issubdtype(y_sequences.dtype, np.floating):
                logger.warning(f"Forçando conversão de y_sequences para float64. Tipo atual: {y_sequences.dtype}")
                y_sequences = y_sequences.astype(np.float64)

            logger.info(f"Dimensões das sequências de treino: X={X_sequences.shape}, y={y_sequences.shape}")
            logger.info(f"Tipos das sequências: X={X_sequences.dtype}, y={y_sequences.dtype}")

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

            # Medir tempo de treinamento
            start_train_time = time.time()

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

            training_time = time.time() - start_train_time
            logger.info(f"Treinamento concluído em {training_time:.2f} segundos")
            logger.info(f"Melhor val_loss: {min(self.history.history['val_loss']):.4f}")

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

            evaluation = self.model.model.evaluate(
                X_test_seq,
                y_test_seq,
                batch_size=self.model.config.batch_size
            )

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
