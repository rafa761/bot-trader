# models\lstm\model.py

from pathlib import Path

import numpy as np
from keras.api.layers import BatchNormalization
from keras.api.layers import Input, LSTM, Dense, Dropout
from keras.api.models import Model, load_model
from keras.api.optimizers import Adam
from keras.api.regularizers import L2

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.base.model import BaseModel
from models.lstm.schemas import LSTMConfig


class LSTMModel(BaseModel):
    """
    Modelo LSTM para previsão de séries temporais financeiras.

    Implementa um modelo de rede neural com arquitetura LSTM (Long Short-Term Memory)
    para previsão de valores futuros com base em sequências de dados históricos.
    """

    def __init__(self, config: LSTMConfig):
        """
        Inicializa o modelo LSTM com a configuração especificada.

        Args:
            config: Configuração do modelo contendo parâmetros como sequence_length,
                   lstm_units, etc.
        """
        super().__init__(config)
        self.config = config
        self.n_features = len(FEATURE_COLUMNS)
        logger.info(f"Modelo LSTM inicializado com {self.n_features} features")
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Constrói a arquitetura do modelo LSTM com base na configuração.
        Versão otimizada para hardware de alto desempenho.
        """
        logger.info("Construindo arquitetura do modelo LSTM...")
        try:
            # Definir input layer
            inputs = Input(shape=(self.config.sequence_length, self.n_features), name="input_layer")

            # Normalização de batch para estabilizar o treinamento
            x = BatchNormalization(name="batch_norm_input")(inputs)

            # Primeira camada LSTM
            is_single_lstm_layer = len(self.config.lstm_units) == 1
            x = LSTM(
                units=self.config.lstm_units[0],
                return_sequences=not is_single_lstm_layer,
                name="lstm",
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=self.config.recurrent_dropout_rate,
                kernel_regularizer=L2(self.config.l2_regularization),
                recurrent_regularizer=L2(self.config.l2_regularization / 2),
                unroll=False
            )(x)
            x = Dropout(self.config.dropout_rate, name="dropout")(x)

            # Camadas LSTM intermediárias
            for i, units in enumerate(self.config.lstm_units[1:], 1):
                return_sequences = i < len(self.config.lstm_units) - 1
                x = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    name=f"lstm_{i}",
                    activation='tanh',
                    recurrent_activation='sigmoid',
                    recurrent_dropout=self.config.recurrent_dropout_rate,
                    kernel_regularizer=L2(self.config.l2_regularization),
                    recurrent_regularizer=L2(self.config.l2_regularization / 2),
                    unroll=False
                )(x)
                x = Dropout(self.config.dropout_rate, name=f"dropout_{i}")(x)

            # Camadas densas expandidas
            for i, units in enumerate(self.config.dense_units):
                x = Dense(
                    units=units,
                    activation='relu',
                    name=f"dense_{i}",
                    kernel_regularizer=L2(self.config.l2_regularization)
                )(x)
                x = BatchNormalization(name=f"batch_norm_dense_{i}")(x)
                x = Dropout(self.config.dropout_rate, name=f"dropout_{i + len(self.config.lstm_units)}")(x)

            # Camada de saída - sem ativação linear
            outputs = Dense(units=1, activation='linear', name="output")(x)

            # Criar modelo
            self.model = Model(inputs=inputs, outputs=outputs)

            # Compilar modelo com otimizador avançado
            optimizer = Adam(
                learning_rate=self.config.learning_rate,
                amsgrad=self.config.use_amsgrad
            )

            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae', 'mse']
            )

            logger.info("Modelo LSTM construído com sucesso")
            logger.info(f"Sumário do modelo:\n{self.model.summary()}")

        except Exception as e:
            logger.error(f"Erro ao construir modelo LSTM: {e}")
            raise

    def predict(self, input_data):
        """
        Realiza previsões com o modelo treinado.

        Args:
            input_data: numpy array com shape (n_samples, sequence_length, n_features)
                        contendo os dados de entrada para previsão.

        Returns:
            numpy array com as previsões realizadas pelo modelo.

        Raises:
            ValueError: Se o modelo ainda não foi treinado.
            Exception: Se ocorrer outro erro durante a previsão.
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")

        try:
            if input_data is None or len(input_data) == 0:
                raise ValueError("Dados de entrada vazios")

            # Verificar se o formato é o correto
            if len(input_data.shape) != 3:
                raise ValueError(f"Formato de entrada inválido. Esperado 3D, recebido: {input_data.shape}")

            # Verificar valores anormais
            if np.isnan(input_data).any():
                logger.warning("Valores NaN detectados nos dados de entrada!")
                input_data = np.nan_to_num(input_data, nan=0.0)

            logger.info(
                f"Estatísticas do input: min={np.min(input_data)}, max={np.max(input_data)}, mean={np.mean(input_data)}")

            # Verificar se o input_data tem a forma correta
            expected_shape = self.model.input_shape
            if input_data.shape[1:] != expected_shape[1:]:
                logger.warning(
                    f"Incompatibilidade de dimensões: modelo espera {expected_shape}, "
                    f"mas os dados são {input_data.shape}"
                )
                # Reajustar o número de features se necessário
                if expected_shape[2] < input_data.shape[2]:
                    logger.info(f"Reduzindo número de features para {expected_shape[2]}")
                    input_data = input_data[:, :, :expected_shape[2]]
                elif expected_shape[2] > input_data.shape[2]:
                    raise ValueError(
                        f"Modelo espera {expected_shape[2]} features, mas os dados têm apenas {input_data.shape[2]}"
                    )

            # Usar batch_size=1 para evitar problemas com previsões em lote
            predictions = self.model.predict(input_data, batch_size=1, verbose=1)

            logger.info(f"Previsões geradas: {predictions}")

            return predictions
        except Exception as e:
            logger.error(f"Erro ao fazer previsões: {e}")
            raise

    def save(self, path: Path):
        """
        Salva o modelo em disco.

        Args:
            path: Caminho onde o modelo será salvo.

        Raises:
            Exception: Se ocorrer algum erro durante o salvamento.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            self.model.save(path)

            logger.info(f"Modelo salvo em {path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise

    @classmethod
    def load(cls, path: Path, config: LSTMConfig = None):
        """
        Carrega o modelo do disco.

        Args:
            path: Caminho de onde o modelo será carregado.
            config: Configuração opcional para inicializar o modelo.
                   Se não fornecida, uma configuração padrão será usada.

        Returns:
            Instância carregada do modelo.

        Raises:
            Exception: Se ocorrer algum erro durante o carregamento.
        """
        try:
            loaded_model = load_model(path)
            logger.info(f"Modelo carregado de {path}")

            # Criar nova instância com configuração fornecida ou padrão
            if config is None:
                config = LSTMConfig(model_name="loaded_model")

            instance = cls(config)
            instance.model = loaded_model

            # Atualizar n_features baseado no modelo carregado
            instance.n_features = loaded_model.input_shape[-1]
            logger.info(f"Modelo carregado espera {instance.n_features} features")

            return instance
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
