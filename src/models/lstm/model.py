from pathlib import Path

import tensorflow as tf

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from models.base.model import BaseModel
from models.lstm.schemas import LSTMConfig


class LSTMModel(BaseModel):
    """Modelo LSTM para previsão de séries temporais"""

    def __init__(self, config: LSTMConfig):
        super().__init__(config)
        self.n_features = len(FEATURE_COLUMNS)
        self.model = None
        self.build_model()

    def build_model(self):
        """Constrói a arquitetura do modelo LSTM"""
        logger.info("Construindo arquitetura do modelo LSTM...")
        try:
            # Definir input layer explicitamente
            inputs = tf.keras.layers.Input(shape=(self.config.sequence_length, self.n_features))

            # Primeira camada LSTM
            x = tf.keras.layers.LSTM(
                units=self.config.lstm_units[0],
                return_sequences=len(self.config.lstm_units) > 1
            )(inputs)
            x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)

            # Camadas LSTM intermediárias
            for units in self.config.lstm_units[1:]:
                x = tf.keras.layers.LSTM(units=units)(x)
                x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)

            # Camadas densas
            for units in self.config.dense_units:
                x = tf.keras.layers.Dense(units=units, activation='relu')(x)
                x = tf.keras.layers.Dropout(self.config.dropout_rate)(x)

            # Camada de saída
            outputs = tf.keras.layers.Dense(units=1)(x)

            # Criar modelo
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

            # Compilar modelo
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss='mse',
                metrics=['mae']
            )

            logger.info("Modelo LSTM construído com sucesso")
            logger.info(f"Sumário do modelo:\n{self.model.summary()}")

        except Exception as e:
            logger.error(f"Erro ao construir modelo LSTM: {e}")
            raise

    def predict(self, input_data):
        """
        Realiza previsões com o modelo treinado

        Args:
            input_data: numpy array com shape (n_samples, sequence_length, n_features)

        Returns:
            numpy array com as previsões
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")

        try:
            predictions = self.model.predict(input_data)
            return predictions
        except Exception as e:
            logger.error(f"Erro ao fazer previsões: {e}")
            raise

    def save(self, path: Path):
        """
        Salva o modelo em disco

        Args:
            path: Caminho onde o modelo será salvo
        """
        try:
            # Garantir que o diretório exista
            path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar o modelo
            self.model.save(path)
            logger.info(f"Modelo salvo em {path}")
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {e}")
            raise

    @classmethod
    def load(cls, path: Path):
        """
        Carrega o modelo do disco

        Args:
            path: Caminho de onde o modelo será carregado

        Returns:
            Instância carregada do modelo
        """
        try:
            # Carregar o modelo
            loaded_model = tf.keras.models.load_model(path)
            logger.info(f"Modelo carregado de {path}")

            # Criar nova instância
            instance = cls(LSTMConfig(model_name="loaded_model"))
            instance.model = loaded_model
            return instance
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise
