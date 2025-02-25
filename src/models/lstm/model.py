# models\lstm\model.py

from pathlib import Path

from keras.api.layers import Input, LSTM, Dense, Dropout
from keras.api.models import Model, load_model

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
        self.config = config  # Garantindo que config seja do tipo LSTMConfig
        # Aqui, usamos len(FEATURE_COLUMNS) para determinar o número de features
        # Se você precisar de um número diferente, atualize este valor ou passe-o na configuração
        self.n_features = len(FEATURE_COLUMNS)
        logger.info(f"Modelo LSTM inicializado com {self.n_features} features")
        self.model = None
        self.build_model()

    def build_model(self):
        """
        Constrói a arquitetura do modelo LSTM com base na configuração.

        Cria uma rede neural com camadas LSTM seguidas por camadas densas,
        conforme especificado na configuração.

        Raises:
            Exception: Se ocorrer algum erro durante a construção do modelo.
        """
        logger.info("Construindo arquitetura do modelo LSTM...")
        try:
            # Definir input layer explicitamente com nome para facilitar debugging
            inputs = Input(shape=(self.config.sequence_length, self.n_features), name="input_layer")

            # OTIMIZAÇÃO: Usando return_sequences=False para primeira camada se houver apenas uma camada LSTM
            # Isso simplifica o modelo e reduz a carga computacional
            is_single_lstm_layer = len(self.config.lstm_units) == 1

            # Primeira camada LSTM
            x = LSTM(
                units=self.config.lstm_units[0],
                return_sequences=not is_single_lstm_layer,
                name="lstm",
                # OTIMIZAÇÃO: Adicionando unroll=True para melhorar desempenho em sequências fixas em CPU
                unroll=True
            )(inputs)
            x = Dropout(self.config.dropout_rate, name="dropout")(x)

            # Camadas LSTM intermediárias
            for i, units in enumerate(self.config.lstm_units[1:], 1):
                return_sequences = i < len(self.config.lstm_units) - 1
                x = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    name=f"lstm_{i}",
                    # OTIMIZAÇÃO: Adicionando unroll=True para melhorar desempenho em sequências fixas em CPU
                    unroll=True
                )(x)
                x = Dropout(self.config.dropout_rate, name=f"dropout_{i}")(x)

            # Camadas densas
            for i, units in enumerate(self.config.dense_units):
                x = Dense(units=units, activation='relu', name=f"dense_{i}")(x)
                x = Dropout(self.config.dropout_rate, name=f"dropout_{i + len(self.config.lstm_units)}")(x)

            # Camada de saída
            outputs = Dense(units=1, name="output")(x)

            # Criar modelo
            self.model = Model(inputs=inputs, outputs=outputs)

            # Compilar modelo
            self.model.compile(
                optimizer='adam',  # OTIMIZAÇÃO: Usando string 'adam' para aproveitar otimizações internas do Keras
                loss='mse',
                metrics=['mae']
            )

            logger.info("Modelo LSTM construído com sucesso")
            logger.info(f"Sumário do modelo:\n{self.model.summary()}")
            logger.info(f"Forma de entrada esperada: {self.model.input_shape}")
            logger.info(f"Forma de saída esperada: {self.model.output_shape}")

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

            # OTIMIZAÇÃO: Usando batch_size definido na configuração para previsões em lote
            predictions = self.model.predict(input_data, batch_size=self.config.batch_size)
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
            # Garantir que o diretório exista
            path.parent.mkdir(parents=True, exist_ok=True)

            # Salvar o modelo
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
            # Carregar o modelo
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
