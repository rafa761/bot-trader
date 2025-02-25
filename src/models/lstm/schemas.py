# models\lstm\schemas.py

from pydantic import Field

from models.base.schemas import ModelConfig, TrainingConfig


class LSTMConfig(ModelConfig):
    """
    Configuração específica para o modelo LSTM.

    Define todos os parâmetros necessários para a construção e treinamento
    de um modelo LSTM para previsão de séries temporais.
    """
    sequence_length: int = Field(16,
                                 description="Número de timesteps para cada sequência")  # OTIMIZAÇÃO: Reduzido de 24 para 16
    lstm_units: list[int] = Field([64, 32],
                                  description="Unidades para cada camada LSTM")  # OTIMIZAÇÃO: Reduzido de [128, 64] para [64, 32]
    dense_units: list[int] = Field([16],
                                   description="Unidades para cada camada densa")  # OTIMIZAÇÃO: Reduzido de [32] para [16]
    dropout_rate: float = Field(0.1, ge=0, le=1, description="Taxa de dropout")  # OTIMIZAÇÃO: Reduzido de 0.2 para 0.1
    learning_rate: float = Field(0.002, gt=0,
                                 description="Taxa de aprendizado")  # OTIMIZAÇÃO: Aumentado de 0.001 para 0.002 para convergência mais rápida
    batch_size: int = Field(64, gt=0, description="Tamanho do batch")  # OTIMIZAÇÃO: Aumentado de 32 para 64
    epochs: int = Field(50, gt=0,
                        description="Número de épocas para treinamento")  # OTIMIZAÇÃO: Reduzido de 100 para 50


class LSTMTrainingConfig(TrainingConfig):
    """
    Configuração específica para treinamento do LSTM.

    Define os parâmetros relacionados ao processo de treinamento,
    como validação, early stopping e ajustes da taxa de aprendizado.
    """
    validation_split: float = Field(0.15, ge=0.1, le=0.3,
                                    description="Fração dos dados de treino usada para validação")  # OTIMIZAÇÃO: Reduzido de 0.2 para 0.15
    early_stopping_patience: int = Field(5, gt=0,
                                         description="Número de épocas para early stopping")  # OTIMIZAÇÃO: Reduzido de 10 para 5
    reduce_lr_patience: int = Field(3, gt=0,
                                    description="Número de épocas para redução do learning rate")  # OTIMIZAÇÃO: Reduzido de 5 para 3
    reduce_lr_factor: float = Field(0.5, gt=0, lt=1, description="Fator de redução do learning rate")
