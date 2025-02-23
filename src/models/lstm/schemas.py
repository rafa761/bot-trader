from pydantic import Field

from models.base.schemas import ModelConfig, TrainingConfig


class LSTMConfig(ModelConfig):
    """Configuração específica para o modelo LSTM"""
    sequence_length: int = Field(24, description="Número de timesteps para cada sequência")
    lstm_units: list[int] = Field([128, 64], description="Unidades para cada camada LSTM")
    dense_units: list[int] = Field([32], description="Unidades para cada camada densa")
    dropout_rate: float = Field(0.2, ge=0, le=1, description="Taxa de dropout")
    learning_rate: float = Field(0.001, gt=0, description="Taxa de aprendizado")
    batch_size: int = Field(32, gt=0, description="Tamanho do batch")
    epochs: int = Field(100, gt=0, description="Número de épocas para treinamento")


class LSTMTrainingConfig(TrainingConfig):
    """Configuração específica para treinamento do LSTM"""
    validation_split: float = Field(0.2, ge=0.1, le=0.3, description="Fração dos dados de treino usada para validação")
    early_stopping_patience: int = Field(10, gt=0, description="Número de épocas para early stopping")
    reduce_lr_patience: int = Field(5, gt=0, description="Número de épocas para redução do learning rate")
    reduce_lr_factor: float = Field(0.5, gt=0, lt=1, description="Fator de redução do learning rate")
