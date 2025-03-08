# models\lstm\schemas.py

from pydantic import Field, BaseModel

from models.base.schemas import ModelConfig, TrainingConfig


class LSTMConfig(ModelConfig):
    """
    Configuração específica para o modelo LSTM - Otimizada para hardware de alto desempenho.
    """
    sequence_length: int = Field(16, description="Número de timesteps para cada sequência")
    lstm_units: list[int] = Field([128, 64], description="Unidades para cada camada LSTM")
    dense_units: list[int] = Field([64, 32], description="Unidades para cada camada densa")
    dropout_rate: float = Field(0.2, ge=0, le=1, description="Taxa de dropout")
    recurrent_dropout_rate: float = Field(0.1, ge=0, le=1, description="Taxa de dropout recorrente")
    l2_regularization: float = Field(0.0001, ge=0, description="Fator de regularização L2")
    learning_rate: float = Field(0.001, gt=0, description="Taxa de aprendizado inicial")
    batch_size: int = Field(128, gt=0, description="Tamanho do batch")
    epochs: int = Field(100, gt=0, description="Número máximo de épocas")
    use_amsgrad: bool = Field(True, description="Usar variante AMSGrad do Adam")

class LSTMTrainingConfig(TrainingConfig):
    """
    Configuração específica para treinamento do LSTM.

    Define os parâmetros relacionados ao processo de treinamento,
    como validação, early stopping e ajustes da taxa de aprendizado.
    """
    validation_split: float = Field(0.15, ge=0.1, le=0.3,
                                    description="Fração dos dados de treino usada para validação")
    early_stopping_patience: int = Field(10, gt=0,
                                         description="Número de épocas para early stopping")
    reduce_lr_patience: int = Field(3, gt=0,
                                    description="Número de épocas para redução do learning rate")
    reduce_lr_factor: float = Field(0.5, gt=0, lt=1, description="Fator de redução do learning rate")
    use_early_stopping: bool = Field(True, description="Usar ou não Early Stopping")
    min_delta: float = Field(0.001, ge=0, description="Melhoria mínima requerida para early stopping")


class OptunaConfig(BaseModel):
    """
    Configuração para tunagem de hiperparâmetros com Optuna.
    """
    enabled: bool = Field(True, description="Habilitar a tunagem de hiperparâmetros")
    n_trials: int = Field(30, gt=0, description="Número de trials para a otimização")
    timeout: int | None = Field(default=None,
                                description="Tempo máximo em segundos para a otimização (None para sem limite)")
    study_name: str = Field("lstm_hyperparameter_study", description="Nome do estudo Optuna")
    storage: str | None = Field(default=None,
                                description="Caminho para o storage do Optuna (None para armazenamento em memória)")
    direction: str = Field("minimize", description="Direção de otimização (minimize para loss)")
    metric: str = Field("val_loss", description="Métrica a ser otimizada")
