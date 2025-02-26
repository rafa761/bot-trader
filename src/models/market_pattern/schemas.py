# models/market_pattern/schemas.py

from pydantic import Field

from models.base.schemas import ModelConfig, TrainingConfig


class MarketPatternConfig(ModelConfig):
    """
    Configuração específica para o modelo de classificação de padrões de mercado.

    Define todos os parâmetros necessários para a construção e treinamento
    de um modelo de classificação para identificação de regimes de mercado.
    """
    sequence_length: int = Field(16,
                                 description="Número de timesteps para cada sequência")
    lstm_units: list[int] = Field([64, 32],
                                  description="Unidades para cada camada LSTM")
    dense_units: list[int] = Field([32, 16],
                                   description="Unidades para cada camada densa")
    dropout_rate: float = Field(0.2, ge=0, le=1,
                                description="Taxa de dropout")
    learning_rate: float = Field(0.001, gt=0,
                                 description="Taxa de aprendizado")
    batch_size: int = Field(64, gt=0,
                            description="Tamanho do batch")
    epochs: int = Field(100, gt=0,
                        description="Número de épocas para treinamento")
    num_classes: int = Field(4, gt=1,
                             description="Número de classes de padrões de mercado")
    class_names: list[str] = Field(
        ["UPTREND", "DOWNTREND", "RANGE", "VOLATILE"],
        description="Nomes das classes de padrões de mercado"
    )


class MarketPatternTrainingConfig(TrainingConfig):
    """
    Configuração específica para treinamento do classificador de padrões de mercado.

    Define os parâmetros relacionados ao processo de treinamento,
    como validação, early stopping e ajustes da taxa de aprendizado.
    """
    validation_split: float = Field(0.2, ge=0.1, le=0.3,
                                    description="Fração dos dados de treino usada para validação")
    early_stopping_patience: int = Field(10, gt=0,
                                         description="Número de épocas para early stopping")
    reduce_lr_patience: int = Field(5, gt=0,
                                    description="Número de épocas para redução do learning rate")
    reduce_lr_factor: float = Field(0.5, gt=0, lt=1,
                                    description="Fator de redução do learning rate")
    class_weight_adjustment: bool = Field(True,
                                          description="Se deve ajustar pesos para classes desbalanceadas")
