# models\base\schemas.py

from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    """Configuração base para todos os modelos"""
    model_name: str = Field(..., description="Nome único do modelo")
    version: str = Field("1.0.0", description="Versão do modelo")
    description: str | None = Field(None, description="Descrição opcional do modelo")

class TrainingConfig(BaseModel):
    """Configuração base para treinamento"""
    test_size: float = Field(0.2, ge=0.1, le=0.5, description="Tamanho do conjunto de teste")
    random_state: int | None = Field(42, description="Seed para reprodutibilidade")
    shuffle: bool = Field(False, description="Se deve embaralhar os dados")


class RetrainingStatus(BaseModel):
    retraining_in_progress: bool = Field(..., description="Indica se o processo de retreinamento está em andamento")
    last_retraining_time: str = Field(..., description="Timestamp ISO do último retreinamento")
    hours_since_last_retraining: float = Field(..., description="Horas transcorridas desde o último retreinamento")
    recent_error_count: int = Field(..., description="Quantidade de erros de previsão recentes monitorados")
    mean_error: float = Field(..., description="Erro médio das previsões recentes")
    tp_error_mean: float = Field(..., description="Erro médio das previsões de Take Profit")
    sl_error_mean: float = Field(..., description="Erro médio das previsões de Stop Loss")
    tp_model_version: str = Field(..., description="Versão atual do modelo de Take Profit")
    sl_model_version: str = Field(..., description="Versão atual do modelo de Stop Loss")
    next_check_in_cycles: int = Field(..., description="Ciclos restantes até a próxima verificação de retreinamento")
    models_updated_flag: bool = Field(..., description="Indica se os modelos foram atualizados no último retreinamento")
