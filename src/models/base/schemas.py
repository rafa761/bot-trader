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
