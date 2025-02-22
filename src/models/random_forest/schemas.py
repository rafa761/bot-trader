from pydantic import Field

from models.base.schemas import ModelConfig


class RandomForestConfig(ModelConfig):
    """Configuração específica para Random Forest"""
    model_type: str = "random_forest"
    n_estimators: int | None = Field(100, gt=0, description="Número de árvores na floresta")
    max_depth: int | None = Field(None, description="Profundidade máxima das árvores")
    min_samples_split: int | None = Field(2, gt=1, description="Mínimo de amostras para dividir um nó")
    min_samples_leaf: int | None = Field(2, description="Mínimo de amostras em um nó folha")
    max_features: str | float | None = Field(1, description="Máximo de features para considerar em cada split")
    random_state: int | None = Field(42, description="Seed para reprodutibilidade")
    feature_columns: list[str] = Field(default_factory=list, description="Colunas de features")
