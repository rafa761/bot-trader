from pathlib import Path

import joblib
import pandas as pd
from joblib import Memory
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from core.constants import CACHE_DIR
from models.base.model import BaseModel
from models.random_forest.schemas import RandomForestConfig

memory = Memory(location=CACHE_DIR, verbose=0)


class RandomForestModel(BaseModel):
    """Implementação concreta do modelo Random Forest"""

    def __init__(self, config: RandomForestConfig):
        super().__init__(config)
        self.config = config
        self.pipeline = self.build_model()

    def build_model(self) -> Pipeline:
        """Constrói o pipeline completo do modelo"""
        return Pipeline([
            ('regressor', RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                max_features=self.config.max_features,
                random_state=self.config.random_state
            ))
        ],
            memory=memory
        )

    def predict(self, input_data: pd.DataFrame) -> pd.Series:
        """Executa previsões"""
        return self.pipeline.predict(input_data)

    def save(self, path: Path) -> None:
        """Salva o modelo em disco"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: Path) -> 'RandomForestModel':
        """Carrega o modelo do disco"""
        pipeline = joblib.load(path)
        model = cls(RandomForestConfig(model_name="loaded_model"))
        model.pipeline = pipeline
        return model
