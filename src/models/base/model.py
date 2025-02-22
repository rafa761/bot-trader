from abc import ABC, abstractmethod
from pathlib import Path

from models.base.schemas import ModelConfig


class BaseModel(ABC):
    """Classe abstrata para todos os modelos"""

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def build_model(self):
        """Método para construir a arquitetura do modelo"""
        pass

    @abstractmethod
    def predict(self, input_data):
        """Método para fazer previsões"""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Salva o modelo em disco"""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path):
        """Carrega o modelo do disco"""
        pass
