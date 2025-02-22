from pathlib import Path
from typing import Generic, TypeVar
import pandas as pd

from core.logger import logger
from models.base.schemas import ModelConfig
from models.base.model import BaseModel
from models.base.trainer import BaseTrainer

class ModelManager:
    """Gerenciador genérico para orquestrar o treinamento de modelos"""
    def __init__(
        self,
        model: BaseModel,
        trainer: BaseTrainer,
        config: ModelConfig
    ):
        self.model = model
        self.trainer = trainer
        self.config = config
        self.metrics: dict | None = None

    def execute_full_pipeline(
        self,
        data: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
        save_path: Path
    ) -> dict:
        """Executa o pipeline completo de treinamento"""
        try:
            logger.info(f"Iniciando pipeline de treinamento do modelo {self.model.config.model_name}...")

            # 1. Preparação dos dados
            logger.info("Preparando os dados...")
            X = data[feature_columns]
            y = data[target_column]
            logger.info("Dados preparados com sucesso.")

            # 2. Divisão dos dados (pode ser injetado se necessário)
            X_train, X_test, y_train, y_test = self._split_data(X, y)

            # 3. Treinamento
            logger.info(f"Treinando o modelo...")
            self.trainer.train(X_train, y_train)
            logger.info("Modelo treinado com sucesso.")

            # 4. Avaliação
            logger.info("Avaliando o modelo...")
            self.metrics = self.trainer.evaluate(X_test, y_test)
            logger.info("Modelo avaliado com sucesso.")
            logger.info(f"Metricas: {self.metrics}")

            # 5. Salvamento
            logger.info("Salvando o modelo...")
            self.model.save(save_path / f"{self.config.model_name}.pkl")
            logger.info("Modelo salvo com sucesso.")

            return self.metrics

        except Exception as e:
            raise RuntimeError(f"Falha no pipeline de treinamento: {e}") from e

    @staticmethod
    def _split_data(
            X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Método interno para divisão dos dados"""
        logger.info("Dividindo os dados...")
        try:
            # Implementação básica - pode ser substituído por estratégia mais complexa
            test_samples = int(len(X) * test_size)

            splitted_data = (
                X.iloc[:-test_samples],
                X.iloc[-test_samples:],
                y.iloc[:-test_samples],
                y.iloc[-test_samples:]
            )
        except Exception as e:
            raise RuntimeError(f"Erro ao dividir os dados: {e}")

        return splitted_data

