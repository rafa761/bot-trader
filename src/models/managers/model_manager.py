# models\managers\model_manager.py

from pathlib import Path

import pandas as pd

from core.logger import logger
from models.base.model import BaseModel
from models.base.schemas import ModelConfig
from models.base.trainer import BaseTrainer


class ModelManager:
    """
    Gerenciador genérico para orquestrar o treinamento de modelos.

    Responsável por coordenar o fluxo completo de treinamento, desde a preparação
    dos dados até a avaliação e salvamento do modelo.
    """

    def __init__(
            self,
            model: BaseModel,
            trainer: BaseTrainer,
            config: ModelConfig
    ):
        """
        Inicializa o gerenciador com o modelo, treinador e configuração especificados.

        Args:
            model: Instância do modelo a ser treinado.
            trainer: Instância do treinador para o modelo.
            config: Configuração do modelo.
        """
        self.model = model
        self.trainer = trainer
        self.config = config
        self.metrics: dict | None = None

    def execute_full_pipeline(
            self,
            data: pd.DataFrame,
            feature_columns: list[str],
            target_column: str,
            save_path: Path,
    ) -> dict:
        """
        Executa o pipeline completo de treinamento.

        Args:
            data: DataFrame com os dados para treinamento.
            feature_columns: Lista de colunas a serem usadas como features.
            target_column: Coluna a ser usada como alvo.
            save_path: Caminho onde o modelo será salvo.

        Returns:
            Dicionário com as métricas de avaliação do modelo.

        Raises:
            RuntimeError: Se ocorrer algum erro durante o pipeline.
        """
        try:
            logger.info(f"Iniciando pipeline de treinamento do modelo {self.config.model_name}...")

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
            model_path = save_path / f"{self.config.model_name}.keras"
            self.model.save(model_path)
            logger.info(f"Modelo salvo com sucesso em {model_path}.")

            return self.metrics

        except Exception as e:
            error_msg = f"Falha no pipeline de treinamento: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    @staticmethod
    def _split_data(
            X: pd.DataFrame,
            y: pd.Series,
            test_size: float = 0.2,
            shuffle: bool = False
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Método interno para divisão dos dados em conjuntos de treino e teste.

        Args:
            X: DataFrame com as features.
            y: Series com os alvos.
            test_size: Proporção dos dados a ser usada para teste (0.0 a 1.0).
            shuffle: Se True, embaralha os dados antes da divisão.

        Returns:
            Tupla contendo (X_train, X_test, y_train, y_test).

        Raises:
            RuntimeError: Se ocorrer algum erro durante a divisão.
        """
        logger.info("Dividindo os dados...")
        try:
            # Para séries temporais, normalmente não embaralhamos os dados
            if shuffle:
                from sklearn.model_selection import train_test_split
                return train_test_split(X, y, test_size=test_size, random_state=42)
            else:
                # Implementação básica - para séries temporais
                test_samples = int(len(X) * test_size)
                train_size = len(X) - test_samples

                return (
                    X.iloc[:train_size],
                    X.iloc[train_size:],
                    y.iloc[:train_size],
                    y.iloc[train_size:]
                )

        except Exception as e:
            error_msg = f"Erro ao dividir os dados: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
