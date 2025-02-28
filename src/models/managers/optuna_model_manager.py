# models\managers\optuna_model_manager.py

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from core.logger import logger
from models.lstm.hyperparameter_tuner import LSTMHyperparameterTuner
from models.managers.model_manager import ModelManager


class OptunaModelManager(ModelManager):
    """
    Extensão do ModelManager que adiciona suporte à tunagem de hiperparâmetros com Optuna.

    Mantém toda a funcionalidade original do ModelManager, adicionando uma etapa
    de tunagem de hiperparâmetros antes do treinamento quando esta está habilitada.
    """

    def execute_full_pipeline(
            self,
            data: pd.DataFrame,
            feature_columns: list[str],
            target_column: str,
            save_path: Path,
            epochs: int = None,
    ) -> dict:
        """
        Executa o pipeline completo de treinamento com tunagem de hiperparâmetros.

        Args:
            data: DataFrame com os dados para treinamento.
            feature_columns: Lista de colunas a serem usadas como features.
            target_column: Coluna a ser usada como alvo.
            save_path: Caminho onde o modelo será salvo.
            epochs: Número de épocas para treinamento, se None usa o valor padrão do modelo

        Returns:
            Dicionário com as métricas de avaliação do modelo.

        Raises:
            RuntimeError: Se ocorrer algum erro durante o pipeline.
        """
        try:
            logger.info(
                f"Iniciando pipeline de treinamento do modelo {self.config.model_name} com suporte à tunagem...")

            # Se epochs foi especificado, salvar temporariamente e restaurar depois
            original_epochs = None
            if epochs is not None and hasattr(self.model, 'config'):
                original_epochs = self.model.config.epochs
                self.model.config.epochs = epochs
                logger.info(f"Usando {epochs} épocas para este treinamento")

            # 1. Preparação dos dados
            logger.info("Preparando os dados...")
            X = data[feature_columns]
            y = data[target_column]
            logger.info("Dados preparados com sucesso.")

            # 2. Divisão dos dados
            X_train, X_test, y_train, y_test = self._split_data(
                X, y,
                test_size=self.trainer.config.test_size,
                shuffle=self.trainer.config.shuffle
            )

            # 3. Tunagem de hiperparâmetros (se habilitada)
            if hasattr(self.model.config, 'optuna_config') and self.model.config.optuna_config.enabled:
                logger.info("Iniciando tunagem de hiperparâmetros com Optuna...")

                # Dividir dados de treino para tunagem
                X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
                    X_train, y_train,
                    test_size=self.trainer.config.validation_split,
                    random_state=42,
                    shuffle=self.trainer.config.shuffle
                )

                # Instanciar e executar o tuner
                tuner = LSTMHyperparameterTuner(
                    model_config=self.model.config,
                    training_config=self.trainer.config,
                    X_train=X_train_tune,
                    y_train=y_train_tune,
                    X_val=X_val_tune,
                    y_val=y_val_tune
                )

                # Executar tunagem e obter configuração otimizada
                optimized_config = tuner.tune()

                # Mostrar os melhores parâmetros
                best_params = tuner.get_best_params()
                logger.info(f"Melhores parâmetros encontrados: {best_params}")
                logger.info(f"Melhor valor da métrica: {tuner.get_best_value()}")

                # Atualizar o modelo com a configuração otimizada
                # Usar a classe concreta do modelo atual para criar uma nova instância
                self.model = type(self.model)(optimized_config)

                # Atualizar a referência do modelo no treinador
                self.trainer.model = self.model

                # Atualizar a configuração armazenada
                self.config = optimized_config

                logger.info("Modelo atualizado com hiperparâmetros otimizados.")

            # 4. Treinamento do modelo (com parâmetros otimizados se a tunagem foi realizada)
            logger.info(f"Treinando o modelo...")
            self.trainer.train(X_train, y_train)
            logger.info("Modelo treinado com sucesso.")

            # 5. Avaliação
            logger.info("Avaliando o modelo...")
            self.metrics = self.trainer.evaluate(X_test, y_test)
            logger.info("Modelo avaliado com sucesso.")
            logger.info(f"Metricas: {self.metrics}")

            # 6. Salvamento (mantendo o mesmo nome e local)
            logger.info("Salvando o modelo...")
            model_path = save_path / f"{self.config.model_name}.keras"
            self.model.save(model_path)
            logger.info(f"Modelo salvo com sucesso em {model_path}.")

            # Restaurar configuração original de épocas
            if original_epochs is not None:
                self.model.config.epochs = original_epochs

            return self.metrics

        except Exception as e:
            error_msg = f"Falha no pipeline de treinamento: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
