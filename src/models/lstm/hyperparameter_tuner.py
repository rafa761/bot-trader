# models\lstm\hyperparameter_tuner.py

import copy
from typing import Any

import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig
from models.lstm.trainer import LSTMTrainer


class LSTMHyperparameterTuner:
    """
    Classe responsável pela tunagem de hiperparâmetros do modelo LSTM usando Optuna.

    Implementa uma otimização bayesiana dos hiperparâmetros do modelo LSTM
    para encontrar a configuração que minimize a métrica de validação especificada.
    """

    def __init__(self,
                 model_config: LSTMConfig,
                 training_config: LSTMTrainingConfig,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_val: pd.DataFrame | None = None,
                 y_val: pd.Series | None = None):
        """
        Inicializa o tuner com as configurações base e dados de treinamento.

        Args:
            model_config: Configuração base do modelo LSTM
            training_config: Configuração de treinamento
            X_train: Features de treinamento
            y_train: Target de treinamento
            X_val: Features de validação (opcional)
            y_val: Target de validação (opcional)
        """
        self.base_model_config = model_config
        self.base_training_config = training_config
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.optuna_config = model_config.optuna_config
        self.best_trial = None
        self.best_value = None
        self.best_model_config = None

    def _create_model_with_params(self, params: dict[str, Any]) -> tuple[LSTMModel, LSTMTrainer]:
        """
        Cria um modelo LSTM e seu treinador com os parâmetros sugeridos pelo Optuna.

        Args:
            params: Dicionário de parâmetros sugeridos pelo Optuna

        Returns:
            Tupla contendo o modelo e o treinador configurados
        """
        # Criar uma cópia da configuração base
        model_config = copy.deepcopy(self.base_model_config)
        training_config = copy.deepcopy(self.base_training_config)

        # Atualizar configuração com parâmetros otimizados
        for key, value in params.items():
            if key.startswith('lstm_units_'):
                layer_idx = int(key.split('_')[-1])
                while len(model_config.lstm_units) <= layer_idx:
                    model_config.lstm_units.append(64)  # valor padrão
                model_config.lstm_units[layer_idx] = value
            elif key.startswith('dense_units_'):
                layer_idx = int(key.split('_')[-1])
                while len(model_config.dense_units) <= layer_idx:
                    model_config.dense_units.append(32)  # valor padrão
                model_config.dense_units[layer_idx] = value
            elif key == 'num_lstm_layers':
                # Ajustar o número de camadas LSTM
                model_config.lstm_units = model_config.lstm_units[:value]
            elif key == 'num_dense_layers':
                # Ajustar o número de camadas densas
                model_config.dense_units = model_config.dense_units[:value]
            elif hasattr(model_config, key):
                setattr(model_config, key, value)
            elif hasattr(training_config, key):
                setattr(training_config, key, value)

        # Criar modelo e treinador
        model = LSTMModel(model_config)
        trainer = LSTMTrainer(model, training_config)

        return model, trainer

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para o Optuna. Define o espaço de busca e avalia o modelo.

        Args:
            trial: Objeto Trial do Optuna

        Returns:
            Valor da métrica a ser otimizada (menor é melhor, por padrão)
        """
        # Definir o espaço de busca dos hiperparâmetros
        params = {
            # Parâmetros do modelo
            'sequence_length': trial.suggest_int('sequence_length', 16, 64, log=True),
            'batch_size': trial.suggest_int('batch_size', 32, 512, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'recurrent_dropout_rate': trial.suggest_float('recurrent_dropout_rate', 0.0, 0.3),
            'l2_regularization': trial.suggest_float('l2_regularization', 1e-6, 1e-3, log=True),

            # Número de camadas
            'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 3),
            'num_dense_layers': trial.suggest_int('num_dense_layers', 1, 3),
        }

        # Definir unidades para cada camada LSTM
        for i in range(params['num_lstm_layers']):
            params[f'lstm_units_{i}'] = trial.suggest_int(f'lstm_units_{i}', 32, 256, log=True)

        # Definir unidades para cada camada densa
        for i in range(params['num_dense_layers']):
            params[f'dense_units_{i}'] = trial.suggest_int(f'dense_units_{i}', 16, 128, log=True)

        # Parâmetros de treinamento
        params['validation_split'] = trial.suggest_float('validation_split', 0.1, 0.3)
        params['early_stopping_patience'] = trial.suggest_int('early_stopping_patience', 5, 20)
        params['reduce_lr_patience'] = trial.suggest_int('reduce_lr_patience', 2, 10)
        params['reduce_lr_factor'] = trial.suggest_float('reduce_lr_factor', 0.1, 0.9)

        try:
            # Criar modelo e treinador com os parâmetros sugeridos
            model, trainer = self._create_model_with_params(params)

            # Treinar o modelo
            trainer.train(self.X_train, self.y_train)

            # Avaliar o modelo
            if self.X_val is not None and self.y_val is not None:
                metrics = trainer.evaluate(self.X_val, self.y_val)
                metric_value = metrics.get(self.optuna_config.metric.replace('val_', 'test_'), float('inf'))
            else:
                # Usar métricas de validação do histórico de treinamento
                history = trainer.history
                if self.optuna_config.metric == 'val_loss':
                    metric_value = min(history.history['val_loss'])
                elif self.optuna_config.metric == 'val_mae':
                    metric_value = min(history.history['val_mae'])
                else:
                    raise ValueError(f"Métrica {self.optuna_config.metric} não suportada")

            # Reportar valores intermediários para pruning
            if history:
                for epoch, val_loss in enumerate(history.history['val_loss']):
                    trial.report(val_loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            return metric_value

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Erro durante a otimização: {e}")
            # Retornar um valor alto para indicar que este trial falhou
            return float('inf')

    def tune(self) -> LSTMConfig:
        """
        Executa a tunagem de hiperparâmetros e retorna a configuração otimizada.

        Returns:
            Configuração otimizada do modelo LSTM
        """
        if not self.optuna_config.enabled:
            logger.info("Tunagem de hiperparâmetros desativada. Usando configuração base.")
            return self.base_model_config

        logger.info(f"Iniciando tunagem de hiperparâmetros com {self.optuna_config.n_trials} trials...")

        # Criar um estudo Optuna
        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            direction=self.optuna_config.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            storage=self.optuna_config.storage,
            load_if_exists=True
        )

        # Executar a otimização
        study.optimize(
            self._objective,
            n_trials=self.optuna_config.n_trials,
            timeout=self.optuna_config.timeout
        )

        # Obter o melhor trial
        self.best_trial = study.best_trial
        self.best_value = study.best_value

        logger.info(f"Tunagem concluída. Melhor valor: {self.best_value}")
        logger.info(f"Melhores parâmetros: {self.best_trial.params}")

        # Criar configuração otimizada
        self.best_model_config = copy.deepcopy(self.base_model_config)

        # Atualizar configuração com os melhores parâmetros
        for key, value in self.best_trial.params.items():
            if key.startswith('lstm_units_'):
                layer_idx = int(key.split('_')[-1])
                while len(self.best_model_config.lstm_units) <= layer_idx:
                    self.best_model_config.lstm_units.append(64)
                self.best_model_config.lstm_units[layer_idx] = value
            elif key.startswith('dense_units_'):
                layer_idx = int(key.split('_')[-1])
                while len(self.best_model_config.dense_units) <= layer_idx:
                    self.best_model_config.dense_units.append(32)
                self.best_model_config.dense_units[layer_idx] = value
            elif key == 'num_lstm_layers':
                # Ajustar o número de camadas LSTM
                self.best_model_config.lstm_units = self.best_model_config.lstm_units[:value]
            elif key == 'num_dense_layers':
                # Ajustar o número de camadas densas
                self.best_model_config.dense_units = self.best_model_config.dense_units[:value]
            elif hasattr(self.best_model_config, key):
                setattr(self.best_model_config, key, value)

        return self.best_model_config

    def get_best_params(self) -> dict[str, Any]:
        """
        Retorna os melhores parâmetros encontrados durante a tunagem.

        Returns:
            Dicionário com os melhores parâmetros
        """
        if self.best_trial is None:
            return {}
        return self.best_trial.params

    def get_best_value(self) -> float:
        """
        Retorna o melhor valor da métrica obtido durante a tunagem.

        Returns:
            Melhor valor da métrica
        """
        if self.best_value is None:
            return float('inf')
        return self.best_value
