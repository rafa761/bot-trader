# lstm_training.py

"""
Módulo de treinamento de modelo LSTM para previsão de movimentos de preço de Bitcoin,
com otimização automática de hiperparâmetros e foco em previsão de take profit e stop loss.
"""

from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from binance.client import Client
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from core.config import config
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.base import DataCollector, LabelCreator


class DataPreprocessor:
    """Classe para preparação de dados para modelos LSTM."""

    def __init__(self):
        """Inicializa os scalers para features e targets."""
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.tp_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sl_scaler = MinMaxScaler(feature_range=(0, 1))

    def create_sequences(self, data, lookback=60):
        """
        Cria sequências de dados temporais para o LSTM.

        Args:
            data (pd.DataFrame): DataFrame com todas as features
            lookback (int): Número de timesteps anteriores para considerar

        Returns:
            tuple: X_sequences, y_tp, y_sl - dados formatados para treinamento
        """
        # Separa features e targets
        X = data[FEATURE_COLUMNS].values
        y_tp = data['TP_pct'].values.reshape(-1, 1)
        y_sl = data['SL_pct'].values.reshape(-1, 1)

        # Normaliza os dados
        X_scaled = self.feature_scaler.fit_transform(X)
        y_tp_scaled = self.tp_scaler.fit_transform(y_tp)
        y_sl_scaled = self.sl_scaler.fit_transform(y_sl)

        # Cria sequências
        X_sequences = []
        y_tp_out = []
        y_sl_out = []

        for i in range(len(X_scaled) - lookback):
            X_sequences.append(X_scaled[i:i + lookback])
            y_tp_out.append(y_tp_scaled[i + lookback])
            y_sl_out.append(y_sl_scaled[i + lookback])

        return np.array(X_sequences), np.array(y_tp_out), np.array(y_sl_out)

    def save_scalers(self, models_dir):
        """Salva os scalers para uso durante a inferência."""
        import joblib

        scalers_dir = Path(models_dir) / 'scalers'
        scalers_dir.mkdir(exist_ok=True)

        joblib.dump(self.feature_scaler, scalers_dir / 'feature_scaler.pkl')
        joblib.dump(self.tp_scaler, scalers_dir / 'tp_scaler.pkl')
        joblib.dump(self.sl_scaler, scalers_dir / 'sl_scaler.pkl')

        logger.info(f"Scalers salvos em {scalers_dir}")

    def inverse_transform_tp(self, y_scaled):
        """Reverte a transformação do scaler para valores reais de TP."""
        return self.tp_scaler.inverse_transform(y_scaled)

    def inverse_transform_sl(self, y_scaled):
        """Reverte a transformação do scaler para valores reais de SL."""
        return self.sl_scaler.inverse_transform(y_scaled)


class LSTMModelBuilder:
    """Classe para construção e treinamento de modelos LSTM."""

    @staticmethod
    def build_model(input_shape, trial=None):
        """
        Constrói o modelo LSTM usando hiperparâmetros do Optuna ou valores padrão.

        Args:
            input_shape (tuple): Formato dos dados de entrada (lookback, n_features)
            trial (optuna.Trial, optional): Trial para otimização de hiperparâmetros

        Returns:
            Sequential: Modelo Keras compilado
        """
        if trial:
            # Hiperparâmetros para otimização
            n_layers = trial.suggest_int('n_layers', 1, 3)
            n_units = [trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)]
            dropouts = [trial.suggest_float(f'dropout_l{i}', 0.0, 0.5) for i in range(n_layers)]
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        else:
            # Hiperparâmetros padrão
            n_layers = 2
            n_units = [128, 64]
            dropouts = [0.2, 0.2]
            learning_rate = 0.001

        model = Sequential()

        # Primeira camada LSTM com retorno de sequências para camadas LSTM subsequentes
        model.add(LSTM(n_units[0], return_sequences=(n_layers > 1),
                       input_shape=input_shape))
        model.add(Dropout(dropouts[0]))

        # Camadas intermediárias LSTM
        for i in range(1, n_layers - 1):
            model.add(LSTM(n_units[i], return_sequences=True))
            model.add(Dropout(dropouts[i]))

        # Última camada LSTM (se houver mais de uma)
        if n_layers > 1:
            model.add(LSTM(n_units[-1]))
            model.add(Dropout(dropouts[-1]))

        # Camada de saída (1 unidade para regressão)
        model.add(Dense(1))

        # Compilação do modelo
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    @staticmethod
    def train_model(model, X_train, y_train, X_val, y_val, model_path, patience=50, epochs=200, batch_size=32):
        """
        Treina o modelo LSTM com early stopping e learning rate adaptativo.

        Args:
            model (Sequential): Modelo Keras a ser treinado
            X_train, y_train: Dados de treinamento
            X_val, y_val: Dados de validação
            model_path (str): Caminho para salvar o melhor modelo
            patience (int): Número de épocas a esperar antes de early stopping
            epochs (int): Número máximo de épocas
            batch_size (int): Tamanho do batch para treinamento

        Returns:
            History: Histórico de treinamento do Keras
        """
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=model_path, save_best_only=True, monitor='val_loss', verbose=1)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return history

    @staticmethod
    def evaluate_model(model, X_test, y_test, preprocessor, model_type):
        """
        Avalia o modelo com métricas relevantes para trading.

        Args:
            model (Sequential): Modelo treinado
            X_test, y_test: Dados de teste
            preprocessor (DataPreprocessor): Preprocessador com scalers
            model_type (str): Tipo do modelo ('tp' ou 'sl')

        Returns:
            dict: Dicionário com métricas de avaliação
        """
        y_pred_scaled = model.predict(X_test)

        # Converte previsões de volta para a escala original
        if model_type == 'tp':
            y_pred = preprocessor.inverse_transform_tp(y_pred_scaled)
            y_true = preprocessor.inverse_transform_tp(y_test)
        else:
            y_pred = preprocessor.inverse_transform_sl(y_pred_scaled)
            y_true = preprocessor.inverse_transform_sl(y_test)

        # Calcula métricas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Calcula acurácia direcional (se o sinal está correto)
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        direction_accuracy = np.mean((y_true_flat > 0) == (y_pred_flat > 0))

        # Logs para depuração
        logger.info(f"[{model_type.upper()}] Primeiras previsões vs. valores reais:")
        for i in range(min(5, len(y_true))):
            logger.info(f"Real: {y_true[i][0]:.4f}, Previsto: {y_pred[i][0]:.4f}")

        logger.info(f"Métricas do modelo {model_type.upper()}:")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"Acurácia direcional: {direction_accuracy:.2%}")

        return {
            'mae': mae,
            'rmse': rmse,
            'direction_accuracy': direction_accuracy
        }


class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros usando Optuna."""

    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape

    def objective(self, trial):
        """
        Função objetivo para Optuna otimizar.

        Args:
            trial (optuna.Trial): Trial atual

        Returns:
            float: Valor da função objetivo (validação loss)
        """
        # Constrói modelo com hiperparâmetros do trial
        model = LSTMModelBuilder.build_model(self.input_shape, trial)

        # Parâmetros de treinamento
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = 50  # Limitar épocas para otimização

        # Callbacks para treinamento
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            TFKerasPruningCallback(trial, 'val_loss')
        ]

        # Treina o modelo
        history = model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )

        # Retorna a melhor loss de validação
        return min(history.history['val_loss'])

    def optimize(self, n_trials=50, study_name=None):
        """
        Executa a otimização de hiperparâmetros.

        Args:
            n_trials (int): Número de trials
            study_name (str, optional): Nome do estudo para armazenamento

        Returns:
            dict: Melhores hiperparâmetros
        """
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(
            direction='minimize',
            pruner=pruner,
            study_name=study_name
        )

        study.optimize(self.objective, n_trials=n_trials)

        logger.info(f"Melhor trial: {study.best_trial.number}")
        logger.info(f"Melhor valor: {study.best_trial.value}")
        logger.info("Melhores hiperparâmetros:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        return study.best_trial.params


def main():
    """Função principal para execução do treinamento do modelo."""
    logger.info("Iniciando treinamento do modelo LSTM para trading de Bitcoin")

    # Configurações de GPU (opcional)
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPUs disponíveis: {len(gpus)}")
        else:
            logger.info("Nenhuma GPU encontrada, usando CPU")
    except Exception as e:
        logger.warning(f"Erro ao configurar GPU: {e}")

    # 1. Coletar dados históricos
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, requests_params={"timeout": 30})
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines(
        config.SYMBOL,
        config.INTERVAL,
        config.MODEL_DATA_TRAINING_START_DATE
    )

    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando.")
        return

    logger.info(f"Dados coletados: {len(df)} registros")

    # 2. Criar labels para TP e SL
    df = LabelCreator.create_labels(df, config.MODEL_DATA_PREDICTION_HORIZON)
    if df.empty or 'TP_pct' not in df.columns:
        logger.error("Falha ao criar labels. Encerrando.")
        return

    # 3. Verificar se todas as features necessárias estão presentes
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        logger.error(f"Features faltando: {missing_features}")
        return

    # 4. Preparar dados para sequências LSTM
    preprocessor = DataPreprocessor()
    lookback = 60  # Janela de observação (parâmetro ajustável)

    # Criar sequências de dados
    X_sequences, y_tp, y_sl = preprocessor.create_sequences(df, lookback=lookback)
    logger.info(f"Sequências criadas: {X_sequences.shape}")

    # 5. Dividir dados em treino, validação e teste (80/10/10)
    # Preserva a ordem temporal dos dados
    train_size = int(0.8 * len(X_sequences))
    val_size = int(0.1 * len(X_sequences))

    X_train, y_tp_train, y_sl_train = (
        X_sequences[:train_size],
        y_tp[:train_size],
        y_sl[:train_size]
    )
    X_val, y_tp_val, y_sl_val = (
        X_sequences[train_size:train_size + val_size],
        y_tp[train_size:train_size + val_size],
        y_sl[train_size:train_size + val_size]
    )
    X_test, y_tp_test, y_sl_test = (
        X_sequences[train_size + val_size:],
        y_tp[train_size + val_size:],
        y_sl[train_size + val_size:]
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Garantir que a pasta de modelos exista
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Salvar os scalers para uso durante a inferência
    preprocessor.save_scalers(TRAINED_MODELS_DIR)

    # Forma de entrada para o LSTM
    input_shape = (X_train.shape[1], X_train.shape[2])

    # 6. Otimização de hiperparâmetros para o modelo TP
    logger.info("Iniciando otimização de hiperparâmetros para modelo TP")
    tp_optimizer = HyperparameterOptimizer(
        X_train, y_tp_train,
        X_val, y_tp_val,
        input_shape
    )
    best_params_tp = tp_optimizer.optimize(n_trials=30, study_name="tp_optimization")

    # 7. Treinar modelo TP com os melhores hiperparâmetros
    logger.info("Treinando modelo TP com os melhores hiperparâmetros")
    model_tp = LSTMModelBuilder.build_model(input_shape, best_params_tp)  # Usar melhores params
    model_tp_path = str(TRAINED_MODELS_DIR / 'lstm_model_tp.h5')

    history_tp = LSTMModelBuilder.train_model(
        model_tp, X_train, y_tp_train, X_val, y_tp_val,
        model_tp_path,
        patience=50,
        epochs=200,
        batch_size=best_params_tp.get('batch_size', 32)
    )

    # Carrega o melhor modelo salvo
    model_tp = load_model(model_tp_path)

    # 8. Otimização de hiperparâmetros para o modelo SL
    logger.info("Iniciando otimização de hiperparâmetros para modelo SL")
    sl_optimizer = HyperparameterOptimizer(
        X_train, y_sl_train,
        X_val, y_sl_val,
        input_shape
    )
    best_params_sl = sl_optimizer.optimize(n_trials=30, study_name="sl_optimization")

    # 9. Treinar modelo SL com os melhores hiperparâmetros
    logger.info("Treinando modelo SL com os melhores hiperparâmetros")
    model_sl = LSTMModelBuilder.build_model(input_shape, best_params_sl)  # Usar melhores params
    model_sl_path = str(TRAINED_MODELS_DIR / 'lstm_model_sl.h5')

    history_sl = LSTMModelBuilder.train_model(
        model_sl, X_train, y_sl_train, X_val, y_sl_val,
        model_sl_path,
        patience=50,
        epochs=200,
        batch_size=best_params_sl.get('batch_size', 32)
    )

    # Carrega o melhor modelo salvo
    model_sl = load_model(model_sl_path)

    # 10. Avaliação final dos modelos no conjunto de teste
    logger.info("Avaliando modelo TP no conjunto de teste")
    tp_metrics = LSTMModelBuilder.evaluate_model(model_tp, X_test, y_tp_test, preprocessor, 'tp')

    logger.info("Avaliando modelo SL no conjunto de teste")
    sl_metrics = LSTMModelBuilder.evaluate_model(model_sl, X_test, y_sl_test, preprocessor, 'sl')

    # 11. Salva métrica de avaliação para referência futura
    import json
    evaluation = {
        'tp_model': tp_metrics,
        'sl_model': sl_metrics,
        'params': {
            'tp': best_params_tp,
            'sl': best_params_sl
        }
    }

    with open(TRAINED_MODELS_DIR / 'model_evaluation.json', 'w') as f:
        json.dump(evaluation, f, indent=4)

    logger.info("Treinamento de modelos LSTM concluído com sucesso!")


if __name__ == "__main__":
    main()
