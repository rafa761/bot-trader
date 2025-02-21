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

    def __init__(self) -> None:
        """Inicializa os scalers para features e targets."""
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.tp_scaler = MinMaxScaler(feature_range=(0, 1))
        self.sl_scaler = MinMaxScaler(feature_range=(0, 1))

    def create_sequences(self, data: pd.DataFrame, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Cria sequências de dados temporais para o LSTM.

        Parameters:
            data: DataFrame com todas as features
            lookback: Número de timesteps anteriores para considerar

        Returns:
            Tuple contendo X_sequences, y_tp, y_sl - dados formatados para treinamento
        """
        # Separa features e targets
        X = data[FEATURE_COLUMNS].values
        y_tp = data["TP_pct"].values.reshape(-1, 1)
        y_sl = data["SL_pct"].values.reshape(-1, 1)

        # Normaliza os dados
        X_scaled = self.feature_scaler.fit_transform(X)
        y_tp_scaled = self.tp_scaler.fit_transform(y_tp)
        y_sl_scaled = self.sl_scaler.fit_transform(y_sl)

        # Cria sequências
        X_sequences: list[np.ndarray] = []
        y_tp_out: list[np.ndarray] = []
        y_sl_out: list[np.ndarray] = []

        for i in range(len(X_scaled) - lookback):
            X_sequences.append(X_scaled[i: i + lookback])
            y_tp_out.append(y_tp_scaled[i + lookback])
            y_sl_out.append(y_sl_scaled[i + lookback])

        return np.array(X_sequences), np.array(y_tp_out), np.array(y_sl_out)

    def save_scalers(self, models_dir: str | Path) -> None:
        """
        Salva os scalers para uso durante a inferência.

        Parameters:
            models_dir: Diretório onde os scalers serão salvos
        """
        import joblib

        scalers_dir = Path(models_dir) / "scalers"
        scalers_dir.mkdir(exist_ok=True)

        joblib.dump(self.feature_scaler, scalers_dir / "feature_scaler.pkl")
        joblib.dump(self.tp_scaler, scalers_dir / "tp_scaler.pkl")
        joblib.dump(self.sl_scaler, scalers_dir / "sl_scaler.pkl")

        logger.info(f"Scalers salvos em {scalers_dir}")

    def inverse_transform_tp(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a transformação do scaler para valores reais de TP.

        Parameters:
            y_scaled: Valores normalizados de TP

        Returns:
            Valores reais de TP
        """
        return self.tp_scaler.inverse_transform(y_scaled)

    def inverse_transform_sl(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Reverte a transformação do scaler para valores reais de SL.

        Parameters:
            y_scaled: Valores normalizados de SL

        Returns:
            Valores reais de SL
        """
        return self.sl_scaler.inverse_transform(y_scaled)


class LSTMModelBuilder:
    """Classe para construção e treinamento de modelos LSTM."""

    @staticmethod
    def build_model(input_shape: tuple[int, int], params: dict | optuna.Trial | None = None) -> Sequential:
        """
        Constrói o modelo LSTM usando hiperparâmetros do Optuna ou valores padrão simplificados.

        Parameters:
            input_shape: Formato dos dados de entrada (lookback, n_features)
            params: Dicionário de parâmetros ou objeto Trial do Optuna (opcional)

        Returns:
            Modelo Keras compilado
        """
        if isinstance(params, optuna.Trial):
            # Hiperparâmetros otimizados pelo Optuna
            n_layers = params.suggest_int("n_layers", 1, 2)  # Reduzido para 2 camadas máximas
            n_units = [params.suggest_int(f"n_units_l{i}", 32, 128) for i in range(n_layers)]  # Menos unidades
            dropouts = [params.suggest_float(f"dropout_l{i}", 0.0, 0.3) for i in range(n_layers)]  # Dropout menor
            learning_rate = params.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        elif isinstance(params, dict):
            # Usar parâmetros pré-definidos (dicionário)
            n_layers = params.get("n_layers", 1)
            n_units = [params.get(f"n_units_l{i}", 64) for i in range(n_layers)]
            dropouts = [params.get(f"dropout_l{i}", 0.2) for i in range(n_layers)]
            learning_rate = params.get("learning_rate", 0.001)
        else:
            # Configuração padrão simplificada para rapidez
            n_layers = 1  # Apenas uma camada para acelerar
            n_units = [64]  # Menos unidades
            dropouts = [0.2]
            learning_rate = 0.001

        model = Sequential()
        model.add(Input(shape=input_shape))
        # Primeira camada LSTM
        model.add(LSTM(n_units[0], return_sequences=(n_layers > 1), input_shape=input_shape))
        model.add(Dropout(dropouts[0]))

        # Camadas adicionais apenas se n_layers > 1
        if n_layers > 1:
            model.add(LSTM(n_units[1]))
            model.add(Dropout(dropouts[1]))

        # Camada de saída
        model.add(Dense(1))

        # Compilação com otimizador Adam
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")

        return model

    @staticmethod
    def train_model(
            model: Sequential,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            model_path: str,
            patience: int,
            epochs: int,
            batch_size: int,
    ) -> tf.keras.callbacks.History:
        """
        Treina o modelo LSTM com early stopping e learning rate adaptativo.

        Parameters:
            model: Modelo Keras a ser treinado
            X_train: Dados de treinamento (features)
            y_train: Dados de treinamento (target)
            X_val: Dados de validação (features)
            y_val: Dados de validação (target)
            model_path: Caminho para salvar o melhor modelo
            patience: Número de épocas para early stopping
            epochs: Número máximo de épocas
            batch_size: Tamanho do batch para treinamento

        Returns:
            Histórico de treinamento do Keras
        """
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=patience // 2, min_lr=1e-6, verbose=1),
            ModelCheckpoint(filepath=model_path, save_best_only=True, monitor="val_loss", verbose=1),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        return history

    @staticmethod
    def evaluate_model(
            model: Sequential, X_test: np.ndarray, y_test: np.ndarray, preprocessor: "DataPreprocessor", model_type: str
    ) -> dict[str, float]:
        """
        Avalia o modelo com métricas relevantes para trading.

        Parameters:
            model: Modelo treinado
            X_test: Dados de teste (features)
            y_test: Dados de teste (target)
            preprocessor: Instância de DataPreprocessor com scalers
            model_type: Tipo do modelo ('tp' ou 'sl')

        Returns:
            Dicionário com métricas de avaliação (mae, rmse, direction_accuracy)
        """
        y_pred_scaled = model.predict(X_test, verbose=0)

        # Converte previsões para escala original
        if model_type == "tp":
            y_pred = preprocessor.inverse_transform_tp(y_pred_scaled)
            y_true = preprocessor.inverse_transform_tp(y_test)
        else:
            y_pred = preprocessor.inverse_transform_sl(y_pred_scaled)
            y_true = preprocessor.inverse_transform_sl(y_test)

        # Calcula métricas
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        direction_accuracy = np.mean((y_true.flatten() > 0) == (y_pred.flatten() > 0))

        # Logs para depuração
        logger.info(f"[{model_type.upper()}] Primeiras previsões vs. valores reais:")
        for i in range(min(5, len(y_true))):
            logger.info(f"Real: {y_true[i][0]:.4f}, Previsto: {y_pred[i][0]:.4f}")
        logger.info(
            f"Métricas do modelo {model_type.upper()}: MAE={mae:.4f}, RMSE={rmse:.4f}, Dir.Acc={direction_accuracy:.2%}")

        return {"mae": mae, "rmse": rmse, "direction_accuracy": direction_accuracy}


class HyperparameterOptimizer:
    """Classe para otimização de hiperparâmetros usando Optuna."""

    def __init__(
            self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            input_shape: tuple[int, int]
    ) -> None:
        """
        Inicializa o otimizador de hiperparâmetros.

        Parameters:
            X_train: Dados de treinamento (features)
            y_train: Dados de treinamento (target)
            X_val: Dados de validação (features)
            y_val: Dados de validação (target)
            input_shape: Formato dos dados de entrada (lookback, n_features)
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape

    def objective(self, trial: optuna.Trial) -> float:
        """
        Função objetivo para Otuna otimizar.

        Parameters:
            trial: Trial atual para teste de hiperparâmetros

        Returns:
            Valor da função objetivo (validação loss)
        """
        model = LSTMModelBuilder.build_model(self.input_shape, trial)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Batch sizes maiores para rapidez

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=config.MODEL_PATIENCE, restore_best_weights=True),
            TFKerasPruningCallback(trial, "val_loss"),
        ]

        history = model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),  # Corrigido: usa atributos da instância
            epochs=config.MODEL_EPOCHS_OPTUNA,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        return min(history.history["val_loss"])

    def optimize(self, n_trials: int, study_name: str | None = None) -> dict[str, int | float]:
        """
        Executa a otimização de hiperparâmetros.

        Parameters:
            n_trials: Número de trials a executar
            study_name: Nome do estudo para armazenamento (opcional)

        Returns:
            Melhores hiperparâmetros encontrados
        """
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction="minimize", pruner=pruner, study_name=study_name)
        study.optimize(self.objective, n_trials=n_trials)

        logger.info(f"Melhor trial: {study.best_trial.number}, Melhor valor: {study.best_trial.value}")
        logger.info("Melhores hiperparâmetros:")
        for key, value in study.best_trial.params.items():
            logger.info(f"    {key}: {value}")

        return study.best_trial.params


def main() -> None:
    """Função principal para execução do treinamento do modelo."""
    logger.info("Iniciando treinamento do modelo LSTM para trading de Bitcoin")

    # Configuração de GPU
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(
                f"GPUs disponíveis: {len(gpus)} - Detalhes: {[tf.config.experimental.get_device_details(gpu) for gpu in gpus]}")
        else:
            logger.info("Nenhuma GPU encontrada, usando CPU")
    except Exception as e:
        logger.warning(f"Erro ao configurar GPU: {e}")

    # Coletar dados históricos da Binance
    client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET, requests_params={"timeout": 30})
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines(config.SYMBOL, config.INTERVAL, config.MODEL_DATA_TRAINING_START_DATE)

    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando.")
        return

    logger.info(f"Dados coletados: {len(df)} registros")

    # Reduzir o tamanho do dataset para acelerar (opcional, configurável)
    if config.MODEL_SAMPLE_FRACTION < 1.0:
        df = df.sample(frac=config.MODEL_SAMPLE_FRACTION, random_state=42)
        logger.info(f"Amostra reduzida para {len(df)} registros ({config.MODEL_SAMPLE_FRACTION * 100:.0f}%)")

    # Criar labels para TP e SL
    df = LabelCreator.create_labels(df, config.MODEL_DATA_PREDICTION_HORIZON)
    if df.empty or "TP_pct" not in df.columns:
        logger.error("Falha ao criar labels. Encerrando.")
        return

    # Verificar features necessárias
    missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_features:
        logger.error(f"Features faltando: {missing_features}")
        return

    # Preparar dados
    preprocessor = DataPreprocessor()
    X_sequences, y_tp, y_sl = preprocessor.create_sequences(df, config.MODEL_DATA_LOOKBACK)
    logger.info(f"Sequências criadas: {X_sequences.shape}")

    # Dividir em treino, validação e teste (80/10/10)
    train_size = int(0.8 * len(X_sequences))
    val_size = int(0.1 * len(X_sequences))

    X_train, y_tp_train, y_sl_train = X_sequences[:train_size], y_tp[:train_size], y_sl[:train_size]
    X_val, y_tp_val, y_sl_val = (
        X_sequences[train_size: train_size + val_size],
        y_tp[train_size: train_size + val_size],
        y_sl[train_size: train_size + val_size],
    )
    X_test, y_tp_test, y_sl_test = (
        X_sequences[train_size + val_size:],
        y_tp[train_size + val_size:],
        y_sl[train_size + val_size:],
    )

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Criar diretório para modelos
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    preprocessor.save_scalers(TRAINED_MODELS_DIR)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Otimização e treinamento do modelo TP
    logger.info("Iniciando otimização de hiperparâmetros para modelo TP")
    tp_optimizer = HyperparameterOptimizer(X_train, y_tp_train, X_val, y_tp_val, input_shape)
    best_params_tp = tp_optimizer.optimize(n_trials=config.MODEL_N_TRIALS, study_name="tp_optimization")

    logger.info("Treinando modelo TP com os melhores hiperparâmetros")
    model_tp = LSTMModelBuilder.build_model(input_shape, best_params_tp)
    model_tp_path = str(TRAINED_MODELS_DIR / "lstm_model_tp.keras")
    history_tp = LSTMModelBuilder.train_model(
        model_tp,
        X_train,
        y_tp_train,
        X_val,
        y_tp_val,
        model_tp_path,
        patience=config.MODEL_PATIENCE,
        epochs=config.MODEL_EPOCHS_TRAINING,
        batch_size=best_params_tp.get("batch_size", config.MODEL_BATCH_SIZE_DEFAULT),
    )
    model_tp = load_model(model_tp_path)

    # Otimização e treinamento do modelo SL
    logger.info("Iniciando otimização de hiperparâmetros para modelo SL")
    sl_optimizer = HyperparameterOptimizer(X_train, y_sl_train, X_val, y_sl_val, input_shape)
    best_params_sl = sl_optimizer.optimize(n_trials=config.MODEL_N_TRIALS, study_name="sl_optimization")

    logger.info("Treinando modelo SL com os melhores hiperparâmetros")
    model_sl = LSTMModelBuilder.build_model(input_shape, best_params_sl)
    model_sl_path = str(TRAINED_MODELS_DIR / "lstm_model_sl.keras")
    history_sl = LSTMModelBuilder.train_model(
        model_sl,
        X_train,
        y_sl_train,
        X_val,
        y_sl_val,
        model_sl_path,
        patience=config.MODEL_PATIENCE,
        epochs=config.MODEL_EPOCHS_TRAINING,
        batch_size=best_params_sl.get("batch_size", config.MODEL_BATCH_SIZE_DEFAULT),
    )
    model_sl = load_model(model_sl_path)

    # Avaliação dos modelos
    logger.info("Avaliando modelo TP no conjunto de teste")
    tp_metrics = LSTMModelBuilder.evaluate_model(model_tp, X_test, y_tp_test, preprocessor, "tp")

    logger.info("Avaliando modelo SL no conjunto de teste")
    sl_metrics = LSTMModelBuilder.evaluate_model(model_sl, X_test, y_sl_test, preprocessor, "sl")

    # Salvar métricas
    import json

    evaluation = {"tp_model": tp_metrics, "sl_model": sl_metrics,
                  "params": {"tp": best_params_tp, "sl": best_params_sl}}
    with open(TRAINED_MODELS_DIR / "model_evaluation.json", "w") as f:
        json.dump(evaluation, f, indent=4)

    logger.info("Treinamento de modelos LSTM concluído com sucesso!")


if __name__ == "__main__":
    main()
