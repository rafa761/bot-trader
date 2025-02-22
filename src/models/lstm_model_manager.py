# lstm_model_manager.py

"""
Gerenciador de modelos LSTM para previsão de TP e SL no bot de trading.
Substitui o model_manager.py anterior, adaptado para os novos modelos LSTM.
"""

import sys

import joblib
import numpy as np
import pandas as pd
from keras.models import load_model

from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger


class LSTMModelManager:
    """
    Classe responsável por carregar, gerenciar e fazer inferência com os modelos LSTM
    para previsão de valores de Take-Profit (TP) e Stop-Loss (SL).
    """

    def __init__(self,
                 tp_model_path=TRAINED_MODELS_DIR / "lstm_model_tp.keras",
                 sl_model_path=TRAINED_MODELS_DIR / "lstm_model_sl.keras",
                 feature_scaler_path=TRAINED_MODELS_DIR / "scalers/feature_scaler.pkl",
                 tp_scaler_path=TRAINED_MODELS_DIR / "scalers/tp_scaler.pkl",
                 sl_scaler_path=TRAINED_MODELS_DIR / "scalers/sl_scaler.pkl",
                 lookback=60):
        """
        Inicializa o gerenciador de modelos LSTM carregando os modelos treinados e scalers.

        Args:
            tp_model_path: Caminho para o modelo TP (arquivo .keras)
            sl_model_path: Caminho para o modelo SL (arquivo .keras)
            feature_scaler_path: Caminho para o scaler de features
            tp_scaler_path: Caminho para o scaler de TP
            sl_scaler_path: Caminho para o scaler de SL
            lookback: Número de timesteps anteriores que o modelo usa
        """
        logger.info("Inicializando LSTMModelManager")
        self.lookback = lookback
        self.tp_model = None
        self.sl_model = None
        self.feature_scaler = None
        self.tp_scaler = None
        self.sl_scaler = None
        self.historical_data = []  # Armazena dados históricos para sequências

        # Carregar os modelos e scalers
        try:
            # Carregar modelos
            self.tp_model = load_model(tp_model_path)
            self.sl_model = load_model(sl_model_path)
            logger.info("Modelos LSTM TP e SL carregados com sucesso.")

            # Carregar scalers
            self.feature_scaler = joblib.load(feature_scaler_path)
            self.tp_scaler = joblib.load(tp_scaler_path)
            self.sl_scaler = joblib.load(sl_scaler_path)
            logger.info("Scalers carregados com sucesso.")
        except FileNotFoundError as e:
            logger.error(f"Arquivo não encontrado: {e}")
            logger.error("Verifique se os modelos e scalers foram treinados e salvos corretamente.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Erro ao carregar modelos ou scalers: {e}", exc_info=True)
            sys.exit(1)

    def update_historical_data(self, new_data_point):
        """
        Atualiza o buffer de dados históricos com um novo ponto de dados.

        Args:
            new_data_point (dict): Dicionário com as features do novo ponto
        """
        # Filtra apenas as features que o modelo espera
        filtered_data = {k: v for k, v in new_data_point.items() if k in FEATURE_COLUMNS}

        # Verifica se todas as features necessárias estão presentes
        if len(filtered_data) != len(FEATURE_COLUMNS):
            missing = set(FEATURE_COLUMNS) - set(filtered_data.keys())
            logger.warning(f"Features faltando em update_historical_data: {missing}")
            return

        # Adiciona ao histórico
        self.historical_data.append(filtered_data)

        # Mantém apenas os últimos 'lookback' pontos
        if len(self.historical_data) > self.lookback:
            self.historical_data = self.historical_data[-self.lookback:]

    def _prepare_sequence(self):
        """
        Prepara uma sequência para inferência usando os dados históricos.

        Returns:
            np.ndarray: Sequência preparada para o modelo LSTM ou None se não houver dados suficientes
        """
        if len(self.historical_data) < self.lookback:
            logger.warning(f"Dados históricos insuficientes: {len(self.historical_data)}/{self.lookback}")
            return None

        # Criar DataFrame com dados históricos
        df_historical = pd.DataFrame(self.historical_data)

        # Extrair apenas as features necessárias
        sequence_data = df_historical[FEATURE_COLUMNS].values

        # Normalizar os dados
        sequence_scaled = self.feature_scaler.transform(sequence_data)

        # Adicionar dimensão de batch
        sequence_reshaped = np.expand_dims(sequence_scaled, axis=0)

        return sequence_reshaped

    def predict_tp_sl(self, df_eval):
        """
        Gera previsões para TP e SL usando os modelos LSTM.

        Args:
            df_eval (pd.DataFrame): DataFrame com as features do ponto atual (usado para atualizar histórico)

        Returns:
            tuple: (predicted_tp_pct, predicted_sl_pct) ou (0.0, 0.0) em caso de erro
        """
        try:
            # Atualiza o histórico com o ponto mais recente
            current_point = df_eval.iloc[0].to_dict()
            self.update_historical_data(current_point)

            # Prepara a sequência
            sequence = self._prepare_sequence()
            if sequence is None:
                logger.warning("Sequência não preparada. Retornando valores neutros.")
                return 0.0, 0.0

            # Previsão TP
            tp_pred_scaled = self.tp_model.predict(sequence, verbose=0)
            tp_pred = self.tp_scaler.inverse_transform(tp_pred_scaled)[0][0]

            # Previsão SL
            sl_pred_scaled = self.sl_model.predict(sequence, verbose=0)
            sl_pred = self.sl_scaler.inverse_transform(sl_pred_scaled)[0][0]

            logger.info(f"LSTM predictions - TP: {tp_pred:.4f}%, SL: {sl_pred:.4f}%")

            return tp_pred, sl_pred

        except Exception as e:
            logger.error(f"Erro ao fazer previsão com LSTM: {e}", exc_info=True)
            return 0.0, 0.0

    def train_models(self, features, target_tp, target_sl):
        """
        Realiza o treinamento incremental dos modelos LSTM.

        Nota: O treinamento completo deve ser feito offline com lstm_training.py.
        Este método permite ajustes incrementais com novos dados.

        Args:
            features (pd.DataFrame): Features para treinamento
            target_tp (pd.Series): Valores alvo para TP
            target_sl (pd.Series): Valores alvo para SL
        """
        try:
            logger.info("Iniciando treinamento incremental dos modelos LSTM")

            # Verifica se há dados suficientes
            if len(features) < self.lookback + 10:
                logger.warning(f"Dados insuficientes para treinamento: {len(features)} pontos")
                return

            # Preparar sequências para treinamento
            X_sequences = []
            y_tp_sequences = []
            y_sl_sequences = []

            # Converter para arrays numpy
            feature_array = features.values
            tp_array = target_tp.values.reshape(-1, 1)
            sl_array = target_sl.values.reshape(-1, 1)

            # Normalizar dados
            X_scaled = self.feature_scaler.transform(feature_array)
            y_tp_scaled = self.tp_scaler.transform(tp_array)
            y_sl_scaled = self.sl_scaler.transform(sl_array)

            # Criar sequências
            for i in range(len(X_scaled) - self.lookback):
                X_sequences.append(X_scaled[i:i + self.lookback])
                y_tp_sequences.append(y_tp_scaled[i + self.lookback])
                y_sl_sequences.append(y_sl_scaled[i + self.lookback])

            X_train = np.array(X_sequences)
            y_tp_train = np.array(y_tp_sequences)
            y_sl_train = np.array(y_sl_sequences)

            # Definir callbacks para treinamento incremental
            callbacks = [
                EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
            ]

            # Treinar modelo TP
            self.tp_model.fit(
                X_train, y_tp_train,
                epochs=10,  # Número de épocas para treinamento incremental
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )

            # Treinar modelo SL
            self.sl_model.fit(
                X_train, y_sl_train,
                epochs=10,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )

            logger.info("Treinamento incremental concluído com sucesso")

            # Salvar os modelos atualizados
            self.tp_model.save(str(TRAINED_MODELS_DIR / 'lstm_model_tp.keras'))
            self.sl_model.save(str(TRAINED_MODELS_DIR / 'lstm_model_sl.keras'))
            logger.info("Modelos atualizados salvos")

        except Exception as e:
            logger.error(f"Erro durante o treinamento incremental: {e}", exc_info=True)
