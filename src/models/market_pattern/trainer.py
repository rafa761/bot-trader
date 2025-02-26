# models/market_pattern/trainer.py

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from core.logger import logger
from models.base.trainer import BaseTrainer
from models.market_pattern.model import MarketPatternClassifier
from models.market_pattern.schemas import MarketPatternTrainingConfig


class MarketPatternTrainer(BaseTrainer):
    """
    Treinador específico para o classificador de padrões de mercado.

    Responsável por preparar os dados, treinar o modelo e avaliar seu desempenho.
    """

    def __init__(self, model: MarketPatternClassifier, config: MarketPatternTrainingConfig):
        """
        Inicializa o treinador com o modelo e configuração especificados.

        Args:
            model: Instância do modelo classificador a ser treinado.
            config: Configuração de treinamento com parâmetros como validation_split,
                   early_stopping_patience, etc.
        """
        self.model = model
        self.config = config
        self.history = None
        self.label_encoder = LabelEncoder()

    def create_market_pattern_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria labels de padrões de mercado com base nos dados históricos.

        Esta função identifica diferentes regimes de mercado baseados em:
        - Direção da tendência (alta/baixa)
        - Volatilidade
        - Consolidação/Range

        Args:
            df: DataFrame com dados históricos de OHLCV e indicadores técnicos

        Returns:
            DataFrame com coluna 'market_pattern' adicionada
        """
        logger.info("Criando labels de padrões de mercado...")

        # Copia o DataFrame para não modificar o original
        result_df = df.copy()

        # Calcular métricas para classificação de regimes
        # 1. Volatilidade (ATR relativo)
        if 'atr' in result_df.columns:
            result_df['atr_pct'] = result_df['atr'] / result_df['close'] * 100
            # Média móvel de ATR para suavizar
            result_df['atr_pct_ma'] = result_df['atr_pct'].rolling(window=5).mean()
        else:
            # Calcular uma aproximação de ATR se não estiver disponível
            result_df['high_low_range'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100
            result_df['atr_pct_ma'] = result_df['high_low_range'].rolling(window=5).mean()

        # 2. Inclinação da tendência (baseada em EMAs)
        if 'ema_short' in result_df.columns and 'ema_long' in result_df.columns:
            # Calcular diferença percentual entre EMAs
            result_df['ema_diff_pct'] = (result_df['ema_short'] - result_df['ema_long']) / result_df['close'] * 100
            # Média móvel da diferença para suavizar
            result_df['ema_diff_pct_ma'] = result_df['ema_diff_pct'].rolling(window=5).mean()
            # Calcular taxa de variação da diferença (inclinação)
            result_df['ema_diff_slope'] = result_df['ema_diff_pct'].pct_change(3) * 100
        else:
            # Calcular EMAs se não estiverem disponíveis
            result_df['ema_9'] = result_df['close'].ewm(span=9, adjust=False).mean()
            result_df['ema_21'] = result_df['close'].ewm(span=21, adjust=False).mean()
            result_df['ema_diff_pct'] = (result_df['ema_9'] - result_df['ema_21']) / result_df['close'] * 100
            result_df['ema_diff_pct_ma'] = result_df['ema_diff_pct'].rolling(window=5).mean()
            result_df['ema_diff_slope'] = result_df['ema_diff_pct'].pct_change(3) * 100

        # 3. Indicador de consolidação/range
        # Calcular Bollinger Width normalizado ou usar o existente
        if 'boll_width' in result_df.columns:
            result_df['boll_width_norm'] = result_df['boll_width'] / result_df['close'] * 100
        else:
            # Calcular bandas de Bollinger se não estiverem disponíveis
            std_20 = result_df['close'].rolling(window=20).std()
            result_df['boll_width'] = 4 * std_20  # 2 desvios acima e abaixo
            result_df['boll_width_norm'] = result_df['boll_width'] / result_df['close'] * 100

        # Definir thresholds para classificação
        volatility_high_threshold = result_df['atr_pct_ma'].quantile(0.7)
        volatility_low_threshold = result_df['atr_pct_ma'].quantile(0.3)
        trend_threshold = 0.5  # Percentual mínimo de diferença EMA para considerar tendência
        consolidation_threshold = result_df['boll_width_norm'].quantile(0.3)  # Bandas estreitas indicam consolidação

        # Classificar cada ponto de dados
        def classify_market_pattern(row):
            if pd.isna(row['atr_pct_ma']) or pd.isna(row['ema_diff_pct_ma']) or pd.isna(row['boll_width_norm']):
                return None

            # Alta volatilidade é prioritária na classificação
            if row['atr_pct_ma'] > volatility_high_threshold:
                return "VOLATILE"

            # Consolidação/Range quando bandas são estreitas e tendência fraca
            if (row['boll_width_norm'] < consolidation_threshold and
                    abs(row['ema_diff_pct_ma']) < trend_threshold and
                    row['atr_pct_ma'] < volatility_low_threshold):
                return "RANGE"

            # Tendência de alta
            if row['ema_diff_pct_ma'] > trend_threshold:
                return "UPTREND"

            # Tendência de baixa
            if row['ema_diff_pct_ma'] < -trend_threshold:
                return "DOWNTREND"

            # Padrão: RANGE para qualquer caso não classificado nos anteriores
            return "RANGE"

        # Aplicar classificação
        result_df['market_pattern'] = result_df.apply(classify_market_pattern, axis=1)

        # Remover NaN (geralmente das primeiras linhas devido a janelas das médias móveis)
        result_df.dropna(subset=['market_pattern'], inplace=True)

        # Estatísticas da distribuição de classes
        pattern_distribution = result_df['market_pattern'].value_counts(normalize=True) * 100
        logger.info(f"Distribuição de padrões de mercado:\n{pattern_distribution}")

        return result_df

    def _prepare_sequences(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepara as sequências para treinamento do classificador.

        Transforma os dados em sequências de tamanho fixo e converte as classes
        em formato one-hot para classificação multiclasse.

        Args:
            df: DataFrame com dados históricos e coluna 'market_pattern'

        Returns:
            Tupla contendo dois arrays numpy: X_seq (sequências de features) e y_seq (labels one-hot)
        """
        logger.info(f"Preparando sequências para treinar classificador de padrões...")

        if 'market_pattern' not in df.columns:
            raise ValueError(
                "DataFrame não contém a coluna 'market_pattern'. Execute create_market_pattern_labels primeiro.")

        # Extrair features e labels
        feature_columns = [col for col in df.columns if col in self.model.config.features]
        X = df[feature_columns].values

        # Codificar classes para formato numérico
        y_labels = df['market_pattern'].values
        self.label_encoder.fit(self.model.config.class_names)
        y_encoded = self.label_encoder.transform(y_labels)

        # Converter para one-hot encoding
        y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=self.model.config.num_classes)

        # Criar sequências
        sequence_length = self.model.config.sequence_length
        stride = 1  # Pode ser ajustado para reduzir número de sequências

        # Calcular número de sequências
        n_samples = (len(X) - sequence_length) // stride

        # Pre-alocar arrays
        X_seq = np.zeros((n_samples, sequence_length, len(feature_columns)), dtype=X.dtype)
        y_seq = np.zeros((n_samples, self.model.config.num_classes), dtype=np.float32)

        # Preencher arrays
        for i in range(n_samples):
            idx = i * stride
            X_seq[i] = X[idx:idx + sequence_length]
            y_seq[i] = y_onehot[idx + sequence_length - 1]  # Classificar o último ponto da sequência

        logger.info(f"Preparadas {n_samples} sequências com shape: X={X_seq.shape}, y={y_seq.shape}")

        return X_seq, y_seq

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, checkpoint_dir: Path = None):
        """
        Treina o modelo classificador de padrões de mercado.

        Args:
            X_train: DataFrame com dados históricos
            y_train: Ignorado (usa apenas X_train, que deve conter os dados necessários)
            checkpoint_dir: Diretório opcional para salvar checkpoints

        Raises:
            Exception: Se ocorrer algum erro durante o treinamento
        """
        logger.info("Iniciando treinamento do classificador de padrões de mercado...")

        try:
            # 1. Criar labels de padrões de mercado
            df_with_patterns = self.create_market_pattern_labels(X_train)

            # 2. Preparar sequências
            X_sequences, y_sequences = self._prepare_sequences(df_with_patterns)

            # 3. Preparar callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.early_stopping_patience,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.reduce_lr_factor,
                    patience=self.config.reduce_lr_patience
                )
            ]

            # Adicionar checkpoint se diretório for fornecido
            if checkpoint_dir:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"{self.model.config.model_name}_checkpoint.h5"
                callbacks.append(
                    ModelCheckpoint(
                        filepath=str(checkpoint_path),
                        save_best_only=True,
                        monitor='val_loss'
                    )
                )

            # 4. Calcular class_weights se habilitado
            class_weights = None
            if self.config.class_weight_adjustment:
                # Obter as classes a partir dos rótulos one-hot
                y_classes = np.argmax(y_sequences, axis=1)
                # Calcular pesos de classe para lidar com desbalanceamento
                class_weights_array = compute_class_weight(
                    class_weight='balanced',
                    classes=np.unique(y_classes),
                    y=y_classes
                )
                class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
                logger.info(f"Pesos de classe calculados: {class_weights}")

            # 5. Treinar modelo
            self.history = self.model.model.fit(
                X_sequences,
                y_sequences,
                batch_size=self.model.config.batch_size,
                epochs=self.model.config.epochs,
                validation_split=self.config.validation_split,
                callbacks=callbacks,
                class_weight=class_weights,
                verbose=1
            )

            logger.info("Treinamento do classificador de padrões concluído com sucesso")

        except Exception as e:
            logger.error(f"Erro durante treinamento do classificador: {e}")
            raise

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Avalia o modelo com dados de teste.

        Args:
            X_test: DataFrame com dados históricos
            y_test: Ignorado (usa apenas X_test, que deve conter os dados necessários)

        Returns:
            Dicionário com métricas de avaliação
        """
        logger.info("Avaliando classificador de padrões de mercado...")

        try:
            # 1. Criar labels de padrões de mercado
            df_with_patterns = self.create_market_pattern_labels(X_test)

            # 2. Preparar sequências
            X_test_seq, y_test_seq = self._prepare_sequences(df_with_patterns)

            # 3. Avaliar modelo
            evaluation = self.model.model.evaluate(
                X_test_seq,
                y_test_seq,
                batch_size=self.model.config.batch_size
            )

            # 4. Preparar métricas
            metrics = {
                'test_loss': float(evaluation[0]),
                'test_accuracy': float(evaluation[1])
            }

            # 5. Fazer previsões para matriz de confusão
            y_pred = self.model.model.predict(X_test_seq)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test_seq, axis=1)

            # 6. Calcular estatísticas por classe
            from sklearn.metrics import classification_report, confusion_matrix

            # Converter índices numéricos para nomes de classes
            class_names = self.model.config.class_names

            # Gerar matriz de confusão
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            cm_str = "\n".join([" ".join([f"{val:4d}" for val in row]) for row in cm])
            logger.info(f"Matriz de confusão:\n{cm_str}")

            # Gerar relatório de classificação
            report = classification_report(
                y_true_classes,
                y_pred_classes,
                target_names=class_names,
                output_dict=True
            )

            # Adicionar relatório às métricas
            metrics['classification_report'] = report

            # Adicionar métricas do histórico de treinamento
            if self.history:
                metrics.update({
                    'final_train_loss': float(self.history.history['loss'][-1]),
                    'final_val_loss': float(self.history.history['val_loss'][-1]),
                    'final_train_accuracy': float(self.history.history['accuracy'][-1]),
                    'final_val_accuracy': float(self.history.history['val_accuracy'][-1]),
                    'best_val_loss': float(min(self.history.history['val_loss'])),
                    'best_val_accuracy': float(max(self.history.history['val_accuracy']))
                })

            logger.info(f"Avaliação do classificador concluída: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Erro durante avaliação do classificador: {e}")
            raise
