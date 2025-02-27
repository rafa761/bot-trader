# data_preprocessor.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

from core.logger import logger


class DataPreprocessor:
    """
    Classe responsável pelo pré-processamento de dados para modelos LSTM,
    incluindo normalização, detecção de outliers e preparação de sequências.
    """

    def __init__(self,
                 feature_columns: list[str],
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 scaling_method: str = 'robust',
                 use_differential: bool = True):
        """
        Inicializa o preprocessador de dados.

        Args:
            feature_columns: Lista de colunas de features a serem processadas
            outlier_method: Método para detecção de outliers ('iqr', 'zscore')
            outlier_threshold: Limite para considerar um valor como outlier
            scaling_method: Método de escalamento ('standard', 'robust', 'minmax')
            use_differential: Se True, usar diferenciação para estacionariedade
        """
        self.feature_columns = feature_columns
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scaling_method = scaling_method
        self.use_differential = use_differential
        self.scalers = {}
        self.original_stats = {}
        self.diff_stats = {}

    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta os parâmetros do preprocessador aos dados de treinamento.

        Args:
            df: DataFrame com os dados de treinamento
        """
        logger.info(f"Ajustando preprocessador para {len(self.feature_columns)} features")

        # Armazenar estatísticas originais para cada coluna
        self.original_stats = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75)
            } for col in self.feature_columns
        }

        # Inicializar e ajustar scalers para cada coluna
        for col in self.feature_columns:
            # Escolher o tipo de scaler baseado na configuração
            if self.scaling_method == 'standard':
                self.scalers[col] = StandardScaler()
            elif self.scaling_method == 'robust':
                self.scalers[col] = RobustScaler()
            else:  # minmax
                self.scalers[col] = StandardScaler()  # Usando StandardScaler como padrão

            # Ajustar o scaler aos dados da coluna
            self.scalers[col].fit(df[[col]])

        # Se usar diferenciação, calcular também estatísticas do dado diferenciado
        if self.use_differential:
            diff_df = df[self.feature_columns].diff().dropna()
            self.diff_stats = {
                col: {
                    'mean': diff_df[col].mean(),
                    'std': diff_df[col].std(),
                    'median': diff_df[col].median(),
                    'min': diff_df[col].min(),
                    'max': diff_df[col].max(),
                    'q1': diff_df[col].quantile(0.25),
                    'q3': diff_df[col].quantile(0.75)
                } for col in self.feature_columns
            }

        logger.info("Preprocessador ajustado com sucesso")

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta e trata outliers no DataFrame.

        Args:
            df: DataFrame a ser processado

        Returns:
            DataFrame com outliers tratados
        """
        df_processed = df.copy()

        for col in self.feature_columns:
            if col not in df_processed.columns:
                continue

            if self.outlier_method == 'iqr':
                # Método IQR (Intervalo Interquartil)
                q1 = self.original_stats[col]['q1']
                q3 = self.original_stats[col]['q3']
                iqr = q3 - q1
                lower_bound = q1 - self.outlier_threshold * iqr
                upper_bound = q3 + self.outlier_threshold * iqr

                # Identificar outliers
                outliers = ((df_processed[col] < lower_bound) |
                            (df_processed[col] > upper_bound))

                if outliers.sum() > 0:
                    logger.info(f"Detectados {outliers.sum()} outliers na coluna {col}")

                    # Substituir outliers pela mediana
                    df_processed.loc[outliers, col] = self.original_stats[col]['median']

            elif self.outlier_method == 'zscore':
                # Método Z-score
                mean = self.original_stats[col]['mean']
                std = self.original_stats[col]['std']

                # Evitar divisão por zero
                if std == 0:
                    continue

                z_scores = np.abs((df_processed[col] - mean) / std)
                outliers = z_scores > self.outlier_threshold

                if outliers.sum() > 0:
                    logger.info(f"Detectados {outliers.sum()} outliers na coluna {col}")

                    # Substituir outliers pela mediana
                    df_processed.loc[outliers, col] = self.original_stats[col]['median']

        return df_processed

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza os dados usando o scaler ajustado.

        Args:
            df: DataFrame a ser normalizado

        Returns:
            DataFrame normalizado
        """
        df_norm = df.copy()

        for col in self.feature_columns:
            if col not in df_norm.columns:
                continue

            # Aplicar o scaler ajustado para a coluna
            if col in self.scalers:
                df_norm[col] = self.scalers[col].transform(df_norm[[col]])

        return df_norm

    def prepare_sequence_for_prediction(
            self,
            df: pd.DataFrame,
            sequence_length: int
    ) -> np.ndarray:
        """
        Prepara uma sequência normalizada e sem outliers para previsão pelo modelo LSTM.

        Args:
            df: DataFrame com dados históricos
            sequence_length: Comprimento da sequência para o LSTM

        Returns:
            Sequência formatada para o modelo LSTM
        """
        if len(df) < sequence_length:
            logger.warning(
                f"Dados insuficientes para gerar sequência. "
                f"Necessário: {sequence_length}, Disponível: {len(df)}"
            )
            return None

        try:
            # 1. Detectar e tratar outliers
            df_cleaned = self.detect_outliers(df)

            # 2. Normalizar os dados
            df_normalized = self.normalize(df_cleaned)

            # 3. Extrair as últimas 'sequence_length' entradas para previsão
            X = df_normalized[self.feature_columns].values[-sequence_length:]

            # 4. Verificar valores NaN e substituir por zeros
            if np.isnan(X).any():
                logger.warning("Valores NaN detectados na sequência. Substituindo por zeros.")
                X = np.nan_to_num(X, nan=0.0)

            # 5. Reformatar para o formato que o LSTM espera [samples, time steps, features]
            X_seq = np.array([X])

            return X_seq

        except Exception as e:
            logger.error(f"Erro ao preparar sequência para previsão: {e}")
            return None

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa um DataFrame completo, incluindo limpeza de outliers e normalização.

        Args:
            df: DataFrame a ser processado

        Returns:
            DataFrame processado
        """
        # Verificar colunas existentes
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colunas ausentes no DataFrame: {missing_cols}")

        # 1. Remover linhas com NaN nas colunas de features
        present_cols = [col for col in self.feature_columns if col in df.columns]
        df_clean = df.dropna(subset=present_cols)

        if len(df_clean) < len(df):
            logger.info(f"Removidas {len(df) - len(df_clean)} linhas com valores NaN")

        # 2. Detectar e tratar outliers
        df_clean = self.detect_outliers(df_clean)

        # 3. Normalizar os dados
        df_normalized = self.normalize(df_clean)

        return df_normalized
