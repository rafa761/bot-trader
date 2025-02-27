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

        # Lista de colunas categóricas que precisam de tratamento especial
        self.categorical_columns = [
            'trend_strength', 'volatility_class', 'volume_class',
            'market_phase', 'supertrend_direction'
        ]

        # Lista de colunas booleanas que precisam de tratamento especial
        self.boolean_columns = [
            'rsi_divergence_bull', 'rsi_divergence_bear', 'squeeze',
            'cloud_green', 'price_above_cloud', 'price_below_cloud',
            'tk_cross_bull', 'tk_cross_bear', 'pivot_resistance', 'pivot_support'
        ]

        # Filtra apenas colunas existentes
        self.categorical_columns = [col for col in self.categorical_columns if col in feature_columns]
        self.boolean_columns = [col for col in self.boolean_columns if col in feature_columns]

        # Colunas numéricas para escalar (exclui categóricas e booleanas)
        self.numeric_columns = [col for col in feature_columns if
                                col not in self.categorical_columns + self.boolean_columns]

    def fit(self, df: pd.DataFrame) -> None:
        """
        Ajusta os parâmetros do preprocessador aos dados de treinamento.

        Args:
            df: DataFrame com os dados de treinamento
        """
        logger.info(f"Ajustando preprocessador para {len(self.feature_columns)} features")

        # Verificar colunas existentes
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Colunas ausentes no DataFrame: {missing_cols}")

        # Filtrar para considerar apenas colunas existentes
        existing_columns = [col for col in self.feature_columns if col in df.columns]
        existing_numeric = [col for col in self.numeric_columns if col in df.columns]

        # Armazenar estatísticas originais apenas para colunas numéricas
        self.original_stats = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'q1': df[col].quantile(0.25),
                'q3': df[col].quantile(0.75)
            } for col in existing_numeric
        }

        # Para colunas categóricas, apenas armazena os valores únicos
        for col in self.categorical_columns:
            if col in df.columns:
                self.original_stats[col] = {
                    'unique_values': df[col].unique().tolist()
                }

        # Para colunas booleanas, não precisa de escalonamento
        for col in self.boolean_columns:
            if col in df.columns:
                self.original_stats[col] = {
                    'is_boolean': True
                }

        # Inicializar e ajustar scalers apenas para colunas numéricas
        for col in existing_numeric:
            # Verificar se há NaN e removê-los temporariamente para ajustar o scaler
            valid_data = df[[col]].dropna()

            if valid_data.empty:
                logger.warning(f"Coluna {col} contém apenas valores NaN. Não é possível ajustar o scaler.")
                continue

            # Escolher o tipo de scaler baseado na configuração
            if self.scaling_method == 'standard':
                self.scalers[col] = StandardScaler()
            elif self.scaling_method == 'robust':
                self.scalers[col] = RobustScaler()
            else:  # minmax
                self.scalers[col] = StandardScaler()  # Usando StandardScaler como padrão

            # Ajustar o scaler aos dados da coluna
            self.scalers[col].fit(valid_data)

        # Se usar diferenciação, calcular também estatísticas do dado diferenciado
        if self.use_differential:
            # Filtrar apenas colunas numéricas para diferenciação
            diff_df = df[existing_numeric].diff().dropna()

            self.diff_stats = {
                col: {
                    'mean': diff_df[col].mean(),
                    'std': diff_df[col].std(),
                    'median': diff_df[col].median(),
                    'min': diff_df[col].min(),
                    'max': diff_df[col].max(),
                    'q1': diff_df[col].quantile(0.25),
                    'q3': diff_df[col].quantile(0.75)
                } for col in existing_numeric
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

        # Tratar apenas colunas numéricas para outliers
        for col in self.numeric_columns:
            if col not in df_processed.columns or col not in self.original_stats:
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

        # Normalizar apenas colunas numéricas
        for col in self.numeric_columns:
            if col not in df_norm.columns or col not in self.scalers:
                continue

            # Lidar com NaN antes de aplicar o scaler
            nan_mask = df_norm[col].isna()
            if nan_mask.any():
                # Armazenar os índices com NaN
                nan_indices = df_norm[nan_mask].index

                # Substituir temporariamente os NaN pela mediana para normalização
                df_norm.loc[nan_indices, col] = self.original_stats[col]['median']

                # Aplicar o scaler
                df_norm[col] = self.scalers[col].transform(df_norm[[col]])

                # Restaurar os NaN
                df_norm.loc[nan_indices, col] = np.nan
            else:
                # Aplicar o scaler normalmente se não houver NaN
                df_norm[col] = self.scalers[col].transform(df_norm[[col]])

        return df_norm

    def prepare_sequence_for_prediction(
            self,
            df: pd.DataFrame,
            sequence_length: int
    ) -> np.ndarray | None:
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
            # Verificar quais features do modelo estão disponíveis nos dados
            features_available = [col for col in self.feature_columns if col in df.columns]

            if len(features_available) < len(self.feature_columns):
                missing = set(self.feature_columns) - set(features_available)
                logger.warning(f"Features ausentes: {missing}")

            # 1. Detectar e tratar outliers
            df_cleaned = self.detect_outliers(df)

            # 2. Normalizar os dados
            df_normalized = self.normalize(df_cleaned)

            # 3. Extrair as últimas 'sequence_length' entradas para previsão
            # Preencher valores ausentes primeiro para evitar problemas
            df_filled = df_normalized[features_available].copy()

            # Preenchimento de valores ausentes
            for col in features_available:
                if col in self.numeric_columns:
                    # Para numéricas, usar 0.0
                    df_filled[col] = df_filled[col].fillna(0.0)
                elif col in self.categorical_columns:
                    # Para categóricas, usar um valor padrão apropriado
                    if col == 'trend_strength':
                        df_filled[col] = df_filled[col].fillna('Moderada')
                    elif col == 'volatility_class':
                        df_filled[col] = df_filled[col].fillna('Média')
                    elif col == 'volume_class':
                        df_filled[col] = df_filled[col].fillna('Normal')
                    elif col == 'market_phase':
                        df_filled[col] = df_filled[col].fillna(0.0)
                    elif col == 'supertrend_direction':
                        df_filled[col] = df_filled[col].fillna(1.0)
                    else:
                        df_filled[col] = df_filled[col].fillna('')
                else:
                    # Para booleanas e outras, usar 0.0
                    df_filled[col] = df_filled[col].fillna(0.0)

            # Converter strings categóricas para valores numéricos
            # (isso é simplificado, na prática você pode precisar de one-hot encoding)
            for col in self.categorical_columns:
                if col in df_filled.columns and df_filled[col].dtype == 'object':
                    # Mapear categorias únicas para valores numéricos
                    if col == 'trend_strength':
                        strength_map = {'Ausente': 0.0, 'Fraca': 0.25, 'Moderada': 0.5, 'Forte': 0.75, 'Extrema': 1.0}
                        df_filled[col] = df_filled[col].map(strength_map).fillna(0.5)
                    elif col == 'volatility_class':
                        vol_map = {'Muito Baixa': 0.0, 'Baixa': 0.25, 'Média': 0.5, 'Alta': 0.75, 'Extrema': 1.0}
                        df_filled[col] = df_filled[col].map(vol_map).fillna(0.5)
                    elif col == 'volume_class':
                        vol_map = {'Muito Baixo': 0.0, 'Baixo': 0.25, 'Normal': 0.5, 'Alto': 0.75, 'Muito Alto': 1.0}
                        df_filled[col] = df_filled[col].map(vol_map).fillna(0.5)

            # Extrair as últimas 'sequence_length' entradas
            X = df_filled.values[-sequence_length:]

            # 5. Verificar valores NaN e substituir por zeros, se houver algum restante
            if np.isnan(X).any():
                logger.warning("Valores NaN detectados na sequência. Substituindo por zeros.")
                X = np.nan_to_num(X, nan=0.0)

            # 6. Reformatar para o formato que o LSTM espera [samples, time steps, features]
            X_seq = np.array([X])

            return X_seq

        except Exception as e:
            logger.error(f"Erro ao preparar sequência para previsão: {e}")
            return None

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Processa um DataFrame completo, incluindo limpeza de outliers, normalização e
        conversão de tipos para garantir que todas as colunas sejam numéricas.

        Args:
            df: DataFrame a ser processado

        Returns:
            DataFrame processado com todas as colunas em formato numérico
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

        # 4. Converter colunas categóricas para numéricas
        for col in self.categorical_columns:
            if col in df_normalized.columns:
                logger.info(f"Convertendo coluna categórica '{col}' para numérica")

                if col == 'trend_strength':
                    # Mapear valores de força de tendência
                    strength_map = {
                        'Ausente': 0.0,
                        'Fraca': 0.25,
                        'Moderada': 0.5,
                        'Forte': 0.75,
                        'Extrema': 1.0
                    }
                    df_normalized[col] = df_normalized[col].map(strength_map).fillna(0.5)

                elif col == 'volatility_class':
                    # Mapear classificação de volatilidade
                    vol_map = {
                        'Muito Baixa': 0.0,
                        'Baixa': 0.25,
                        'Média': 0.5,
                        'Alta': 0.75,
                        'Extrema': 1.0
                    }
                    df_normalized[col] = df_normalized[col].map(vol_map).fillna(0.5)

                elif col == 'volume_class':
                    # Mapear classificação de volume
                    vol_class_map = {
                        'Muito Baixo': 0.0,
                        'Baixo': 0.25,
                        'Normal': 0.5,
                        'Alto': 0.75,
                        'Muito Alto': 1.0
                    }
                    df_normalized[col] = df_normalized[col].map(vol_class_map).fillna(0.5)

                elif col == 'market_phase':
                    # Garantir que market_phase é float (já deve ser)
                    df_normalized[col] = df_normalized[col].astype(float)

                elif col == 'supertrend_direction':
                    # Garantir que supertrend_direction é float (já deve ser)
                    df_normalized[col] = df_normalized[col].astype(float)

                else:
                    # Para outras colunas categóricas, convertemos para one-hot encoding simples
                    # ou para valores de 0 a N-1
                    if df_normalized[col].dtype == 'object':
                        unique_values = df_normalized[col].unique()
                        value_map = {val: idx / len(unique_values) for idx, val in enumerate(unique_values)}
                        df_normalized[col] = df_normalized[col].map(value_map).fillna(0.0)

        # 5. Verificar e garantir que todas as colunas são numéricas
        non_numeric_cols = [col for col in df_normalized.columns if
                            not np.issubdtype(df_normalized[col].dtype, np.number)]

        if non_numeric_cols:
            logger.warning(f"Convertendo colunas não numéricas para float: {non_numeric_cols}")
            for col in non_numeric_cols:
                try:
                    # Tentar converter para float
                    df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce').fillna(0.0)
                except Exception as e:
                    logger.error(f"Não foi possível converter a coluna {col} para numérico: {e}")
                    # Se falhar, remover a coluna
                    if col in self.feature_columns:
                        logger.warning(
                            f"Removendo coluna {col} das features por não ser possível convertê-la para numérico")
                        self.feature_columns.remove(col)

        # 6. Verificar novamente se há valores NaN e substituí-los por zeros
        if df_normalized.isnull().values.any():
            logger.warning("Ainda existem valores NaN após normalização. Substituindo por zeros.")
            df_normalized = df_normalized.fillna(0.0)

        # 7. Verificar se há valores infinitos e substituí-los
        if (~np.isfinite(df_normalized.values)).any():
            logger.warning("Valores infinitos encontrados. Substituindo por valores extremos.")
            df_normalized = df_normalized.replace([np.inf, -np.inf], [1e10, -1e10])

        # 8. Verificação final de tipos de dados
        for col in df_normalized.columns:
            if not np.issubdtype(df_normalized[col].dtype, np.number):
                logger.error(f"ALERTA: A coluna {col} ainda não é numérica após todo o processamento!")

                # Último recurso: forçar conversão ou remover
                try:
                    df_normalized[col] = df_normalized[col].astype(float)
                except:
                    if col in df_normalized.columns:
                        logger.warning(f"Removendo coluna problemática: {col}")
                        df_normalized = df_normalized.drop(columns=[col])

        # 9. Garantir que todas as colunas são float64 para consistência
        df_normalized = df_normalized.astype(np.float64)

        return df_normalized
