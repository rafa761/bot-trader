# models\lstm\main.py

import numpy as np
import pandas as pd
from binance import Client

from core.config import settings
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig, OptunaConfig
from models.lstm.trainer import LSTMTrainer
from models.managers.optuna_model_manager import OptunaModelManager
from repositories.data_handler import DataCollector, LabelCreator, TechnicalIndicatorAdder
from repositories.data_preprocessor import DataPreprocessor


def diagnose_data(df: pd.DataFrame, feature_columns: list[str], target_column: str):
    """
    Diagnostica problemas nos dados e formato para o modelo LSTM.

    Args:
        df: DataFrame com os dados.
        feature_columns: Lista de colunas a serem usadas como features.
        target_column: Coluna a ser usada como alvo.
    """
    logger.info("Realizando diagnóstico dos dados...")
    logger.info(f"Shape do DataFrame: {df.shape}")
    logger.info(f"Colunas disponíveis: {df.columns.tolist()}")

    # Verificar se todas as colunas necessárias existem
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        logger.error(f"Colunas de features ausentes: {missing_features}")

    if target_column not in df.columns:
        logger.error(f"Coluna alvo '{target_column}' não encontrada no DataFrame")

    # Verificar valores nulos
    null_counts = df[feature_columns + [target_column]].isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Valores nulos encontrados:\n{null_counts[null_counts > 0]}")

    # Verificar estatísticas do target para validação
    if target_column in df.columns:
        logger.info(f"Estatísticas do alvo ({target_column}):")
        logger.info(f"Média: {df[target_column].mean()}")
        logger.info(f"Mediana: {df[target_column].median()}")
        logger.info(f"Mínimo: {df[target_column].min()}")
        logger.info(f"Máximo: {df[target_column].max()}")
        logger.info(f"Desvio padrão: {df[target_column].std()}")

        # Verificar distribuição para potencial normalização
        if df[target_column].std() > 10 or df[target_column].max() > 100:
            logger.warning(f"Alvo ({target_column}) pode precisar de normalização devido à alta variância")

    # Verificar dimensões para LSTM
    X = df[feature_columns].values
    y = df[target_column].values

    logger.info(f"Dimensões das features (X): {X.shape}")
    logger.info(f"Dimensões do alvo (y): {y.shape}")
    logger.info(f"Número total de features: {len(feature_columns)}")

    # Simulação de preparação de sequências para verificar dimensões
    sequence_length = 16  # Valor otimizado para CPU conforme configuração
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:(i + sequence_length)])
        y_seq.append(y[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    logger.info(f"Dimensões simuladas das sequências de X: {X_seq.shape}")
    logger.info(f"Dimensões simuladas das sequências de y: {y_seq.shape}")

    # Se estamos acrescentando y às features em algum lugar, alertar
    if X_seq.shape[2] != len(feature_columns):
        logger.warning(
            f"Alerta: O número de features nas sequências ({X_seq.shape[2]}) "
            f"não corresponde ao número de colunas de features ({len(feature_columns)})"
        )


def setup_model_and_trainer(target_column: str = 'take_profit_pct'):
    """
    Configura e inicializa o modelo LSTM e seu treinador com parâmetros otimizáveis.

    Args:
        target_column: Coluna alvo para treinamento ('take_profit_pct' ou 'stop_loss_pct')

    Returns:
        Tupla contendo o modelo, o treinador e as configurações.
    """
    # Configuração base do modelo com suporte a tunagem automática
    model_config = LSTMConfig(
        model_name=f"lstm_btc_{target_column}",
        version="1.2.0",
        description=f"Modelo LSTM para previsão de {target_column} do Bitcoin (otimizado com Optuna)",
        sequence_length=24,
        lstm_units=[128, 64],
        dense_units=[32],
        dropout_rate=0.2,
        recurrent_dropout_rate=0.1,
        l2_regularization=0.0001,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        # Configuração da tunagem de hiperparâmetros
        optuna_config=OptunaConfig(
            enabled=True,  # Habilitada por padrão
            n_trials=30,  # 30 trials para explorar o espaço de parâmetros
            timeout=3600,  # Limite de tempo de 1 hora
            study_name=f"lstm_btc_{target_column}_study",
            direction="minimize",
            metric="val_loss"
        )
    )

    # Configuração de treinamento
    training_config = LSTMTrainingConfig(
        validation_split=0.15,
        early_stopping_patience=10,
        reduce_lr_patience=3,
        reduce_lr_factor=0.5,
        use_early_stopping=True,
        min_delta=0.001,
        test_size=0.2,
        random_state=42,
        shuffle=False
    )

    # Instanciar modelo e treinador com a configuração inicial
    model = LSTMModel(model_config)
    trainer = LSTMTrainer(model, training_config)

    return model, trainer, model_config


def collect_and_prepare_data():
    """
    Coleta dados históricos da Binance e prepara-os para treinamento
    seguindo a sequência adequada:
    1. Coleta dados brutos OHLCV
    2. Remove outliers nos dados OHLCV
    3. Calcula indicadores técnicos nos dados OHLCV limpos
    4. Cria labels baseadas nos indicadores
    5. Realiza normalização e processamento final de todas as features
    """
    try:
        # 1. Coletar dados brutos (sem indicadores)
        client = Client(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            requests_params={"timeout": 30}
        )

        data_collector = DataCollector(client)
        # Obter apenas dados OHLCV sem indicadores técnicos
        raw_df = data_collector.get_historical_klines(add_indicators=False)

        if raw_df.empty:
            logger.error("Não foi possível coletar dados históricos")
            return None

        logger.info(f"Dados brutos coletados: {len(raw_df)} registros")

        # 2. Processar os dados OHLCV para remover outliers
        # Criar um preprocessador temporário apenas para OHLCV
        ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv_preprocessor = DataPreprocessor(
            feature_columns=ohlcv_columns,
            outlier_method='iqr',
            scaling_method='robust'
        )
        ohlcv_preprocessor.fit(raw_df)

        # Detectar e remover outliers apenas (sem normalizar ainda)
        cleaned_df = ohlcv_preprocessor.detect_outliers(raw_df)

        logger.info(f"Outliers removidos dos dados OHLCV. Processando indicadores...")

        # 3. Calcular indicadores técnicos nos dados OHLCV limpos
        df_with_indicators = TechnicalIndicatorAdder.add_technical_indicators(cleaned_df)

        if df_with_indicators.empty:
            logger.error("Falha ao adicionar indicadores técnicos")
            return None

        logger.info(f"Indicadores técnicos calculados com sucesso: {len(df_with_indicators.columns)} colunas")

        # 4. Criar labels baseadas nos dados com indicadores limpos
        df_with_labels = LabelCreator.create_labels(df_with_indicators)

        if df_with_labels.empty:
            logger.error("Não foi possível criar labels")
            return None

        logger.info("Labels criadas com sucesso")

        # 5. Normalizar e processar todas as features para o modelo
        full_preprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            outlier_method='iqr',
            scaling_method='robust'
        )
        full_preprocessor.fit(df_with_labels)

        # Processar o DataFrame para o modelo (normalização, conversão de tipos, etc.)
        df_processed = full_preprocessor.process_dataframe(df_with_labels)

        logger.info(
            f"Dados processados com sucesso. "
            f"Tamanho final: {len(df_processed)} linhas, {len(df_processed.columns)} colunas"
        )

        data_collector.save_to_csv(df_processed)

        return df_processed

    except Exception as e:
        logger.error(f"Erro ao coletar e preparar dados: {e}", exc_info=True)
        return None

def main():
    """
    Função principal que executa o pipeline completo com tunagem de hiperparâmetros.

    Configura o modelo, coleta dados, otimiza hiperparâmetros, treina o modelo e avalia seu desempenho.
    """
    try:
        # Coletar e preparar dados
        df = collect_and_prepare_data()
        if df is None:
            logger.error("Falha ao preparar dados. Abortando execução.")
            return

        # Lista de targets para treinar modelos
        targets = ['take_profit_pct', 'stop_loss_pct']
        metrics_results = {}

        # Treinar um modelo para cada target
        for target in targets:
            logger.info(f"Treinando modelo para previsão de {target}")

            # Executar diagnóstico dos dados
            diagnose_data(df, FEATURE_COLUMNS, target)

            # Configurar modelo e trainer com suporte a tunagem
            model, trainer, model_config = setup_model_and_trainer(target)

            # Usar o OptunaModelManager em vez do ModelManager padrão
            manager = OptunaModelManager(model, trainer, model_config)

            # Executar pipeline com tunagem automática de hiperparâmetros
            metrics = manager.execute_full_pipeline(
                data=df,
                feature_columns=FEATURE_COLUMNS,
                target_column=target,
                save_path=TRAINED_MODELS_DIR,
            )

            metrics_results[target] = metrics
            logger.info(f"Treinamento para {target} concluído com sucesso.")

        logger.info("Pipeline completo concluído com sucesso.")
        logger.info(f"Métricas finais: {metrics_results}")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
