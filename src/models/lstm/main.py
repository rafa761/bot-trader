# models\lstm\main.py

from pathlib import Path

import numpy as np
import pandas as pd
from binance import Client

from core.config import settings
from core.constants import TRAINED_MODELS_DIR, FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig
from models.lstm.trainer import LSTMTrainer
from repositories.data_handler import DataCollector, LabelCreator


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

    # Verificar dimensões para LSTM
    X = df[feature_columns].values
    y = df[target_column].values

    logger.info(f"Dimensões das features (X): {X.shape}")
    logger.info(f"Dimensões do alvo (y): {y.shape}")
    logger.info(f"Número total de features: {len(feature_columns)}")

    # Simulação de preparação de sequências para verificar dimensões
    sequence_length = 24  # Valor padrão, deve corresponder ao definido na configuração
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


def setup_model_and_trainer():
    """
    Configura e inicializa o modelo LSTM e seu treinador.

    Returns:
        Tupla contendo o modelo, o treinador e as configurações.
    """
    # Configurar modelo e trainer
    model_config = LSTMConfig(
        model_name="lstm_btc_predictor",
        version="1.0.0",
        description="Modelo LSTM para previsão de preços do Bitcoin",
        # Parâmetros específicos do LSTM
        sequence_length=24,
        lstm_units=[128, 64],
        dense_units=[32],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=100
    )

    training_config = LSTMTrainingConfig(
        validation_split=0.2,
        early_stopping_patience=10,
        reduce_lr_patience=5,
        reduce_lr_factor=0.5
    )

    # Criar modelo e trainer
    model = LSTMModel(model_config)
    trainer = LSTMTrainer(model, training_config)

    return model, trainer, model_config


def collect_and_prepare_data():
    """
    Coleta dados históricos da Binance e prepara-os para treinamento.

    Returns:
        DataFrame pandas contendo os dados preparados ou None se ocorrer erro.
    """
    try:
        # Coletar dados com timeout adequado
        client = Client(
            settings.BINANCE_API_KEY,
            settings.BINANCE_API_SECRET,
            requests_params={"timeout": 30}
        )

        data_collector = DataCollector(client)
        df = data_collector.get_historical_klines()

        if df.empty:
            logger.error("Não foi possível coletar dados históricos")
            return None

        # Criar labels
        df = LabelCreator.create_labels(df)

        if df.empty:
            logger.error("Não foi possível criar labels")
            return None

        return df

    except Exception as e:
        logger.error(f"Erro ao coletar e preparar dados: {e}")
        return None


def main():
    """
    Função principal que executa o pipeline completo.

    Configura o modelo, coleta dados, treina o modelo e avalia seu desempenho.
    """
    try:
        # Configurar modelo e trainer
        model, trainer, model_config = setup_model_and_trainer()

        # Coletar e preparar dados
        df = collect_and_prepare_data()
        if df is None:
            logger.error("Falha ao preparar dados. Abortando execução.")
            return

        # Executar diagnóstico dos dados
        diagnose_data(df, FEATURE_COLUMNS, 'take_profit_pct')

        # Importar o gerenciador e executar o pipeline
        from models.managers.model_manager import ModelManager

        # Criar diretório para modelos treinados se não existir
        models_dir = Path(TRAINED_MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)

        # Criar checkpoint dir dentro do diretório de modelos
        checkpoint_dir = models_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Criar gerenciador e executar pipeline
        manager = ModelManager(model, trainer, model_config)
        metrics = manager.execute_full_pipeline(
            data=df,
            feature_columns=FEATURE_COLUMNS,
            target_column='take_profit_pct',  # Pode ser take_profit_pct ou stop_loss_pct
            save_path=models_dir,
            checkpoint_dir=checkpoint_dir
        )

        logger.info(f"Pipeline concluído com sucesso. Métricas finais: {metrics}")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
