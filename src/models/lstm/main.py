# models\lstm\main.py

import numpy as np
import pandas as pd
from binance import Client

from core.config import settings
from core.constants import TRAINED_MODELS_DIR, FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig
from models.lstm.trainer import LSTMTrainer
from models.managers.model_manager import ModelManager
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


def setup_model_and_trainer(target_column='take_profit_pct'):
    """
    Configura e inicializa o modelo LSTM e seu treinador com parâmetros otimizados para CPU.

    Args:
        target_column: Coluna alvo para treinamento ('take_profit_pct' ou 'stop_loss_pct')

    Returns:
        Tupla contendo o modelo, o treinador e as configurações.
    """
    # Configurar modelo e trainer
    model_config = LSTMConfig(
        model_name=f"lstm_btc_{target_column}",
        version="1.1.0",
        description=f"Modelo LSTM para previsão de {target_column} do Bitcoin",
        # Parâmetros específicos do LSTM otimizados para CPU
        sequence_length=16,  # OTIMIZAÇÃO: Reduzido de 24 para 16 (menos contexto temporal, mais rápido)
        lstm_units=[64, 32],  # OTIMIZAÇÃO: Reduzido de [128, 64] para [64, 32] (menos computação)
        dense_units=[16],  # OTIMIZAÇÃO: Reduzido de [32] para [16] (menos computação)
        dropout_rate=0.1,  # OTIMIZAÇÃO: Reduzido de 0.2 para 0.1 (menos regularização, mais rápido)
        learning_rate=0.002,  # OTIMIZAÇÃO: Aumentado para convergência mais rápida
        batch_size=64,  # OTIMIZAÇÃO: Aumentado para melhor utilização de cache/memória
        epochs=50  # OTIMIZAÇÃO: Reduzido para treinamento mais rápido
    )

    training_config = LSTMTrainingConfig(
        validation_split=0.15,  # OTIMIZAÇÃO: Reduzido para treinamento mais rápido
        early_stopping_patience=5,  # OTIMIZAÇÃO: Interrompe treinamento mais cedo
        reduce_lr_patience=3,  # OTIMIZAÇÃO: Reduz learning rate mais cedo
        reduce_lr_factor=0.5,  # Mantido em 0.5
        # Parâmetros padrão
        test_size=0.2,
        random_state=42,
        shuffle=False  # Importante para séries temporais
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

            # Configurar modelo e trainer
            model, trainer, model_config = setup_model_and_trainer(target)

            # Criar gerenciador e executar pipeline
            manager = ModelManager(model, trainer, model_config)
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
