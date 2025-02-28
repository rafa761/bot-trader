# models/market_pattern/main.py

from binance import Client

from core.config import settings
from core.constants import TRAINED_MODELS_DIR, FEATURE_COLUMNS
from core.logger import logger
from models.managers.model_manager import ModelManager
from models.market_pattern.model import MarketPatternClassifier
from models.market_pattern.schemas import MarketPatternConfig, MarketPatternTrainingConfig
from models.market_pattern.trainer import MarketPatternTrainer
from repositories.data_handler import DataCollector


def setup_market_pattern_model_and_trainer():
    """
    Configura e inicializa o modelo classificador de padrões de mercado e seu treinador.

    Returns:
        Tupla contendo o modelo, o treinador e as configurações.
    """
    # Configurar modelo
    model_config = MarketPatternConfig(
        model_name="market_pattern_classifier",
        version="1.0.0",
        description="Modelo para classificação de padrões de mercado do Bitcoin",
        sequence_length=16,
        lstm_units=[64, 32],
        dense_units=[32, 16],
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=64,
        epochs=100,
        num_classes=4,
        class_names=["UPTREND", "DOWNTREND", "RANGE", "VOLATILE"]
    )

    # Configurar treinador
    training_config = MarketPatternTrainingConfig(
        validation_split=0.2,
        early_stopping_patience=10,
        reduce_lr_patience=5,
        reduce_lr_factor=0.5,
        test_size=0.2,
        random_state=42,
        shuffle=False,  # Importante para séries temporais
        class_weight_adjustment=True
    )

    # Criar modelo e trainer
    model = MarketPatternClassifier(model_config)
    trainer = MarketPatternTrainer(model, training_config)

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

        # Configurar coleta de dados com período mais longo para melhor generalização
        from repositories.data_handler import DataCollectorConfig
        config = DataCollectorConfig(
            symbol=settings.SYMBOL,
            interval=settings.INTERVAL,
            start_str="1 Jan, 2022",  # Período mais longo para capturar diferentes regimes de mercado
            end_str=None,
            cache_retention_days=7
        )

        data_collector = DataCollector(client, config)
        df = data_collector.get_historical_klines()

        if df.empty:
            logger.error("Não foi possível coletar dados históricos")
            return None

        # Verificar dados
        logger.info(f"Dados coletados: {len(df)} candles de {df.index[0]} até {df.index[-1]}")

        return df

    except Exception as e:
        logger.error(f"Erro ao coletar e preparar dados: {e}")
        return None


def main():
    """
    Função principal que executa o pipeline completo de treinamento
    do classificador de padrões de mercado.
    """
    try:
        logger.info("Iniciando pipeline de treinamento do classificador de padrões de mercado...")

        # 1. Coletar e preparar dados
        df = collect_and_prepare_data()
        if df is None:
            logger.error("Falha ao preparar dados. Abortando execução.")
            return

        # 2. Configurar modelo e trainer
        model, trainer, model_config = setup_market_pattern_model_and_trainer()

        # 3. Atualizar modelo com features corretas
        model.config.features = FEATURE_COLUMNS
        model.n_features = len(FEATURE_COLUMNS)

        # 4. Criar gerenciador e executar pipeline
        manager = ModelManager(model, trainer, model_config)

        # 5. Executar pipeline completo
        metrics = manager.execute_full_pipeline(
            data=df,
            feature_columns=FEATURE_COLUMNS,
            target_column=None,  # Não usado para o classificador de padrões
            save_path=TRAINED_MODELS_DIR,
        )

        logger.info("Treinamento do classificador de padrões de mercado concluído com sucesso.")
        logger.info(f"Métricas finais: {metrics}")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
