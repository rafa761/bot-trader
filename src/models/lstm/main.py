from pathlib import Path

from binance import Client

from core.config import settings
from core.constants import TRAINED_MODELS_DIR, FEATURE_COLUMNS
from core.logger import logger
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig, LSTMTrainingConfig
from models.lstm.trainer import LSTMTrainer
from repositories.data_handler import DataCollector, LabelCreator


def main():
    try:
        # Configurar modelo e trainer
        model_config = LSTMConfig(
            model_name="lstm_btc_predictor",
            version="1.0.0",
            description="Modelo LSTM para previsão de preços do Bitcoin"
        )

        training_config = LSTMTrainingConfig()

        # Coletar dados
        client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, requests_params={"timeout": 20})
        data_collector = DataCollector(client)
        df = data_collector.get_historical_klines()

        if df.empty:
            logger.error("Não foi possível coletar dados históricos")
            return

        # Criar labels
        df = LabelCreator.create_labels(df)

        if df.empty:
            logger.error("Não foi possível criar labels")
            return

        # Criar modelo e trainer
        model = LSTMModel(model_config)
        trainer = LSTMTrainer(model, training_config)

        # Criar gerenciador e executar pipeline
        from models.managers.model_manager import ModelManager

        manager = ModelManager(model, trainer, model_config)
        metrics = manager.execute_full_pipeline(
            data=df,
            feature_columns=FEATURE_COLUMNS,
            target_column='take_profit_pct',
            save_path=Path(TRAINED_MODELS_DIR)
        )

        logger.info(f"Pipeline concluído com sucesso. Métricas finais: {metrics}")

    except Exception as e:
        logger.error(f"Erro durante execução: {e}")
        raise


if __name__ == "__main__":
    main()
