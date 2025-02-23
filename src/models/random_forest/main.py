from binance.client import Client

from core.config import settings
from core.constants import FEATURE_COLUMNS, TRAINED_MODELS_DIR
from core.logger import logger
from models.managers.model_manager import ModelManager
from models.random_forest import RandomForestModel, RandomForestTrainer, RandomForestConfig
from repositories.data_handler import DataCollector, LabelCreator


# ---------------------------- Fluxo Principal ----------------------------
def main():
    # Configurações dos modelos
    tp_config = RandomForestConfig(
        model_name="tp_predictor",
        description="Modelo para previsão de Take Profit",
        version="1.0.0",
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        feature_columns=FEATURE_COLUMNS,
    )

    sl_config = RandomForestConfig(
        model_name="sl_predictor",
        description="Modelo para previsão de Stop Loss",
        version="1.0.0",
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        feature_columns=FEATURE_COLUMNS
    )

    # 1. Coleta e preparação de dados
    client = Client(settings.BINANCE_API_KEY, settings.BINANCE_API_SECRET, requests_params={"timeout": 20})
    data_collector = DataCollector(client)
    df = data_collector.get_historical_klines()
    if df.empty:
        logger.error("Não foi possível coletar dados históricos. Encerrando.")
        return

    # 2. Criação de labels
    df = LabelCreator.create_labels(df)
    if df.empty:
        logger.error("Não foi possível criar labels. Encerrando.")
        return

    # 3. Treinamento do modelo TP
    tp_model = RandomForestModel(tp_config)
    tp_trainer = RandomForestTrainer(tp_model)
    tp_manager = ModelManager(tp_model, tp_trainer, tp_config)

    # 4. Treinamento do modelo SL
    sl_model = RandomForestModel(sl_config)
    sl_trainer = RandomForestTrainer(sl_model)
    sl_manager = ModelManager(sl_model, sl_trainer, sl_config)

    # 5. Execução dos pipelines
    try:
        tp_metrics = tp_manager.execute_full_pipeline(
            data=df,
            feature_columns=FEATURE_COLUMNS,
            target_column='take_profit_pct',
            save_path=TRAINED_MODELS_DIR
        )

        sl_metrics = sl_manager.execute_full_pipeline(
            data=df,
            feature_columns=FEATURE_COLUMNS,
            target_column='stop_loss_pct',
            save_path=TRAINED_MODELS_DIR
        )

        logger.info(f"Métricas do modelo TP: {tp_metrics}")
        logger.info(f"Métricas do modelo SL: {sl_metrics}")

    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")
        return

    logger.info("Processo de treinamento concluído.")


if __name__ == "__main__":
    main()
