# main.py

import asyncio
import threading

from binance import ThreadedWebsocketManager

from core.constants import FEATURE_COLUMNS
from core.logger import logger
from dashboard.dashboard import create_dashboard
from models.managers.model_manager import ModelManager
from models.random_forest import RandomForestConfig, RandomForestModel, RandomForestTrainer
from services.trading_bot import TradingBot


def main() -> None:
    """
    Função principal que inicializa o TradingBot, a aplicação Dash,
    e mantém o fluxo de execução.
    """
    logger.info("Iniciando TradingBot..aguarde")

    # Configuração do modelo RandomForest
    base_config = RandomForestConfig(
        model_name="random_forest_model",
        description="Modelo para previsão de Take Profit e Stop Loss",
        version="1.0.0",
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
        feature_columns=FEATURE_COLUMNS,
    )

    # Modelos e treinadores para TP e SL
    tp_config = base_config.model_copy(update={"model_name": "tp_model"})
    sl_config = base_config.model_copy(update={"model_name": "sl_model"})

    tp_model = RandomForestModel(tp_config)
    sl_model = RandomForestModel(sl_config)

    tp_trainer = RandomForestTrainer(tp_model)
    sl_trainer = RandomForestTrainer(sl_model)

    # Criação do ModelManager
    tp_model_manager = ModelManager(tp_model, tp_trainer, tp_config)
    sl_model_manager = ModelManager(sl_model, sl_trainer, sl_config)

    # Passa ambos os ModelManagers para o TradingBot
    bot = TradingBot(tp_model_manager=tp_model_manager, sl_model_manager=sl_model_manager)

    # Inicia WebSocket Manager
    logger.info("Iniciando WebsocketManager...")
    twm = ThreadedWebsocketManager(
        api_key=bot.binance_client.client.API_KEY,
        api_secret=bot.binance_client.client.API_SECRET,
        testnet=True
    )
    twm.start()
    logger.info("WebsocketManager iniciado com sucesso.")

    # Cria o dashboard
    logger.info("Iniciando Dashboard...")
    dashboard_app = create_dashboard(bot.data_handler)

    # Executa o Dash em thread separada
    dash_thread = threading.Thread(
        target=dashboard_app.run_server,
        kwargs={"debug": False, "use_reloader": False},
        daemon=True
    )
    dash_thread.start()
    logger.info("Dashboard iniciado com sucesso")

    try:
        logger.info("Bot iniciado")
        # Executa o loop principal do bot
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot interrompido manualmente.")
    finally:
        logger.info("Bot finalizado")
        twm.stop()


if __name__ == "__main__":
    main()
