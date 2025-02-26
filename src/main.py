# main.py

import asyncio
import threading

from binance import ThreadedWebsocketManager

from core.constants import TRAINED_MODELS_DIR
from core.logger import logger
from dashboard.dashboard import create_dashboard
from models.lstm.model import LSTMModel
from models.lstm.schemas import LSTMConfig
from services.trading_bot import TradingBot


def main() -> None:
    """
    Função principal que inicializa o TradingBot com modelos LSTM pré-treinados,
    a aplicação Dash, e mantém o fluxo de execução.
    """
    logger.info("Iniciando TradingBot..aguarde")

    # Carregamento dos modelos LSTM pré-treinados
    try:
        # Configurações básicas para carregamento dos modelos
        tp_config = LSTMConfig(
            model_name="lstm_btc_take_profit_pct",
            version="1.1.0",
            description="Modelo LSTM para previsão de take profit do Bitcoin"
        )

        sl_config = LSTMConfig(
            model_name="lstm_btc_stop_loss_pct",
            version="1.1.0",
            description="Modelo LSTM para previsão de stop loss do Bitcoin"
        )

        # Caminhos para os modelos treinados
        tp_model_path = TRAINED_MODELS_DIR / "lstm_btc_take_profit_pct.keras"
        sl_model_path = TRAINED_MODELS_DIR / "lstm_btc_stop_loss_pct.keras"

        # Verificação de existência dos arquivos
        if not tp_model_path.exists():
            logger.error(f"Modelo take profit não encontrado em {tp_model_path}")
            return

        if not sl_model_path.exists():
            logger.error(f"Modelo stop loss não encontrado em {sl_model_path}")
            return

        # Carregamento dos modelos
        tp_model = LSTMModel.load(tp_model_path, tp_config)
        sl_model = LSTMModel.load(sl_model_path, sl_config)

        logger.info("Modelos LSTM carregados com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelos LSTM: {e}", exc_info=True)
        return

    # Inicializa o TradingBot com os modelos carregados
    bot = TradingBot(tp_model=tp_model, sl_model=sl_model)

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
