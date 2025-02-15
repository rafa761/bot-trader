# main.py

"""
Módulo principal que faz a integração final, iniciando o bot de trading
e a aplicação Dash.
"""

import asyncio
import threading

from binance import ThreadedWebsocketManager

from dashboard import create_dashboard
from logger import logger
from trading_bot import TradingBot


def main() -> None:
    """
    Função principal que inicializa o TradingBot, a aplicação Dash,
    e mantém o fluxo de execução.
    """
    logger.info("Iniciando TradingBot..aguarde")
    bot = TradingBot()

    # Inicia WebSocket Manager (caso queira receber dados via stream, etc.)
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
