# main.py

import sys
import threading

from binance import ThreadedWebsocketManager

from config import API_KEY, API_SECRET
from dashboard import create_dashboard
from logger import logger
from trading_bot import TradingBot


def main():
    symbol = 'BTCUSDT'
    interval = '1m'

    # Instancia o bot
    bot = TradingBot(symbol=symbol, interval=interval)

    # Coleta dados históricos iniciais
    df_init = bot.data_handler.get_latest_data(symbol, interval, limit=5000)
    if df_init.empty or len(df_init) < 1000:
        logger.error(
            f"Dados históricos insuficientes para inicialização. Necessário >= 1000, obtido: {len(df_init)}"
        )
        sys.exit(1)
    else:
        # Calcula indicadores
        df_init = bot.data_handler.add_technical_indicators(df_init)
        bot.data_handler.historical_df = df_init
        logger.info(f"Dados históricos coletados. Linhas: {len(df_init)}")

    # Iniciar WebSocket
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
    twm.start()

    # Inicia stream de kline contínuo
    twm.start_kline_futures_socket(
        callback=bot.handle_socket_message,
        symbol=symbol.lower(),
        interval='1m'
    )
    logger.info(f"WebSocket iniciado para {symbol} (intervalo {interval}).")

    # Thread do bot (aprendizado online + trading)
    trading_thread = threading.Thread(
        target=bot.online_learning_and_trading,
        daemon=True
    )
    trading_thread.start()

    # Inicia o Dashboard
    app = create_dashboard(bot)
    app.run_server(debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
