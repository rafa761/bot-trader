# main.py

import asyncio
import signal
import sys

from core.logger import logger
from services.trading_bot import TradingBot

# Variável global para armazenar referência ao bot
bot_instance = None


async def async_main() -> None:
    """
    Função principal assíncrona que inicializa o TradingBot com modelos LSTM pré-treinados
    e mantém o fluxo de execução.
    """
    global bot_instance

    logger.info("Iniciando TradingBot..aguarde")

    # Inicializa o TradingBot com os modelos carregados
    bot = TradingBot()
    bot_instance = bot  # Armazena a referência global ao bot

    try:
        logger.info("Bot iniciado")
        # Executa o loop principal do bot
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Bot interrompido manualmente. A limpeza será tratada pelo CleanupHandler.")
    except Exception as e:
        logger.error(f"Erro durante execução: {e}", exc_info=True)
    finally:
        logger.info("Bot finalizado")


def register_sigterm_handler():
    """
    Registra o handler para sinal SIGTERM (enviado por gestores de processos como systemd).

    Nota: O sinal SIGINT (Ctrl+C) já é tratado pelo CleanupHandler interno do bot.
    """

    def sigterm_handler(signum, frame):
        logger.info(f"Recebido sinal SIGTERM ({signum}). Iniciando encerramento ordenado.")
        # Forçar saída do programa - deixar os manipuladores de saída limpa funcionarem
        sys.exit(0)

    # Registrar para SIGTERM (o sinal enviado por gestores de processos como systemd)
    signal.signal(signal.SIGTERM, sigterm_handler)
    logger.info("Manipulador SIGTERM registrado.")


def main() -> None:
    """Função de entrada para execução do bot"""
    try:
        # Registrar manipulador para SIGTERM
        register_sigterm_handler()

        # Executar o bot
        asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("Aplicação interrompida pelo usuário via KeyboardInterrupt.")
        # A limpeza já será tratada pelo CleanupHandler do bot
    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)


if __name__ == "__main__":
    main()
