# services/cleanup_handler.py

import asyncio
import signal
import sys
import threading
from typing import Any, Literal

from core.logger import logger
from services.binance.binance_client import BinanceClient


class CleanupHandler:
    """
    Gerenciador de limpeza para cancelar ordens e fechar posições quando o bot é interrompido.
    Captura sinais de interrupção (SIGINT) e executa a limpeza das posições antes de encerrar.
    """

    def __init__(self, binance_client: BinanceClient, symbol: str):
        """
        Inicializa o manipulador de limpeza.

        Args:
            binance_client: Cliente da Binance para executar as operações de cancelamento
            symbol: Par de trading a ser considerado nas operações de limpeza
        """
        self.client = binance_client
        self.symbol = symbol
        self.original_sigint_handler = None
        self.is_cleaning_up = False
        # Armazenar loop principal para referência
        self.main_loop = asyncio.get_running_loop() if asyncio.get_event_loop().is_running() else None

    def register(self) -> None:
        """
        Registra o manipulador de sinal SIGINT (Ctrl+C) para executar a limpeza.
        """
        # Armazenar o manipulador original para poder chamá-lo após nossa limpeza
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)

        # Definir nosso manipulador personalizado
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info("Manipulador de interrupção registrado para fechamento seguro de posições")

    def _signal_handler(self, sig: int, frame: Any) -> None:
        """
        Manipulador de sinal que será chamado quando o usuário pressionar Ctrl+C.

        Args:
            sig: Número do sinal recebido
            frame: Frame atual de execução
        """
        if self.is_cleaning_up:
            logger.warning("Interrupção secundária detectada. Encerrando imediatamente.")
            sys.exit(1)

        self.is_cleaning_up = True
        logger.info("Interrupção detectada. Iniciando procedimento de limpeza...")

        # Executar a limpeza em um novo thread para não bloquear o loop de eventos principal
        cleanup_thread = threading.Thread(target=self._run_cleanup_in_new_loop)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        cleanup_thread.join()  # Aguardar a conclusão da limpeza antes de continuar

        logger.info("Limpeza concluída. Encerrando o bot.")

        # Chamar o manipulador original para encerrar o programa normalmente
        if callable(self.original_sigint_handler):
            self.original_sigint_handler(sig, frame)
        else:
            sys.exit(0)

    def _run_cleanup_in_new_loop(self) -> None:
        """
        Cria um novo loop de eventos e executa a limpeza.
        Esta função é executada em uma nova thread.
        """
        try:
            # Criar um novo loop para esta thread
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            # Executar a limpeza no novo loop
            new_loop.run_until_complete(self._cleanup())
            new_loop.close()
        except Exception as e:
            logger.error(f"Erro durante execução da limpeza: {e}", exc_info=True)

    async def _cleanup(self) -> None:
        """
        Executa a limpeza das posições e ordens abertas.

        1. Cancela todas as ordens abertas
        2. Fecha todas as posições abertas
        """
        try:
            logger.info("Cancelando todas as ordens abertas...")
            await self.cancel_all_orders()

            logger.info("Fechando todas as posições abertas...")
            await self.close_all_positions()

            logger.info("Limpeza concluída com sucesso.")
        except Exception as e:
            logger.error(f"Erro durante limpeza: {e}")

    async def cancel_all_orders(self) -> None:
        """Cancela todas as ordens abertas para o símbolo configurado."""
        try:
            # Verificar se o cliente está inicializado
            if not self.client.is_client_initialized():
                await self.client.initialize()

            # Cancelar todas as ordens abertas
            result = await self.client.client.futures_cancel_all_open_orders(symbol=self.symbol)
            logger.info(f"Ordens canceladas: {result}")
        except Exception as e:
            logger.error(f"Erro ao cancelar ordens: {e}")

    async def close_all_positions(self) -> None:
        """Fecha todas as posições abertas para o símbolo configurado."""
        try:
            # Verificar se o cliente está inicializado
            if not self.client.is_client_initialized():
                await self.client.initialize()

            # Obter posições abertas
            positions = await self.client.client.futures_position_information(symbol=self.symbol)

            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt == 0:
                    continue  # Pular posições vazias

                # Determinar direção da posição e parâmetros para fechamento
                if position_amt > 0:  # Posição LONG
                    side: Literal["SELL", "BUY"] = "SELL"
                    position_side = "LONG"
                else:  # Posição SHORT
                    side: Literal["SELL", "BUY"] = "BUY"
                    position_side = "SHORT"

                # Quantidade absoluta (remover sinal)
                quantity = abs(position_amt)

                # Executar ordem para fechar a posição
                try:
                    close_order = await self.client.client.futures_create_order(
                        symbol=self.symbol,
                        side=side,
                        positionSide=position_side,
                        type="MARKET",
                        quantity=f"{quantity}"
                    )
                    logger.info(f"Posição fechada: {position_side} {quantity} {self.symbol} - Ordem: {close_order}")
                except Exception as order_error:
                    logger.warning(f"Primeiro método falhou: {order_error}. Tentando método alternativo...")
                    try:
                        # Método alternativo: usar closePosition=True
                        close_order = await self.client.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            positionSide=position_side,
                            type="MARKET",
                            closePosition=True
                        )
                        logger.info(f"Posição fechada (método alternativo): {position_side} {quantity} {self.symbol}")
                    except Exception as alt_error:
                        logger.error(f"Falha em todos os métodos de fechamento de posição: {alt_error}")

        except Exception as e:
            logger.error(f"Erro ao fechar posições: {e}")

    async def execute_cleanup(self) -> None:
        """
        Executa a limpeza manualmente (sem esperar pelo sinal de interrupção).
        Útil para chamadas programáticas de limpeza.
        """
        self.is_cleaning_up = True
        await self._cleanup()
