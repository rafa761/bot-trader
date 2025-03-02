# services/cleanup_handler.py

import asyncio
import signal
import sys
from typing import Any

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

        # Criar e executar a tarefa de limpeza
        try:
            loop = asyncio.get_event_loop()

            # Se o loop estiver sendo fechado, criar um novo
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Executar a limpeza de forma síncrona (bloqueante)
            loop.run_until_complete(self._cleanup())

        except Exception as e:
            logger.error(f"Erro durante a limpeza na interrupção: {e}")
        finally:
            logger.info("Limpeza concluída. Encerrando o bot.")

            # Chamar o manipulador original para encerrar o programa normalmente
            if callable(self.original_sigint_handler):
                self.original_sigint_handler(sig, frame)
            else:
                sys.exit(0)

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
            if not self.client._initialized:
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
            if not self.client._initialized:
                await self.client.initialize()

            # Obter posições abertas
            positions = await self.client.client.futures_position_information(symbol=self.symbol)

            for position in positions:
                position_amt = float(position['positionAmt'])
                if position_amt == 0:
                    continue  # Pular posições vazias

                # Determinar direção da posição e parâmetros para fechamento
                if position_amt > 0:  # Posição LONG
                    side = "SELL"
                    position_side = "LONG"
                else:  # Posição SHORT
                    side = "BUY"
                    position_side = "SHORT"

                # Quantidade absoluta (remover sinal)
                quantity = abs(position_amt)

                # Executar ordem para fechar a posição SEM o parâmetro reduceOnly
                try:
                    close_order = await self.client.client.futures_create_order(
                        symbol=self.symbol,
                        side=side,
                        positionSide=position_side,
                        type="MARKET",
                        quantity=f"{quantity}"  # Removido reduceOnly=True
                    )

                    logger.info(f"Posição fechada: {position_side} {quantity} {self.symbol} - Ordem: {close_order}")
                except Exception as order_error:
                    # Tentar uma abordagem alternativa se a primeira falhar
                    logger.warning(f"Primeiro método falhou: {order_error}. Tentando método alternativo...")

                    try:
                        # Método alternativo: usar closePosition=True
                        close_order = await self.client.client.futures_create_order(
                            symbol=self.symbol,
                            side=side,
                            positionSide=position_side,
                            type="MARKET",
                            quantity=f"{quantity}",
                            closePosition=True  # Usar closePosition em vez de reduceOnly
                        )
                        logger.info(f"Posição fechada (método alternativo): {position_side} {quantity} {self.symbol}")
                    except Exception as alt_error:
                        logger.error(f"Erro ao usar método alternativo: {alt_error}")

                        # Terceira tentativa: usar apenas quantidade, sem flags adicionais
                        try:
                            close_order = await self.client.client.futures_create_order(
                                symbol=self.symbol,
                                side=side,
                                positionSide=position_side,
                                type="MARKET",
                                quantity=f"{quantity}"
                            )
                            logger.info(f"Posição fechada (método básico): {position_side} {quantity} {self.symbol}")
                        except Exception as basic_error:
                            logger.error(f"Falha em todos os métodos de fechamento de posição: {basic_error}")

        except Exception as e:
            logger.error(f"Erro ao fechar posições: {e}")

    async def execute_cleanup(self) -> None:
        """
        Executa a limpeza manualmente (sem esperar pelo sinal de interrupção).
        Útil para chamadas programáticas de limpeza.
        """
        self.is_cleaning_up = True
        await self._cleanup()
