# services/binance/binance_client.py

import asyncio
import math
from typing import Any, Dict, Optional, Tuple

from binance import AsyncClient
from binance.exceptions import BinanceAPIException

from core.config import settings
from core.logger import logger


class BinanceClient:
    """
    Classe responsável por gerenciar a conexão assíncrona com a API da Binance (Futuros),
    bem como fornecer métodos para colocar ordens, checar posições e obter
    dados de preço de forma assíncrona.
    """

    def __init__(self):
        """
        Construtor que prepara o cliente assíncrono da Binance.
        O cliente é inicializado como None e deve ser criado com await initialize().
        """
        self.client = None

    async def initialize(self) -> None:
        """
        Inicializa o cliente assíncrono da Binance.
        Deve ser chamado após a criação da instância e antes de usar qualquer outro método.
        """
        if not self.is_client_initialized():
            logger.info("Iniciando client assíncrono da Binance...")

            is_development = settings.BINANCE_ENVIRONMENT == "development"
            binance_api_key = settings.BINANCE_API_KEY_TESTNET
            binance_api_secret = settings.BINANCE_API_SECRET_TESTNET
            if not is_development:
                logger.info("CUIDADO - Usando ambiente de produção da Binance")
                binance_api_key = settings.BINANCE_API_KEY
                binance_api_secret = settings.BINANCE_API_SECRET

            self.client = await AsyncClient.create(
                api_key=binance_api_key,
                api_secret=binance_api_secret,
                testnet=is_development
            )
            logger.info("Client assíncrono da Binance iniciado")

    def is_client_initialized(self) -> bool:
        """Verifica se o cliente Binance está inicializado."""
        return hasattr(self, 'client') and self.client is not None

    async def close(self) -> None:
        """
        Fecha a conexão do cliente assíncrono.
        Deve ser chamado quando não precisar mais do cliente.
        """
        if self.client and self.is_client_initialized():
            await self.client.close_connection()
            logger.info("Conexão do client assíncrono da Binance fechada")

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Define o valor de alavancagem para um determinado símbolo de Futuros de forma assíncrona.

        Args:
            symbol: Par de trading (exemplo: "BTCUSDT")
            leverage: Valor inteiro de alavancagem (exemplo: 5)
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        try:
            await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Alavancagem do {symbol} definida para {leverage}x")
        except BinanceAPIException as e:
            logger.warning(f"Não foi possível setar alavancagem={leverage} para {symbol}. Erro: {e}")

    async def get_symbol_filters(self, symbol: str) -> tuple[float | None, float | None]:
        """
        Obtém os filtros de preço (tickSize) e quantidade (stepSize) do par em Futuros de forma assíncrona.

        Args:
            symbol: Par de trading, ex.: "BTCUSDT"

        Returns:
            Tuple[Optional[float], Optional[float]]: (tick_size, step_size), ou (None, None) se não encontrar.
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        try:
            info = await self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    filters = {f["filterType"]: f for f in s["filters"]}
                    tick_size = float(filters["PRICE_FILTER"]["tickSize"])
                    step_size = float(filters["LOT_SIZE"]["stepSize"])
                    return tick_size, step_size
        except Exception as e:
            logger.error(f"Erro ao obter symbol filters para {symbol}: {e}", exc_info=True)
        return None, None

    async def get_futures_last_price(self, symbol: str) -> float:
        """
        Obtém o Last Price atual de um símbolo de Futuros de forma assíncrona.

        Args:
            symbol: Par de trading (exemplo: "BTCUSDT")

        Returns:
            float: Último preço como float, ou 0.0 em caso de erro
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        try:
            ticker = await self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Erro ao obter last price para {symbol}: {e}", exc_info=True)
            return 0.0

    async def get_open_position_by_side(self, symbol: str, desired_side: str) -> dict[str, Any] | None:
        """
        Retorna a posição aberta para o lado desejado ("LONG" ou "SHORT") de forma assíncrona.
        Em modo hedge, permite distinguir as posições.

        Args:
            symbol: Par de trading, ex.: "BTCUSDT"
            desired_side: "LONG" ou "SHORT"

        Returns:
            Optional[Dict[str, Any]]: Dicionário com informações da posição, ou None se não houver
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        try:
            positions = await self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if (
                        pos["positionSide"].upper() == desired_side.upper()
                        and math.fabs(float(pos["positionAmt"])) > 1e-9  # Use a small epsilon value (1e-9)
                ):
                    return pos
            return None
        except Exception as e:
            logger.error(f"Erro ao checar posição para {desired_side}: {e}", exc_info=True)
            return None

    async def place_order_with_retry(
            self,
            symbol: str,
            side: str,
            quantity: float,
            position_side: str,
            step_size: float,
            max_attempts: int = 3,
            min_notional: float = 100.0
    ) -> dict[str, Any] | None:
        """
        Cria uma ordem MARKET com retentativas em caso de erro (ex.: margem insuficiente) de forma assíncrona.
        Formata a quantidade para a precisão definida pelo stepSize.

        Args:
            symbol: Par de trading (ex.: "BTCUSDT")
            side: "BUY" ou "SELL"
            quantity: Quantidade desejada em float
            position_side: "LONG" ou "SHORT"
            step_size: Valor de step size para arredondar a quantidade
            max_attempts: Número máximo de tentativas de enviar a ordem
            min_notional: Valor mínimo notional requerido pela Binance (preço x quantidade)

        Returns:
            Optional[Dict[str, Any]]: Resposta da ordem (dict) ou None se falhar em todas as tentativas
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        attempt, backoff_time = 0, 2
        order_resp = None

        formatted_qty = self.format_quantity_for_step_size(quantity, step_size)

        # Obter o preço atual para verificar o valor notional
        current_price = await self.get_futures_last_price(symbol)

        # Calcular valor notional inicial
        formatted_qty = self.format_quantity_for_step_size(quantity, step_size)
        notional_value = float(formatted_qty) * current_price

        # Verificar se atende ao valor mínimo notional
        if notional_value < min_notional:
            logger.warning(
                f"Valor notional ({notional_value:.2f} USDT) abaixo do mínimo da Binance ({min_notional} USDT). "
                f"Quantidade: {formatted_qty} {symbol}. Preço: {current_price} USDT"
            )

            # Calcular nova quantidade com MARGEM DE SEGURANÇA para garantir valor mínimo
            # Usamos ceiling para garantir arredondamento para cima
            min_qty_raw = (min_notional * 1.1) / current_price  # 10% acima do mínimo para segurança

            # Calcular quantos passos de step_size precisamos
            steps = math.ceil(min_qty_raw / step_size)
            min_qty = steps * step_size  # Garantir múltiplo exato de step_size

            new_notional = min_qty * current_price

            logger.info(
                f"Ajustando quantidade de {formatted_qty} para {min_qty:.3f} {symbol} "
                f"(valor: {new_notional:.2f} USDT) para atender ao mínimo da Binance"
            )

            formatted_qty = f"{min_qty:.3f}"  # Com 3 casas decimais para 0.001 step_size

        while attempt < max_attempts:
            try:
                order_resp = await self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=formatted_qty,
                    positionSide=position_side
                )
                logger.info(f"Ordem {side} {position_side} executada com sucesso: {symbol} {formatted_qty}")
                return order_resp
            except BinanceAPIException as e:
                logger.error(f"Erro da API ao colocar ordem MARKET: {e}")
                if e.code == -2019:  # margem insuficiente
                    quantity *= 0.9
                    formatted_qty = self.format_quantity_for_step_size(quantity, step_size)
                    await asyncio.sleep(backoff_time)
                    backoff_time *= 2
                attempt += 1
            except Exception as e:
                logger.error(f"Erro inesperado ao colocar ordem MARKET: {e}", exc_info=True)
                break

        logger.error("Não foi possível colocar a ordem após várias tentativas.")
        return None

    async def place_tp_sl_orders(
            self,
            symbol: str,
            side: str,
            position_side: str,
            tp_price: str,
            sl_price: str
    ) -> tuple[dict | None, dict | None]:
        """
        Cria ordens de TAKE_PROFIT e STOP_MARKET de forma assíncrona.

        Args:
            symbol: Par de trading (ex.: "BTCUSDT")
            side: "BUY" ou "SELL"
            position_side: "LONG" ou "SHORT"
            tp_price: Preço formatado para Take-Profit
            sl_price: Preço formatado para Stop-Loss

        Returns:
            Tuple[Optional[Dict], Optional[Dict]]: Tupla contendo (resposta_tp, resposta_sl)
        """
        if not self.is_client_initialized():
            raise RuntimeError("Cliente Binance não foi inicializado. Chame await initialize() primeiro.")

        tp_order, sl_order = None, None

        # Cria ordem TAKE_PROFIT
        try:
            tp_order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="TAKE_PROFIT_MARKET",
                stopPrice=tp_price,
                closePosition=True,
                positionSide=position_side
            )
            logger.info(f"Ordem TAKE_PROFIT criada: {tp_order}")
        except Exception as e:
            logger.error(f"Erro ao criar TAKE_PROFIT: {e}", exc_info=True)

        # Cria ordem STOP
        try:
            sl_order = await self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type="STOP_MARKET",
                stopPrice=sl_price,
                closePosition=True,
                positionSide=position_side
            )
            logger.info(f"Ordem STOP (SL) criada: {sl_order}")
        except Exception as e:
            logger.error(f"Erro ao criar STOP (SL): {e}", exc_info=True)

        return tp_order, sl_order

    @staticmethod
    def format_quantity_for_step_size(qty: float, step_size: float) -> str:
        """
        Formata a quantidade com a precisão baseada no step_size.

        Args:
            qty: Quantidade em float
            step_size: Valor de step size

        Returns:
            str: Quantidade formatada em string
        """
        # Calcular quantos steps completos cabem na quantidade (arredondando para baixo)
        steps = math.floor(qty / step_size)
        # Multiplicar para obter o valor exato múltiplo do step_size
        adjusted_qty = steps * step_size

        # Determinar número de casas decimais para formatação
        decimals = 0
        if '.' in str(step_size):
            decimals = len(str(step_size).split('.')[-1])

        # Formatar com a precisão correta
        return f"{adjusted_qty:.{decimals}f}"
