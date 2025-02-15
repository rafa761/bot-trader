# binance_client.py

"""
Este módulo encapsula a criação do Client da Binance e métodos de acesso
relacionados (ordens, preços, posições, etc.).
"""

import time

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import config
from logger import logger


class BinanceClient:
    """
    Classe responsável por gerenciar a conexão com a API da Binance (Futuros),
    bem como fornecer métodos para colocar ordens, checar posições e obter
    dados de preço.
    """

    def __init__(self):
        """
        Construtor que inicializa o client do Binance a partir das configurações.
        Utiliza a rede de teste (testnet=True) para Futuros.
        """
        logger.info("Iniciando client da Binance...")
        self.client = Client(
            api_key=config.BINANCE_API_KEY_TESTNET,
            api_secret=config.BINANCE_API_SECRET_TESTNET,
            testnet=True
        )
        logger.info("Client da Binance iniciado")


    def set_leverage(self, symbol: str, leverage: int) -> None:
        """
        Define o valor de alavancagem para um determinado símbolo de Futuros.

        :param symbol: Par de trading (exemplo: "BTCUSDT")
        :param leverage: Valor inteiro de alavancagem (exemplo: 5)
        """
        try:
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
        except BinanceAPIException as e:
            logger.warning(f"Não foi possível setar alavancagem={leverage} para {symbol}. Erro: {e}")

    def get_symbol_filters(self, symbol: str) -> tuple[float | None, float | None]:
        """
        Obtém os filtros de preço (tickSize) e quantidade (stepSize) do par em Futuros.

        :param symbol: Par de trading, ex.: "BTCUSDT"
        :return: (tick_size, step_size), ou (None, None) se não encontrar.
        """
        try:
            info = self.client.futures_exchange_info()
            for s in info["symbols"]:
                if s["symbol"] == symbol:
                    filters = {f["filterType"]: f for f in s["filters"]}
                    tick_size = float(filters["PRICE_FILTER"]["tickSize"])
                    step_size = float(filters["LOT_SIZE"]["stepSize"])
                    return tick_size, step_size
        except Exception as e:
            logger.error(f"Erro ao obter symbol filters para {symbol}: {e}", exc_info=True)
        return None, None

    def get_futures_last_price(self, symbol: str) -> float:
        """
        Obtém o Last Price atual de um símbolo de Futuros.

        :param symbol: Par de trading (exemplo: "BTCUSDT")
        :return: Último preço como float, ou 0.0 em caso de erro
        """
        try:
            ticker = self.client.futures_symbol_ticker(symbol=symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Erro ao obter last price para {symbol}: {e}", exc_info=True)
            return 0.0

    def get_open_position_by_side(self, symbol: str, desired_side: str) -> dict | None:
        """
        Retorna a posição aberta para o lado desejado ("LONG" ou "SHORT").
        Em modo hedge, permite distinguir as posições.

        :param symbol: Par de trading, ex.: "BTCUSDT"
        :param desired_side: "LONG" ou "SHORT"
        :return: Dicionário com informações da posição, ou None se não houver
        """
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            for pos in positions:
                if (
                        pos["positionSide"].upper() == desired_side.upper()
                        and float(pos["positionAmt"]) != 0.0
                ):
                    return pos
            return None
        except Exception as e:
            logger.error(f"Erro ao checar posição para {desired_side}: {e}", exc_info=True)
            return None

    def place_order_with_retry(
            self,
            symbol: str,
            side: str,
            quantity: float,
            position_side: str,
            step_size: float,
            max_attempts: int = 3
    ) -> dict | None:
        """
        Cria uma ordem MARKET com retentativas em caso de erro (ex.: margem insuficiente).
        Formata a quantidade para a precisão definida pelo stepSize.

        :param symbol: Par de trading (ex.: "BTCUSDT")
        :param side: "BUY" ou "SELL"
        :param quantity: Quantidade desejada em float
        :param position_side: "LONG" ou "SHORT"
        :param step_size: Valor de step size para arredondar a quantidade
        :param max_attempts: Número máximo de tentativas de enviar a ordem
        :return: Resposta da ordem (dict) ou None se falhar em todas as tentativas
        """
        attempt, backoff_time = 0, 2
        order_resp = None

        formatted_qty = self.format_quantity_for_step_size(quantity, step_size)
        while attempt < max_attempts:
            try:
                order_resp = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="MARKET",
                    quantity=formatted_qty,
                    positionSide=position_side
                )
                return order_resp
            except BinanceAPIException as e:
                logger.error(f"Erro da API ao colocar ordem MARKET: {e}")
                if e.code == -2019:  # margem insuficiente
                    quantity *= 0.9
                    formatted_qty = self.format_quantity_for_step_size(quantity, step_size)
                    time.sleep(backoff_time)
                    backoff_time *= 2
                attempt += 1
            except Exception as e:
                logger.error(f"Erro inesperado ao colocar ordem MARKET: {e}", exc_info=True)
                break

        logger.error("Não foi possível colocar a ordem após várias tentativas.")
        return None

    @staticmethod
    def format_quantity_for_step_size(qty: float, step_size: float) -> str:
        """
        Formata a quantidade com a precisão baseada no step_size.

        :param qty: Quantidade em float
        :param step_size: Valor de step size
        :return: Quantidade formatada em string
        """
        decimals = 0
        if '.' in str(step_size):
            decimals = len(str(step_size).split('.')[-1])
        return f"{qty:.{decimals}f}"
