# binance_client.py

import sys
import requests
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from logger import logger
from config import API_KEY, API_SECRET

# Ajusta parâmetros globais de requests, se necessário
REQUESTS_PARAMS = {"timeout": 10}


class BinanceClientService:
    """
    Encapsula a lógica de criação e uso do Client da Binance.
    """

    def __init__(self, api_key: str = API_KEY, api_secret: str = API_SECRET, testnet: bool = True):
        """
        Inicializa o cliente da Binance.

        :param api_key: Chave da API da Binance.
        :param api_secret: Segredo da API da Binance.
        :param testnet: Indica se o cliente está rodando no ambiente de testes.
        """
        self.client = Client(api_key, api_secret, testnet=testnet, requests_params=REQUESTS_PARAMS)
        self.testnet = testnet

    def get_futures_account_info(self) -> dict:
        """
        Retorna informações da conta de futuros.

        :return: Um dicionário contendo informações sobre a conta de futuros.
        """
        return self.client.futures_account()

    def get_futures_mark_price(self, symbol: str) -> dict:
        """
        Retorna o preço de marcação (mark price) de um determinado símbolo.

        :param symbol: O símbolo do ativo.
        :return: Um dicionário contendo o mark price.
        """
        return self.client.futures_mark_price(symbol=symbol)

    def create_futures_order(self, symbol: str, side: str, quantity: float, position_side: str) -> dict:
        """
        Cria uma ordem de mercado de futuros.

        :param symbol: O símbolo do ativo.
        :param side: O lado da ordem ('BUY' ou 'SELL').
        :param quantity: A quantidade a ser negociada.
        :param position_side: O lado da posição ('LONG' ou 'SHORT').
        :return: Um dicionário contendo os detalhes da ordem criada.
        """
        try:
            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                positionSide=position_side
            )
            return order
        except BinanceAPIException as e:
            logger.error(f"Erro da API da Binance ao criar ordem: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro inesperado ao criar ordem: {e}", exc_info=True)
            raise
