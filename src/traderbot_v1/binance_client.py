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
        self.client = Client(api_key, api_secret, testnet=testnet, requests_params=REQUESTS_PARAMS)
        self.testnet = testnet

    def get_futures_account_info(self):
        """Retorna informações da conta de futuros."""
        return self.client.futures_account()

    def get_futures_mark_price(self, symbol: str):
        """Retorna o mark price de um determinado símbolo."""
        return self.client.futures_mark_price(symbol=symbol)

    def create_futures_order(self, symbol: str, side: str, quantity: float, position_side: str):
        """
        Cria uma ordem de mercado de futuros.
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
