# config.py

import asyncio
import os
import sys

import certifi
from dotenv import load_dotenv

# Ajuste a variável de ambiente para o certificado SSL
os.environ['SSL_CERT_FILE'] = certifi.where()

# Para Windows: ajusta a policy do event loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Carrega variáveis do arquivo .env
load_dotenv()

API_KEY = os.environ.get('BINANCE_API_KEY_TESTNET')
API_SECRET = os.environ.get('BINANCE_API_SECRET_TESTNET')

# Nova variável de ambiente para a API de notícias
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

# Flag para habilitar a análise de sentimento
SENTIMENT_ANALYSIS_ENABLED = bool(os.environ.get('SENTIMENT_ANALYSIS_ENABLED', 'False').lower() in ('true', '1', 'yes'))

# Verificações de chaves
if not API_KEY or not API_SECRET:
    raise EnvironmentError(
        "Chaves da Binance não encontradas. Ajuste o arquivo .env com BINANCE_API_KEY_TESTNET e BINANCE_API_SECRET_TESTNET."
    )

# Parâmetros de Trading e Modelo
CAPITAL_INICIAL = 1000.0
RISK_PER_TRADE = 0.02
LEVERAGE = 25
TRANSACTION_COST = 0.0004
SLIPPAGE = 0.0001
SMA_WINDOW_SHORT = 5
SMA_WINDOW_LONG = 10

# Limite de perda diária (ex: 2% do capital)
DAILY_LOSS_LIMIT = -0.02 * CAPITAL_INICIAL
