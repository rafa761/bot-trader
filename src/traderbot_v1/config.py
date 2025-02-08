# config.py

import os
import sys
import certifi
from dotenv import load_dotenv
import asyncio

# --------------------------------------
# 1. Definição de Variáveis de Ambiente
# --------------------------------------

# Ajusta a variável de ambiente para o certificado SSL
os.environ['SSL_CERT_FILE'] = certifi.where()

# Para Windows: ajusta a policy do event loop
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Carrega variáveis do arquivo .env
load_dotenv()

API_KEY = os.environ.get('BINANCE_API_KEY_TESTNET')
API_SECRET = os.environ.get('BINANCE_API_SECRET_TESTNET')

if not API_KEY or not API_SECRET:
    raise EnvironmentError(
        "Chaves de API não encontradas. "
        "Por favor, defina as variáveis de ambiente BINANCE_API_KEY_TESTNET e BINANCE_API_SECRET_TESTNET no arquivo .env."
    )

# --------------------------------------
# 2. Parâmetros de Trading e do Modelo
# --------------------------------------

CAPITAL_INICIAL = 1000.0       # Capital inicial em dólares
RISK_PER_TRADE = 0.02          # Risco de 2% do capital por trade
LEVERAGE = 25                  # Alavancagem
TRANSACTION_COST = 0.0004      # Custo de transação
SLIPPAGE = 0.0001              # Slippage estimado
SMA_WINDOW_SHORT = 5
SMA_WINDOW_LONG = 10

# Exemplo de limite de perda diária (2% do capital):
DAILY_LOSS_LIMIT = -0.02 * CAPITAL_INICIAL
