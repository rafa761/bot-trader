# logger.py

import os
import logging
import logging.handlers

# Cria pasta de logs caso não exista
if not os.path.exists('logs'):
    os.makedirs('logs')

# Configuração básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('logs/trading_app.log', maxBytes=5*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
