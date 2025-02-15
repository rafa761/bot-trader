# logger.py

import logging.handlers
import os
from pathlib import Path

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

class InMemoryLogger(logging.Handler):
    def __init__(self, capacity=5000):
        super().__init__()
        self.capacity = capacity
        self.records = []

    def emit(self, record):
        formatted = self.format(record)
        self.records.append(formatted)
        # Se ultrapassar o limite, descarta o mais antigo
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def get_logs(self):
        return self.records


# Instancia o handler em memória
memory_logger = InMemoryLogger(capacity=5000)

# Defina o mesmo formato que você usa no basicConfig
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
memory_logger.setFormatter(formatter)

# Agora, configure o logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler(
            logs_dir / 'trading_app.log',
            maxBytes=5 * 1024 * 1024,
            backupCount=5
        ),
        logging.StreamHandler()
    ]
)

# Adiciona o handler de memória ao logger root
logging.getLogger().addHandler(memory_logger)

logger = logging.getLogger(__name__)
