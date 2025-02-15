# logger.py

import logging
from logging.handlers import RotatingFileHandler
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


# Define o formato do log
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Instancia o handler em memória
memory_logger = InMemoryLogger(capacity=5000)
memory_logger.setFormatter(formatter)


def setup_logging():
    # Configura o logging basico
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                logs_dir / "trading_app.log",
                maxBytes=5 * 1024 * 1024,  # 5mb
                backupCount=5,
            ),
            logging.StreamHandler()
        ]
    )

    # Adiciona o handler de memória ao logger root
    logging.getLogger().addHandler(memory_logger)

    return logging.getLogger(__name__)


logger = setup_logging()
