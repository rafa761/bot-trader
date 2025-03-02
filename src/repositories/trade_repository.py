# repositories/trade_repository.py

import datetime
import json
import sqlite3
from pathlib import Path

from core.logger import logger
from services.performance_monitor import Trade, TradeStatus, TradeResult


class TradeRepository:
    """
    Repositório responsável pelo acesso e persistência dos dados de trades.

    Esta classe fornece uma camada de abstração para interagir com o
    banco de dados de trades, seguindo o padrão Repository do SOLID.
    """

    def __init__(self, db_path: str = None):
        """
        Inicializa o repositório de trades.

        Args:
            db_path: Caminho para o banco de dados SQLite. Se None,
                    será usado o caminho padrão na pasta do projeto.
        """
        # Definir caminho do banco de dados
        if db_path is None:
            project_dir = Path(__file__).resolve().parent.parent
            db_path = str(project_dir / "data" / "trade_performance.db")

        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Inicializa o banco de dados SQLite para armazenar trades."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Criar tabela de trades se não existir
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    signal_id TEXT,
                    data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Criar índices para melhorar performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id)")

            conn.commit()
            conn.close()

            logger.info(f"Banco de dados de trades inicializado em {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados de trades: {e}")

    def save(self, trade: Trade) -> None:
        """
        Salva um trade no banco de dados.

        Args:
            trade: O trade a ser salvo.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Verificar se já existe
            cursor.execute("SELECT 1 FROM trades WHERE trade_id = ?", (trade.trade_id,))
            exists = cursor.fetchone() is not None

            if exists:
                # Atualizar
                cursor.execute(
                    "UPDATE trades SET data = ?, signal_id = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                    (json.dumps(trade.to_dict()), trade.signal_id, trade.trade_id)
                )
                logger.info(f"Trade {trade.trade_id} atualizado no banco de dados")
            else:
                # Inserir novo
                cursor.execute(
                    "INSERT INTO trades (trade_id, signal_id, data) VALUES (?, ?, ?)",
                    (trade.trade_id, trade.signal_id, json.dumps(trade.to_dict()))
                )
                logger.info(f"Trade {trade.trade_id} inserido no banco de dados")

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Erro ao salvar trade no banco de dados: {e}")

    def get_by_id(self, trade_id: str) -> Trade | None:
        """
        Recupera um trade pelo ID.

        Args:
            trade_id: ID do trade a ser recuperado.

        Returns:
            O trade encontrado ou None se não existir.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades WHERE trade_id = ?", (trade_id,))
            result = cursor.fetchone()

            conn.close()

            if result:
                trade_data = json.loads(result[0])
                return Trade.from_dict(trade_data)

            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar trade {trade_id}: {e}")
            return None

    def get_by_signal_id(self, signal_id: str) -> Trade | None:
        """
        Recupera um trade pelo signal_id.

        Args:
            signal_id: ID do sinal associado ao trade.

        Returns:
            O trade encontrado ou None se não existir.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades WHERE signal_id = ?", (signal_id,))
            result = cursor.fetchone()

            conn.close()

            if result:
                trade_data = json.loads(result[0])
                return Trade.from_dict(trade_data)

            return None
        except Exception as e:
            logger.error(f"Erro ao recuperar trade por signal_id {signal_id}: {e}")
            return None

    def get_all(self) -> list[Trade]:
        """
        Recupera todos os trades.

        Returns:
            Lista com todos os trades.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades ORDER BY created_at DESC")
            results = cursor.fetchall()

            conn.close()

            trades = []
            for result in results:
                trade_data = json.loads(result[0])
                trades.append(Trade.from_dict(trade_data))

            return trades
        except Exception as e:
            logger.error(f"Erro ao recuperar todos os trades: {e}")
            return []

    def get_open_trades(self) -> list[Trade]:
        """
        Recupera todos os trades abertos.

        Returns:
            Lista com todos os trades abertos.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades")
            results = cursor.fetchall()

            conn.close()

            open_trades = []
            for result in results:
                trade_data = json.loads(result[0])
                trade = Trade.from_dict(trade_data)
                if trade.status == TradeStatus.OPEN:
                    open_trades.append(trade)

            return open_trades
        except Exception as e:
            logger.error(f"Erro ao recuperar trades abertos: {e}")
            return []

    def get_trades_by_result(self, result: TradeResult) -> list[Trade]:
        """
        Recupera todos os trades com um determinado resultado.

        Args:
            result: Resultado desejado (WIN, LOSS, BREAKEVEN).

        Returns:
            Lista com os trades encontrados.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades")
            results = cursor.fetchall()

            conn.close()

            filtered_trades = []
            for row in results:
                trade_data = json.loads(row[0])
                trade = Trade.from_dict(trade_data)
                if trade.result == result:
                    filtered_trades.append(trade)

            return filtered_trades
        except Exception as e:
            logger.error(f"Erro ao recuperar trades por resultado {result}: {e}")
            return []

    def get_trades_by_period(self, start_date: datetime.datetime, end_date: datetime.datetime) -> list[Trade]:
        """
        Recupera todos os trades em um período específico.

        Args:
            start_date: Data de início do período.
            end_date: Data de fim do período.

        Returns:
            Lista com os trades no período.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT data FROM trades")
            results = cursor.fetchall()

            conn.close()

            filtered_trades = []
            for row in results:
                trade_data = json.loads(row[0])
                trade = Trade.from_dict(trade_data)

                # Verificar se a data de entrada está no período
                if (trade.entry_time and start_date <= trade.entry_time <= end_date):
                    filtered_trades.append(trade)

            return filtered_trades
        except Exception as e:
            logger.error(f"Erro ao recuperar trades por período: {e}")
            return []

    def delete(self, trade_id: str) -> bool:
        """
        Remove um trade do banco de dados.

        Args:
            trade_id: ID do trade a ser removido.

        Returns:
            True se o trade foi removido com sucesso, False caso contrário.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM trades WHERE trade_id = ?", (trade_id,))
            deleted = cursor.rowcount > 0

            conn.commit()
            conn.close()

            if deleted:
                logger.info(f"Trade {trade_id} removido com sucesso")
            else:
                logger.warning(f"Trade {trade_id} não encontrado para remoção")

            return deleted
        except Exception as e:
            logger.error(f"Erro ao remover trade {trade_id}: {e}")
            return False

    def get_latest_trades(self, limit: int = 10) -> list[Trade]:
        """
        Recupera os trades mais recentes.

        Args:
            limit: Número máximo de trades a serem retornados.

        Returns:
            Lista com os trades mais recentes.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "SELECT data FROM trades ORDER BY created_at DESC LIMIT ?",
                (limit,)
            )
            results = cursor.fetchall()

            conn.close()

            trades = []
            for result in results:
                trade_data = json.loads(result[0])
                trades.append(Trade.from_dict(trade_data))

            return trades
        except Exception as e:
            logger.error(f"Erro ao recuperar trades recentes: {e}")
            return []
