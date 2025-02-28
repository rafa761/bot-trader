# services/performance_monitor.py

import datetime
import json
import os
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.config import settings
from core.logger import logger


class TradeResult(str, Enum):
    """Enumeração para os possíveis resultados de um trade."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"  # Para trades que saem próximo do ponto de entrada
    UNKNOWN = "UNKNOWN"  # Para trades que ainda não foram concluídos


class TradeStatus(str, Enum):
    """Enumeração para o status atual de um trade."""
    OPEN = "OPEN"  # Trade ainda está aberto
    CLOSED = "CLOSED"  # Trade foi fechado (TP, SL ou saída manual)
    CANCELLED = "CANCELLED"  # Trade foi cancelado antes de ser executado
    ERROR = "ERROR"  # Ocorreu um erro durante a execução do trade


class Trade(BaseModel):
    """Modelo para representar um trade completo."""
    trade_id: str
    signal_id: str | None = None
    direction: str
    entry_time: datetime.datetime
    entry_price: float
    exit_time: datetime.datetime | None = None
    exit_price: float | None = None

    # TP/SL previstos e reais
    predicted_tp_pct: float
    predicted_sl_pct: float
    actual_tp_pct: float | None = None
    actual_sl_pct: float | None = None

    # Preços alvos calculados
    tp_target_price: float
    sl_target_price: float

    # Resultado
    status: TradeStatus = TradeStatus.OPEN
    result: TradeResult = TradeResult.UNKNOWN
    profit_loss_pct: float | None = None
    profit_loss_absolute: float | None = None
    profit_loss_r: float | None = None  # Lucro/perda em R-múltiplos

    # Quantidade e margem
    quantity: float
    leverage: int  # Alavancagem utilizada
    margin_used: float  # Margem utilizada para o trade

    # Detalhes adicionais
    entry_score: float | None = None
    rr_ratio: float | None = None

    # Condições de mercado
    market_trend: str | None = None  # UPTREND, DOWNTREND, NEUTRAL
    market_volatility: float | None = None  # ATR%
    market_strength: str | None = None  # STRONG_TREND, WEAK_TREND

    # Metadados
    notes: str | None = None
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"

    def calculate_profit_loss(self) -> tuple[float, float, float]:
        """
        Calcula o lucro/perda em percentual e absoluto.

        Returns:
            Tuple[float, float, float]: (profit_loss_pct, profit_loss_absolute, profit_loss_r)
        """
        if self.exit_price is None or self.status != TradeStatus.CLOSED:
            return 0.0, 0.0, 0.0

        # Calcular P&L em percentual
        if self.direction == "LONG":
            profit_loss_pct = ((self.exit_price / self.entry_price) - 1) * 100
        else:  # SHORT
            profit_loss_pct = ((self.entry_price / self.exit_price) - 1) * 100

        # Calcular P&L absoluto (considerando alavancagem)
        notional_value = self.entry_price * self.quantity
        profit_loss_absolute = notional_value * (profit_loss_pct / 100) * self.leverage

        # Calcular P&L em R-múltiplos (proporção em relação ao risco inicial)
        # R=1 significa que o trade ganhou exatamente o valor arriscado
        initial_risk_pct = self.predicted_sl_pct
        if initial_risk_pct > 0:
            profit_loss_r = profit_loss_pct / initial_risk_pct
        else:
            profit_loss_r = 0.0

        return profit_loss_pct, profit_loss_absolute, profit_loss_r

    def update_status_and_result(self) -> None:
        """Atualiza o status e resultado do trade."""
        if self.exit_price is None:
            self.status = TradeStatus.OPEN
            self.result = TradeResult.UNKNOWN
            return

        self.status = TradeStatus.CLOSED

        # Calcular P&L
        self.profit_loss_pct, self.profit_loss_absolute, self.profit_loss_r = self.calculate_profit_loss()

        # Determinar resultado (com uma pequena margem para breakeven)
        if abs(self.profit_loss_pct) < 0.05:  # Menos de 0.05% é considerado breakeven
            self.result = TradeResult.BREAKEVEN
        elif self.profit_loss_pct > 0:
            self.result = TradeResult.WIN
        else:
            self.result = TradeResult.LOSS

    def to_dict(self) -> dict[str, Any]:
        """Converte o trade para um dicionário."""
        result = self.dict()

        # Converter datetime para string ISO
        if result["entry_time"]:
            result["entry_time"] = result["entry_time"].isoformat()
        if result["exit_time"]:
            result["exit_time"] = result["exit_time"].isoformat()

        # Converter Enum para string
        result["status"] = result["status"].value
        result["result"] = result["result"].value

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Trade':
        """Cria um objeto Trade a partir de um dicionário."""
        # Converter string ISO para datetime
        if "entry_time" in data and data["entry_time"]:
            data["entry_time"] = datetime.datetime.fromisoformat(data["entry_time"])
        if "exit_time" in data and data["exit_time"]:
            data["exit_time"] = datetime.datetime.fromisoformat(data["exit_time"])

        # Converter string para Enum
        if "status" in data:
            data["status"] = TradeStatus(data["status"])
        if "result" in data:
            data["result"] = TradeResult(data["result"])

        return cls(**data)


class PerformanceMetrics(BaseModel):
    """Modelo para representar métricas de desempenho."""
    # Métricas gerais
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0

    # Win rate
    win_rate: float = 0.0  # (winning_trades / total_trades)

    # Dados de lucro/perda
    total_profit_loss: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    average_profit: float = 0.0
    average_loss: float = 0.0
    average_trade: float = 0.0

    # Profit factor = total_profit / abs(total_loss)
    profit_factor: float = 0.0

    # Expectancy = (win_rate * average_profit) - ((1-win_rate) * average_loss)
    expectancy: float = 0.0

    # Risk/Reward
    average_rr_ratio: float = 0.0

    # Drawdown
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0

    # Métricas por direção
    long_trades: int = 0
    long_wins: int = 0
    long_win_rate: float = 0.0
    short_trades: int = 0
    short_wins: int = 0
    short_win_rate: float = 0.0

    # Métricas por tendência
    trend_aligned_trades: int = 0
    trend_aligned_wins: int = 0
    trend_aligned_win_rate: float = 0.0
    counter_trend_trades: int = 0
    counter_trend_wins: int = 0
    counter_trend_win_rate: float = 0.0

    # Série temporal para gráfico de equity
    equity_curve: list[float] = Field(default_factory=list)

    # Streak (sequência de ganhos/perdas)
    current_streak: int = 0  # Positivo para wins, negativo para losses
    max_win_streak: int = 0
    max_loss_streak: int = 0

    # Métricas de tempo
    avg_trade_duration_minutes: float = 0.0

    # Métricas de qualidade de previsão
    tp_prediction_accuracy: float = 0.0  # Quão próximos os TPs reais são dos previstos
    sl_prediction_accuracy: float = 0.0  # Quão próximos os SLs reais são dos previstos

    def to_dict(self) -> dict[str, Any]:
        """Converte as métricas para um dicionário."""
        result = self.dict()

        # Converter arrays numpy para listas Python, se necessário
        if isinstance(result["equity_curve"], np.ndarray):
            result["equity_curve"] = result["equity_curve"].tolist()

        return result


class TradePerformanceMonitor:
    """
    Monitor de performance para análise e registro de trades.

    Esta classe rastreia todos os trades executados, calcula métricas de desempenho
    e fornece insights para melhorar a estratégia de trading.
    """

    def __init__(self, db_path: str = None):
        """
        Inicializa o monitor de performance.

        Args:
            db_path: Caminho para o banco de dados SQLite. Se None,
                    será usado o caminho padrão na pasta do projeto.
        """
        self.trades: list[Trade] = []
        self.metrics = PerformanceMetrics()

        # Definir caminho do banco de dados
        if db_path is None:
            project_dir = Path(__file__).resolve().parent.parent.parent
            db_path = str(project_dir / "data" / "trade_performance.db")

        self.db_path = db_path
        self._initialize_db()

        # Carregar trades existentes do banco de dados
        self._load_trades_from_db()

        logger.info(f"Monitor de Performance inicializado - {len(self.trades)} trades carregados")

    def _initialize_db(self) -> None:
        """Inicializa o banco de dados SQLite."""
        # Certificar que o diretório existe
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

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

            # Criar tabela de métricas acumuladas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Criar índices para melhorar performance de consultas
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")

            conn.commit()
            conn.close()

            logger.info(f"Banco de dados inicializado em {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")

    def _load_trades_from_db(self) -> None:
        """Carrega trades existentes do banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT trade_id, data FROM trades")
            rows = cursor.fetchall()

            for row in rows:
                trade_id, data_json = row
                trade_data = json.loads(data_json)
                trade = Trade.from_dict(trade_data)
                self.trades.append(trade)

            conn.close()

            # Após carregar os trades, calcular métricas
            if self.trades:
                self.calculate_metrics()

            logger.info(f"Carregados {len(self.trades)} trades do banco de dados")
        except Exception as e:
            logger.error(f"Erro ao carregar trades do banco de dados: {e}")

    def add_trade(self, trade: Trade) -> None:
        """
        Adiciona um novo trade ao monitor.

        Args:
            trade: O trade a ser adicionado.
        """
        # Verificar se o trade já existe
        if any(t.trade_id == trade.trade_id for t in self.trades):
            logger.warning(f"Trade {trade.trade_id} já existe no monitor. Atualizando...")
            self.update_trade(trade)
            return

        # Adicionar a lista em memória
        self.trades.append(trade)

        # Salvar no banco de dados
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO trades (trade_id, signal_id, data) VALUES (?, ?, ?)",
                (trade.trade_id, trade.signal_id, json.dumps(trade.to_dict()))
            )

            conn.commit()
            conn.close()

            # Recalcular métricas
            self.calculate_metrics()

            logger.info(f"Trade {trade.trade_id} adicionado ao monitor")
        except Exception as e:
            logger.error(f"Erro ao adicionar trade ao banco de dados: {e}")

    def update_trade(self, trade: Trade) -> None:
        """
        Atualiza um trade existente.

        Args:
            trade: O trade com informações atualizadas.
        """
        # Atualizar o trade em memória
        for i, t in enumerate(self.trades):
            if t.trade_id == trade.trade_id:
                self.trades[i] = trade
                break
        else:
            logger.warning(f"Trade {trade.trade_id} não encontrado para atualização. Adicionando...")
            self.add_trade(trade)
            return

        # Atualizar no banco de dados
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE trades SET data = ?, signal_id = ?, updated_at = CURRENT_TIMESTAMP WHERE trade_id = ?",
                (json.dumps(trade.to_dict()), trade.signal_id, trade.trade_id)
            )

            conn.commit()
            conn.close()

            # Recalcular métricas
            self.calculate_metrics()

            logger.info(f"Trade {trade.trade_id} atualizado")
        except Exception as e:
            logger.error(f"Erro ao atualizar trade no banco de dados: {e}")

    def get_trade(self, trade_id: str) -> Trade | None:
        """
        Recupera um trade pelo ID.

        Args:
            trade_id: ID do trade a ser recuperado.

        Returns:
            O trade encontrado ou None se não existir.
        """
        for trade in self.trades:
            if trade.trade_id == trade_id:
                return trade
        return None

    def get_trade_by_signal_id(self, signal_id: str) -> Trade | None:
        """
        Recupera um trade pelo signal_id.

        Args:
            signal_id: ID do sinal associado ao trade.

        Returns:
            O trade encontrado ou None se não existir.
        """
        for trade in self.trades:
            if trade.signal_id == signal_id:
                return trade
        return None

    def get_trades_dataframe(self) -> pd.DataFrame:
        """
        Converte todos os trades para um DataFrame do Pandas.

        Returns:
            DataFrame com todos os trades.
        """
        if not self.trades:
            return pd.DataFrame()

        # Converter cada trade para um dicionário
        trades_data = [trade.to_dict() for trade in self.trades]

        # Criar DataFrame
        df = pd.DataFrame(trades_data)

        # Converter colunas de data para datetime
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = pd.to_datetime(df['exit_time'])

        return df

    def register_trade_exit(self, trade_id: str, exit_price: float, exit_time: datetime.datetime = None) -> None:
        """
        Registra a saída de um trade.

        Args:
            trade_id: ID do trade.
            exit_price: Preço de saída.
            exit_time: Timestamp da saída (se None, será usado o timestamp atual).
        """
        trade = self.get_trade(trade_id)
        if not trade:
            logger.warning(f"Trade {trade_id} não encontrado para registrar saída")
            return

        # Atualizar informações de saída
        trade.exit_price = exit_price
        trade.exit_time = exit_time or datetime.datetime.now()

        # Calcular TP/SL reais
        if trade.direction == "LONG":
            actual_pct_change = ((exit_price / trade.entry_price) - 1) * 100
        else:  # SHORT
            actual_pct_change = ((trade.entry_price / exit_price) - 1) * 100

        # Determinar se saiu por TP ou SL
        if actual_pct_change >= 0:
            trade.actual_tp_pct = actual_pct_change
            trade.actual_sl_pct = 0
        else:
            trade.actual_tp_pct = 0
            trade.actual_sl_pct = abs(actual_pct_change)

        # Atualizar status e resultado
        trade.update_status_and_result()

        # Atualizar o trade
        self.update_trade(trade)

        logger.info(
            f"Saída registrada para trade {trade_id}: "
            f"Preço={exit_price}, P&L={trade.profit_loss_pct:.2f}%, "
            f"Resultado={trade.result.value}"
        )

    def register_trade_from_signal(
            self,
            signal_id: str,
            direction: str,
            entry_price: float,
            quantity: float,
            tp_target_price: float,
            sl_target_price: float,
            predicted_tp_pct: float,
            predicted_sl_pct: float,
            market_trend: str = None,
            market_volatility: float = None,
            market_strength: str = None,
            entry_score: float = None,
            rr_ratio: float = None,
            leverage: int = settings.LEVERAGE,
            entry_time: datetime.datetime = None,
            trade_id: str = None
    ) -> Trade:
        """
        Registra um novo trade a partir de um sinal.

        Args:
            signal_id: ID do sinal que gerou o trade.
            direction: Direção do trade (LONG/SHORT).
            entry_price: Preço de entrada.
            quantity: Quantidade negociada.
            tp_target_price: Preço alvo para take profit.
            sl_target_price: Preço alvo para stop loss.
            predicted_tp_pct: Percentual previsto para take profit.
            predicted_sl_pct: Percentual previsto para stop loss.
            market_trend: Tendência do mercado (UPTREND/DOWNTREND/NEUTRAL).
            market_volatility: Volatilidade do mercado (ATR%).
            market_strength: Força da tendência (STRONG_TREND/WEAK_TREND).
            entry_score: Pontuação de qualidade da entrada.
            rr_ratio: Razão risk/reward.
            leverage: Alavancagem utilizada.
            entry_time: Timestamp da entrada (se None, será usado o timestamp atual).
            trade_id: ID do trade (se None, será gerado um ID baseado no timestamp).

        Returns:
            Objeto Trade criado.
        """
        # Gerar ID de trade baseado no timestamp se não fornecido
        if not trade_id:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            trade_id = f"TRADE_{timestamp}_{np.random.randint(1000, 9999)}"

        # Criar o objeto Trade
        trade = Trade(
            trade_id=trade_id,
            signal_id=signal_id,
            direction=direction,
            entry_time=entry_time or datetime.datetime.now(),
            entry_price=entry_price,
            tp_target_price=tp_target_price,
            sl_target_price=sl_target_price,
            predicted_tp_pct=predicted_tp_pct,
            predicted_sl_pct=predicted_sl_pct,
            quantity=quantity,
            leverage=leverage,
            margin_used=(entry_price * quantity) / leverage,
            market_trend=market_trend,
            market_volatility=market_volatility,
            market_strength=market_strength,
            entry_score=entry_score,
            rr_ratio=rr_ratio,
            symbol=settings.SYMBOL,
            timeframe=settings.INTERVAL
        )

        # Adicionar ao monitor
        self.add_trade(trade)

        logger.info(
            f"Novo trade registrado: ID={trade_id}, Sinal={signal_id}, "
            f"{direction} a {entry_price}, Qty={quantity}, "
            f"TP={tp_target_price}({predicted_tp_pct:.2f}%), "
            f"SL={sl_target_price}({predicted_sl_pct:.2f}%)"
        )

        return trade

    def calculate_metrics(self) -> PerformanceMetrics:
        """
        Calcula todas as métricas de desempenho com base nos trades registrados.

        Returns:
            Objeto PerformanceMetrics atualizado.
        """
        # Resetar métricas
        self.metrics = PerformanceMetrics()

        # Filtrar apenas trades fechados
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]

        if not closed_trades:
            logger.info("Sem trades fechados para calcular métricas")
            return self.metrics

        # Ordenar por data de saída
        closed_trades.sort(key=lambda t: t.exit_time or datetime.datetime.min)

        # Métricas básicas
        self.metrics.total_trades = len(closed_trades)
        self.metrics.winning_trades = sum(1 for t in closed_trades if t.result == TradeResult.WIN)
        self.metrics.losing_trades = sum(1 for t in closed_trades if t.result == TradeResult.LOSS)
        self.metrics.breakeven_trades = sum(1 for t in closed_trades if t.result == TradeResult.BREAKEVEN)

        # Win rate
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        # Lucro/perda
        total_profit = sum(t.profit_loss_absolute for t in closed_trades if t.profit_loss_absolute > 0)
        total_loss = sum(t.profit_loss_absolute for t in closed_trades if t.profit_loss_absolute < 0)

        self.metrics.total_profit = total_profit
        self.metrics.total_loss = total_loss
        self.metrics.total_profit_loss = total_profit + total_loss

        # Médias
        if self.metrics.winning_trades > 0:
            self.metrics.average_profit = total_profit / self.metrics.winning_trades

        if self.metrics.losing_trades > 0:
            self.metrics.average_loss = total_loss / self.metrics.losing_trades

        if self.metrics.total_trades > 0:
            self.metrics.average_trade = self.metrics.total_profit_loss / self.metrics.total_trades

        # Profit factor
        if total_loss != 0:
            self.metrics.profit_factor = abs(total_profit / total_loss)

        # Expectancy
        if self.metrics.win_rate > 0:
            avg_win = self.metrics.average_profit
            avg_loss = abs(self.metrics.average_loss)
            self.metrics.expectancy = (self.metrics.win_rate * avg_win) - ((1 - self.metrics.win_rate) * avg_loss)

        # Métricas por direção
        long_trades = [t for t in closed_trades if t.direction == "LONG"]
        short_trades = [t for t in closed_trades if t.direction == "SHORT"]

        self.metrics.long_trades = len(long_trades)
        self.metrics.long_wins = sum(1 for t in long_trades if t.result == TradeResult.WIN)
        if self.metrics.long_trades > 0:
            self.metrics.long_win_rate = self.metrics.long_wins / self.metrics.long_trades

        self.metrics.short_trades = len(short_trades)
        self.metrics.short_wins = sum(1 for t in short_trades if t.result == TradeResult.WIN)
        if self.metrics.short_trades > 0:
            self.metrics.short_win_rate = self.metrics.short_wins / self.metrics.short_trades

        # Métricas por tendência
        aligned_trades = [
            t for t in closed_trades if (
                    (t.market_trend == "UPTREND" and t.direction == "LONG") or
                    (t.market_trend == "DOWNTREND" and t.direction == "SHORT")
            )
        ]

        counter_trades = [
            t for t in closed_trades if (
                    (t.market_trend == "UPTREND" and t.direction == "SHORT") or
                    (t.market_trend == "DOWNTREND" and t.direction == "LONG")
            )
        ]

        self.metrics.trend_aligned_trades = len(aligned_trades)
        self.metrics.trend_aligned_wins = sum(1 for t in aligned_trades if t.result == TradeResult.WIN)
        if self.metrics.trend_aligned_trades > 0:
            self.metrics.trend_aligned_win_rate = self.metrics.trend_aligned_wins / self.metrics.trend_aligned_trades

        self.metrics.counter_trend_trades = len(counter_trades)
        self.metrics.counter_trend_wins = sum(1 for t in counter_trades if t.result == TradeResult.WIN)
        if self.metrics.counter_trend_trades > 0:
            self.metrics.counter_trend_win_rate = self.metrics.counter_trend_wins / self.metrics.counter_trend_trades

        # Calcular curva de equity
        initial_capital = settings.CAPITAL
        equity = [initial_capital]
        current_capital = initial_capital

        for trade in closed_trades:
            if trade.profit_loss_absolute is not None:
                current_capital += trade.profit_loss_absolute
                equity.append(current_capital)

        self.metrics.equity_curve = equity

        # Calcular drawdown máximo
        peak = initial_capital
        max_drawdown = 0
        current_drawdown = 0

        for value in equity:
            if value > peak:
                peak = value

            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

            # Atualizar drawdown atual
            if value == equity[-1]:
                current_drawdown = drawdown

        self.metrics.max_drawdown_pct = max_drawdown
        self.metrics.current_drawdown_pct = current_drawdown

        # Calcular streaks
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for trade in closed_trades:
            if trade.result == TradeResult.WIN:
                if current_streak <= 0:
                    current_streak = 1
                else:
                    current_streak += 1
                max_win_streak = max(max_win_streak, current_streak)
            elif trade.result == TradeResult.LOSS:
                if current_streak >= 0:
                    current_streak = -1
                else:
                    current_streak -= 1
                max_loss_streak = max(max_loss_streak, abs(current_streak))
            else:  # BREAKEVEN
                continue  # Não afeta streaks

        self.metrics.current_streak = current_streak
        self.metrics.max_win_streak = max_win_streak
        self.metrics.max_loss_streak = max_loss_streak

        # Calcular duração média dos trades
        durations = []
        for trade in closed_trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60  # minutos
                durations.append(duration)

        if durations:
            self.metrics.avg_trade_duration_minutes = sum(durations) / len(durations)

        # Calcular precisão das previsões de TP/SL
        tp_errors = []
        sl_errors = []

        for trade in closed_trades:
            if trade.result == TradeResult.WIN and trade.actual_tp_pct and trade.predicted_tp_pct:
                # Calcular erro de previsão de TP
                error = abs(trade.actual_tp_pct - trade.predicted_tp_pct) / max(0.1, trade.predicted_tp_pct)
                tp_errors.append(min(error, 1.0))  # Limitar a 100% de erro

            if trade.result == TradeResult.LOSS and trade.actual_sl_pct and trade.predicted_sl_pct:
                # Calcular erro de previsão de SL
                error = abs(trade.actual_sl_pct - trade.predicted_sl_pct) / max(0.1, trade.predicted_sl_pct)
                sl_errors.append(min(error, 1.0))  # Limitar a 100% de erro

        if tp_errors:
            self.metrics.tp_prediction_accuracy = 1.0 - (sum(tp_errors) / len(tp_errors))

        if sl_errors:
            self.metrics.sl_prediction_accuracy = 1.0 - (sum(sl_errors) / len(sl_errors))

        # Calcular R:R médio
        rr_values = [t.rr_ratio for t in closed_trades if t.rr_ratio is not None]
        if rr_values:
            self.metrics.average_rr_ratio = sum(rr_values) / len(rr_values)

        # Salvar métricas no banco de dados
        self._save_metrics_to_db()

        return self.metrics

    def _save_metrics_to_db(self) -> None:
        """Salva as métricas atuais no banco de dados."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO metrics (data) VALUES (?)",
                (json.dumps(self.metrics.to_dict()),)
            )

            conn.commit()
            conn.close()

            logger.debug("Métricas salvas no banco de dados")
        except Exception as e:
            logger.error(f"Erro ao salvar métricas no banco de dados: {e}")

    def get_weekly_performance_summary(self) -> dict[str, Any]:
        """
        Obtém um resumo de desempenho da semana atual.

        Returns:
            Dicionário com resumo de desempenho semanal.
        """
        # Obter data de início da semana atual (segunda-feira)
        today = datetime.datetime.now()
        start_of_week = today - datetime.timedelta(days=today.weekday())
        start_of_week = datetime.datetime(start_of_week.year, start_of_week.month, start_of_week.day)

        # Filtrar trades da semana atual
        weekly_trades = [
            t for t in self.trades
            if t.status == TradeStatus.CLOSED and t.exit_time and t.exit_time >= start_of_week
        ]

        # Calcular métricas da semana
        total_trades = len(weekly_trades)
        winning_trades = sum(1 for t in weekly_trades if t.result == TradeResult.WIN)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_profit = sum(t.profit_loss_absolute for t in weekly_trades if t.profit_loss_absolute > 0)
        total_loss = sum(t.profit_loss_absolute for t in weekly_trades if t.profit_loss_absolute < 0)
        total_pnl = total_profit + total_loss

        # Resumo
        return {
            "period": f"{start_of_week.strftime('%d/%m/%Y')} até {today.strftime('%d/%m/%Y')}",
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "best_trade": max(
                (t.profit_loss_absolute for t in weekly_trades if t.profit_loss_absolute is not None),
                default=0
            ),
            "worst_trade": min(
                (t.profit_loss_absolute for t in weekly_trades if t.profit_loss_absolute is not None),
                default=0
            )
        }

    def get_performance_report(self) -> dict[str, Any]:
        """
        Gera um relatório completo de desempenho.

        Returns:
            Dicionário com métricas completas de desempenho.
        """
        # Garantir que as métricas estão atualizadas
        self.calculate_metrics()

        # Criar relatório
        report = {
            "general": {
                "total_trades": self.metrics.total_trades,
                "winning_trades": self.metrics.winning_trades,
                "losing_trades": self.metrics.losing_trades,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "expectancy": self.metrics.expectancy,
                "total_profit_loss": self.metrics.total_profit_loss,
                "max_drawdown": self.metrics.max_drawdown_pct,
                "avg_trade_duration_minutes": self.metrics.avg_trade_duration_minutes
            },
            "by_direction": {
                "long": {
                    "trades": self.metrics.long_trades,
                    "wins": self.metrics.long_wins,
                    "win_rate": self.metrics.long_win_rate
                },
                "short": {
                    "trades": self.metrics.short_trades,
                    "wins": self.metrics.short_wins,
                    "win_rate": self.metrics.short_win_rate
                }
            },
            "by_trend": {
                "aligned": {
                    "trades": self.metrics.trend_aligned_trades,
                    "wins": self.metrics.trend_aligned_wins,
                    "win_rate": self.metrics.trend_aligned_win_rate
                },
                "counter": {
                    "trades": self.metrics.counter_trend_trades,
                    "wins": self.metrics.counter_trend_wins,
                    "win_rate": self.metrics.counter_trend_win_rate
                }
            },
            "streaks": {
                "current": self.metrics.current_streak,
                "max_win": self.metrics.max_win_streak,
                "max_loss": self.metrics.max_loss_streak
            },
            "prediction_accuracy": {
                "tp": self.metrics.tp_prediction_accuracy,
                "sl": self.metrics.sl_prediction_accuracy
            },
            "weekly_summary": self.get_weekly_performance_summary()
        }

        return report

    def log_performance_summary(self) -> None:
        """
        Registra um resumo de desempenho no log.
        """
        if not self.trades:
            logger.info("Sem trades para gerar resumo de desempenho")
            return

        # Garantir que as métricas estão atualizadas
        self.calculate_metrics()

        # Criar mensagem de log
        logger.info("=" * 80)
        logger.info("RESUMO DE DESEMPENHO")
        logger.info("=" * 80)
        logger.info(f"Total de trades: {self.metrics.total_trades}")
        logger.info(f"Win rate: {self.metrics.win_rate:.2%}")
        logger.info(f"Profit factor: {self.metrics.profit_factor:.2f}")
        logger.info(f"P&L total: ${self.metrics.total_profit_loss:.2f}")
        logger.info(f"Drawdown máximo: {self.metrics.max_drawdown_pct:.2f}%")
        logger.info(f"Lucro médio por trade: ${self.metrics.average_profit:.2f}")
        logger.info(f"Perda média por trade: ${self.metrics.average_loss:.2f}")

        logger.info("-" * 40)
        logger.info("Desempenho por direção:")
        logger.info(
            f"LONG: {self.metrics.long_win_rate:.2%} win rate ({self.metrics.long_wins}/{self.metrics.long_trades})")
        logger.info(
            f"SHORT: {self.metrics.short_win_rate:.2%} win rate ({self.metrics.short_wins}/{self.metrics.short_trades})")

        logger.info("-" * 40)
        logger.info("Desempenho por tendência:")
        logger.info(
            f"A favor: {self.metrics.trend_aligned_win_rate:.2%} win rate ({self.metrics.trend_aligned_wins}/{self.metrics.trend_aligned_trades})")
        logger.info(
            f"Contra: {self.metrics.counter_trend_win_rate:.2%} win rate ({self.metrics.counter_trend_wins}/{self.metrics.counter_trend_trades})")

        logger.info("-" * 40)
        logger.info(f"Acurácia de previsão TP: {self.metrics.tp_prediction_accuracy:.2%}")
        logger.info(f"Acurácia de previsão SL: {self.metrics.sl_prediction_accuracy:.2%}")
        logger.info("=" * 80)
