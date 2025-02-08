# trading_bot.py

import time
from datetime import datetime, timezone
from threading import Lock

import numpy as np
import pandas as pd
from binance.exceptions import BinanceAPIException

from binance_client import BinanceClientService
from data_handler import DataHandler
from logger import logger
from model_manager import ModelManager
from risk_manager import dynamic_risk_adjustment  # Para risco dinâmico
from trading_strategy import TradingStrategy


class TradingBot:
    """
    Classe principal que gerencia o bot de trading.
    Integra WebSocket, coloca ordens, registra trades e executa operações automáticas.
    """

    def __init__(self, symbol: str = 'BTCUSDT', interval: str = '1m') -> None:
        """
        Inicializa o bot de trading com os parâmetros fornecidos.

        :param symbol: O símbolo do par de trading.
        :param interval: O intervalo de tempo dos candles.
        """
        self.symbol: str = symbol
        self.interval: str = interval

        # Locks para garantir a segurança em operações multithreading
        self.price_lock: Lock = Lock()
        self.volume_lock: Lock = Lock()
        self.data_lock: Lock = Lock()

        # Dados de mercado atuais
        self.current_real_price: float | None = None
        self.current_volume: float | None = None

        # Objetos principais
        self.binance_service: BinanceClientService = BinanceClientService()
        self.data_handler: DataHandler = DataHandler(self.binance_service)
        self.strategy: TradingStrategy = TradingStrategy()
        self.model_manager: ModelManager = ModelManager()

        # DataFrames de resultados
        self.trade_results: pd.DataFrame = pd.DataFrame()
        self.backtest_results: pd.DataFrame = pd.DataFrame()
        self.performance_metrics: list[dict] = []

    def place_order(self, side: str, quantity: float) -> dict | None:
        """
        Cria uma ordem de mercado de futuros.

        :param side: 'BUY' para compra, 'SELL' para venda.
        :param quantity: Quantidade a ser negociada.
        :return: Dicionário contendo os detalhes da ordem se bem-sucedida, senão None.
        """
        max_attempts = 2
        attempt = 0
        backoff_time = 2

        while attempt < max_attempts:
            try:
                # Obter informações da conta
                account_info = self.binance_service.get_futures_account_info()
                available_balance = 0
                for asset in account_info['assets']:
                    if asset['asset'] == 'USDT':
                        available_balance = float(asset['availableBalance'])
                        break

                # Obter o mark price
                mark_price = float(self.binance_service.get_futures_mark_price(symbol=self.symbol)['markPrice'])
                estimated_margin = (quantity / self.strategy.leverage) * mark_price

                if available_balance < estimated_margin:
                    logger.warning("Margem insuficiente. Ajustando quantidade.")
                    quantity = (available_balance * self.strategy.leverage) / mark_price * 0.9
                    quantity = round(quantity, 3)
                    if quantity <= 0:
                        logger.error("Saldo insuficiente para qualquer ordem. Ordem cancelada.")
                        return None

                position_side = 'LONG' if side == 'BUY' else 'SHORT'
                order = self.binance_service.create_futures_order(
                    symbol=self.symbol,
                    side=side,
                    quantity=quantity,
                    position_side=position_side
                )

                if order and order['status'] == 'FILLED':
                    return order
                else:
                    return None

            except BinanceAPIException as e:
                # -2019 -> Código de erro para margem insuficiente
                if e.code == -2019:
                    logger.error(f"Erro da API da Binance (margem insuficiente): {e}")
                    attempt += 1
                    logger.warning(f"Tentando novamente após {backoff_time}s.")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                    continue
                else:
                    logger.error(f"Erro da API da Binance: {e}")
                    return None
            except Exception as e:
                logger.error(f"Erro inesperado ao colocar ordem: {e}", exc_info=True)
                return None

        logger.error("Número máximo de tentativas atingido. Falha ao colocar ordem.")
        return None

    def save_trade_results(self) -> None:
        """
        Salva os resultados das operações de trading em um arquivo CSV.
        """
        with self.data_lock:
            self.trade_results.to_csv('trade_results.csv', index=True)

    def save_performance_metrics(self) -> None:
        """
        Salva as métricas de desempenho do bot em um arquivo CSV.
        """
        pd.DataFrame(self.performance_metrics).to_csv('perf_metric_data/performance_metrics.csv', index=True)

    def handle_socket_message(self, msg: dict) -> None:
        """
        Processa a mensagem recebida do WebSocket e atualiza o estado do bot.

        :param msg: Dicionário contendo os dados do WebSocket.
        """
        logger.info(f"Mensagem recebida no WebSocket: {msg}")
        try:
            if msg['e'] == 'continuous_kline':
                k = msg['k']
                is_closed = k['x']
                if is_closed:
                    close_price = float(k['c'])
                    volume = float(k['v'])
                    timestamp = pd.to_datetime(k['t'], unit='ms')

                    with self.price_lock, self.volume_lock:
                        self.current_real_price = close_price
                        self.current_volume = volume

                    logger.info(f"Kline fechado - Preço de Fechamento: {close_price}, Volume: {volume}")

                    # Atualiza o DataFrame histórico
                    new_row = {
                        'timestamp': timestamp,
                        'open': float(k['o']),
                        'high': float(k['h']),
                        'low': float(k['l']),
                        'close': float(k['c']),
                        'volume': float(k['v'])
                    }
                    with self.data_lock:
                        self.data_handler.update_historical_df(new_row)
        except Exception as e:
            logger.error(f"Erro ao processar mensagem do WebSocket: {e}", exc_info=True)

    def save_trade(
            self,
            exit_price, profit, position, tp_value, sl_value,
            tp_percent, sl_percent, current_features
    ):
        """
        Salva informações de trade no DataFrame global e em CSV.
        """
        exit_time = datetime.now(timezone.utc)
        trade_info = {
            'interval': self.interval,
            'symbol': self.symbol,
            'entry_time': self.strategy.entry_time,
            'exit_time': exit_time,
            'entry_price': self.strategy.entry_price,
            'exit_price': exit_price,
            'profit': profit,
            'position': position,
            'take_profit_value': tp_value,
            'stop_loss_value': sl_value,
            'features': current_features.to_dict('records')[0],
            'predicted_tp_percent': tp_percent,
            'predicted_sl_percent': sl_percent
        }
        with self.data_lock:
            self.trade_results = pd.concat([self.trade_results, pd.DataFrame([trade_info])], ignore_index=True)
            self.save_trade_results()

    def online_learning_and_trading(self):
        """
        Método principal que faz loop infinito de aprendizado e execução de trades.
        Pode ajustar risco dinamicamente com base em análise de sentimento e volatilidade.
        """
        logger.info(f"Iniciando aprendizado online e trading automático para {self.symbol}")
        training_window = 500

        while True:
            # Captura preço e volume atuais
            with self.price_lock, self.volume_lock:
                real_price = self.current_real_price
                real_volume = self.current_volume

            if real_price is None or real_volume is None:
                logger.warning("Preço ou volume ainda não disponíveis. Aguardando...")
                time.sleep(10)
                continue

            # Copia o DF histórico e verifica se temos dados suficientes
            with self.data_lock:
                df = self.data_handler.historical_df.copy()

            if df.empty or len(df) < training_window + 50:
                logger.error(f"Dados insuficientes: necessário {training_window + 50}, obtido {len(df)}")
                time.sleep(60)
                continue

            # Inicializa modelos se necessário
            if not self.model_manager.model_initialized:
                features = df[['sma_short', 'sma_long', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband']]
                target_tp = df['close'].pct_change().shift(-1) * 100
                target_sl = df['close'].pct_change().shift(-1) * 100

                self.model_manager.train_models(features[:-1], target_tp[:-1], target_sl[:-1])

            # Obter indicadores atuais
            current_features = self.data_handler.get_current_features()
            if current_features.empty:
                logger.warning("Nenhum indicador disponível. Aguardando próximo ciclo.")
                time.sleep(10)
                continue

            current_features.replace([np.inf, -np.inf], np.nan, inplace=True)
            if current_features.isnull().values.any():
                logger.warning("Dados contêm NaN/inf. Aguardando...")
                time.sleep(10)
                continue

            # Faz previsões de TP e SL
            predicted_tp_percent, predicted_sl_percent = self.model_manager.predict_tp_sl(current_features)
            if not predicted_tp_percent or not predicted_sl_percent:
                time.sleep(10)
                continue

            # Calcula preço TP e SL
            take_profit_price = real_price * (1 + predicted_tp_percent / 100)
            stop_loss_price = real_price * (1 - predicted_sl_percent / 100)

            logger.info(f"TP previsto: {predicted_tp_percent:.2f}%, Preço: {take_profit_price:.2f}")
            logger.info(f"SL previsto: {predicted_sl_percent:.2f}%, Preço: {stop_loss_price:.2f}")

            # Exemplo de cálculo de volatilidade atual (se quiser usar ATR ou outra métrica):
            # Usando 'atr' se você tiver calculado no DataHandler
            if 'atr' in df.columns and not df['atr'].dropna().empty:
                current_volatility = df['atr'].iloc[-1]
            else:
                # fallback se não houver atr
                current_volatility = 1.0

            # ===========================================
            # DECISÕES DE ENTRADA
            # ===========================================
            if self.strategy.position is None:
                sma_short = current_features['sma_short'].values[0]
                sma_long = current_features['sma_long'].values[0]

                # Condição de Compra
                if self.strategy.should_buy(sma_short, sma_long):
                    logger.info("Condição de COMPRA atendida.")
                    # Obter saldo
                    account_info = self.binance_service.get_futures_account_info()
                    available_balance = float([
                                                  x for x in account_info['assets'] if x['asset'] == 'USDT'
                                              ][0]['availableBalance'])

                    # Ajuste dinâmico do risco e alavancagem
                    adjusted_risk, adjusted_leverage = dynamic_risk_adjustment(df, current_volatility)
                    self.strategy.leverage = adjusted_leverage

                    # Cálculo de quantidade (exemplo simplificado)
                    # Risco = capital * adjusted_risk, assumindo SL ~ predicted_sl_percent
                    risk_amount = self.strategy.capital * adjusted_risk
                    effective_sl_percent = max(predicted_sl_percent, 0.1)
                    quantity = (risk_amount / (real_price * (effective_sl_percent / 100))) * adjusted_leverage
                    quantity = round(quantity, 3)

                    max_quantity = (available_balance * adjusted_leverage) / real_price
                    max_quantity = round(max_quantity, 3)
                    quantity = min(quantity, max_quantity)

                    if quantity <= 0:
                        logger.error("Quantidade após ajuste <= 0. Aguardando...")
                        time.sleep(60)
                        continue

                    order = self.place_order('BUY', quantity)
                    if order and order['status'] == 'FILLED':
                        self.strategy.position = 'long'
                        self.strategy.entry_price = float(order['avgPrice'])
                        self.strategy.entry_time = datetime.now(timezone.utc)
                        self.strategy.take_profit_target = round(take_profit_price, 2)
                        self.strategy.trailing_stop_loss = round(stop_loss_price, 2)
                        self.strategy.quantity = quantity
                        logger.info(f"Compra executada a {self.strategy.entry_price}, qty={quantity}")

                # Condição de Venda
                elif self.strategy.should_sell(sma_short, sma_long):
                    logger.info("Condição de VENDA atendida.")

                    account_info = self.binance_service.get_futures_account_info()
                    available_balance = float([
                                                  x for x in account_info['assets'] if x['asset'] == 'USDT'
                                              ][0]['availableBalance'])

                    # Ajuste dinâmico de risco e alavancagem
                    adjusted_risk, adjusted_leverage = dynamic_risk_adjustment(df, current_volatility)
                    self.strategy.leverage = adjusted_leverage

                    risk_amount = self.strategy.capital * adjusted_risk
                    effective_sl_percent = max(predicted_sl_percent, 0.1)
                    quantity = (risk_amount / (real_price * (effective_sl_percent / 100))) * adjusted_leverage
                    quantity = round(quantity, 3)

                    max_quantity = (available_balance * adjusted_leverage) / real_price
                    max_quantity = round(max_quantity, 3)
                    quantity = min(quantity, max_quantity)

                    if quantity <= 0:
                        logger.error("Quantidade após ajuste <= 0. Aguardando...")
                        time.sleep(60)
                        continue

                    order = self.place_order('SELL', quantity)
                    if order and order['status'] == 'FILLED':
                        self.strategy.position = 'short'
                        self.strategy.entry_price = float(order['avgPrice'])
                        self.strategy.entry_time = datetime.now(timezone.utc)
                        self.strategy.take_profit_target = round(take_profit_price, 2)
                        self.strategy.trailing_stop_loss = round(stop_loss_price, 2)
                        self.strategy.quantity = quantity
                        logger.info(f"Venda executada a {self.strategy.entry_price}, qty={quantity}")

            # ===========================================
            # GERENCIAMENTO DE POSIÇÃO (se já estiver aberta)
            # ===========================================
            else:
                position = self.strategy.position
                entry_price = self.strategy.entry_price
                quantity = self.strategy.quantity
                take_profit_target = self.strategy.take_profit_target
                trailing_stop_loss = self.strategy.trailing_stop_loss

                if position == 'long':
                    # TP
                    if real_price >= take_profit_target:
                        # Fecha posição com lucro
                        order = self.place_order('SELL', quantity)
                        if order and order['status'] == 'FILLED':
                            exit_price = float(order['avgPrice'])
                            profit = (exit_price - entry_price) * quantity * self.strategy.leverage
                            self.strategy.capital += profit
                            self.strategy.daily_profit += profit

                            logger.info(f"Fechou LONG em TP. Lucro: {profit:.2f}")
                            self.save_trade(
                                exit_price, profit, 'long',
                                take_profit_target, trailing_stop_loss,
                                predicted_tp_percent, predicted_sl_percent,
                                current_features
                            )
                            self.strategy.reset_position()

                    # SL
                    elif real_price <= trailing_stop_loss:
                        # Fecha posição com perda
                        order = self.place_order('SELL', quantity)
                        if order and order['status'] == 'FILLED':
                            exit_price = float(order['avgPrice'])
                            profit = (exit_price - entry_price) * quantity * self.strategy.leverage
                            self.strategy.capital += profit
                            self.strategy.daily_profit += profit

                            logger.info(f"Fechou LONG em SL. Lucro: {profit:.2f}")
                            self.save_trade(
                                exit_price, profit, 'long',
                                take_profit_target, trailing_stop_loss,
                                predicted_tp_percent, predicted_sl_percent,
                                current_features
                            )
                            self.strategy.reset_position()

                    # Ajusta trailing stop se ultrapassou a projeção do TP
                    elif real_price > entry_price * (1 + predicted_tp_percent / 100):
                        new_trailing = real_price * (1 - predicted_sl_percent / 100)
                        if new_trailing > trailing_stop_loss:
                            self.strategy.trailing_stop_loss = round(new_trailing, 2)
                            logger.info(f"Trailing Stop ajustado para {self.strategy.trailing_stop_loss}")

                elif position == 'short':
                    # TP
                    if real_price <= take_profit_target:
                        # Fecha posição com lucro
                        order = self.place_order('BUY', quantity)
                        if order and order['status'] == 'FILLED':
                            exit_price = float(order['avgPrice'])
                            profit = (entry_price - exit_price) * quantity * self.strategy.leverage
                            self.strategy.capital += profit
                            self.strategy.daily_profit += profit

                            logger.info(f"Fechou SHORT em TP. Lucro: {profit:.2f}")
                            self.save_trade(
                                exit_price, profit, 'short',
                                take_profit_target, trailing_stop_loss,
                                predicted_tp_percent, predicted_sl_percent,
                                current_features
                            )
                            self.strategy.reset_position()

                    # SL
                    elif real_price >= trailing_stop_loss:
                        # Fecha posição com perda
                        order = self.place_order('BUY', quantity)
                        if order and order['status'] == 'FILLED':
                            exit_price = float(order['avgPrice'])
                            profit = (entry_price - exit_price) * quantity * self.strategy.leverage
                            self.strategy.capital += profit
                            self.strategy.daily_profit += profit

                            logger.info(f"Fechou SHORT em SL. Lucro: {profit:.2f}")
                            self.save_trade(
                                exit_price, profit, 'short',
                                take_profit_target, trailing_stop_loss,
                                predicted_tp_percent, predicted_sl_percent,
                                current_features
                            )
                            self.strategy.reset_position()

                    # Ajusta trailing stop
                    elif real_price < entry_price * (1 - predicted_tp_percent / 100):
                        new_trailing = real_price * (1 + predicted_sl_percent / 100)
                        if new_trailing < trailing_stop_loss:
                            self.strategy.trailing_stop_loss = round(new_trailing, 2)
                            logger.info(f"Trailing Stop ajustado para {self.strategy.trailing_stop_loss}")

            # Verifica se o limite de perda diária foi atingido
            if self.strategy.check_daily_loss_limit():
                break

            # Salva métricas de performance e atualiza DF de backtest
            with self.data_lock:
                self.backtest_results = self.trade_results.copy()
                if not self.trade_results.empty and 'profit' in self.trade_results.columns:
                    total_profit = self.trade_results['profit'].sum()
                    win_rate = (self.trade_results['profit'] > 0).mean()
                    avg_profit = self.trade_results['profit'].mean()
                else:
                    total_profit = 0
                    win_rate = 0
                    avg_profit = 0
                self.performance_metrics.append({
                    'timestamp': datetime.now(),
                    'cumulative_profit': total_profit,
                    'win_rate': win_rate,
                    'average_profit': avg_profit
                })
                self.save_performance_metrics()

            time.sleep(1)
