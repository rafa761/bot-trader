# trading_bot.py

import os
import certifi

# Definir a variável de ambiente antes de importar outros módulos
os.environ['SSL_CERT_FILE'] = certifi.where()

import asyncio
import sys
import logging
import logging.handlers
from binance.client import Client
from binance import ThreadedWebsocketManager
from threading import Lock
import threading
from binance.exceptions import BinanceAPIException
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
import pandas as pd
import ta  # Biblioteca para indicadores técnicos
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor  # Utilizando XGBoost
import joblib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import math
import requests
import numpy as np
import schedule
import ssl

# ---------------------------
# 1. Configuração Inicial
# ---------------------------

# Definir a política do event loop para Windows
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar o logging com RotatingFileHandler para evitar crescimento indefinido
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('logs/trading_app.log', maxBytes=5*1024*1024, backupCount=5),
        logging.StreamHandler()  # Também loga no console
    ]
)

# Carregar as chaves de API do arquivo .env
API_KEY ='b5361ba39e9ba47bcdc7976ca427714d2dd32544755b9bbffd0e50313cb905ef'
API_SECRET = '80837417a3f7ec3a27be068c62496bee78a11bc9c2c023a848d657a579a67094'

# Verificar se as chaves de API foram definidas
if not API_KEY or not API_SECRET:
    logging.error("Chaves de API não encontradas. Por favor, defina as variáveis de ambiente 'BINANCE_API_KEY_TESTNET' e 'BINANCE_API_SECRET_TESTNET' no arquivo .env.")
    sys.exit(1)

# Inicializar o cliente Binance para trading na Testnet
client = Client(API_KEY, API_SECRET, testnet=True, requests_params={"timeout": 10})

# Cliente para dados de futuros na Testnet
data_client = Client(API_KEY, API_SECRET, testnet=True, requests_params={"timeout": 10})

# ---------------------------
# 2. Variáveis Globais e Locks
# ---------------------------

current_real_price = None
current_volume = None
price_lock = Lock()
volume_lock = Lock()
data_lock = Lock()

# DataFrames globais para resultados
backtest_results = pd.DataFrame()
trade_results = pd.DataFrame()
performance_metrics = []

# ---------------------------
# 3. Parâmetros do Modelo e Trading
# ---------------------------

capital = 1000  # Capital inicial em dólares
risk_per_trade = 0.02  # 1% do capital por trade
leverage = 25  # Alavancagem
transaction_cost = 0.0004  # Custo de transação
slippage = 0.0001  # Slippage estimado
sma_window_short = 5
sma_window_long = 10

daily_loss_limit = -0.02 * capital  # Limite de perda diária (por exemplo, 2% do capital)

# ---------------------------
# 4. Função para Adicionar Indicadores Técnicos
# ---------------------------

def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos ao DataFrame fornecido.
    """
    logging.info("Calculando indicadores técnicos usando 'ta'")
    try:
        df['sma_short'] = ta.trend.sma_indicator(df['close'], window=sma_window_short)
        df['sma_long'] = ta.trend.sma_indicator(df['close'], window=sma_window_long)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['bollinger_hband'] = ta.volatility.bollinger_hband(df['close'], window=20)
        df['bollinger_lband'] = ta.volatility.bollinger_lband(df['close'], window=20)
        df.dropna(inplace=True)
        logging.info("Indicadores técnicos calculados com sucesso")
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular indicadores técnicos: {e}", exc_info=True)
        return df

# ---------------------------
# 5. Função para Processar Mensagens do WebSocket
# ---------------------------

def handle_socket_message(msg):
    logging.info(f"Mensagem recebida no WebSocket: {msg}")
    global current_real_price, current_volume, historical_df
    try:
        if msg['e'] == 'continuous_kline':
            k = msg['k']
            is_closed = k['x']
            if is_closed:
                close_price = float(k['c'])
                volume = float(k['v'])
                timestamp = pd.to_datetime(k['t'], unit='ms')  # Obter timestamp do kline
                with price_lock, volume_lock:
                    current_real_price = close_price
                    current_volume = volume
                logging.info(f"Kline fechado - Preço de Fechamento: {close_price}, Volume: {volume}")
                logging.info(f"Preço real atualizado via WebSocket: {current_real_price}, Volume: {current_volume}")
                # Atualiza o DataFrame histórico
                timestamp = pd.to_datetime(k['t'], unit='ms')  # Definir timestamp
                new_row = {
                    'timestamp': timestamp,  # Incluir timestamp
                    'open': float(k['o']),
                    'high': float(k['h']),
                    'low': float(k['l']),
                    'close': float(k['c']),
                    'volume': float(k['v'])
                }
                with data_lock:
                    # Substituição de append por pd.concat
                    historical_df = pd.concat([historical_df, pd.DataFrame([new_row])], ignore_index=True)
                    # Remover possíveis duplicatas
                    historical_df.drop_duplicates(subset='timestamp', keep='last', inplace=True)
                    # Calcula os indicadores técnicos
                    historical_df = add_technical_indicators(historical_df)
                    logging.info("DataFrame histórico atualizado com indicadores técnicos.")
    except Exception as e:
        logging.error(f"Erro ao processar mensagem do WebSocket: {e}", exc_info=True)

# ---------------------------
# 6. Funções para Gerenciar Ordens e Resultados
# ---------------------------

def place_order(symbol, side, quantity, leverage):
    """
    Coloca uma ordem de mercado e ajusta a quantidade em caso de margem insuficiente.
    """
    max_attempts = 2  # Número máximo de tentativas
    attempt = 0
    backoff_time = 2  # Tempo de espera inicial em segundos

    while attempt < max_attempts:
        try:
            # Obter informações sobre a posição existente
            account_info = client.futures_account()
            available_balance = 0
            for asset in account_info['assets']:
                if asset['asset'] == 'USDT':
                    available_balance = float(asset['availableBalance'])
                    break

            # Obter o preço de marca para calcular a margem necessária
            mark_price = float(client.futures_mark_price(symbol=symbol)['markPrice'])
            estimated_margin = (quantity / leverage) * mark_price

            # Verificar se a margem é suficiente
            if available_balance < estimated_margin:
                logging.warning("Margem insuficiente. Ajustando quantidade para caber na margem disponível.")
                # Ajustar a quantidade para uma porcentagem segura do saldo disponível
                quantity = (available_balance * leverage) / mark_price * 0.9  # 90% do valor possível
                quantity = round(quantity, 3)
                if quantity <= 0:
                    logging.error("Saldo insuficiente para qualquer ordem. Ordem cancelada.")
                    return None

            # Colocar a ordem de mercado
            order = client.futures_create_order(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=quantity,
                positionSide='LONG' if side == 'BUY' else 'SHORT'
            )

            # Se a ordem foi colocada com sucesso, sair da função
            return order

        except BinanceAPIException as e:
            if e.code == -2019:  # Código de erro para margem insuficiente
                logging.error(f"Erro da API da Binance ao colocar ordem: {e}")
                attempt += 1
                logging.warning(f"Tentativa de ajuste de margem insuficiente. Tentando novamente após {backoff_time} segundos...")
                time.sleep(backoff_time)
                backoff_time *= 2  # Aumentar o tempo de espera para o próximo backoff
                continue
            else:
                logging.error(f"Erro da API da Binance: {e}")
                return None

        except Exception as e:
            logging.error(f"Erro inesperado ao colocar ordem: {e}", exc_info=True)
            return None

    # Se atingir o número máximo de tentativas, registrar o erro e parar as tentativas
    logging.error("Número máximo de tentativas atingido. Não foi possível colocar a ordem devido à margem insuficiente.")
    return None

def save_trade_results():
    """
    Salva os resultados das trades em um arquivo CSV.
    """
    with data_lock:
        trade_results.to_csv('trade_results.csv', index=True)

def save_performance_metrics():
    """
    Salva as métricas de desempenho em um arquivo CSV.
    """
    pd.DataFrame(performance_metrics).to_csv('performance_metrics.csv', index=True)

# ---------------------------
# 7. Função de Aprendizado Online e Trading Automático
# ---------------------------

def online_learning_and_trading(symbol, interval, capital):
    logging.info("Função online_learning_and_trading iniciada")
    global backtest_results, trade_results, performance_metrics
    logging.info(f"Iniciando aprendizado online e trading automático para {symbol}")
    position = None
    entry_price = 0
    take_profit_target = None
    trailing_stop_loss = None
    quantity = 0
    entry_time = None
    daily_profit = 0
    model_initialized = False
    training_window = 500  # Janela de treinamento para aprendizado online

    # Carregar os modelos pré-treinados
    try:
        pipeline_tp = joblib.load('model_tp.pkl')
        pipeline_sl = joblib.load('model_sl.pkl')
        logging.info("Pipelines dos modelos de Take-Profit e Stop-Loss carregados com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao carregar os modelos: {e}")
        sys.exit(1)

    while True:
        trade_executed = True

        with price_lock, volume_lock:
            real_price = current_real_price
            real_volume = current_volume

        if real_price is None or real_volume is None:
            logging.warning("Preço ou volume real ainda não está disponível via WebSocket.")
            time.sleep(10)
            continue

        # Inicializar o modelo se ainda não foi inicializado
        if not model_initialized:
            with data_lock:
                df = historical_df.copy()
            if df.empty or len(df) < training_window + 50:
                logging.error(f"Dados insuficientes: necessário {training_window + 50}, obtido: {len(historical_df)}")
                time.sleep(60)
                continue
            # Selecionar características e targets
            features = df[['sma_short', 'sma_long', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband']]
            target_tp = df['close'].pct_change().shift(-1) * 100  # Retornos futuros para TP
            target_sl = df['close'].pct_change().shift(-1) * 100  # Retornos futuros para SL
            # Treinar os modelos de Take-Profit e Stop-Loss
            pipeline_tp.fit(features[:-1], target_tp[:-1])
            pipeline_sl.fit(features[:-1], target_sl[:-1])
            model_initialized = True
            logging.info("Modelos treinados com sucesso.")

        # Obter os dados mais recentes para fazer a previsão
        with data_lock:
            if historical_df.empty:
                logging.warning("historical_df está vazio. Aguardando próximo ciclo.")
                time.sleep(1)
                continue
            current_features = historical_df.tail(1)[['sma_short', 'sma_long', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband']]
            timestamp = datetime.now(timezone.utc)
            current_indicators = {
                'sma_short': current_features['sma_short'].values[0],
                'sma_long': current_features['sma_long'].values[0],
                'rsi': current_features['rsi'].values[0],
                'macd': current_features['macd'].values[0],
                'bollinger_hband': current_features['bollinger_hband'].values[0],
                'bollinger_lband': current_features['bollinger_lband'].values[0]
            }

        if current_features.empty:
            logging.warning("current_features está vazio. Aguardando próximo ciclo.")
            time.sleep(1)
            continue

        X_current = current_features.copy()

        # Tratar valores infinitos ou NaN
        X_current.replace([np.inf, -np.inf], np.nan, inplace=True)
        if X_current.isnull().values.any():
            logging.warning("Dados de entrada contêm valores NaN ou infinitos. Aguardando próximo ciclo.")
            time.sleep(1)
            continue

        # Fazer previsões de Take-Profit e Stop-Loss
        try:
            predicted_tp_percent = pipeline_tp.predict(X_current)[0]
            predicted_sl_percent = abs(pipeline_sl.predict(X_current)[0])
        except Exception as e:
            logging.error(f"Erro ao fazer previsões: {e}", exc_info=True)
            time.sleep(1)
            continue

        # Verificar se as previsões são válidas
        if predicted_tp_percent <= 0 or predicted_sl_percent <= 0:
            logging.warning("Previsões inválidas, aguardando próximo ciclo")
            time.sleep(1)
            continue

        # Calcular os preços de Take-Profit e Stop-Loss
        take_profit_price = real_price * (1 + predicted_tp_percent / 100)
        stop_loss_price = real_price * (1 - predicted_sl_percent / 100)

        logging.info(f"Take-Profit previsto: {predicted_tp_percent:.2f}%, Valor de Take-Profit: {take_profit_price:.2f}")
        logging.info(f"Stop-Loss previsto: {predicted_sl_percent:.2f}%, Valor de Stop-Loss: {stop_loss_price:.2f}")

        # Condições para entrar em uma posição
        if position is None:
            # Condição de compra: SMA curta acima da SMA longa
            if current_features['sma_short'].values[0] > current_features['sma_long'].values[0]:
                logging.info("Condição de compra atendida")
                # Calcular a quantidade a ser comprada
                risk_amount = capital * risk_per_trade
                effective_sl_percent = max(predicted_sl_percent, 0.1)  # Evitar divisão por zero
                quantity = (risk_amount / (real_price * (effective_sl_percent / 100))) * leverage
                quantity = round(quantity, 3)  # Ajustar para o incremento mínimo

                # Obter o saldo disponível
                try:
                    account_info = client.futures_account()
                    available_balance = float(account_info['availableBalance'])
                    logging.info(f"Saldo disponível: {available_balance} USDT")
                except Exception as e:
                    logging.error(f"Erro ao obter saldo disponível: {e}", exc_info=True)
                    time.sleep(1)
                    continue

                # Calcular a quantidade máxima com base no saldo disponível
                max_quantity = (available_balance * leverage) / real_price
                max_quantity = round(max_quantity, 3)

                # Ajustar a quantidade se necessário
                quantity = min(quantity, max_quantity)
                quantity = round(quantity, 3)

                if quantity <= 0:
                    logging.error("Quantidade calculada é zero ou negativa após ajuste. Aguardando próximo ciclo.")
                    time.sleep(60)
                    continue

                # Colocar a ordem de compra
                order = place_order(symbol, 'BUY', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    position = "long"
                    entry_price = float(order['avgPrice'])
                    entry_time = timestamp
                    take_profit_target = round(take_profit_price, 1)
                    trailing_stop_loss = round(stop_loss_price, 1)
                    logging.info(f"Compra executada: {quantity} {symbol} a {entry_price}")
                    logging.info(f"take_profit_target: {take_profit_target}, trailing_stop_loss: {trailing_stop_loss}")
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de compra")
                    time.sleep(1)
                    continue

            # Condição de venda: SMA curta abaixo da SMA longa
            elif current_features['sma_short'].values[0] < current_features['sma_long'].values[0]:
                logging.info("Condição de venda atendida")
                # Calcular a quantidade a ser vendida
                risk_amount = capital * risk_per_trade
                effective_sl_percent = max(predicted_sl_percent, 0.1)
                quantity = (risk_amount / (real_price * (effective_sl_percent / 100))) * leverage
                quantity = round(quantity, 3)

                # Obter o saldo disponível
                try:
                    account_info = client.futures_account()
                    available_balance = float(account_info['availableBalance'])
                    logging.info(f"Saldo disponível: {available_balance} USDT")
                except Exception as e:
                    logging.error(f"Erro ao obter saldo disponível: {e}", exc_info=True)
                    time.sleep(1)
                    continue

                # Calcular a quantidade máxima com base no saldo disponível
                max_quantity = (available_balance * leverage) / real_price
                max_quantity = round(max_quantity, 3)

                # Ajustar a quantidade se necessário
                quantity = min(quantity, max_quantity)
                quantity = round(quantity, 3)

                if quantity <= 0:
                    logging.error("Quantidade calculada é zero ou negativa após ajuste. Aguardando próximo ciclo.")
                    time.sleep(60)
                    continue

                # Colocar a ordem de venda
                order = place_order(symbol, 'SELL', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    position = "short"
                    entry_price = float(order['avgPrice'])
                    entry_time = timestamp
                    take_profit_target = round(take_profit_price, 1)
                    trailing_stop_loss = round(stop_loss_price, 1)
                    logging.info(f"Venda executada: {quantity} {symbol} a {entry_price}")
                    logging.info(f"take_profit_target: {take_profit_target}, trailing_stop_loss: {trailing_stop_loss}")
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de venda")
                    time.sleep(1)
                    continue

        # Gerenciamento da posição aberta
        if position == "long":
            logging.info("Verificando condições para posição long")

            # Verificação se o preço atingiu o Take-Profit
            if real_price >= take_profit_target:
                # Fecha toda a posição com lucro
                order = place_order(symbol, 'SELL', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    exit_price = float(order['avgPrice'])
                    exit_time = datetime.now(timezone.utc)
                    profit = (exit_price - entry_price) * quantity * leverage
                    capital += profit
                    daily_profit += profit
                    logging.info(f"Venda executada no Take-Profit: {quantity} {symbol} a {exit_price}, Lucro: {profit}")
                    # Salvar os resultados
                    results = {
                        'interval': interval,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'position': 'long',
                        'take_profit_value': take_profit_target,
                        'stop_loss_value': trailing_stop_loss,
                        'features': current_features.to_dict('records')[0],
                        'predicted_tp_percent': predicted_tp_percent,
                        'predicted_sl_percent': predicted_sl_percent,
                        'indicators': current_indicators,
                        'parameters': {
                            'risk_per_trade': risk_per_trade,
                            'leverage': leverage,
                            'transaction_cost': transaction_cost,
                            'slippage': slippage,
                            'sma_window_short': sma_window_short,
                            'sma_window_long': sma_window_long
                        }
                    }
                    with data_lock:
                        trade_results = pd.concat([trade_results, pd.DataFrame([results])], ignore_index=True)
                        # Salvar os resultados das trades
                        save_trade_results()
                    position = None
                    take_profit_target = None
                    trailing_stop_loss = None
                    quantity = 0
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de venda no Take-Profit")

            # Verificação se o preço atingiu o Stop-Loss
            elif real_price <= trailing_stop_loss:
                # Fecha toda a posição com perda limitada
                order = place_order(symbol, 'SELL', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    exit_price = float(order['avgPrice'])
                    exit_time = datetime.now(timezone.utc)
                    profit = (exit_price - entry_price) * quantity * leverage
                    capital += profit
                    daily_profit += profit
                    logging.info(f"Venda executada no Stop-Loss: {quantity} {symbol} a {exit_price}, Lucro: {profit}")
                    # Salvar os resultados
                    results = {
                        'interval': interval,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'position': 'long',
                        'take_profit_value': take_profit_target,
                        'stop_loss_value': trailing_stop_loss,
                        'features': current_features.to_dict('records')[0],
                        'predicted_tp_percent': predicted_tp_percent,
                        'predicted_sl_percent': predicted_sl_percent,
                        'indicators': current_indicators,
                        'parameters': {
                            'risk_per_trade': risk_per_trade,
                            'leverage': leverage,
                            'transaction_cost': transaction_cost,
                            'slippage': slippage,
                            'sma_window_short': sma_window_short,
                            'sma_window_long': sma_window_long
                        }
                    }
                    with data_lock:
                        trade_results = pd.concat([trade_results, pd.DataFrame([results])], ignore_index=True)
                        # Salvar os resultados das trades
                        save_trade_results()
                    position = None
                    take_profit_target = None
                    trailing_stop_loss = None
                    quantity = 0
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de venda no Stop-Loss")

            # Ajustar o trailing stop-loss se o preço se mover a favor
            elif real_price > entry_price * (1 + predicted_tp_percent / 100):
                new_trailing_stop_loss = real_price * (1 - predicted_sl_percent / 100)
                if new_trailing_stop_loss > trailing_stop_loss:
                    trailing_stop_loss = round(new_trailing_stop_loss, 1)
                    logging.info(f"Trailing Stop-Loss ajustado para: {trailing_stop_loss}")

        elif position == "short":
            logging.info("Verificando condições para posição short")

            # Verificação se o preço atingiu o Take-Profit
            if real_price <= take_profit_target:
                # Fecha toda a posição com lucro
                order = place_order(symbol, 'BUY', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    exit_price = float(order['avgPrice'])
                    exit_time = datetime.now(timezone.utc)
                    profit = (entry_price - exit_price) * quantity * leverage
                    capital += profit
                    daily_profit += profit
                    logging.info(f"Compra executada no Take-Profit: {quantity} {symbol} a {exit_price}, Lucro: {profit}")
                    # Salvar os resultados
                    results = {
                        'interval': interval,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'position': 'short',
                        'take_profit_value': take_profit_target,
                        'stop_loss_value': trailing_stop_loss,
                        'features': current_features.to_dict('records')[0],
                        'predicted_tp_percent': predicted_tp_percent,
                        'predicted_sl_percent': predicted_sl_percent,
                        'indicators': current_indicators,
                        'parameters': {
                            'risk_per_trade': risk_per_trade,
                            'leverage': leverage,
                            'transaction_cost': transaction_cost,
                            'slippage': slippage,
                            'sma_window_short': sma_window_short,
                            'sma_window_long': sma_window_long
                        }
                    }
                    with data_lock:
                        trade_results = pd.concat([trade_results, pd.DataFrame([results])], ignore_index=True)
                        # Salvar os resultados das trades
                        save_trade_results()
                    position = None
                    take_profit_target = None
                    trailing_stop_loss = None
                    quantity = 0
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de compra no Take-Profit")

            # Verificação se o preço atingiu o Stop-Loss
            elif real_price >= trailing_stop_loss:
                # Fecha toda a posição com perda limitada
                order = place_order(symbol, 'BUY', quantity, leverage)
                if order and order['status'] == 'FILLED':
                    exit_price = float(order['avgPrice'])
                    exit_time = datetime.now(timezone.utc)
                    profit = (entry_price - exit_price) * quantity * leverage
                    capital += profit
                    daily_profit += profit
                    logging.info(f"Compra executada no Stop-Loss: {quantity} {symbol} a {exit_price}, Lucro: {profit}")
                    # Salvar os resultados
                    results = {
                        'interval': interval,
                        'symbol': symbol,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'profit': profit,
                        'position': 'short',
                        'take_profit_value': take_profit_target,
                        'stop_loss_value': trailing_stop_loss,
                        'features': current_features.to_dict('records')[0],
                        'predicted_tp_percent': predicted_tp_percent,
                        'predicted_sl_percent': predicted_sl_percent,
                        'indicators': current_indicators,
                        'parameters': {
                            'risk_per_trade': risk_per_trade,
                            'leverage': leverage,
                            'transaction_cost': transaction_cost,
                            'slippage': slippage,
                            'sma_window_short': sma_window_short,
                            'sma_window_long': sma_window_long
                        }
                    }
                    with data_lock:
                        trade_results = pd.concat([trade_results, pd.DataFrame([results])], ignore_index=True)
                        # Salvar os resultados das trades
                        save_trade_results()
                    position = None
                    take_profit_target = None
                    trailing_stop_loss = None
                    quantity = 0
                    trade_executed = True
                else:
                    logging.error("Falha ao executar ordem de compra no Stop-Loss")

            # Ajustar o trailing stop-loss se o preço se mover a favor
            elif real_price < entry_price * (1 - predicted_tp_percent / 100):
                new_trailing_stop_loss = real_price * (1 + predicted_sl_percent / 100)
                if new_trailing_stop_loss < trailing_stop_loss:
                    trailing_stop_loss = round(new_trailing_stop_loss, 1)
                    logging.info(f"Trailing Stop-Loss ajustado para: {trailing_stop_loss}")

        # Verificar se o limite de perda diária foi atingido
        if daily_profit <= daily_loss_limit:
            logging.warning("Limite de perda diária atingido. Parando as operações por hoje.")
            break

        # Atualizar os resultados para o dashboard
        with data_lock:
            backtest_results = trade_results.copy()
            logging.info(f"backtest_results atualizado com {len(backtest_results)} registros")
            logging.info(f"trade_results atualizado com {len(trade_results)} registros")

        # Calcular métricas de desempenho
        with data_lock:
            if not trade_results.empty and 'profit' in trade_results.columns:
                total_profit = trade_results['profit'].sum()
                win_rate = (trade_results['profit'] > 0).mean()
                average_profit = trade_results['profit'].mean()
            else:
                total_profit = 0
                win_rate = 0
                average_profit = 0
            performance_metrics.append({
                'timestamp': datetime.now(),
                'cumulative_profit': total_profit,
                'win_rate': win_rate,
                'average_profit': average_profit
            })
            # Salvar métricas de performance
            save_performance_metrics()

        # Espera antes da próxima iteração
        if trade_executed:
            logging.info("Trade executado, aguardando período curto antes do próximo ciclo")
            time.sleep(1)
        else:
            logging.info("Nenhum trade executado, aguardando próximo ciclo completo")
            time.sleep(1)

# ---------------------------
# 8. Função para Coletar Dados Históricos
# ---------------------------

def get_latest_data(symbol='BTCUSDT', interval='1m', limit=5000):
    """
    Coleta dados históricos mais recentes da Binance Futures Testnet.
    """
    logging.info(f"Coletando {limit} dados mais recentes para {symbol} com intervalo {interval}")
    try:
        max_limit_per_call = 1500  # Limite máximo da API por chamada para futuros
        data = []
        endTime = None
        while limit > 0:
            current_limit = min(limit, max_limit_per_call)
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': current_limit
            }
            if endTime:
                params['endTime'] = endTime
            klines = data_client.futures_klines(**params)
            logging.info(f"Linhas retornadas nesta chamada: {len(klines)}")
            if not klines:
                break  # Sem mais dados disponíveis
            data.extend(klines)
            limit -= current_limit
            # Atualizar o 'endTime' para coletar dados anteriores
            endTime = klines[0][0] - 1  # Timestamp do primeiro kline menos 1 ms
        logging.info(f"Número total de linhas coletadas: {len(data)}")
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.dropna(inplace=True)
        # Ordenar os dados em ordem cronológica crescente
        df.sort_values('timestamp', inplace=True)
        logging.info(f"Dados mais recentes de {symbol} coletados com sucesso")
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    except requests.exceptions.Timeout:
        logging.warning(f"Timeout ao coletar dados, tentativa {attempt + 1} de {max_attempts}")
        attempt += 1
        time.sleep(60)  # Aguardar antes de tentar novamente
    except BinanceAPIException as e:
        logging.error(f"Erro da API da Binance: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Erro ao coletar dados: {e}", exc_info=True)
        return pd.DataFrame()

        logging.error("Número máximo de tentativas atingido. Não foi possível coletar dados suficientes.")
        return pd.DataFrame()

# ---------------------------
# 9. Inicialização do WebSocket e Threads
# ---------------------------

# Definir o símbolo e o intervalo desejado
symbol = 'BTCUSDT'
interval = '1m'

# Inicializar o DataFrame histórico com dados suficientes
historical_df = get_latest_data(symbol=symbol, interval=interval, limit=5000)

if len(historical_df) < 1000:
    logging.error(f"Dados históricos insuficientes para inicialização. Necessário: 550, obtido: {len(historical_df)}")
    sys.exit(1)
    time.sleep(60) 
else:
    historical_df = add_technical_indicators(historical_df)
    logging.info(f"Dados históricos coletados com sucesso. Número de linhas: {len(historical_df)}")

# ---------------------------
# 10. Configuração da Interface com Dash
# ---------------------------

# Layout da Interface com Dash Bootstrap Components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Calcular métricas de performance para exibir no dashboard
def calculate_performance_metrics():
    total_profit = trade_results['profit'].sum()
    win_rate = (trade_results['profit'] > 0).mean()
    average_profit = trade_results['profit'].mean()
    return total_profit, win_rate, average_profit

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Dashboard de Trading com IA', className='text-center text-primary mb-4'), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.H3('Métricas de Performance'),
            html.Div(id='performance-metrics'),
        ], width=12),
    ]),

    dbc.Row([
        dbc.Col([
            html.H3('Análise por Período Gráfico'),
            dcc.Interval(id='interval-component', interval=10000, n_intervals=0),
            dcc.Graph(id='prediction-graph'),
            html.Div(id='trade-table')
        ], width=12),
    ]),

    dbc.Row([
        dbc.Col([
            html.H3('Resultados do Backtest'),
            dcc.Graph(id='backtest-results-graph')
        ], width=12),
    ]),

    dbc.Row([
        dbc.Col([
            html.H3('Histórico de Trades'),
            dcc.Graph(id='trades-history-graph')
        ], width=12),
    ]),
], fluid=True)

# ---------------------------
# 11. Callbacks do Dash
# ---------------------------

# Callback para atualizar as métricas de performance
@app.callback(
    Output('performance-metrics', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_metrics(n):
    with data_lock:
        if trade_results.empty:
            return html.Div('Nenhum trade realizado ainda.')
        total_profit, win_rate, average_profit = calculate_performance_metrics()
        return html.Div([
            html.P(f'Total de Lucro: ${total_profit:.2f}'),
            html.P(f'Taxa de Sucesso: {win_rate:.2%}'),
            html.P(f'Lucro Médio por Trade: ${average_profit:.2f}'),
        ])

# Callback para atualizar gráfico de previsão
@app.callback(
    Output('prediction-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_prediction_graph(n):
    with data_lock:
        figure = go.Figure()
        if not trade_results.empty:
            for interval_period in trade_results['interval'].unique():
                period_data = trade_results[trade_results['interval'] == interval_period]
                figure.add_trace(go.Scatter(
                    x=period_data['entry_time'],
                    y=period_data['entry_price'],
                    mode='markers',
                    marker=dict(color='green'),
                    name=f'Entradas - {interval_period}'
                ))
                figure.add_trace(go.Scatter(
                    x=period_data['exit_time'],
                    y=period_data['exit_price'],
                    mode='markers',
                    marker=dict(color='red'),
                    name=f'Saídas - {interval_period}'
                ))
        figure.update_layout(title='Entradas e Saídas por Período Gráfico', xaxis_title='Data', yaxis_title='Preço')
        return figure

# Callback para atualizar a tabela de trades
@app.callback(
    Output('trade-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_trade_table(n):
    with data_lock:
        if trade_results.empty:
            return html.Div('Nenhum trade realizado ainda.')

        # Exibir apenas colunas relevantes na tabela
        columns_to_display = ['interval', 'symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'profit', 'position']
        return dbc.Table.from_dataframe(trade_results[columns_to_display], striped=True, bordered=True, hover=True)

# Callback para gerar gráfico do backtest
@app.callback(
    Output('backtest-results-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_backtest_results_graph(n):
    with data_lock:
        if backtest_results.empty:
            return go.Figure()

        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=backtest_results['exit_time'],
            y=backtest_results['profit'].cumsum(),
            mode='lines',
            name='Lucro Acumulado'
        ))
        figure.update_layout(title='Resultados do Backtest', xaxis_title='Data', yaxis_title='Lucro Acumulado')
        return figure

# Callback para gerar gráfico do histórico de trades
@app.callback(
    Output('trades-history-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_trades_history(n):
    with data_lock:
        if trade_results.empty:
            return go.Figure()

        figure = go.Figure()

        figure.add_trace(go.Scatter(
            x=trade_results['entry_time'],
            y=trade_results['entry_price'],
            mode='markers',
            marker=dict(color='green'),
            name='Entradas'
        ))

        figure.add_trace(go.Scatter(
            x=trade_results['exit_time'],
            y=trade_results['exit_price'],
            mode='markers',
            marker=dict(color='red'),
            name='Saídas'
        ))

        figure.update_layout(title='Histórico de Trades', xaxis_title='Data', yaxis_title='Preço')

        return figure

# ---------------------------
# 12. Função Principal para Executar o Bot
# ---------------------------

if __name__ == '__main__':
    # Inicializar o WebSocket Manager com testnet=True
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
    twm.start()

    # Iniciar o stream de kline contínuo para futuros perpétuos
    twm.start_kline_futures_socket(
        callback=handle_socket_message,
        symbol=symbol.lower(),
        interval=Client.KLINE_INTERVAL_1MINUTE
    )

    logging.info(f"WebSocket iniciado para {symbol} com intervalo {interval}.")

    # Executa o aprendizado online e trading automático em uma thread separada
    trading_thread = threading.Thread(target=online_learning_and_trading, args=(symbol, interval, capital), daemon=True)
    trading_thread.start()

    # Executa o servidor Dash sem debug e sem reloader
    app.run_server(debug=False, use_reloader=False)
