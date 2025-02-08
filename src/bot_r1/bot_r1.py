import os
import certifi
import asyncio
import sys
import logging
import logging.handlers
from binance.client import Client
from binance import ThreadedWebsocketManager
from threading import Lock, Thread
from binance.exceptions import BinanceAPIException
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
import pandas as pd
import ta
import numpy as np
import joblib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import requests
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import sys
import asyncio

# Configuração específica para Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Configurações iniciais
os.environ['SSL_CERT_FILE'] = certifi.where()
load_dotenv()
nltk.download('vader_lexicon')

# Configuração de logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler('logs/trading_app.log', maxBytes=5*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)

# Chaves de API
API_KEY = 'b5361ba39e9ba47bcdc7976ca427714d2dd32544755b9bbffd0e50313cb905ef'
API_SECRET = '80837417a3f7ec3a27be068c62496bee78a11bc9c2c023a848d657a579a67094'
NEWS_API_KEY = '9ac6e0ed93dc493c9c0bfe862cd3804e'

if not API_KEY or not API_SECRET:
    logging.error("Chaves de API não encontradas. Verifique o arquivo .env.")
    sys.exit(1)

# Inicialização do cliente Binance
client = Client(API_KEY, API_SECRET, testnet=True, requests_params={"timeout": 10})
data_client = Client(API_KEY, API_SECRET, testnet=True, requests_params={"timeout": 10})

# Variáveis globais e locks
current_real_price = None
current_volume = None
price_lock = Lock()
volume_lock = Lock()
data_lock = Lock()
historical_df = pd.DataFrame()
trade_results = pd.DataFrame()
performance_metrics = []

# Parâmetros do modelo R1
r1_hyperparams = {
    'volatility_window': 14,
    'dynamic_leverage': True,
    'max_leverage': 25,
    'sentiment_analysis': True,
    'trend_confirmation': {
        'adx_threshold': 25,
        'volume_spike_multiplier': 2.5
    }
}

def get_latest_data(symbol, interval, limit):
    """Obtém dados históricos da Binance Futures com paginação"""
    max_limit = 1000  # Limite máximo por requisição
    df = pd.DataFrame()
    
    try:
        # Calcula quantas requisições são necessárias
        num_requests = (limit + max_limit - 1) // max_limit
        
        for i in range(num_requests):
            # Calcula o intervalo de tempo para cada requisição
            end_time = int(time.time() * 1000) - (i * max_limit * 60 * 1000)
            start_time = end_time - (max_limit * 60 * 1000)
            
            klines = data_client.futures_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=max_limit
            )
            
            temp_df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            df = pd.concat([df, temp_df], ignore_index=True)
            
        # Processamento dos dados
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df.tail(limit)  # Garante o limite exato
    except Exception as e:
        logging.error(f"Erro ao obter dados históricos: {e}")
        return pd.DataFrame()

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def get_market_sentiment():
    if r1_hyperparams['sentiment_analysis'] and NEWS_API_KEY:
        try:
            url = f"https://newsapi.org/v2/everything?q=bitcoin&apiKey={NEWS_API_KEY}"
            response = requests.get(url).json()
            articles = [article['title'] for article in response.get('articles', [])]
            sia = SentimentIntensityAnalyzer()
            sentiment_scores = [sia.polarity_scores(article)['compound'] for article in articles]
            return np.mean(sentiment_scores) if sentiment_scores else 0.5
        except Exception as e:
            logging.error(f"Erro ao obter sentimento: {e}")
            return 0.5
    return 0.5

def dynamic_risk_adjustment(df, current_volatility):
    base_risk = 0.02
    sentiment = get_market_sentiment()
    risk_multiplier = np.clip(
        (current_volatility / df['atr'].mean()) * (sentiment + 0.5),
        0.5, 2.0
    )
    adjusted_risk = base_risk * risk_multiplier
    adjusted_leverage = min(
        r1_hyperparams['max_leverage'],
        int(25 * (1 / risk_multiplier))
    )
    return adjusted_risk, adjusted_leverage

def add_technical_indicators(df):
    # Garantir tamanho mínimo do DataFrame
    min_data_points = max(20, r1_hyperparams['volatility_window'] + 10)
    
    if len(df) < min_data_points:
        raise ValueError(f"DataFrame precisa ter pelo menos {min_data_points} registros para calcular indicadores")
    
    # Calcular indicadores na ordem correta
    df['sma_short'] = ta.trend.sma_indicator(df['close'], window=5)
    df['sma_long'] = ta.trend.sma_indicator(df['close'], window=10)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    df['macd'] = ta.trend.macd_diff(df['close'])
    df['bollinger_hband'] = ta.volatility.bollinger_hband(df['close'], window=20)
    df['bollinger_lband'] = ta.volatility.bollinger_lband(df['close'], window=20)
    df['atr'] = calculate_atr(df, r1_hyperparams['volatility_window'])
    
    # ADX requer cálculo especial
    df['adx'] = ta.trend.ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    ).adx()
    
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df = df.dropna().copy()  # Evitar SettingWithCopyWarning
    return df

def place_order(symbol, side, quantity, leverage):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        return order
    except BinanceAPIException as e:
        logging.error(f"Erro na ordem: {e}")
        return None

def intelligent_position_management(position, entry_price, current_price, atr):
    take_profit = 3 * atr
    stop_loss = 1.5 * atr
    
    if position == 'long':
        return entry_price + take_profit, current_price - stop_loss
    else:
        return entry_price - take_profit, current_price + stop_loss

def handle_socket_message(msg):
    global current_real_price, current_volume
    try:
        if msg['e'] == 'kline':
            kline = msg['k']
            with price_lock:
                current_real_price = float(kline['c'])
            with volume_lock:
                current_volume = float(kline['v'])
    except Exception as e:
        logging.error(f"Erro no WebSocket: {e}")

def online_learning_and_trading(symbol, interval, capital):
    global historical_df, trade_results, performance_metrics
    position = None
    entry_price = 0
    quantity = 0
    entry_time = None

    while True:
        try:
            with price_lock, volume_lock:
                price = current_real_price
                volume = current_volume

            if historical_df.empty:
                time.sleep(10)
                continue

            current_features = historical_df.iloc[-1]
            current_volatility = current_features['atr'] / current_features['close']
            risk, leverage = dynamic_risk_adjustment(historical_df, current_volatility)

            adx = current_features['adx']
            volume_spike = current_features['volume'] > (current_features['volume_ma'] * r1_hyperparams['trend_confirmation']['volume_spike_multiplier'])
            strong_trend = adx > r1_hyperparams['trend_confirmation']['adx_threshold']

            if position is None:
                buy_cond = (current_features['sma_short'] > current_features['sma_long'] and 
                           (strong_trend or volume_spike) and 
                           get_market_sentiment() > 0.6)
                
                sell_cond = (current_features['sma_short'] < current_features['sma_long'] and 
                            (strong_trend or volume_spike) and 
                            get_market_sentiment() < 0.4)

                if buy_cond or sell_cond:
                    side = 'BUY' if buy_cond else 'SELL'
                    quantity = (capital * risk) / (price * current_volatility) * leverage
                    quantity = round(quantity, 3)
                    
                    order = place_order(symbol, side, quantity, leverage)
                    if order:
                        position = 'long' if side == 'BUY' else 'short'
                        entry_price = float(order['avgPrice'])
                        entry_time = datetime.now(timezone.utc)
                        logging.info(f"Ordem {side} executada: {quantity}@{entry_price}")

            else:
                current_profit = (price - entry_price) * quantity * (-1 if position == 'short' else 1)
                time_in_trade = (datetime.now(timezone.utc) - entry_time).total_seconds() / 3600

                take_profit, stop_loss = intelligent_position_management(
                    position, entry_price, price, current_features['atr']
                )

                exit_cond = (
                    (position == 'long' and (price >= take_profit or price <= stop_loss)) or
                    (position == 'short' and (price <= take_profit or price >= stop_loss)) or
                    time_in_trade > 24
                )

                if exit_cond:
                    side = 'SELL' if position == 'long' else 'BUY'
                    order = place_order(symbol, side, quantity, leverage)
                    if order:
                        exit_price = float(order['avgPrice'])
                        profit = (exit_price - entry_price) * quantity * (-1 if position == 'short' else 1)
                        trade_results = pd.concat([trade_results, pd.DataFrame([{
                            'entry_time': entry_time,
                            'exit_time': datetime.now(timezone.utc),
                            'side': position,
                            'quantity': quantity,
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'profit': profit
                        }])], ignore_index=True)
                        logging.info(f"Posição fechada. Lucro: {profit:.2f} USDT")
                        position = None

            time.sleep(5)
        except Exception as e:
            logging.error(f"Erro no loop principal: {e}")
            time.sleep(10)

def create_dash_app(df):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
    
    app.layout = dbc.Container([
        dbc.Row(dbc.Col(html.H1("Dashboard de Trading", className="text-center my-4"))),
        
        dbc.Row([
            dbc.Col([
                html.Div(id='live-updates', className="card p-3 mb-3"),
                dcc.Interval(id='update-interval', interval=10*1000)
            ], width=6),
            
            dbc.Col([
                dcc.Graph(id='price-chart', figure={
                    'data': [go.Scatter(x=df.index, y=df['close'], name='Preço')],
                    'layout': go.Layout(title='Preço em Tempo Real')
                })
            ], width=6)
        ])
    ], fluid=True)

    @app.callback(
        Output('live-updates', 'children'),
        Input('update-interval', 'n_intervals')
    )
    def update_metrics(n):
        metrics = [
            html.H4("Métricas Chave"),
            html.P(f"Preço Atual: {current_real_price or 'N/A'}"),
            html.P(f"Último Lucro: {trade_results['profit'].iloc[-1] if not trade_results.empty else 'N/A'}"),
            html.P(f"Taxa de Acerto: {(trade_results['profit'] > 0).mean()*100 if not trade_results.empty else 0:.1f}%")
        ]
        return metrics

    return app

if __name__ == '__main__':
    symbol = 'BTCUSDT'
    interval = '1h'
    
    # Obter dados históricos
    historical_df = get_latest_data(symbol=symbol, interval=interval, limit=2000)
    if historical_df.empty:
        logging.error("Falha ao obter dados históricos. Verifique a conexão ou parâmetros.")
        sys.exit(1)
    historical_df = add_technical_indicators(historical_df)

    # Iniciar WebSocket
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET)
    twm.start()
    twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR)

    # Iniciar thread de trading
    trading_thread = Thread(target=online_learning_and_trading, args=(symbol, interval, 1000), daemon=True)
    trading_thread.start()

    # Iniciar dashboard
    app = create_dash_app(historical_df)
    app.run_server(debug=True, port=8050)
