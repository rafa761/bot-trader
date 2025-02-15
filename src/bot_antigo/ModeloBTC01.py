import os
import sys
import math
import asyncio
import logging
import logging.handlers
import threading
import time
import platform
import pandas as pd
import numpy as np
import ta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import joblib
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from binance.client import Client
from binance import ThreadedWebsocketManager
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import requests

# -----------------------------------------------------------
# 1. Configurações Iniciais
# -----------------------------------------------------------
load_dotenv()

API_KEY = 'b5361ba39e9ba47bcdc7976ca427714d2dd32544755b9bbffd0e50313cb905ef'
API_SECRET = '80837417a3f7ec3a27be068c62496bee78a11bc9c2c023a848d657a579a67094'

if not API_KEY or not API_SECRET:
    print("Chaves de API não encontradas. Verifique seu .env ou variáveis de ambiente.")
    sys.exit(1)

if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler("logs/trading_app.log", maxBytes=5*1024*1024, backupCount=5),
        logging.StreamHandler(sys.stdout)
    ]
)

client = Client(API_KEY, API_SECRET, testnet=True)

if platform.system().lower().startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

symbol = "BTCUSDT"
interval = "15m"  # você pode alterar para 1m, 15m, etc.
capital = 1000
leverage = 5
risk_per_trade = 0.20  # 1%

data_lock = threading.Lock()
historical_df = pd.DataFrame()

# -----------------------------------------------------------
# 2. Funções Auxiliares
# -----------------------------------------------------------

def get_symbol_filters(client, symbol):
    """Obtém tickSize (preço) e stepSize (quantidade) do par em Futuros."""
    info = client.futures_exchange_info()
    for s in info["symbols"]:
        if s["symbol"] == symbol:
            filters = {f["filterType"]: f for f in s["filters"]}
            tick_size = float(filters["PRICE_FILTER"]["tickSize"])
            step_size = float(filters["LOT_SIZE"]["stepSize"])
            return tick_size, step_size
    return None, None

def adjust_price_to_tick_size(price, tick_size):
    """Arredonda 'price' para baixo (floor) ao múltiplo do tick_size."""
    return math.floor(price / tick_size) * tick_size

def format_price_for_tick_size(price, tick_size):
    """Formata 'price' com a quantidade correta de casas decimais baseada no tick_size."""
    decimals = 0
    if '.' in str(tick_size):
        decimals = len(str(tick_size).split('.')[-1])
    return f"{price:.{decimals}f}"

def adjust_quantity_to_step_size(qty, step_size):
    """Arredonda 'qty' para o múltiplo do step_size."""
    return math.floor(qty / step_size) * step_size

def format_quantity_for_step_size(qty, step_size):
    """Formata a quantidade com a precisão baseada no step_size."""
    decimals = 0
    if '.' in str(step_size):
        decimals = len(str(step_size).split('.')[-1])
    return f"{qty:.{decimals}f}"

def calculate_trade_quantity(capital, current_price, leverage, risk_per_trade):
    """Calcula a quantidade de trade: (capital * risco) / current_price * leverage."""
    risk_amount = capital * risk_per_trade
    quantity = (risk_amount / current_price) * leverage
    return quantity

def add_technical_indicators(df):
    """Adiciona indicadores técnicos usando a biblioteca 'ta'."""
    try:
        df["sma_short"] = ta.trend.sma_indicator(df["close"], window=5)
        df["sma_long"]  = ta.trend.sma_indicator(df["close"], window=10)
        df["rsi"]       = ta.momentum.rsi(df["close"], window=14)
        df["macd"]      = ta.trend.macd_diff(df["close"])
        df["boll_hband"] = ta.volatility.bollinger_hband(df["close"], window=20)
        df["boll_lband"] = ta.volatility.bollinger_lband(df["close"], window=20)

        if len(df) >= 14:
            df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()

        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular indicadores: {e}", exc_info=True)
        return df

def update_historical_data(df, new_row):
    """Atualiza o DataFrame histórico com nova linha e recalcula indicadores."""
    try:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.drop_duplicates(subset="timestamp", keep="last", inplace=True)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df = add_technical_indicators(df)
        return df
    except Exception as e:
        logging.error(f"Erro ao atualizar histórico: {e}", exc_info=True)
        return df

def get_open_position_by_side(symbol, desired_side):
    """
    Retorna a posição aberta para o lado desejado ("LONG" ou "SHORT").
    Em modo hedge, permite distinguir as posições.
    """
    try:
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            if pos["positionSide"].upper() == desired_side.upper() and float(pos["positionAmt"]) != 0.0:
                return pos
        return None
    except Exception as e:
        logging.error(f"Erro ao checar posição para {desired_side}: {e}", exc_info=True)
        return None

def place_order_with_retry(symbol, side, quantity, position_side, max_attempts=3):
    """
    Cria uma ordem MARKET com retentativas em caso de erro (ex.: margem insuficiente).
    Formata a quantidade para a precisão definida pelo stepSize.
    """
    attempt, backoff_time = 0, 2
    order_resp = None
    formatted_qty = format_quantity_for_step_size(quantity, STEP_SIZE)
    while attempt < max_attempts:
        try:
            order_resp = client.futures_create_order(
                symbol=symbol,
                side=side,
                type="MARKET",
                quantity=formatted_qty,
                positionSide=position_side
            )
            return order_resp
        except BinanceAPIException as e:
            logging.error(f"Erro da API ao colocar ordem MARKET: {e}")
            if e.code == -2019:  # margem insuficiente
                quantity *= 0.9
                formatted_qty = format_quantity_for_step_size(quantity, STEP_SIZE)
                time.sleep(backoff_time)
                backoff_time *= 2
            attempt += 1
        except Exception as e:
            logging.error(f"Erro inesperado ao colocar ordem MARKET: {e}", exc_info=True)
            break
    logging.error("Não foi possível colocar a ordem após várias tentativas.")
    return None

def get_futures_last_price(symbol):
    """Obtém o Last Price atual do ticker de Futuros."""
    try:
        ticker = client.futures_symbol_ticker(symbol=symbol)
        return float(ticker["price"])
    except Exception as e:
        logging.error(f"Erro ao obter last price: {e}")
        return 0.0

# -----------------------------------------------------------
# 3. Coleta de Dados Históricos (Futuros)
# -----------------------------------------------------------
async def get_latest_data(symbol="BTCUSDT", interval="15", limit=1000):
    logging.info(f"Coletando {limit} velas de {symbol} (intervalo={interval})")
    attempt, max_attempts = 0, 5
    while attempt < max_attempts:
        try:
            klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            df = pd.DataFrame(klines, columns=[
                "timestamp","open","high","low","close","volume",
                "close_time","quote_asset_volume","number_of_trades",
                "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
            df.dropna(inplace=True)
            df.reset_index(drop=True, inplace=True)
            return df[["timestamp","open","high","low","close","volume"]]
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout ao coletar dados, tentativa {attempt+1}")
            await asyncio.sleep(3)
        except BinanceAPIException as e:
            logging.error(f"Erro API Binance: {e}")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Erro ao coletar dados: {e}", exc_info=True)
            return pd.DataFrame()
        attempt += 1
    logging.error("Tentativas esgotadas. Não foi possível coletar dados.")
    return pd.DataFrame()

# -----------------------------------------------------------
# 4. Função Principal de Trading
# -----------------------------------------------------------
async def online_learning_and_trading(symbol, interval, capital):
    training_interval = 50
    cycle_count = 0
    model_initialized = False

    try:
        pipeline_tp = joblib.load("model_tp.pkl")
        pipeline_sl = joblib.load("model_sl.pkl")
        logging.info("Modelos TP e SL carregados com sucesso.")
        logging.info(f"Pipeline TP: {pipeline_tp} e Pipeline SL: {pipeline_sl}")
    except FileNotFoundError:
        logging.error("Modelos TP e SL não encontrados.")
        sys.exit(1)

    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except BinanceAPIException as e:
        logging.warning(f"Não foi possível setar alavancagem={leverage}: {e}")

    global TICK_SIZE, STEP_SIZE
    TICK_SIZE, STEP_SIZE = get_symbol_filters(client, symbol)
    if not TICK_SIZE or not STEP_SIZE:
        logging.error("Não foi possível obter tickSize/stepSize. Encerrando.")
        sys.exit(1)
    logging.info(f"{symbol} -> tickSize={TICK_SIZE}, stepSize={STEP_SIZE}")

    global historical_df
    
    while True:
        cycle_count += 1

        new_data = await get_latest_data(symbol, interval, limit=2)
        if not new_data.empty:
            with data_lock:
                if historical_df.empty:
                    historical_df = await get_latest_data(symbol, interval, limit=1000)
                    historical_df = add_technical_indicators(historical_df)
                else:
                    for i in range(len(new_data)):
                        row = new_data.iloc[i]
                        new_row = {
                            "timestamp": row["timestamp"],
                            "open": row["open"],
                            "high": row["high"],
                            "low": row["low"],
                            "close": row["close"],
                            "volume": row["volume"]
                        }
                        historical_df = update_historical_data(historical_df, new_row)
                        
        if (not model_initialized) or (cycle_count % training_interval == 0):
            with data_lock:
                df_train = historical_df.copy()
            if len(df_train) >= 300:
                df_train["pct_change_next"] = df_train["close"].pct_change().shift(-1) * 100
                features = df_train[["sma_short","sma_long","rsi","macd","boll_hband","boll_lband","atr"]].copy()
                features.dropna(inplace=True)
                target_tp = df_train["pct_change_next"].copy()
                target_sl = df_train["pct_change_next"].copy()

                min_len = min(len(features), len(target_tp))
                features = features.iloc[:min_len]
                target_tp = target_tp.iloc[:min_len]
                target_sl = target_sl.iloc[:min_len]

                features = features.iloc[:-1]
                target_tp = target_tp.iloc[:-1]
                target_sl = target_sl.iloc[:-1]

                pipeline_tp.fit(features, target_tp)
                pipeline_sl.fit(features, target_sl)
                model_initialized = True
                logging.info("Modelos TP/SL re-treinados.")
                logging.info(f"target_tp: {target_tp}, target_sl:{target_sl}, features:{features}")
            else:
                logging.info("Ainda não há dados suficientes para treinar.")

        open_long = get_open_position_by_side(symbol, "LONG")
        open_short = get_open_position_by_side(symbol, "SHORT")

        if open_long is not None or open_short is not None:
            logging.info("Já existe posição aberta. Aguardando fechamento para abrir novo trade.")
        else:
            if model_initialized and len(historical_df) > 0:
                current_price = get_futures_last_price(symbol)
                if current_price <= 0:
                    logging.warning("Falha ao obter last price. Nenhum trade.")
                    await asyncio.sleep(5)
                    continue

                with data_lock:
                    df_eval = historical_df.copy()
                last_row = df_eval.iloc[-1]
                X_eval = pd.DataFrame([
                    [
                        last_row["sma_short"],
                        last_row["sma_long"],
                        last_row["rsi"],
                        last_row["macd"],
                        last_row["boll_hband"],
                        last_row["boll_lband"],
                        last_row["atr"]
                    ]
                ], columns=["sma_short", "sma_long", "rsi", "macd", "boll_hband", "boll_lband", "atr"])

                predicted_tp_pct = pipeline_tp.predict(X_eval)[0]
                predicted_sl_pct = pipeline_sl.predict(X_eval)[0]
                logging.info(f"Predicted TP: {predicted_tp_pct:.2f}%, Predicted SL: {predicted_sl_pct:.2f}%")  

                direction = None
                if predicted_tp_pct > 0.2:
                    direction = "LONG"
                    logging.info("Sinal de compra (LONG) detectado.")
                elif predicted_tp_pct < -0.2:
                    direction = "SHORT"
                    logging.info("Sinal de venda (SHORT) detectado.")
                else:
                    logging.info("Sinal neutro, não abrir trade.")
                    direction = None

                if direction:
                    if direction == "LONG":
                        tp_price = current_price * (1 + max(abs(predicted_tp_pct)/100, 0.02))
                        sl_price = current_price * (1 - max(abs(predicted_sl_pct)/100, 0.02))
                        side = "BUY"
                        position_side = "LONG"
                    else:  # SHORT
                        tp_price = current_price * (1 - max(abs(predicted_tp_pct)/100, 0.02))
                        sl_price = current_price * (1 + max(abs(predicted_sl_pct)/100, 0.02))
                        side = "SELL"
                        position_side = "SHORT"

                    qty = calculate_trade_quantity(capital, current_price, leverage, risk_per_trade)
                    qty_adj = adjust_quantity_to_step_size(qty, STEP_SIZE)
                    if qty_adj <= 0:
                        logging.warning("Qty ajustada <= 0. Trade abortado.")
                        await asyncio.sleep(5)
                        continue

                    formatted_qty = format_quantity_for_step_size(qty_adj, STEP_SIZE)
                    logging.info(f"Abrindo {direction} c/ qty={formatted_qty}, lastPrice={current_price:.2f}...")

                    order_resp = place_order_with_retry(symbol, side, qty_adj, position_side)
                    if order_resp:
                        logging.info(f"Ordem de abertura executada: {order_resp}")

                        if direction == "LONG":
                            if tp_price <= current_price:
                                tp_price = tp_price
                            tp_price_adj = adjust_price_to_tick_size(tp_price, TICK_SIZE)
                            if tp_price_adj <= current_price:
                                tp_price_adj = current_price + (TICK_SIZE * 10)
                            tp_str = format_price_for_tick_size(tp_price_adj, TICK_SIZE)
                            logging.info(f"[DEBUG] Final TP Price = {tp_str} (current={current_price})")

                            if sl_price >= current_price:
                                sl_price = sl_price
                            sl_price_adj = adjust_price_to_tick_size(sl_price, TICK_SIZE)
                            if sl_price_adj >= current_price:
                                sl_price_adj = current_price - (TICK_SIZE * 10)
                            sl_str = format_price_for_tick_size(sl_price_adj, TICK_SIZE)
                            logging.info(f"[DEBUG] Final SL Price = {sl_str} (current={current_price})")

                            try:
                                tp_order = client.futures_create_order(
                                    symbol=symbol,
                                    side="SELL",
                                    type="TAKE_PROFIT_MARKET",
                                    stopPrice=tp_str,
                                    closePosition=True,
                                    positionSide="LONG"
                                )
                                logging.info(f"Ordem TAKE_PROFIT criada: {tp_order}")
                            except Exception as e:
                                logging.error(f"Erro ao criar TAKE_PROFIT: {e}", exc_info=True)

                            try:
                                sl_order = client.futures_create_order(
                                    symbol=symbol,
                                    side="SELL",
                                    type="STOP_MARKET",
                                    stopPrice=sl_str,
                                    closePosition=True,
                                    positionSide="LONG"
                                )
                                logging.info(f"Ordem STOP (SL) criada: {sl_order}")
                            except Exception as e:
                                logging.error(f"Erro ao criar STOP (SL): {e}", exc_info=True)

                        else:  # Para SHORT
                            if tp_price >= current_price:
                                tp_price = tp_price
                            tp_price_adj = adjust_price_to_tick_size(tp_price, TICK_SIZE)
                            if tp_price_adj >= current_price:
                                tp_price_adj = current_price - (TICK_SIZE * 10)
                            tp_str = format_price_for_tick_size(tp_price_adj, TICK_SIZE)
                            logging.info(f"[DEBUG] Final TP Price (SHORT) = {tp_str} (current={current_price})")

                            if sl_price <= current_price:
                                sl_price = sl_price
                            sl_price_adj = adjust_price_to_tick_size(sl_price, TICK_SIZE)
                            if sl_price_adj <= current_price:
                                sl_price_adj = current_price + (TICK_SIZE * 10)
                            sl_str = format_price_for_tick_size(sl_price_adj, TICK_SIZE)
                            logging.info(f"[DEBUG] Final SL Price (SHORT) = {sl_str} (current={current_price})")

                            try:
                                tp_order = client.futures_create_order(
                                    symbol=symbol,
                                    side="BUY",
                                    type="TAKE_PROFIT_MARKET",
                                    stopPrice=tp_str,
                                    closePosition=True,
                                    positionSide="SHORT"
                                )
                                logging.info(f"Ordem TAKE_PROFIT criada: {tp_order}")
                            except Exception as e:
                                logging.error(f"Erro ao criar TAKE_PROFIT: {e}", exc_info=True)

                            try:
                                sl_order = client.futures_create_order(
                                    symbol=symbol,
                                    side="BUY",
                                    type="STOP_MARKET",
                                    stopPrice=sl_str,
                                    closePosition=True,
                                    positionSide="SHORT"
                                )
                                logging.info(f"Ordem STOP (SL) criada: {sl_order}")
                            except Exception as e:
                                logging.error(f"Erro ao criar STOP (SL): {e}", exc_info=True)
                    else:
                        logging.info("Modelos não inicializados ou histórico insuficiente. Nenhum trade.")
                else:
                    if open_long:
                        open_pos = open_long
                    elif open_short:
                        open_pos = open_short
                    else:
                        open_pos = None

                    if open_pos:
                        amt = float(open_pos["positionAmt"])
                        entry_price = float(open_pos["entryPrice"])
                        side_pos = open_pos["positionSide"]
                        logging.info(f"Posição em aberto: {amt} @ {entry_price} ({side_pos}). Aguardando TP/SL...")
                    else:
                        logging.info("Nenhuma posição aberta encontrada.")

                await asyncio.sleep(5)

            await asyncio.sleep(5)

# -------------------------------------------------------------
# 5. Dash para Visualização (Opcional)
# -------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.H3("Bot com Gap Forçado (Futuros Testnet)"),
    dcc.Interval(id="interval-component", interval=60000, n_intervals=0),
    dcc.Graph(id="price-chart")
])

@app.callback(Output("price-chart", "figure"),
              [Input("interval-component", "n_intervals")])
def update_graph(n):
    global historical_df
    with data_lock:
        df_plot = historical_df.copy()
    if df_plot.empty:
        return go.Figure()
    fig = go.Figure(data=[go.Candlestick(
        x=df_plot["timestamp"],
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name=symbol
    )])
    fig.update_layout(title=f"Histórico {symbol} (Testnet)", xaxis_rangeslider_visible=False)
    return fig

# -------------------------------------------------------------
# 6. Execução Principal
# -------------------------------------------------------------
if __name__ == "__main__":
    twm = ThreadedWebsocketManager(api_key=API_KEY, api_secret=API_SECRET, testnet=True)
    twm.start()

    dash_thread = threading.Thread(
        target=app.run_server,
        kwargs={"debug": False, "use_reloader": False},
        daemon=True
    )
    dash_thread.start()

    try:
        asyncio.run(online_learning_and_trading(symbol, interval, capital))
    except KeyboardInterrupt:
        logging.info("Bot interrompido manualmente.")
    finally:
        twm.stop()
