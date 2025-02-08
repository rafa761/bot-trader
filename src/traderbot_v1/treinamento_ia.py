from pathlib import Path
import os
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from datetime import datetime
import time
import ta

# ---------------------------- Configuração Inicial ----------------------------

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

API_KEY ='vEXmhiuKEoKPMJBVSYnejP0PYKQfkzGCMB2yM9qIn4W3kmlTc1MSfGQT0V1VHseK'
API_SECRET = 'h7mzLHYCsVv0qshySmGaQoKFBZaKTEob56QnheGmJI5sQSxAwojUywpQVvFo5BGf'

# Verificar se as chaves de API foram definidas
if not API_KEY or not API_SECRET:
    raise ValueError("Chaves de API não encontradas. Defina BINANCE_API_KEY e BINANCE_API_SECRET no arquivo .env.")

# Configurar o cliente da Binance
client = Client(API_KEY, API_SECRET, requests_params={"timeout": 20})

# Configurar o logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_bot.log"),
        logging.StreamHandler()
    ]
)

# ---------------------------- Funções Auxiliares ----------------------------

def get_historical_klines(symbol, interval, start_str, end_str=None):
    """
    Obtém dados históricos de candles da Binance.
    """
    try:
        logging.info(f"Coletando dados históricos para {symbol} com intervalo {interval} desde {start_str}")
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
        logging.info(f"Coleta de dados concluída: {len(df)} registros coletados.")
        return df
    except BinanceAPIException as e:
        logging.error(f"Erro ao coletar dados históricos: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Erro inesperado ao coletar dados históricos: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """
    Adiciona indicadores técnicos ao DataFrame.
    """
    try:
        logging.info("Adicionando indicadores técnicos ao DataFrame.")
        # Média Móvel Simples (curta e longa)
        df['sma_short'] = ta.trend.SMAIndicator(close=df['close'], window=10).sma_indicator()
        df['sma_long'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        # Índice de Força Relativa (RSI)
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # Average True Range (ATR)
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
        
        # Remove linhas com valores NaN
        df.dropna(inplace=True)
        logging.info("Indicadores técnicos adicionados com sucesso.")
        return df
    except Exception as e:
        logging.error(f"Erro ao adicionar indicadores técnicos: {e}")
        return df

def create_labels(df, horizon=12):
    """
    Cria labels para TP e SL com base no movimento de preço futuro.
    
    :param df: DataFrame com dados históricos e indicadores técnicos.
    :param horizon: Número de períodos futuros para considerar ao definir TP e SL.
    :return: DataFrame com novas colunas 'TP_pct' e 'SL_pct'.
    """
    try:
        logging.info(f"Criando labels para TP e SL com horizon={horizon} períodos.")
        df['future_high'] = df['high'].rolling(window=horizon).max().shift(-horizon)
        df['future_low'] = df['low'].rolling(window=horizon).min().shift(-horizon)
        
        # Definir TP como o percentual de aumento até o máximo futuro
        df['TP_pct'] = ((df['future_high'] - df['close']) / df['close']) * 100
        
        # Definir SL como o percentual de queda até o mínimo futuro
        df['SL_pct'] = ((df['close'] - df['future_low']) / df['close']) * 100
        
        # Remover colunas auxiliares
        df.drop(['future_high', 'future_low'], axis=1, inplace=True)
        
        # Remover linhas com NaN
        df.dropna(inplace=True)
        logging.info("Labels para TP e SL criados com sucesso.")
        return df
    except Exception as e:
        logging.error(f"Erro ao criar labels: {e}")
        return df

def preprocess_data(df, feature_columns):
    """
    Normaliza as features e as labels.
    
    :param df: DataFrame com dados e labels.
    :param feature_columns: Lista de colunas que serão usadas como features.
    :return: Scalers e dados normalizados.
    """
    try:
        logging.info("Iniciando pré-processamento dos dados.")
        X = df[feature_columns]
        y_tp = df['TP_pct']
        y_sl = df['SL_pct']
        
        # Inicializar os scalers
        scaler_X = StandardScaler()
        scaler_y_tp = StandardScaler()
        scaler_y_sl = StandardScaler()
        
        # Ajustar e transformar as features
        X_scaled = scaler_X.fit_transform(X)
        
        # Ajustar e transformar os labels
        y_tp_scaled = scaler_y_tp.fit_transform(y_tp.values.reshape(-1, 1)).flatten()
        y_sl_scaled = scaler_y_sl.fit_transform(y_sl.values.reshape(-1, 1)).flatten()
        
        # Concatenar as features e labels em um novo DataFrame
        data_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=df.index)
        data_scaled['TP_pct'] = y_tp_scaled
        data_scaled['SL_pct'] = y_sl_scaled
        
        logging.info("Pré-processamento concluído com sucesso.")
        return scaler_X, scaler_y_tp, scaler_y_sl, data_scaled
    except Exception as e:
        logging.error(f"Erro no pré-processamento dos dados: {e}")
        return None, None, None, df

def split_data(data_scaled, feature_columns, test_size=0.2):
    """
    Divide os dados em conjuntos de treino e teste.
    
    :param data_scaled: DataFrame com dados normalizados.
    :param feature_columns: Lista de colunas que serão usadas como features.
    :param test_size: Proporção dos dados que será usada para teste.
    :return: Conjuntos de treino e teste para TP e SL.
    """
    try:
        logging.info("Dividindo os dados em conjuntos de treino e teste.")
        X = data_scaled[feature_columns]
        y_tp = data_scaled['TP_pct']
        y_sl = data_scaled['SL_pct']
        
        # Dividir os dados em treino e teste
        X_train, X_test, y_tp_train, y_tp_test = train_test_split(X, y_tp, test_size=test_size, shuffle=False)
        _, _, y_sl_train, y_sl_test = train_test_split(X, y_sl, test_size=test_size, shuffle=False)
        
        logging.info("Divisão dos dados concluída.")
        return X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test
    except Exception as e:
        logging.error(f"Erro ao dividir os dados: {e}")
        return None, None, None, None, None, None

def train_model(X_train, y_train, model_name):
    """
    Treina um modelo de regressão e salva o modelo treinado.
    
    :param X_train: Dados de treino.
    :param y_train: Labels de treino.
    :param model_name: Nome para salvar o modelo.
    :return: Modelo treinado.
    """
    try:
        logging.info(f"Iniciando treinamento do modelo para {model_name}.")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, f'model_{model_name}.pkl')
        logging.info(f"Modelo para {model_name} treinado e salvo como model_{model_name}.pkl.")
        return model
    except Exception as e:
        logging.error(f"Erro ao treinar o modelo para {model_name}: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name):
    """
    Avalia o modelo usando o conjunto de teste.
    
    :param model: Modelo treinado.
    :param X_test: Dados de teste.
    :param y_test: Labels de teste.
    :param model_name: Nome do modelo para exibir nos logs.
    :return: Mean Absolute Error.
    """
    try:
        logging.info(f"Avaliação do modelo para {model_name}.")
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"Mean Absolute Error para {model_name}: {mae:.2f}%")
        return mae
    except Exception as e:
        logging.error(f"Erro ao avaliar o modelo para {model_name}: {e}")
        return None

# ---------------------------- Fluxo Principal ----------------------------

def main():
    # Parâmetros
    symbol = 'BTCUSDT'
    interval = Client.KLINE_INTERVAL_15MINUTE
    start_date = '1 Jan, 2020'  # Data de início
    horizon = 12  # Número de períodos futuros para definir TP e SL (15m * 12 = 3 horas)
    feature_columns = ['sma_short', 'sma_long', 'rsi', 'atr', 'volume']
    
    # 1. Coletar dados históricos
    df = get_historical_klines(symbol, interval, start_date)
    if df.empty:
        logging.error("Não foi possível coletar dados históricos. Encerrando o script.")
        return
    
    # 2. Adicionar indicadores técnicos
    df = add_technical_indicators(df)
    
    # 3. Criar labels para TP e SL
    df = create_labels(df, horizon)
    
    # 4. Pré-processar os dados
    scaler_X, scaler_y_tp, scaler_y_sl, data_scaled = preprocess_data(df, feature_columns)
    if data_scaled is None:
        logging.error("Pré-processamento falhou. Encerrando o script.")
        return
    
    # 5. Dividir os dados em treino e teste
    X_train, X_test, y_tp_train, y_tp_test, y_sl_train, y_sl_test = split_data(data_scaled, feature_columns)
    if X_train is None:
        logging.error("Divisão dos dados falhou. Encerrando o script.")
        return
    
    # 6. Treinar modelos
    model_tp = train_model(X_train, y_tp_train, 'tp')
    model_sl = train_model(X_train, y_sl_train, 'sl')
    
    if model_tp is None or model_sl is None:
        logging.error("Treinamento dos modelos falhou. Encerrando o script.")
        return
    
    # 7. Avaliar modelos
    mae_tp = evaluate_model(model_tp, X_test, y_tp_test, 'tp')
    mae_sl = evaluate_model(model_sl, X_test, y_sl_test, 'sl')
    
    # 8. Salvar scalers
    try:
        # Define o diretório onde os arquivos serão salvos
        train_data_dir = Path('train_data')

        # Cria o diretório se ele não existir
        train_data_dir.mkdir(parents=True, exist_ok=True)

        # Salva os scalers usando joblib
        joblib.dump(scaler_X, train_data_dir / 'scaler_X.pkl')
        joblib.dump(scaler_y_tp, train_data_dir / 'scaler_y_tp.pkl')
        joblib.dump(scaler_y_sl, train_data_dir / 'scaler_y_sl.pkl')

        logging.info("Scalers salvos como scaler_X.pkl, scaler_y_tp.pkl e scaler_y_sl.pkl.")
    except Exception as e:
        logging.error(f"Erro ao salvar scalers: {e}")
    
    logging.info("Processo de treinamento concluído com sucesso.")

if __name__ == "__main__":
    main()
