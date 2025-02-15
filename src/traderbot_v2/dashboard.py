# dashboard.py

"""
Este módulo constrói a aplicação Dash para visualização:
  1) de velas (candlestick) a partir do DataFrame de histórico (data_handler.historical_df)
  2) de um gráfico em tempo real dos logs provenientes do memory_logger
"""

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from config import config
from data_handler import DataHandler
from logger import memory_logger


def create_dashboard(data_handler: DataHandler) -> dash.Dash:
    """
    Cria e retorna uma instância da aplicação Dash.

    :param data_handler: Objeto DataHandler, de onde os dados serão lidos.
                        Ele deve conter:
                         - historical_df (com colunas: timestamp, open, high, low, close)
                         - data_lock (para acesso thread-safe)
    :return: Aplicação Dash configurada
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # =========================================================================
    # Layout
    # =========================================================================

    app.layout = dbc.Container(fluid=True, children=[
        dbc.Row([
            dbc.Col(html.H3(f"Bot de Trading {config.SYMBOL} (Futuros Testnet)"), width=12),
        ]),

        # 1) Gráfico de Candlestick
        dbc.Row([
            dbc.Col([
                html.H4("Gráfico de Velas (Candlestick)"),
                # Intervalo de atualização
                dcc.Interval(id="interval-candles", interval=60_000, n_intervals=0),
                dcc.Graph(id="candlestick-graph")
            ], width=12),
        ]),

        # 2) Gráfico de Logs em Tempo Real
        dbc.Row([
            dbc.Col([
                html.H4("Logs em Tempo Real"),
                dcc.Interval(id='log-interval', interval=5_000, n_intervals=0),
                html.Div(
                    id='log-output',
                    style={
                        'whiteSpace': 'pre-wrap',
                        'height': '300px',
                        'overflowY': 'scroll',
                        'backgroundColor': '#f8f9fa',
                        'border': '1px solid #ced4da',
                        'padding': '10px'
                    }
                )
            ], width=12),
        ]),
    ])

    # =========================================================================
    #  Callbacks
    # =========================================================================

    # -------------------------------------------------------------------------
    # A) Atualiza o Gráfico de Candlestick
    # -------------------------------------------------------------------------
    @app.callback(
        Output("candlestick-graph", "figure"),
        Input("interval-candles", "n_intervals")
    )
    def update_candlestick_chart(n: int) -> go.Figure:
        """
        Callback para atualizar o gráfico de velas do BTC (ou outro símbolo)
        com base no histórico em data_handler.historical_df.
        """
        with data_handler.data_lock:
            df_plot = data_handler.historical_df.copy()

        if df_plot.empty:
            return go.Figure()

        fig = go.Figure(data=[go.Candlestick(
            x=df_plot["timestamp"],
            open=df_plot["open"],
            high=df_plot["high"],
            low=df_plot["low"],
            close=df_plot["close"],
            name=config.SYMBOL
        )])
        fig.update_layout(
            title=f"Histórico de Preços - {config.SYMBOL}",
            xaxis_rangeslider_visible=False,
            xaxis_title="Data/Hora",
            yaxis_title="Preço"
        )
        return fig

    # -------------------------------------------------------------------------
    # B) Atualiza o Gráfico de Logs em Tempo Real
    # -------------------------------------------------------------------------
    @app.callback(
        Output('log-output', 'children'),
        Input('log-interval', 'n_intervals')
    )
    def update_log_output(n: int) -> go.Figure:
        """
        Constrói um gráfico (scatter) no qual cada linha do log é plotada
        com base em seu timestamp (eixo X) e nível de log (eixo Y).
        """
        # 1) Obtém todas as linhas do logger em memória
        lines = memory_logger.get_logs()

        # 2) Mapeia níveis de log para estilo (emoji e cor)
        log_styles = {
            'DEBUG': {'emoji': '🔎', 'color': '#6c757d'},  # cinza
            'INFO': {'emoji': 'ℹ️', 'color': '#000000'},  # preto
            'WARNING': {'emoji': '⚠️', 'color': 'orange'},
            'ERROR': {'emoji': '🛑', 'color': 'red'},
            'CRITICAL': {'emoji': '💥', 'color': 'red'},
        }

        def parse_log_level(line: str) -> str:
            """Identifica o nível de log na string formatada (simples substring)."""
            if ' - ERROR - ' in line:
                return 'ERROR'
            elif ' - WARNING - ' in line:
                return 'WARNING'
            elif ' - INFO - ' in line:
                return 'INFO'
            elif ' - DEBUG - ' in line:
                return 'DEBUG'
            elif ' - CRITICAL - ' in line:
                return 'CRITICAL'
            else:
                return 'INFO'

        # 3) Para cada linha, identificamos nível e geramos um componente html.Div
        styled_logs = []
        for line in lines:
            level = parse_log_level(line)
            style_info = log_styles.get(level, log_styles['INFO'])
            styled_line = f"{style_info['emoji']} {line}"

            styled_logs.append(
                html.Div(
                    styled_line,
                    style={
                        'color': style_info['color'],
                        'fontFamily': 'monospace',
                        'fontSize': '12px',
                        'marginBottom': '2px'
                    }
                )
            )

        # Retorna uma lista de Divs, cada um representando uma linha de log
        return styled_logs

    return app
