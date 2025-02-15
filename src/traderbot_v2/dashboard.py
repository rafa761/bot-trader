# dashboard.py

"""
Este módulo constrói a aplicação Dash para visualização das velas,
além de registrar callbacks necessários.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import dash_bootstrap_components as dbc

from data_handler import DataHandler
from config import config


def create_dashboard(data_handler: DataHandler) -> dash.Dash:
    """
    Cria e retorna uma instância da aplicação Dash.

    :param data_handler: Objeto DataHandler, de onde os dados serão lidos
    :return: Aplicação Dash configurada
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = html.Div([
        html.H3("Bot de Trading BTC (Futuros Testnet)"),
        dcc.Interval(id="interval-component", interval=60000, n_intervals=0),
        dcc.Graph(id="price-chart")
    ])

    @app.callback(Output("price-chart", "figure"),
                  [Input("interval-component", "n_intervals")])
    def update_graph(n: int) -> go.Figure:
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
            title=f"Histórico {config.SYMBOL} (Testnet)",
            xaxis_rangeslider_visible=False
        )
        return fig

    return app
