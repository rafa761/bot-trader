# dashboard.py

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from logger import memory_logger


def create_dashboard(trading_bot):
    """
    Cria e retorna um objeto Dash configurado.
    `trading_bot` é uma instância da classe TradingBot,
    de onde serão puxados os dados de trade_results, backtest_results etc.
    """

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    data_lock = trading_bot.data_lock

    def calculate_performance_metrics():
        total_profit = trading_bot.trade_results['profit'].sum()
        win_rate = (trading_bot.trade_results['profit'] > 0).mean()
        average_profit = trading_bot.trade_results['profit'].mean()
        return total_profit, win_rate, average_profit

    # ========= Layout =========
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H1(
                    'Dashboard de Trading com IA',
                    className='text-center text-primary mb-4'
                ), width=12
            )
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

        # -----------------------------
        # SEÇÃO: Logs do Console
        # -----------------------------
        dbc.Row([
            dbc.Col([
                html.H3('Logs do Console'),
                dcc.Interval(id='log-interval', interval=5000, n_intervals=0),

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
    ], fluid=True)

    # ========= Callbacks =========

    @app.callback(
        Output('performance-metrics', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_performance_metrics(n):
        with data_lock:
            if trading_bot.trade_results.empty:
                return "Nenhum trade realizado ainda."
            total_profit, win_rate, avg_profit = calculate_performance_metrics()
            return html.Div([
                html.P(f'Total de Lucro: ${total_profit:.2f}'),
                html.P(f'Taxa de Sucesso: {win_rate:.2%}'),
                html.P(f'Lucro Médio por Trade: ${avg_profit:.2f}')
            ])

    @app.callback(
        Output('prediction-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_prediction_graph(n):
        with data_lock:
            fig = go.Figure()
            df_trades = trading_bot.trade_results
            if not df_trades.empty:
                for interval_period in df_trades['interval'].unique():
                    period_data = df_trades[df_trades['interval'] == interval_period]
                    fig.add_trace(go.Scatter(
                        x=period_data['entry_time'],
                        y=period_data['entry_price'],
                        mode='markers',
                        marker=dict(color='green'),
                        name=f'Entradas - {interval_period}'
                    ))
                    fig.add_trace(go.Scatter(
                        x=period_data['exit_time'],
                        y=period_data['exit_price'],
                        mode='markers',
                        marker=dict(color='red'),
                        name=f'Saídas - {interval_period}'
                    ))
            fig.update_layout(
                title='Entradas e Saídas por Período Gráfico',
                xaxis_title='Data',
                yaxis_title='Preço'
            )
            return fig

    @app.callback(
        Output('trade-table', 'children'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_trade_table(n):
        with data_lock:
            df_trades = trading_bot.trade_results
            if df_trades.empty:
                return "Nenhum trade realizado ainda."

            columns_to_display = [
                'interval', 'symbol', 'entry_time',
                'exit_time', 'entry_price', 'exit_price',
                'profit', 'position'
            ]
            return dbc.Table.from_dataframe(
                df_trades[columns_to_display],
                striped=True,
                bordered=True,
                hover=True
            )

    @app.callback(
        Output('backtest-results-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_backtest_results_graph(n):
        with data_lock:
            if trading_bot.backtest_results.empty:
                return go.Figure()

            fig = go.Figure()
            df_backtest = trading_bot.backtest_results
            fig.add_trace(go.Scatter(
                x=df_backtest['exit_time'],
                y=df_backtest['profit'].cumsum(),
                mode='lines',
                name='Lucro Acumulado'
            ))
            fig.update_layout(
                title='Resultados do Backtest',
                xaxis_title='Data',
                yaxis_title='Lucro Acumulado'
            )
            return fig

    @app.callback(
        Output('trades-history-graph', 'figure'),
        [Input('interval-component', 'n_intervals')]
    )
    def update_trades_history(n):
        with data_lock:
            df_trades = trading_bot.trade_results
            fig = go.Figure()
            if not df_trades.empty:
                fig.add_trace(go.Scatter(
                    x=df_trades['entry_time'],
                    y=df_trades['entry_price'],
                    mode='markers',
                    marker=dict(color='green'),
                    name='Entradas'
                ))
                fig.add_trace(go.Scatter(
                    x=df_trades['exit_time'],
                    y=df_trades['exit_price'],
                    mode='markers',
                    marker=dict(color='red'),
                    name='Saídas'
                ))
            fig.update_layout(
                title='Histórico de Trades',
                xaxis_title='Data',
                yaxis_title='Preço'
            )
            return fig

    # ====== CALLBACK: para exibir logs no Div 'log-output' ======
    @app.callback(
        Output('log-output', 'children'),
        [Input('log-interval', 'n_intervals')]
    )
    def update_log_output(n):
        """
        Retorna as últimas linhas do nosso logger em memória,
        estilizadas conforme o nível do log (INFO, WARNING, ERROR, etc.).
        """

        # 1) Obtém todas as linhas do logger em memória
        lines = memory_logger.get_logs()

        # 2) Definimos um dict de estilos e emojis por nível
        log_styles = {
            'DEBUG': {'emoji': '🔎', 'color': '#6c757d'},  # cinza
            'INFO': {'emoji': 'ℹ️', 'color': '#000000'},  # preto
            'WARNING': {'emoji': '⚠️', 'color': 'orange'},
            'ERROR': {'emoji': '🛑', 'color': 'red'},
            'CRITICAL': {'emoji': '💥', 'color': 'red'},
        }

        def parse_log_level(line):
            """Tenta identificar o nível de log na string formatada."""
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

            # Exemplo: substituindo parte do prefixo para ficar mais limpo
            # mas você pode apenas manter a linha inteira
            # line sem mudança: f"{style_info['emoji']} {line}"
            # ou processar datas, etc.

            # Aqui, vamos exibir:
            # [EMOJI] 2023-10-10 10:10:10,999 - LEVEL - Mensagem...
            styled_logs.append(
                html.Div(
                    f"{style_info['emoji']} {line}",
                    style={
                        'color': style_info['color'],
                        'fontFamily': 'monospace',
                        'fontSize': '12px',
                        'marginBottom': '2px'
                    }
                )
            )

        # Retornamos uma lista de Divs, cada um representando uma linha
        return styled_logs

    return app
