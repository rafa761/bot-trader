# dashboard/dashboard.py

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, dash_table

from services.performance_monitor import TradePerformanceMonitor


def create_info_icon(tooltip_id, tooltip_text):
    """
    Cria um ícone de informação com tooltip.

    Args:
        tooltip_id: ID único para o tooltip
        tooltip_text: Texto explicativo para o tooltip

    Returns:
        Um componente html.Div contendo o ícone e o tooltip
    """
    return html.Div([
        html.I(
            className="fas fa-question-circle text-info ml-2",
            id=tooltip_id,
            style={"cursor": "pointer", "margin-left": "5px"}
        ),
        dbc.Tooltip(
            tooltip_text,
            target=tooltip_id,
            placement="right"
        )
    ], style={"display": "inline-block"})


def create_technical_analysis_tab(data_handler):
    """
    Cria a aba de Análise Técnica com gráficos e indicadores.

    Args:
        data_handler: Objeto DataHandler para acesso aos dados

    Returns:
        Componente dbc.Tab para análise técnica
    """
    return dbc.Tab(label="Análise Técnica", children=[
        dbc.Row([
            dbc.Col([
                html.H3(
                    ["Análise Técnica", create_info_icon(
                        "technical-analysis-info",
                        "Esta aba exibe gráficos de preços e diversos indicadores técnicos "
                        "utilizados pelo bot para tomar decisões de trading."
                    )],
                    className="mt-3 mb-3 d-flex align-items-center"
                ),
            ]),
        ]),

        # Controles para seleção de indicadores
        dbc.Row([
            dbc.Col([
                html.Label("Selecione o Período:"),
                dcc.Dropdown(
                    id="timeframe-selector",
                    options=[
                        {"label": "1 Hora", "value": "1h"},
                        {"label": "4 Horas", "value": "4h"},
                        {"label": "1 Dia", "value": "1d"},
                        {"label": "1 Semana", "value": "1w"},
                    ],
                    value="1d",
                    clearable=False,
                    className="mb-3"
                ),
            ], width=3),
            dbc.Col([
                html.Label("Selecione Indicadores:"),
                dcc.Dropdown(
                    id="indicator-selector",
                    options=[
                        {"label": "RSI", "value": "rsi"},
                        {"label": "MACD", "value": "macd"},
                        {"label": "Bandas de Bollinger", "value": "bollinger"},
                        {"label": "Médias Móveis", "value": "moving_avg"},
                        {"label": "Volume", "value": "volume"},
                    ],
                    value=["rsi", "moving_avg"],
                    multi=True,
                    className="mb-3"
                ),
            ], width=5),
            dbc.Col([
                html.Label("Atualização:"),
                dbc.Button(
                    "Atualizar Gráficos",
                    id="update-chart-button",
                    color="primary",
                    className="mt-1"
                ),
            ], width=4),
        ]),

        # Gráfico principal de preços
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Gráfico de Preços",
                        create_info_icon(
                            "price-chart-info",
                            "Exibe a evolução do preço ao longo do tempo com indicadores selecionados."
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="price-chart", style={"height": "500px"}),
                    ])
                ]),
            ], width=12),
        ], className="mb-4"),

        # Painel de indicadores
        dbc.Row([
            # Indicadores Osciladores (RSI, Estocástico, etc.)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Osciladores",
                        create_info_icon(
                            "oscillators-info",
                            "Os osciladores como RSI, Estocástico e MACD são úteis para identificar "
                            "condições de sobrecompra e sobrevenda do mercado."
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="oscillators-chart", style={"height": "250px"}),
                    ])
                ]),
            ], width=6),

            # Indicadores de Momento (MACD, etc.)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Indicadores de Momento",
                        create_info_icon(
                            "momentum-indicators-info",
                            "Indicadores de momento como o MACD ajudam a identificar a força "
                            "e direção da tendência atual."
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="momentum-indicators-chart", style={"height": "250px"}),
                    ])
                ]),
            ], width=6),
        ], className="mb-4"),

        # Painel de volume e volatilidade
        dbc.Row([
            # Volume
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Volume",
                        create_info_icon(
                            "volume-info",
                            "O volume de negociação é um indicador importante para confirmar "
                            "a força de movimentos de preço e tendências."
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="volume-chart", style={"height": "250px"}),
                    ])
                ]),
            ], width=6),

            # Volatilidade (ATR)
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        "Volatilidade (ATR)",
                        create_info_icon(
                            "volatility-info",
                            "O Average True Range (ATR) mede a volatilidade do mercado "
                            "e é usado para calcular tamanhos de posição e níveis de stop loss."
                        )
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="volatility-chart", style={"height": "250px"}),
                    ])
                ]),
            ], width=6),
        ]),

        # Intervalo para atualização automática dos gráficos
        dcc.Interval(
            id="chart-update-interval",
            interval=60 * 1000,  # a cada 60 segundos
            n_intervals=0
        ),
    ])


def create_performance_tab(performance_monitor):
    """
    Cria a aba de Performance com métricas e gráficos.

    Args:
        performance_monitor: Objeto TradePerformanceMonitor

    Returns:
        Componente dbc.Tab para performance
    """
    return dbc.Tab(label="Performance", children=[
        html.Div([
            html.H3([
                "Métricas de Performance",
                create_info_icon(
                    "performance-metrics-info",
                    "Esta seção exibe as principais métricas de performance do seu bot de trading, "
                    "incluindo win rate, profit factor, total de trades e P&L total."
                )
            ], className="mt-3 mb-3 d-flex align-items-center"),

            # Cards com métricas principais
            dbc.Row([
                # Win Rate
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            "Win Rate",
                            create_info_icon(
                                "win-rate-info",
                                "Porcentagem de trades vencedores em relação ao total. "
                                "Um win rate acima de 50% é geralmente considerado bom."
                            )
                        ]),
                        dbc.CardBody(
                            html.H3(id="win-rate-display", className="text-center")
                        )
                    ]),
                    width=3
                ),

                # Profit Factor
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            "Profit Factor",
                            create_info_icon(
                                "profit-factor-info",
                                "Razão entre o lucro total e a perda total. "
                                "Um valor acima de 1.5 é geralmente considerado bom."
                            )
                        ]),
                        dbc.CardBody(
                            html.H3(id="profit-factor-display", className="text-center")
                        )
                    ]),
                    width=3
                ),

                # Total Trades
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            "Total Trades",
                            create_info_icon(
                                "total-trades-info",
                                "Número total de trades realizados pelo bot. "
                                "Um número maior proporciona mais confiabilidade nas métricas."
                            )
                        ]),
                        dbc.CardBody(
                            html.H3(id="total-trades-display", className="text-center")
                        )
                    ]),
                    width=3
                ),

                # P&L Total
                dbc.Col(
                    dbc.Card([
                        dbc.CardHeader([
                            "P&L Total",
                            create_info_icon(
                                "pnl-total-info",
                                "Profit & Loss total acumulado desde o início das operações, "
                                "considerando todos os trades fechados."
                            )
                        ]),
                        dbc.CardBody(
                            html.H3(id="pnl-total-display", className="text-center")
                        )
                    ]),
                    width=3
                )
            ], className="mb-4"),

            # Gráficos
            dbc.Row([
                # Gráfico de curva de equity
                dbc.Col([
                    html.H4([
                        "Curva de Equity",
                        create_info_icon(
                            "equity-curve-info",
                            "Mostra a evolução do capital ao longo dos trades. "
                            "Uma curva ascendente e suave é ideal."
                        )
                    ], className="text-center d-flex align-items-center justify-content-center"),
                    dcc.Graph(id="equity-curve-graph")
                ], width=8),

                # Gráfico de Win Rate por tipo
                dbc.Col([
                    html.H4([
                        "Win Rate por Tipo",
                        create_info_icon(
                            "win-rate-by-type-info",
                            "Compara o win rate entre diferentes tipos de trades (LONG/SHORT, "
                            "a favor/contra tendência, etc.)."
                        )
                    ], className="text-center d-flex align-items-center justify-content-center"),
                    dcc.Graph(id="win-rate-by-type")
                ], width=4)
            ], className="mb-4"),

            # Tabela de trades recentes
            html.H4([
                "Trades Recentes",
                create_info_icon(
                    "recent-trades-info",
                    "Lista dos trades mais recentes com informações detalhadas sobre "
                    "entradas, saídas, resultado e métricas específicas."
                )
            ], className="mt-4 d-flex align-items-center"),
            html.Div(id="recent-trades-table"),

            # Botão para atualizar os dados
            dbc.Button(
                "Atualizar Dados",
                id="update-performance-button",
                color="primary",
                className="mt-3 mb-4"
            ),

            # Intervalo para atualização automática
            dcc.Interval(
                id="performance-update-interval",
                interval=30 * 1000,  # a cada 30 segundos
                n_intervals=0
            )
        ])
    ])


def create_dashboard(data_handler):
    """
    Cria o dashboard da aplicação com análise técnica e métricas de desempenho.

    Args:
        data_handler: Instância de DataHandler para acesso aos dados.

    Returns:
        Aplicação Dash
    """
    # Carregar CSS do Font Awesome para ícones
    external_stylesheets = [
        dbc.themes.DARKLY,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    ]

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    # Inicializar monitor de performance para o dashboard
    performance_monitor = TradePerformanceMonitor()

    # Layout do dashboard com abas
    app.layout = dbc.Container([
        html.H1([
            "Trading Bot Dashboard",
            create_info_icon(
                "dashboard-main-info",
                "Este dashboard fornece uma visão abrangente do desempenho do bot de trading, "
                "incluindo análise técnica em tempo real e métricas de performance."
            )
        ], className="mt-3 mb-4 text-center d-flex align-items-center justify-content-center"),

        dbc.Tabs([
            create_technical_analysis_tab(data_handler),
            create_performance_tab(performance_monitor)
        ])
    ], fluid=True)

    # Callbacks para atualizar os elementos da interface

    # Callback para atualizar o gráfico de preços principal
    @app.callback(
        Output("price-chart", "figure"),
        [Input("chart-update-interval", "n_intervals"),
         Input("update-chart-button", "n_clicks"),
         Input("timeframe-selector", "value"),
         Input("indicator-selector", "value")]
    )
    def update_price_chart(n_intervals, n_clicks, timeframe, indicators):
        try:
            df = data_handler.historical_df.copy()

            if df.empty:
                # Retornar gráfico vazio se não houver dados
                return go.Figure().update_layout(
                    template="plotly_dark",
                    title="Sem dados de preço disponíveis"
                )

            # Criar figura base com o gráfico de candlestick
            fig = go.Figure()

            # Adicionar candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="OHLC"
                )
            )

            # Adicionar indicadores selecionados
            if indicators and "moving_avg" in indicators:
                if 'sma_short' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['sma_short'],
                        line=dict(color='rgba(255, 165, 0, 0.7)', width=1.5),
                        name='SMA Curta'
                    ))
                if 'sma_long' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['sma_long'],
                        line=dict(color='rgba(75, 0, 130, 0.7)', width=1.5),
                        name='SMA Longa'
                    ))
                if 'ema_short' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['ema_short'],
                        line=dict(color='rgba(255, 0, 255, 0.7)', width=1.5),
                        name='EMA Curta'
                    ))

            if indicators and "bollinger" in indicators:
                if 'boll_hband' in df.columns and 'boll_lband' in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['boll_hband'],
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1),
                        name='Bollinger Superior'
                    ))
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['boll_lband'],
                        line=dict(color='rgba(0, 255, 0, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(0, 255, 0, 0.05)',
                        name='Bollinger Inferior'
                    ))

            # Layout principal
            fig.update_layout(
                title=f"Gráfico de Preços ({timeframe})",
                xaxis_title="Data/Hora",
                yaxis_title="Preço",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=500
            )

            # Adicionar barras verticais para trades
            # (precisaríamos recuperar os trades do performance_monitor)

            return fig

        except Exception as e:
            print(f"Erro ao atualizar gráfico de preços: {e}")
            # Retornar gráfico vazio em caso de erro
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Erro ao carregar gráfico de preços"
            )

    # Callback para atualizar o gráfico de osciladores
    @app.callback(
        Output("oscillators-chart", "figure"),
        [Input("chart-update-interval", "n_intervals"),
         Input("update-chart-button", "n_clicks"),
         Input("indicator-selector", "value")]
    )
    def update_oscillators_chart(n_intervals, n_clicks, indicators):
        try:
            df = data_handler.historical_df.copy()

            if df.empty:
                return go.Figure().update_layout(
                    template="plotly_dark",
                    title="Sem dados de osciladores disponíveis"
                )

            fig = go.Figure()

            # Adicionar RSI se selecionado e disponível
            if indicators and "rsi" in indicators and 'rsi' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['rsi'],
                    line=dict(color='rgba(255, 255, 0, 0.8)', width=1.5),
                    name='RSI'
                ))

                # Adicionar linhas horizontais para níveis de sobrecompra/sobrevenda
                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    y0=70,
                    x1=df.index[-1],
                    y1=70,
                    line=dict(color="red", width=1, dash="dash"),
                )

                fig.add_shape(
                    type="line",
                    x0=df.index[0],
                    y0=30,
                    x1=df.index[-1],
                    y1=30,
                    line=dict(color="green", width=1, dash="dash"),
                )

                # Definir eixo Y para valores de 0 a 100
                fig.update_layout(yaxis_range=[0, 100])

            # Adicionar Estocástico se disponível
            if 'stoch_k' in df.columns and 'stoch_d' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['stoch_k'],
                    line=dict(color='rgba(0, 191, 255, 0.8)', width=1.5),
                    name='Estocástico %K'
                ))

                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['stoch_d'],
                    line=dict(color='rgba(255, 127, 80, 0.8)', width=1.5),
                    name='Estocástico %D'
                ))

            fig.update_layout(
                title="Osciladores",
                xaxis_title="Data/Hora",
                yaxis_title="Valor",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=40, b=0),
                height=250
            )

            return fig

        except Exception as e:
            print(f"Erro ao atualizar gráfico de osciladores: {e}")
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Erro ao carregar osciladores"
            )

    # Callback para atualizar o gráfico de indicadores de momento
    @app.callback(
        Output("momentum-indicators-chart", "figure"),
        [Input("chart-update-interval", "n_intervals"),
         Input("update-chart-button", "n_clicks"),
         Input("indicator-selector", "value")]
    )
    def update_momentum_chart(n_intervals, n_clicks, indicators):
        try:
            df = data_handler.historical_df.copy()

            if df.empty:
                return go.Figure().update_layout(
                    template="plotly_dark",
                    title="Sem dados de momento disponíveis"
                )

            fig = go.Figure()

            # Adicionar MACD se selecionado e disponível
            if indicators and "macd" in indicators:
                if 'macd' in df.columns and 'macd_signal' in df.columns and 'macd_histogram' in df.columns:
                    # Histograma MACD
                    colors = ['green' if val >= 0 else 'red' for val in df['macd_histogram']]

                    fig.add_trace(go.Bar(
                        x=df.index,
                        y=df['macd_histogram'],
                        marker_color=colors,
                        name='Histograma MACD'
                    ))

                    # Linha MACD e Linha de Sinal
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['macd'],
                        line=dict(color='rgba(0, 255, 255, 0.8)', width=1.5),
                        name='MACD'
                    ))

                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['macd_signal'],
                        line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5),
                        name='Sinal MACD'
                    ))

            # Adicionar linha zero
            fig.add_shape(
                type="line",
                x0=df.index[0],
                y0=0,
                x1=df.index[-1],
                y1=0,
                line=dict(color="gray", width=1, dash="dash"),
            )

            fig.update_layout(
                title="Indicadores de Momento",
                xaxis_title="Data/Hora",
                yaxis_title="Valor",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=40, b=0),
                height=250
            )

            return fig

        except Exception as e:
            print(f"Erro ao atualizar gráfico de momento: {e}")
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Erro ao carregar indicadores de momento"
            )

    # Callback para atualizar o gráfico de volume
    @app.callback(
        Output("volume-chart", "figure"),
        [Input("chart-update-interval", "n_intervals"),
         Input("update-chart-button", "n_clicks")]
    )
    def update_volume_chart(n_intervals, n_clicks):
        try:
            df = data_handler.historical_df.copy()

            if df.empty:
                return go.Figure().update_layout(
                    template="plotly_dark",
                    title="Sem dados de volume disponíveis"
                )

            fig = go.Figure()

            # Cor do volume baseada na direção do preço
            colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
                      for i in range(len(df))]

            # Adicionar barras de volume
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color=colors,
                name='Volume'
            ))

            # Adicionar média de volume se disponível
            if 'volume_sma' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['volume_sma'],
                    line=dict(color='rgba(255, 255, 255, 0.8)', width=1.5),
                    name='Média de Volume'
                ))

            fig.update_layout(
                title="Volume de Negociação",
                xaxis_title="Data/Hora",
                yaxis_title="Volume",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=40, b=0),
                height=250
            )

            return fig

        except Exception as e:
            print(f"Erro ao atualizar gráfico de volume: {e}")
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Erro ao carregar dados de volume"
            )

    # Callback para atualizar o gráfico de volatilidade
    @app.callback(
        Output("volatility-chart", "figure"),
        [Input("chart-update-interval", "n_intervals"),
         Input("update-chart-button", "n_clicks")]
    )
    def update_volatility_chart(n_intervals, n_clicks):
        try:
            df = data_handler.historical_df.copy()

            if df.empty:
                return go.Figure().update_layout(
                    template="plotly_dark",
                    title="Sem dados de volatilidade disponíveis"
                )

            fig = go.Figure()

            # Adicionar ATR se disponível
            if 'atr' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['atr'],
                    line=dict(color='rgba(255, 0, 0, 0.8)', width=1.5),
                    name='ATR'
                ))

            # Adicionar ATR percentual se disponível
            if 'atr_pct' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['atr_pct'],
                    line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5),
                    name='ATR %'
                ))

            fig.update_layout(
                title="Volatilidade (ATR)",
                xaxis_title="Data/Hora",
                yaxis_title="Valor",
                template="plotly_dark",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=40, b=0),
                height=250
            )

            return fig

        except Exception as e:
            print(f"Erro ao atualizar gráfico de volatilidade: {e}")
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Erro ao carregar dados de volatilidade"
            )

    # Callbacks para a aba de Performance (mantidos do código original)
    @app.callback(
        Output("win-rate-display", "children"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_win_rate(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics
        if metrics.total_trades > 0:
            return f"{metrics.win_rate:.1%}"
        return "N/A"

    @app.callback(
        Output("profit-factor-display", "children"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_profit_factor(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics
        if metrics.profit_factor > 0:
            return f"{metrics.profit_factor:.2f}"
        return "N/A"

    @app.callback(
        Output("total-trades-display", "children"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_total_trades(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics
        return f"{metrics.total_trades}"

    @app.callback(
        Output("pnl-total-display", "children"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_pnl_total(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics
        if metrics.total_profit_loss != 0:
            color = "text-success" if metrics.total_profit_loss > 0 else "text-danger"
            return html.Span(f"${metrics.total_profit_loss:.2f}", className=color)
        return "$ 0.00"

    @app.callback(
        Output("equity-curve-graph", "figure"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_equity_curve(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics

        if not metrics.equity_curve:
            # Retornar gráfico vazio se não houver dados
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Sem dados de equity disponíveis"
            )

        # Criar DataFrame para o gráfico
        equity_df = pd.DataFrame({
            "Trade": list(range(len(metrics.equity_curve))),
            "Equity": metrics.equity_curve
        })

        fig = px.line(
            equity_df,
            x="Trade",
            y="Equity",
            title="Curva de Equity"
        )

        # Adicionar linha de tendência
        if len(equity_df) > 2:
            fig.add_trace(
                px.scatter(
                    equity_df,
                    x="Trade",
                    y="Equity",
                    trendline="ols"
                ).data[1]
            )

        fig.update_layout(
            template="plotly_dark",
            xaxis_title="Número do Trade",
            yaxis_title="Capital ($)",
            hovermode="x"
        )

        return fig

    @app.callback(
        Output("win-rate-by-type", "figure"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_win_rate_by_type(_n_intervals, _n_clicks):
        metrics = performance_monitor.metrics

        # Criar dados para o gráfico de barras
        categories = []
        win_rates = []
        colors = []

        # Win Rate Global
        if metrics.total_trades > 0:
            categories.append("Global")
            win_rates.append(metrics.win_rate * 100)
            colors.append("#636EFA")  # Azul

        # Win Rate LONG
        if metrics.long_trades > 0:
            categories.append("LONG")
            win_rates.append(metrics.long_win_rate * 100)
            colors.append("#00CC96")  # Verde

        # Win Rate SHORT
        if metrics.short_trades > 0:
            categories.append("SHORT")
            win_rates.append(metrics.short_win_rate * 100)
            colors.append("#EF553B")  # Vermelho

        # Win Rate Trend Aligned
        if metrics.trend_aligned_trades > 0:
            categories.append("Com Tendência")
            win_rates.append(metrics.trend_aligned_win_rate * 100)
            colors.append("#AB63FA")  # Roxo

        # Win Rate Counter Trend
        if metrics.counter_trend_trades > 0:
            categories.append("Contra Tendência")
            win_rates.append(metrics.counter_trend_win_rate * 100)
            colors.append("#FFA15A")  # Laranja

        if not categories:
            # Retornar gráfico vazio se não houver dados
            return go.Figure().update_layout(
                template="plotly_dark",
                title="Sem dados de Win Rate disponíveis"
            )

        # Criar gráfico de barras
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=win_rates,
                marker_color=colors
            )
        ])

        fig.update_layout(
            template="plotly_dark",
            title="Win Rate por Tipo de Trade",
            xaxis_title="Tipo de Trade",
            yaxis_title="Win Rate (%)",
            yaxis_range=[0, 100]
        )

        return fig

    @app.callback(
        Output("recent-trades-table", "children"),
        [Input("performance-update-interval", "n_intervals"),
         Input("update-performance-button", "n_clicks")]
    )
    def update_recent_trades_table(_n_intervals, _n_clicks):
        # Obter trades recentes
        df = performance_monitor.get_trades_dataframe()

        if df.empty:
            return html.P("Nenhum trade registrado ainda.")

        # Ordenar por data de entrada (mais recente primeiro)
        if 'entry_time' in df.columns:
            df = df.sort_values('entry_time', ascending=False)

        # Limitar a 10 trades mais recentes
        df = df.head(10)

        # Selecionar e renomear colunas para exibição
        display_cols = [
            'trade_id', 'direction', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'profit_loss_pct', 'profit_loss_absolute',
            'result', 'rr_ratio'
        ]

        # Verificar quais colunas existem no DataFrame
        display_cols = [col for col in display_cols if col in df.columns]

        if not display_cols:
            return html.P("Dados de trades indisponíveis.")

        # Renomear colunas para exibição
        rename_map = {
            'trade_id': 'ID',
            'direction': 'Direção',
            'entry_time': 'Entrada',
            'entry_price': 'Preço Entrada',
            'exit_time': 'Saída',
            'exit_price': 'Preço Saída',
            'profit_loss_pct': 'P&L %',
            'profit_loss_absolute': 'P&L $',
            'result': 'Resultado',
            'rr_ratio': 'R:R'
        }

        # Aplicar renomeação para colunas que existem
        rename_cols = {col: rename_map[col] for col in display_cols if col in rename_map}
        df_display = df[display_cols].rename(columns=rename_cols)

        # Formatar datas e valores numéricos
        for col in df_display.columns:
            if 'time' in col.lower() and pd.api.types.is_datetime64_any_dtype(df_display[col]):
                df_display[col] = df_display[col].dt.strftime('%d/%m/%Y %H:%M')
            elif 'price' in col.lower() or 'p&l' in col.lower():
                df_display[col] = df_display[col].round(2)

        # Criar tabela Dash com estilos
        table = dash_table.DataTable(
            id='trades-table',
            columns=[
                {"name": col, "id": col} for col in df_display.columns
            ],
            data=df_display.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'color': 'white',
                'fontWeight': 'bold'
            },
            style_cell={
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white',
                'textAlign': 'left',
                'padding': '8px'
            },
            style_data_conditional=[
                {
                    'if': {
                        'filter_query': '{Resultado} contains "WIN"'
                    },
                    'backgroundColor': 'rgba(0, 255, 0, 0.2)'
                },
                {
                    'if': {
                        'filter_query': '{Resultado} contains "LOSS"'
                    },
                    'backgroundColor': 'rgba(255, 0, 0, 0.2)'
                },
                {
                    'if': {
                        'filter_query': '{P&L $} > 0'
                    },
                    'color': 'green'
                },
                {
                    'if': {
                        'filter_query': '{P&L $} < 0'
                    },
                    'color': 'red'
                }
            ]
        )

        return table

    return app
