# Trading Bot Bitcoin
Projeto de um bot para fazer trade de bitcoin

# Configuração da conta Binance Testnet

1 - Criar conta na [Binance Testnet](https://testnet.binancefuture.com/en/futures/BTCUSDT)
2 - habilitar o Hedge Mode em "Settings > Position Mode"

# Setup do Projeto

1 - Instalar Python versao 3.12+

2 - Criar venv do Python

3 - Rodar o seguinte comando instalar as dependencias

```shell
pip install -r requirements.txt
```

# Estrutura do Trading Bot V1

* **config.py** cuida de variáveis de ambiente e configurações.
* **logger.py** centraliza a configuração de logging.
* **binance_client.py** encapsula a criação do Client da Binance e métodos de acesso.
* **data_handler.py** gerencia coleta de dados e atualização do DataFrame histórico.
* **model_manager.py** carrega/treina modelos e faz predições de TP e SL.
* **trading_strategy.py** define a lógica de entrada e saída (compra/venda) e o cálculo de quantidade.
* **trading_bot.py** coordena tudo: WebSocket, coloca ordens, registra trades e faz loop de trading.
* **dashboard.py** constrói a aplicação Dash e seus callbacks.
* **main.py** faz a “cola” final: inicia tudo e mantém o fluxo.


# Informações extras

* Criar conta no ambiente de teste da Binance
https://testnet.binancefuture.com/en/futures/BTCUSDT
