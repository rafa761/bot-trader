# Trading Bot Bitcoin
Projeto de um bot para fazer trade de bitcoin

## Configuração da conta Binance Testnet

1 - Criar conta na [Binance Testnet](https://testnet.binancefuture.com/en/futures/BTCUSDT)
2 - habilitar o Hedge Mode em "Settings > Position Mode"

## Setup do Projeto

1 - Instalar Python versao 3.12+

2 - Criar venv do Python

3 - Rodar o seguinte comando instalar as dependencias

```shell
pip install -r requirements-dev.txt
```

## Pre commit hooks

### Instalar

```bash
pre-commit install
pre-commit install --hook-type prepare-commit-msg
```

### Atualizar

```bash
pre-commit autoupdate
```

## Estrutura do Projeto

```shell
bot-trader/
│── docs/                          # Documentação do projeto
│── src/                           # Código-fonte principal
│   │── core/                      # Módulo central do bot (configurações e utilitários)
│   │   │── config.py              # Configurações do bot (Pydantic V2)
│   │   │── constants.py           # Definição de constantes globais
│   │   │── logger.py              # Configuração do logging
│   │── models/                    # Módulo para machine learning
│   │   │── ai_training.py         # Treinamento dos modelos de IA
│   │   │── model_manager.py       # Gerenciamento do modelo de IA (carregamento, predição)
│   │── services/                  # Serviços principais do bot
│   │   │── binance_client.py      # Cliente para integração com a Binance
│   │   │── trading_strategy.py    # Implementação da lógica de trading
│   │── repositories/              # Camada de abstração de dados
│   │   │── data_handler.py        # Coleta e manipulação de dados do mercado
│   │── dashboard/                 # Interface web (Dash) para visualizar dados do bot
│   │   │── dashboard.py           # Aplicação Dash para gráficos de candles e logs
│   │── main.py                    # Ponto de entrada principal do bot
```