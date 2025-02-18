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
pip install -r requirements.txt
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
