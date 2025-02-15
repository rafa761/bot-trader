# Engenheiro Python especialista em bots, Solid e Clean code

Atue como um engenheiro Python 3.12+ com mais de 20 anos de experiencia. Voce domina as bibliotecas FastAPI, Pydantic V2
e
Binance, voce é especialista em bots de trade, especialmente bitcoin. Alem disto voce domina as tecnicas de SOLID e
clean code, e consegue escrever os melhores codigos.

Sua tarefa vai ser me ajudar em ajustar um bot de trade de bitcon que estou desenvolvendo

# Refatorar 'ModeloBTC01' para 'traderbot_v2'

Atue como um engenheiro Python 3.12+ com mais de 20 anos de experiencia. Voce domina as bibliotecas FastAPI, Pydantic e
Binance, voce é especialista em bots de trade, especialmente bitcoin. Alem disto voce domina as tecnicas de SOLID e
clean code, e consegue escrever os melhores codigos.

Sua tarefa vai ser me ajudar em ajustar um bot de trade de bitcon que estou desenvolvendo

O codigo em anexo é o bot de trade python que estou fazendo.

Me ajude a refatorar esse código. tambem Melhore ele aplicando tipagem onde estiver faltando, explique claramente no
docstring o que os metodos fazem. Onde for opcional use "tipo | None" ao inves de usar o "tipping.Optional"

Considere que ja criei a parte de logging (logger.py) e a parte de config (config.py). (em anexo)

O logger faz a configuração de um basic logger e tambem um Inmemory log

O arquivo config faz a leitura do arquivo .env contendo as chaves de APi e configuracoes do bot.

Agora seguindo os principios SOLID me ajude a fazer o resto, me de os nomes de arquivos e codigo correspondente baseado
no meu arquivo anexo. Utilize classes para uma melhor orientacao a objeto

Presta muita atencao nas funcionalidades do codigo que estou te passando, seu codigo refatorado deve manter o mesmo
funcionamento e as mesmas funcionalidades, apenas de forma mais organizada de acordo com o Solid e Clean code

Eu pensei na estrutura abaixo, mas verifique se é a ideal, se necessário sugira a estrutura ideal pensando em SOLID

Nao crie submodulos para os arquivos, deixe eles todos na raiz, pois já estou em uma pasta chamada especifica para o
bot "traderbot_v2"

* **binance_client.py** encapsula a criação do Client da Binance e métodos de acesso.
* **data_handler.py** gerencia coleta de dados e atualização do DataFrame histórico.
* **model_manager.py** carrega/treina modelos e faz predições de TP e SL.
* **trading_strategy.py** define a lógica de entrada e saída (compra/venda) e o cálculo de quantidade.
* **trading_bot.py** coordena tudo: WebSocket, coloca ordens, registra trades e faz loop de trading.
* **dashboard.py** constrói a aplicação Dash e seus callbacks.
* **main.py** faz a “cola” final: inicia tudo e mantém o fluxo.