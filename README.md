# D3 Teste - Predição dos casos diários de Covid19


## Visão Geral
Este repositório foi criado com o objetivo de cumprir com os objetivos propostos pelo [teste da D3_Company](https://test.d3.do/):

    Criar um método ou função que dado um numero N de dias faça a previsão da quantidade diárias de novos casos de Covid19 no Mundo. Tal que N seja um número inteiro e maior do que 0
Para realizar a tarefa de predição, utilizou-se um modelo estatistico de Machine Learning que através dos dados históricos possibilitou estimar os valores futuros.

## Instalação
A versão do python utilizada foi a `3.8.5`.

Para criação de um ambiente virtual e instalação das dependências:
```bash
python3 -m venv env
source env/bin/activate
```
```bash
pip install --upgrade pip
pip install -r -requirements.txt
```
Para executar os testes:
```bash
python pytest
```

## Definição do Modelo
Para definir qual modelo seria utilizado foi realizado um breve estudo disponivel em três `jupyter notebooks` no diretório `eda - covid19`.

#### `EDA - Covid19.ipynb`: 
Analise exploratória inicial dos dados, onde é realizado o filtro para obter as informações referentes ao mundo todo. Foram realizados testes com a bilbioteca `pandas` para obter os dados diretamente de um `URL` público disponivilizado pelo Our World in Data.

Treinou-se 2 modelos para uma comparação visual de qual conseguiria representar de forma mais fiel o os surgimentos de novos casos no mundo: `auto_arima` e `prophet`, ambos estatisticos. 

#### `Comparison ARIMA x auto_arima.ipynb`:
Comparação entre um modelo `ARIMA` e o modelo `auto_arima`. Como o `auto_arima` se trata de um algoritimo que testa diferentes combinações de modelos `ARIMA`, optou-se por utiliza-lo apenas na fase de hiper parametrização e deixar a implementação de fato com o modelo `ARIMA`, mais performático e com o mesmo resultado.


#### `Evaluate ARIMA model.ipynb`:
Availiação da precisão do modelo treinado utilizando cross validation.

## Guia de Uso
Todo sistema foi pensado para ser usado de forma simples, tanto por uma interface por linha de comando ou diretamente na código, para tanto foram desenvolvidas duas classes:

#### `predict_covid.data_provider.DataProvider`:
Carrega o `dataset`com os dados históricos provenientes da plataforma Our World in Data (OWID). A classe permite tanto carregar os dados a partir de um arquivo `.csv` disponivel no repositório como carregar os dados diretamente de um `url` público disponibilizado pelo OWID.

#### `predict_covid.model.CovidModel`:
Treina um modelo `ARIMA` com os parâmetaros obtidos através do estudo disponivel em `eda - covid19`. A classe permite salvar o modelo treinado em disco no formato de um arquivo binário, o que permite facilmente a sua reutilização, além de adotar o padrão `singleton` para garantir que o modelo não será treinado ou carregado do disco desnecessáriamente, utilizando a mesma instancia em toda a aplicação.

### Variaveis de Ambiente
Para o seu correto funcionamento é necessário configurar algumas variaveis de ambiente (que por conveniência estará disponivel no arquivo `.env`):
* `DATASET_PATH`: Caminho onde está salvo o `dataset` local.
* `DATASET_GITHUB_URL`: `URL`que aponta para o `dataset` disponivilizado pelo OWID.
* `MODEL_PATH`: Caminho onde o modelo será salvo em um arquivo binário com a extensão `.pkl`. 
* `USE_REMOTE_DATASET`: Flag que sinaliza se o sistema usará os dados locais ou os dados remotos.
* `FLASK_APP`: Caminho para o arquivo que contem o app flask.

### Interface de linha de comando CLI
A interface de linha de comando está disponivel a partir do arquivo `predict.py` e dispõe de duas funcionalidades: Fazer uma nova predição e atualizar o modelo.

Para fazer uma nova predição:
```bash
python predict.py predict --days 5
```
E saída deverá ser:
```
1 ->   855684
2 ->   798035
3 ->   770671
4 ->   799985
5 ->   851167
```
Para atualizar o modelo:
```bash
python predict.py update
```
E a saída deverá ser:
```
Updated Model
```
Visto que os dados de casos de covid são atualizados diariamente, sempre que o modelo for atualizado será a partir dos dados mais atuais disponibilizados pelo OWID.

### Usando a classe `CovidModel`
Além da interface de linha de comando, é possivel importar a classe `CovidModel` e utilizada acoplada em outros códigos, APIs REST, serviços em backgroud, clientes RPC e etc.
Para tal segue abaixo um exemplo simples de utilização:
```python
from predict_covid.model import CovidModel



if __name__ == '__main__':
    
    model = CovidModel()
    model.load_model()

    forecast = model.predict(days=2)
    print(forecast)
```
Diferentemente da CLI, a saída do método `.predict` é um dicionário, o retorno do código acima seria:
```python
{'1': 855684, '2': 798035}
```
## Executar com Docker
Para exemplificar o uso do sistema dentro um container docker, foi criado uma simples API HTTP utilizando o framework `flask`.

Para criar a imagem, basta executar o comando:
```
docker build --tag forecast_covid .
```
E em seguida para executar:
```
docker run --env-file ./.env -p 8000:8000 forecast_covid
```

### Utilização da API
A API HTTP disponbilizá duas rotas na porta `8000`, uma para predição e outra para update. Ambas aceitam somente requisições do tipo `GET`.

Para realizar a predição:
```
curl 127.0.0.1:8000/forecast-covid/predict/2/
```
E o resultado deverá ser:
```python
{'1': 855684, '2': 798035}
```

Para atualizar o modelo:
```
curl 127.0.0.1:8000/forecast-covid/update/
```
E o resultado deverá ser:
```python
{'message': 'Updated model'}
```

## Proposta de Arquitetura e Infraestrutura
Assim como foi feito no Dockerfile, para distribuição desse modelo eu aconselharia a utilização de uma API HTTP, entretanto com algumas modificações de modo a melhorar a performance e a escalabilidade.

A primeira alteração seria primeiramente hospedar todos os serviços em algum serviço de nuvem compativel, e por tanto, em vez de salvar o modelo em binário em disco, salvar no serviço de storage disponivel, facilitando a sua manutenção e interação com outros micro serviços além da API principal.

Seria interessante também, dado a natureza das informações que são atualizadas diariamentes, que o modelo fosse atualizado periodicamente, e pra tal tarefa poderia ser utilizado um script executado por um `cron job` em intervalos pré-determinados.

Diferente de outros modelos de Machine Learning, o algoritimo `ARIMA` permite que o modelo seja atualizado sempre que dados novos estiverem disponiveis, sem precisar que o mesmo seja treinado do zero novamente. 

Além dos recursos citados, a API também pode contar com um sistema de cache externo (como Redis ou Memcached) para salvar os resultados já calculados não sendo necessários calculalos de novo. A adoção desses recursos descentralizados (como usar uma memória cache externa, e salvar o modelo binário em um storage) permite que multiplas instâncias da API possam utilizar-se do mesmo modelo base, podendo incrementar de forma significativa a sua disponibilidade.
