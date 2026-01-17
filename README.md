# Projeto de IA para Análise de Crédito

## Visão Geral

Este projeto tem como objetivo desenvolver um **modelo de Inteligência Artificial para análise e previsão de score de crédito**, utilizando técnicas de *Machine Learning* em Python. A solução simula um cenário real de avaliação de risco de crédito, apoiando a tomada de decisão por meio de modelos preditivos baseados em dados históricos de clientes.

O projeto percorre todas as etapas essenciais de um pipeline de dados aplicado à ciência de dados, desde a leitura e tratamento da base até a comparação de modelos e geração de novas previsões.

---

## Objetivo do Projeto

* Prever o **score de crédito** de clientes com base em variáveis socioeconômicas e comportamentais;
* Comparar diferentes algoritmos de *Machine Learning*;
* Avaliar o desempenho dos modelos e selecionar a melhor abordagem;
* Demonstrar, de forma prática, a aplicação de IA em problemas de negócio ligados a crédito e risco.

---

## Tecnologias e Bibliotecas Utilizadas

* **Python**
* **Pandas** – manipulação e análise de dados
* **Scikit-learn** – construção, treino e avaliação dos modelos de Machine Learning

---

## Estrutura do Projeto e Explicação do Código

### 1. Importação das Bibliotecas e Leitura dos Dados

Nesta etapa inicial, são importadas as bibliotecas necessárias e realizada a leitura da base de dados contendo informações dos clientes.

```python
import pandas as pd

tabela = pd.read_csv("clientes.csv")
display(tabela)
display(tabela.info())
```

Essa análise inicial permite entender a estrutura do dataset, tipos de variáveis e possíveis ajustes necessários antes da modelagem.

---

### 2. Tratamento e Preparação dos Dados

Aqui ocorre a preparação da base para aplicação dos modelos de IA. A variável **score_credito** é definida como o alvo da previsão.

```python
# Quero prever a coluna score_credito
```

Essa etapa é fundamental para garantir que os dados estejam no formato adequado para o treinamento dos algoritmos.

---

### 3. Separação entre Variáveis Explicativas e Variável Alvo

Os dados são divididos em:

* **X**: variáveis independentes (características dos clientes)
* **y**: variável dependente (score de crédito)

Além disso, o conjunto é separado em dados de treino e teste.

```python
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
```

---

### 4. Criação e Treinamento dos Modelos de Machine Learning

São testados diferentes algoritmos para identificar o modelo com melhor desempenho:

* **Árvore de Decisão**
* **KNN (K-Nearest Neighbors)**

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

modelo_arvore = DecisionTreeClassifier()
modelo_knn = KNeighborsClassifier()

modelo_arvore.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

---

### 5. Avaliação dos Modelos

Os modelos são avaliados com base na métrica de **acurácia**, permitindo comparar o desempenho de cada abordagem.

```python
from sklearn.metrics import accuracy_score

previsao_arvore = modelo_arvore.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

print(accuracy_score(y_teste, previsao_arvore))
print(accuracy_score(y_teste, previsao_knn))
```

A partir dos resultados, é possível identificar qual modelo apresenta melhor capacidade preditiva.

---

### 6. Geração de Novas Previsões

Após a escolha do melhor modelo, ele é utilizado para realizar previsões em novos dados de clientes.

```python
previsao = modelo.predict(novos_clientes)
display(novos_clientes)
display(previsao)
```

Essa etapa simula a aplicação prática do modelo em um ambiente real de análise de crédito.

---

## Resultados Esperados

* Identificação do modelo mais eficiente para previsão de score de crédito;
* Demonstração prática do uso de IA aplicada a risco de crédito;
* Base sólida para evolução do projeto com métricas adicionais, ajuste de hiperparâmetros e validação cruzada.

---

## Possíveis Evoluções do Projeto

* Inclusão de novos algoritmos (Random Forest, XGBoost, Logistic Regression);
* Análise de importância das variáveis;
* Validação cruzada e *tuning* de hiperparâmetros;
* Integração com pipelines automatizados e deploy do modelo.

---

## Autor

**Jorge Ferreira da Silva Neto**
Analista de Dados | MBA em Data Science e Analytics
