# Algoritmos de aprendizado de maquina, do zero

Implementacoes didaticas, em NumPy, de Naive Bayes e de regressao logistica, com a mesma conta escrita de novo no scikit-learn para comparar. Os doze cadernos baixam o proprio conjunto de dados, rodam no Google Colab e explicam o metodo em portugues.

O material e de Joao Pedro, cientista de dados em Manaus.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-11557c?logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![Licenca MIT](https://img.shields.io/badge/licenca-MIT-green)](LICENSE)

## Cadernos

Cada linha abre o notebook e, ao lado, o mesmo arquivo no Colab.

| Notebook | Conjunto de dados | Algoritmo | Colab |
| --- | --- | --- | --- |
| [01. Filtro antispam](notebooks/01-filtro-antispam-naive-bayes.ipynb) | SMS Spam Collection, UCI 228 | Naive Bayes multinomial | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/01-filtro-antispam-naive-bayes.ipynb) |
| [02. Sentimentos no IMDb](notebooks/02-sentimentos-imdb-naive-bayes.ipynb) | Stanford aclImdb (amostra reprodutivel do corpus de Maas et al., 2011) | Naive Bayes multinomial | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/02-sentimentos-imdb-naive-bayes.ipynb) |
| [03. Noticia falsa](notebooks/03-detector-fake-news-naive-bayes.ipynb) | Noticias reais e falsas de George McIntire, 2016 | Naive Bayes multinomial | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/03-detector-fake-news-naive-bayes.ipynb) |
| [04. Cadernos de noticias](notebooks/04-classificador-noticias-naive-bayes.ipynb) | 20 Newsgroups, quatro grupos (esporte, politica, ciencia, tecnologia) | Naive Bayes multinomial | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/04-classificador-noticias-naive-bayes.ipynb) |
| [05. Iris](notebooks/05-iris-regressao-logistica.ipynb) | Iris, comprimento e largura da petala | Regressao logistica (sigmoide e softmax) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/05-iris-regressao-logistica.ipynb) |
| [06. Diabetes](notebooks/06-diabetes-regressao-logistica.ipynb) | Pima Indians Diabetes, OpenML 37 | Regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/06-diabetes-regressao-logistica.ipynb) |
| [07. Credito](notebooks/07-aprovacao-emprestimos-regressao-logistica.ipynb) | German Credit (Statlog), OpenML `credit-g` | Regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/07-aprovacao-emprestimos-regressao-logistica.ipynb) |
| [08. Cancelamento](notebooks/08-churn-regressao-logistica.ipynb) | Telco Customer Churn, amostra publica da IBM | Regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/08-churn-regressao-logistica.ipynb) |
| [09. Doenca cardiaca](notebooks/09-doenca-cardiaca-regressao-logistica.ipynb) | Heart Disease da UCI, OpenML 43398 | Regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/09-doenca-cardiaca-regressao-logistica.ipynb) |
| [10. Arvores da Amazonia Central](notebooks/10-plantas-amazonia-regressao-logistica.ipynb) | Censo florestal de Cunha et al. 2021, EIDC (dado real) | Regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/10-plantas-amazonia-regressao-logistica.ipynb) |
| [11. Os dois metodos, lado a lado](notebooks/11-comparando-logistic-regression-e-naive-bayes.ipynb) | SMS Spam Collection, UCI 228 | Naive Bayes e regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/11-comparando-logistic-regression-e-naive-bayes.ipynb) |
| [12. Fronteiras de decisao](notebooks/12-fronteiras-de-decisao.ipynb) | Nuvens e luas sinteticas, mais a Iris real | Gaussiano e regressao logistica | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JPEDROPS092/machine-learning-algorithms/blob/main/notebooks/12-fronteiras-de-decisao.ipynb) |

## Como rodar na sua maquina

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
jupyter notebook notebooks/
```

O `pytest` compara Naive Bayes, TF-IDF, metricas e a regressao logistica com o scikit-learn em dados pequenos. Nao baixa os conjuntos dos cadernos.

No Colab, a primeira celula de cada notebook instala as bibliotecas e clona este repositorio para o `src/mlalgos` entrar no caminho. Os arquivos grandes ficam numa pasta `data/`, listada no `.gitignore`.

## O que tem no codigo

- `MultinomialNB` e `GaussianNB`: verossimilhanca no logaritmo, prior empirico, Laplace no multinomial e `var_smoothing` no gaussiano.
- `LogisticRegression`: sigmoide no caso binario, softmax no multiclasse, entropia cruzada e descida de gradiente com L2 so nos pesos.
- `BagOfWords` e `TfidfVectorizer`: tokenizacao, contagem e TF-IDF com a formula suave `log((1+n)/(1+df))+1` e norma L2.
- `metrics`: matriz de confusao, precisao, revocacao, F1 e ROC-AUC.

## Estrutura

```text
src/mlalgos/          algoritmos em NumPy
tests/                comparacao com o scikit-learn
notebooks/            doze cadernos, um assunto cada
.github/workflows/    pytest em cada push
requirements.txt
```

## Serie no blog

Os textos que acompanham estes cadernos ficam no blog de Joao Pedro, em [https://www.jpedrops.com/blog](https://www.jpedrops.com/blog). O endereco do site esta em `siteUrl` no arquivo `data/site-metadata.ts` do repositorio [jp-site](https://github.com/JPEDROPS092/jp-site). A serie esta sendo publicada em paralelo; a pagina do blog e o ponto de entrada.

## Licenca

MIT. Veja [LICENSE](LICENSE).
