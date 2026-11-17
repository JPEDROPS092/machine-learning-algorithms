"""Algoritmos classicos de aprendizado de maquina, escritos com NumPy.

O pacote existe para estudo. Cada modelo abaixo reimplementa a conta
que o scikit-learn faz por dentro, com os nomes das etapas a vista:
verossimilhanca, suavizacao de Laplace, sigmoide, entropia cruzada
e descida de gradiente.
"""

from mlalgos.logistic_regression import LogisticRegression
from mlalgos.metrics import (
    confusion_matrix,
    curva_roc,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from mlalgos.naive_bayes import GaussianNB, MultinomialNB
from mlalgos.text import BagOfWords, TfidfVectorizer, tokenize

__all__ = [
    "BagOfWords",
    "GaussianNB",
    "LogisticRegression",
    "MultinomialNB",
    "TfidfVectorizer",
    "confusion_matrix",
    "curva_roc",
    "f1_score",
    "precision_score",
    "recall_score",
    "roc_auc_score",
    "tokenize",
]
