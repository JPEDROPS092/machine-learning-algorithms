"""Tokenizacao, saco de palavras e TF-IDF, sem o scikit-learn.

O padrao de token e o mesmo do CountVectorizer: sequencias de duas ou
mais letras ou digitos, ja em minusculas. Acentos continuam no token,
entao "nao" e "não" nao sao a mesma palavra.
"""

from __future__ import annotations

import re

import numpy as np

# (?u) deixa \w unicode. \w\w+ exige pelo menos dois caracteres.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")


def tokenize(texto):
    """Devolve a lista de tokens de um texto."""
    if texto is None:
        return []
    return _TOKEN.findall(str(texto).lower())


class BagOfWords:
    """Matriz de contagens. Cada coluna e uma palavra do vocabulario.

    Parametros
    ----------
    max_features : int ou None
        Se nao for None, fica so com as palavras de maior contagem total
        no ajuste. O desempate e alfabetico, e as colunas saem em ordem
        alfabetica.
    """

    def __init__(self, max_features=None):
        if max_features is not None and max_features < 1:
            raise ValueError("max_features precisa ser pelo menos 1.")
        self.max_features = max_features
        self.vocabulary_ = None

    def fit(self, documentos):
        contagem_total = {}
        for documento in documentos:
            for token in tokenize(documento):
                contagem_total[token] = contagem_total.get(token, 0) + 1

        if not contagem_total:
            raise ValueError("Nenhum token encontrado nos documentos.")

        ordenadas = sorted(contagem_total, key=lambda t: (-contagem_total[t], t))
        if self.max_features is not None:
            ordenadas = ordenadas[: self.max_features]
        vocabulario = sorted(ordenadas)
        self.vocabulary_ = {token: i for i, token in enumerate(vocabulario)}
        self._contagem_total = {t: contagem_total[t] for t in vocabulario}
        return self

    def transform(self, documentos):
        if self.vocabulary_ is None:
            raise ValueError("Chame fit antes de transform.")
        documentos = list(documentos)
        n = len(documentos)
        matriz = np.zeros((n, len(self.vocabulary_)), dtype=float)
        for i, documento in enumerate(documentos):
            for token in tokenize(documento):
                coluna = self.vocabulary_.get(token)
                if coluna is not None:
                    matriz[i, coluna] += 1.0
        return matriz

    def fit_transform(self, documentos):
        documentos = list(documentos)
        return self.fit(documentos).transform(documentos)


class TfidfVectorizer(BagOfWords):
    """TF-IDF com a mesma formula suave do scikit-learn.

    idf(t) = log( (1 + n) / (1 + df(t)) ) + 1

    tf e a contagem crua. Cada linha e normalizada para norma L2 igual a 1,
    para o tamanho do texto nao dominar o cosseno.
    """

    def fit(self, documentos):
        documentos = list(documentos)
        super().fit(documentos)
        n = len(documentos)
        df = np.zeros(len(self.vocabulary_), dtype=float)
        for documento in documentos:
            vistos = set()
            for token in tokenize(documento):
                coluna = self.vocabulary_.get(token)
                if coluna is not None and token not in vistos:
                    df[coluna] += 1.0
                    vistos.add(token)
        # Suavizacao: soma 1 no numerador e no denominador, depois soma 1 no idf.
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.df_ = df
        self.idf_ = idf
        self.n_documents_ = n
        return self

    def transform(self, documentos):
        contagens = super().transform(documentos)
        tfidf = contagens * self.idf_
        norma = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norma[norma == 0.0] = 1.0
        return tfidf / norma

    def fit_transform(self, documentos):
        documentos = list(documentos)
        return self.fit(documentos).transform(documentos)
