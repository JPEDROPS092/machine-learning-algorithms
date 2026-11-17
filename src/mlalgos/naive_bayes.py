"""Naive Bayes multinomial e gaussiano, apenas com NumPy.

A hipotese ingênua e que, dada a classe, os atributos sao independentes.
Com ela, a verossimilhanca de um exemplo vira um produto, e o produto
vira uma soma quando trabalhamos no logaritmo. Some alpha (Laplace) para
nenhuma palavra nunca vista zerar a conta inteira.
"""

from __future__ import annotations

import numpy as np


def _como_matriz(X, nome="X"):
    matriz = np.asarray(X, dtype=float)
    if matriz.ndim != 2:
        raise ValueError(f"{nome} precisa ter duas dimensoes: (amostras, atributos).")
    if matriz.shape[0] == 0 or matriz.shape[1] == 0:
        raise ValueError(f"{nome} esta vazio.")
    if np.isnan(matriz).any():
        raise ValueError(f"{nome} contem NaN.")
    return matriz


def _como_rotulos(y, n_amostras):
    rotulos = np.asarray(y)
    if rotulos.ndim != 1:
        rotulos = rotulos.reshape(-1)
    if rotulos.shape[0] != n_amostras:
        raise ValueError("X e y precisam ter o mesmo numero de linhas.")
    return rotulos


def _logsumexp(matriz):
    """log(sum(exp(v))) estavel, por linha."""
    pico = np.max(matriz, axis=1, keepdims=True)
    return pico.ravel() + np.log(np.sum(np.exp(matriz - pico), axis=1))


class MultinomialNB:
    """Naive Bayes para contagens, como um saco de palavras.

    Parametros
    ----------
    alpha : float
        Forca da suavizacao de Laplace (Lidstone). O padrao 1 soma um
        a cada contagem de palavra em cada classe.
    """

    def __init__(self, alpha=1.0):
        if alpha < 0:
            raise ValueError("alpha nao pode ser negativo.")
        self.alpha = float(alpha)

    def fit(self, X, y):
        X = _como_matriz(X)
        if np.any(X < 0):
            raise ValueError("MultinomialNB so aceita contagens nao negativas.")
        y = _como_rotulos(y, X.shape[0])

        classes, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        n_amostras, n_atributos = X.shape

        # Contagem bruta: quantas vezes cada palavra apareceu em cada classe.
        feature_count = np.zeros((n_classes, n_atributos), dtype=float)
        class_count = np.zeros(n_classes, dtype=float)
        for k in range(n_classes):
            linhas = X[y_idx == k]
            feature_count[k] = linhas.sum(axis=0)
            class_count[k] = linhas.shape[0]

        # Laplace: (contagem + alpha) / (total da classe + alpha * |V|).
        suavizada = feature_count + self.alpha
        total_classe = suavizada.sum(axis=1, keepdims=True)
        feature_log_prob = np.log(suavizada) - np.log(total_classe)

        # Prior empirico P(y) = n_y / n, tambem no log.
        class_log_prior = np.log(class_count) - np.log(class_count.sum())

        self.classes_ = classes
        self.class_count_ = class_count
        self.feature_count_ = feature_count
        self.feature_log_prob_ = feature_log_prob
        self.class_log_prior_ = class_log_prior
        self.n_features_in_ = n_atributos
        return self

    def _log_conjunto(self, X):
        X = _como_matriz(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Numero de atributos diferente do ajuste.")
        # log P(x, y) = log P(y) + soma_j x_j * log P(palavra_j | y)
        return X @ self.feature_log_prob_.T + self.class_log_prior_

    def predict_log_proba(self, X):
        conjunto = self._log_conjunto(X)
        return conjunto - _logsumexp(conjunto)[:, None]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_conjunto(X), axis=1)]


class GaussianNB:
    """Naive Bayes em que cada atributo, dada a classe, e gaussiano.

    Parametros
    ----------
    var_smoothing : float
        Fracao da maior variancia dos dados somada a cada variancia.
        Evita divisao por zero quando uma classe nao varia num atributo.
        O mesmo papel do ``var_smoothing`` do scikit-learn (padrao 1e-9).
    """

    def __init__(self, var_smoothing=1e-9):
        if var_smoothing < 0:
            raise ValueError("var_smoothing nao pode ser negativo.")
        self.var_smoothing = float(var_smoothing)

    def fit(self, X, y):
        X = _como_matriz(X)
        y = _como_rotulos(y, X.shape[0])
        classes = np.unique(y)
        n_classes = len(classes)
        n_atributos = X.shape[1]

        theta = np.zeros((n_classes, n_atributos), dtype=float)
        var = np.zeros((n_classes, n_atributos), dtype=float)
        class_count = np.zeros(n_classes, dtype=float)

        for i, classe in enumerate(classes):
            Xi = X[y == classe]
            theta[i] = Xi.mean(axis=0)
            # Variancia de maxima verossimilhanca, divisao por n e nao por n-1.
            var[i] = Xi.var(axis=0)
            class_count[i] = Xi.shape[0]

        # epsilon e uma folga proporcional a maior variancia de X, igual
        # para todas as classes. Sem ela, variancia zero quebra o log.
        epsilon = self.var_smoothing * np.var(X, axis=0).max()
        var = var + epsilon

        self.classes_ = classes
        self.theta_ = theta
        self.var_ = var
        self.epsilon_ = epsilon
        self.class_count_ = class_count
        self.class_prior_ = class_count / class_count.sum()
        self.n_features_in_ = n_atributos
        return self

    def _log_conjunto(self, X):
        X = _como_matriz(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Numero de atributos diferente do ajuste.")
        n_classes = len(self.classes_)
        conjunto = np.empty((X.shape[0], n_classes), dtype=float)
        for i in range(n_classes):
            # log da gaussiana, somando atributos porque sao independentes:
            # -0.5 * log(2 pi sigma^2) - 0.5 * (x - mu)^2 / sigma^2
            log_norm = -0.5 * np.sum(np.log(2.0 * np.pi * self.var_[i]))
            quadrado = ((X - self.theta_[i]) ** 2) / self.var_[i]
            verossimilhanca = log_norm - 0.5 * np.sum(quadrado, axis=1)
            conjunto[:, i] = np.log(self.class_prior_[i]) + verossimilhanca
        return conjunto

    def predict_log_proba(self, X):
        conjunto = self._log_conjunto(X)
        return conjunto - _logsumexp(conjunto)[:, None]

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_conjunto(X), axis=1)]
