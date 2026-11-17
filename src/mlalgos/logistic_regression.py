"""Regressao logistica binaria e multiclasse, com descida de gradiente.

Binaria: a sigmoide transforma o escore linear em probabilidade e o
treino minimiza a entropia cruzada com penalidade L2 nos pesos.
Multiclasse: a softmax faz o mesmo papel para mais de duas classes.
O vies nao entra na penalidade L2.
"""

from __future__ import annotations

import numpy as np


def sigmoide(z):
    """sigmoide(z) = 1 / (1 + exp(-z)), estavel para z muito positivo ou negativo."""
    z = np.asarray(z, dtype=float)
    saida = np.empty_like(z, dtype=float)
    positivo = z >= 0
    saida[positivo] = 1.0 / (1.0 + np.exp(-z[positivo]))
    expoente = np.exp(z[~positivo])
    saida[~positivo] = expoente / (1.0 + expoente)
    return saida


def softmax(z):
    """softmax por linha. Subtrair o maximo evita exp estourar."""
    z = np.asarray(z, dtype=float)
    deslocado = z - np.max(z, axis=1, keepdims=True)
    expoente = np.exp(deslocado)
    return expoente / np.sum(expoente, axis=1, keepdims=True)


class LogisticRegression:
    """Classificador linear treinado por descida de gradiente em lote.

    A funcao de custo, ja na media das amostras, e

        J = entropia cruzada + (l2 / 2) * ||w||^2

    ``l2`` e o lambda da formula. Para chegar perto do ``C`` do
    scikit-learn (que penaliza a soma, nao a media), use ``l2 = 1 / (n * C)``
    com ``C = 1``, isto e, ``l2 = 1 / n``.

    Parametros
    ----------
    learning_rate : float
        Tamanho do passo. Em atributos padronizados, valores entre 0,5 e 2
        costumam descer sem oscilar.
    n_iterations : int
        Numero maximo de passadas pelo conjunto inteiro.
    l2 : float
        Forca da penalidade L2. Zero desliga a regularizacao.
    fit_intercept : bool
        Se verdadeiro, estima um vies. O vies nao e penalizado.
    tol : float ou None
        Se nao for None, para quando a perda absoluta cai menos que ``tol``
        de uma iteracao para a outra.
    """

    def __init__(
        self,
        learning_rate=0.5,
        n_iterations=1000,
        l2=0.0,
        fit_intercept=True,
        tol=None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate precisa ser positivo.")
        if n_iterations < 1:
            raise ValueError("n_iterations precisa ser pelo menos 1.")
        if l2 < 0:
            raise ValueError("l2 nao pode ser negativo.")
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)
        self.tol = tol

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("X precisa ter duas dimensoes: (amostras, atributos).")
        if y.ndim != 1:
            y = y.reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y precisam ter o mesmo numero de linhas.")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X esta vazio.")
        if np.isnan(X).any():
            raise ValueError("X contem NaN. Trate os valores ausentes antes do ajuste.")

        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Preciso de pelo menos duas classes para treinar.")

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.loss_history_ = []

        if len(classes) == 2:
            self._fit_binario(X, y)
        else:
            self._fit_multiclasse(X, y)
        return self

    def _passo_aceitavel(self, perda):
        if not np.isfinite(perda):
            raise ValueError(
                "A perda virou NaN ou infinito. Reduza learning_rate: o passo passou do minimo."
            )
        if self.tol is not None and len(self.loss_history_) > 0:
            if abs(self.loss_history_[-1] - perda) < self.tol:
                self.loss_history_.append(float(perda))
                return False
        self.loss_history_.append(float(perda))
        return True

    def _fit_binario(self, X, y):
        # A classe positiva e classes_[1], a maior na ordenacao do np.unique.
        y_bin = (y == self.classes_[1]).astype(float)
        n, d = X.shape
        w = np.zeros(d, dtype=float)
        b = 0.0

        for _ in range(self.n_iterations):
            z = X @ w + b
            p = np.clip(sigmoide(z), 1e-12, 1.0 - 1e-12)
            # Entropia cruzada binaria, media das amostras.
            perda = -np.mean(y_bin * np.log(p) + (1.0 - y_bin) * np.log(1.0 - p))
            perda += 0.5 * self.l2 * np.dot(w, w)
            if not self._passo_aceitavel(perda):
                break
            # dJ/dw = X^T (p - y) / n + lambda w. O vies nao leva lambda.
            erro = (p - y_bin) / n
            w -= self.learning_rate * (X.T @ erro + self.l2 * w)
            if self.fit_intercept:
                b -= self.learning_rate * np.sum(erro)

        self.coef_ = w
        self.intercept_ = float(b) if self.fit_intercept else 0.0
        self.n_iter_ = len(self.loss_history_)

    def _fit_multiclasse(self, X, y):
        indice = {classe: i for i, classe in enumerate(self.classes_)}
        y_idx = np.array([indice[valor] for valor in y], dtype=int)
        n, d = X.shape
        k = len(self.classes_)
        W = np.zeros((k, d), dtype=float)
        b = np.zeros(k, dtype=float)
        um_quente = np.zeros((n, k), dtype=float)
        um_quente[np.arange(n), y_idx] = 1.0

        for _ in range(self.n_iterations):
            escores = X @ W.T + b
            P = np.clip(softmax(escores), 1e-12, 1.0)
            P = P / P.sum(axis=1, keepdims=True)
            correta = P[np.arange(n), y_idx]
            perda = -np.mean(np.log(correta)) + 0.5 * self.l2 * np.sum(W * W)
            if not self._passo_aceitavel(perda):
                break
            # dJ/dW = (P - Y)^T X / n + lambda W
            erro = (P - um_quente) / n
            W -= self.learning_rate * (erro.T @ X + self.l2 * W)
            if self.fit_intercept:
                b -= self.learning_rate * erro.sum(axis=0)

        self.coef_ = W
        self.intercept_ = b if self.fit_intercept else np.zeros(k, dtype=float)
        self.n_iter_ = len(self.loss_history_)

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X precisa ter o mesmo numero de atributos do ajuste.")
        if self.coef_.ndim == 1:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_.T + self.intercept_

    def predict_proba(self, X):
        escore = self.decision_function(X)
        if self.coef_.ndim == 1:
            p1 = sigmoide(escore)
            return np.column_stack([1.0 - p1, p1])
        return softmax(escore)

    def predict(self, X):
        if self.coef_.ndim == 1:
            positivo = self.predict_proba(X)[:, 1] >= 0.5
            return np.where(positivo, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        y = np.asarray(y).reshape(-1)
        return float(np.mean(self.predict(X) == y))
