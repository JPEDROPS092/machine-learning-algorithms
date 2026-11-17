"""A regressao logistica do zero precisa chegar perto do scikit-learn."""

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.preprocessing import StandardScaler

from mlalgos import LogisticRegression


def _perda_binaria(X, y_bin, w, b, l2):
    z = X @ w + b
    z = np.clip(z, -500, 500)
    p = 1.0 / (1.0 + np.exp(-z))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(
        -np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p)) + 0.5 * l2 * np.dot(w, w)
    )


def test_binaria_concorda_com_sklearn():
    X, y = make_classification(
        n_samples=180,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    X = StandardScaler().fit_transform(X)
    l2 = 1.0 / len(X)
    nosso = LogisticRegression(learning_rate=1.0, n_iterations=800, l2=l2).fit(X, y)
    deles = SkLogistic(C=1.0, solver="lbfgs", max_iter=500).fit(X, y)

    assert nosso.score(X, y) >= 0.95
    assert nosso.score(X, y) >= deles.score(X, y) - 0.02
    acordo = np.mean(nosso.predict(X) == deles.predict(X))
    assert acordo >= 0.95
    # A perda desce a partir da inicializacao em zero.
    assert nosso.loss_history_[-1] < nosso.loss_history_[0]
    # E fica perto da perda no coeficiente do scikit-learn, que resolve o mesmo custo.
    perda_deles = _perda_binaria(X, y.astype(float), deles.coef_.ravel(), float(deles.intercept_[0]), l2)
    assert nosso.loss_history_[-1] <= perda_deles + 0.02


def test_multiclasse_iris_perto_do_sklearn():
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data[:, 2:4])
    y = iris.target
    l2 = 1.0 / len(X)
    nosso = LogisticRegression(learning_rate=0.8, n_iterations=1500, l2=l2).fit(X, y)
    deles = SkLogistic(C=1.0, solver="lbfgs", max_iter=500).fit(X, y)
    assert nosso.score(X, y) >= 0.93
    assert abs(nosso.score(X, y) - deles.score(X, y)) <= 0.05
    proba = nosso.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba.shape == (len(X), 3)


def test_gradiente_binario_por_diferenca_finita():
    gerador = np.random.default_rng(2)
    X = gerador.normal(size=(30, 3))
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    modelo = LogisticRegression(learning_rate=0.2, n_iterations=1, l2=0.05).fit(X, y)
    # Um passo a partir de zero. Recalcula o gradiente analitico e compara
    # com diferenca finita na perda inicial.
    y_bin = y.astype(float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    l2 = 0.05
    p = np.full(n, 0.5)
    erro = (p - y_bin) / n
    grad_w = X.T @ erro + l2 * w
    grad_b = np.sum(erro)

    eps = 1e-6
    base = _perda_binaria(X, y_bin, w, b, l2)
    numerico_w = np.empty(d)
    for j in range(d):
        passo = w.copy()
        passo[j] += eps
        numerico_w[j] = (_perda_binaria(X, y_bin, passo, b, l2) - base) / eps
    numerico_b = (_perda_binaria(X, y_bin, w, b + eps, l2) - base) / eps
    np.testing.assert_allclose(grad_w, numerico_w, atol=1e-5)
    np.testing.assert_allclose(grad_b, numerico_b, atol=1e-5)
    # O passo unico do modelo usa exatamente esse gradiente.
    np.testing.assert_allclose(modelo.coef_, w - 0.2 * grad_w, atol=1e-8)
    np.testing.assert_allclose(modelo.intercept_, b - 0.2 * grad_b, atol=1e-8)


def test_rejeita_uma_classe_so():
    X = np.ones((4, 2))
    y = np.zeros(4)
    try:
        LogisticRegression().fit(X, y)
    except ValueError as erro:
        assert "duas classes" in str(erro)
    else:
        raise AssertionError("deveria ter recusado um y de uma classe so")
