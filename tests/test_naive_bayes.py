"""Os dois Naive Bayes precisam bater com o scikit-learn em dado pequeno."""

import numpy as np
from sklearn.naive_bayes import GaussianNB as SkGaussian
from sklearn.naive_bayes import MultinomialNB as SkMultinomial

from mlalgos import GaussianNB, MultinomialNB


def test_multinomial_bate_com_sklearn():
    gerador = np.random.default_rng(0)
    X = gerador.integers(0, 6, size=(60, 8)).astype(float)
    y = gerador.integers(0, 3, size=60)
    # Garante as tres classes.
    y[0], y[1], y[2] = 0, 1, 2

    nosso = MultinomialNB(alpha=1.0).fit(X, y)
    deles = SkMultinomial(alpha=1.0).fit(X, y)

    np.testing.assert_allclose(nosso.feature_log_prob_, deles.feature_log_prob_, atol=1e-10)
    np.testing.assert_allclose(nosso.class_log_prior_, deles.class_log_prior_, atol=1e-10)
    consulta = gerador.integers(0, 6, size=(15, 8)).astype(float)
    np.testing.assert_allclose(
        nosso.predict_log_proba(consulta),
        deles.predict_log_proba(consulta),
        atol=1e-8,
    )
    np.testing.assert_array_equal(nosso.predict(consulta), deles.predict(consulta))


def test_multinomial_laplace_nao_zera_palavra_nova():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0]])
    y = np.array([0, 0, 1])
    modelo = MultinomialNB(alpha=1.0).fit(X, y)
    # A classe 0 nunca viu o segundo atributo, mas a probabilidade fica positiva.
    assert modelo.feature_log_prob_[0, 1] > -np.inf
    proba = modelo.predict_proba([[0.0, 5.0]])
    assert np.isclose(proba.sum(), 1.0)


def test_gaussiano_bate_com_sklearn():
    gerador = np.random.default_rng(1)
    X = gerador.normal(size=(80, 4))
    y = np.array([0] * 40 + [1] * 40)
    X[:40] += np.array([1.0, -1.0, 0.2, 0.0])

    nosso = GaussianNB(var_smoothing=1e-9).fit(X, y)
    deles = SkGaussian(var_smoothing=1e-9).fit(X, y)

    np.testing.assert_allclose(nosso.theta_, deles.theta_, atol=1e-10)
    np.testing.assert_allclose(nosso.var_, deles.var_, atol=1e-10)
    np.testing.assert_allclose(nosso.class_prior_, deles.class_prior_, atol=1e-10)
    consulta = gerador.normal(size=(20, 4))
    np.testing.assert_allclose(
        nosso.predict_log_proba(consulta),
        deles.predict_log_proba(consulta),
        atol=1e-8,
    )
    np.testing.assert_array_equal(nosso.predict(consulta), deles.predict(consulta))


def test_gaussiano_com_rotulos_texto():
    X = np.array([[0.0], [0.1], [3.0], [3.2]])
    y = np.array(["a", "a", "b", "b"])
    modelo = GaussianNB().fit(X, y)
    assert modelo.predict([[0.0]])[0] == "a"
    assert modelo.predict([[4.0]])[0] == "b"
