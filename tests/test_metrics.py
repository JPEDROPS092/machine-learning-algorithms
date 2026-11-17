"""Metricas do zero comparadas as do scikit-learn."""

import numpy as np
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall
from sklearn.metrics import roc_auc_score as sk_auc

from mlalgos import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def test_matriz_e_medidas_binarias():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))
    assert precision_score(y_true, y_pred) == sk_precision(y_true, y_pred, zero_division=0)
    assert recall_score(y_true, y_pred) == sk_recall(y_true, y_pred, zero_division=0)
    assert f1_score(y_true, y_pred) == sk_f1(y_true, y_pred, zero_division=0)


def test_macro_e_weighted_multiclasse():
    y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1])
    y_pred = np.array([0, 2, 2, 1, 1, 0, 2, 0])
    for media in ("macro", "weighted"):
        assert precision_score(y_true, y_pred, average=media) == sk_precision(
            y_true, y_pred, average=media, zero_division=0
        )
        assert recall_score(y_true, y_pred, average=media) == sk_recall(
            y_true, y_pred, average=media, zero_division=0
        )
        assert np.isclose(
            f1_score(y_true, y_pred, average=media),
            sk_f1(y_true, y_pred, average=media, zero_division=0),
        )


def test_roc_auc_binaria_e_multiclasse():
    gerador = np.random.default_rng(4)
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    escore = np.array([0.1, 0.4, 0.35, 0.8, 0.7, 0.9, 0.2, 0.05])
    # Inclui empate para exercitar o posto medio.
    escore_empate = escore.copy()
    escore_empate[1] = 0.35
    assert np.isclose(roc_auc_score(y_true, escore), sk_auc(y_true, escore))
    assert np.isclose(roc_auc_score(y_true, escore_empate), sk_auc(y_true, escore_empate))

    y_multi = np.array([0, 1, 2, 1, 0, 2])
    proba = gerador.random((6, 3))
    proba = proba / proba.sum(axis=1, keepdims=True)
    nosso = roc_auc_score(y_multi, proba, average="macro")
    deles = sk_auc(y_multi, proba, multi_class="ovr", average="macro")
    assert np.isclose(nosso, deles)


def test_divisao_por_zero_devolve_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert precision_score(y_true, y_pred, pos_label=1) == 0.0
    assert recall_score(y_true, y_pred, pos_label=1) == 0.0
    assert f1_score(y_true, y_pred, pos_label=1) == 0.0
