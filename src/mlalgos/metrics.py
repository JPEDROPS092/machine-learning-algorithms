"""Matriz de confusao, precisao, revocacao, F1 e ROC-AUC.

Tudo em NumPy, para o caderno mostrar de onde sai cada numero. A area
sob a ROC usa o estadistico de Wilcoxon-Mann-Whitney, que trata empate
de escore com posto medio e coincide com o scikit-learn em dados comuns.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None):
    """Linhas sao a classe verdadeira, colunas a classe prevista."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true e y_pred precisam ter o mesmo tamanho.")
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    else:
        labels = np.asarray(labels)
    indice = {classe: i for i, classe in enumerate(labels)}
    matriz = np.zeros((len(labels), len(labels)), dtype=int)
    for verdadeiro, previsto in zip(y_true, y_pred):
        if verdadeiro in indice and previsto in indice:
            matriz[indice[verdadeiro], indice[previsto]] += 1
    return matriz


def _rotulos_presentes(y_true, y_pred):
    return np.unique(np.concatenate([np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)]))


def _contagens(y_true, y_pred, classe):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    verdadeiro_pos = y_true == classe
    previsto_pos = y_pred == classe
    tp = int(np.sum(verdadeiro_pos & previsto_pos))
    fp = int(np.sum(~verdadeiro_pos & previsto_pos))
    fn = int(np.sum(verdadeiro_pos & ~previsto_pos))
    suporte = int(np.sum(verdadeiro_pos))
    return tp, fp, fn, suporte


def _dividir(numerador, denominador):
    if denominador == 0:
        return 0.0
    return float(numerador) / float(denominador)


def precision_score(y_true, y_pred, average="binary", pos_label=1):
    return _media_por_classe(y_true, y_pred, average, pos_label, _precisao_de)


def recall_score(y_true, y_pred, average="binary", pos_label=1):
    return _media_por_classe(y_true, y_pred, average, pos_label, _revocacao_de)


def f1_score(y_true, y_pred, average="binary", pos_label=1):
    return _media_por_classe(y_true, y_pred, average, pos_label, _f1_de)


def _precisao_de(y_true, y_pred, classe):
    tp, fp, _, _ = _contagens(y_true, y_pred, classe)
    return _dividir(tp, tp + fp)


def _revocacao_de(y_true, y_pred, classe):
    tp, _, fn, _ = _contagens(y_true, y_pred, classe)
    return _dividir(tp, tp + fn)


def _f1_de(y_true, y_pred, classe):
    precisao = _precisao_de(y_true, y_pred, classe)
    revocacao = _revocacao_de(y_true, y_pred, classe)
    return _dividir(2.0 * precisao * revocacao, precisao + revocacao)


def _media_por_classe(y_true, y_pred, average, pos_label, funcao):
    if average == "binary":
        return funcao(y_true, y_pred, pos_label)
    classes = _rotulos_presentes(y_true, y_pred)
    valores = np.array([funcao(y_true, y_pred, classe) for classe in classes], dtype=float)
    if average == "macro":
        return float(np.mean(valores)) if len(valores) else 0.0
    if average == "weighted":
        suportes = np.array([_contagens(y_true, y_pred, classe)[3] for classe in classes], dtype=float)
        total = suportes.sum()
        if total == 0:
            return 0.0
        return float(np.sum(valores * suportes) / total)
    raise ValueError("average precisa ser 'binary', 'macro' ou 'weighted'.")


def _binario(y_true, pos_label):
    y = np.asarray(y_true).reshape(-1)
    return (y == pos_label).astype(np.int64)


def _postos_medios(escores):
    """Posto 1-based com media nos empates, na ordem original."""
    escores = np.asarray(escores, dtype=float)
    ordem = np.argsort(escores, kind="mergesort")
    ordenados = escores[ordem]
    postos = np.empty(len(escores), dtype=float)
    i = 0
    n = len(escores)
    while i < n:
        j = i
        while j + 1 < n and ordenados[j + 1] == ordenados[i]:
            j += 1
        # Postos de i+1 ate j+1; a media e o centro desse intervalo.
        medio = (i + j) / 2.0 + 1.0
        postos[ordem[i : j + 1]] = medio
        i = j + 1
    return postos


def _auc_binario(y_true, y_score, pos_label):
    y = _binario(y_true, pos_label)
    escore = np.asarray(y_score, dtype=float).reshape(-1)
    if y.shape[0] != escore.shape[0]:
        raise ValueError("y_true e y_score precisam ter o mesmo tamanho.")
    n_pos = int(y.sum())
    n_neg = int(y.shape[0] - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC-AUC precisa das duas classes no y_true.")
    postos = _postos_medios(escore)
    soma_pos = postos[y == 1].sum()
    return float((soma_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_score(y_true, y_score, pos_label=1, average=None):
    """Area sob a curva ROC.

    ``y_score`` pode ser um vetor (binario) ou uma matriz de probabilidades
    com uma coluna por classe, na ordem de ``np.unique(y_true)``. No caso
    multiclasse, ``average='macro'`` faz a media simples das AUCs um-contra-o-resto.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim == 1:
        return _auc_binario(y_true, y_score, pos_label)
    classes = np.unique(y_true)
    if y_score.shape != (y_true.shape[0], len(classes)):
        raise ValueError(
            "Para multiclasse, y_score deve ter uma coluna por classe, na ordem de np.unique(y_true)."
        )
    if average not in (None, "macro"):
        raise ValueError("No caso multiclasse use average='macro'.")
    areas = [_auc_binario(y_true, y_score[:, i], classe) for i, classe in enumerate(classes)]
    return float(np.mean(areas))


def curva_roc(y_true, y_score, pos_label=1):
    """Devolve taxa de falso positivo, taxa de verdadeiro positivo e limiares.

    O primeiro ponto e (0, 0), quando o limiar esta acima de todo escore.
    """
    y = _binario(y_true, pos_label)
    escore = np.asarray(y_score, dtype=float).reshape(-1)
    if y.shape[0] != escore.shape[0]:
        raise ValueError("y_true e y_score precisam ter o mesmo tamanho.")
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("A curva ROC precisa das duas classes.")

    ordem = np.argsort(-escore, kind="mergesort")
    y = y[ordem]
    escore = escore[ordem]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)

    # Um limiar por valor distinto de escore. O ultimo indice de cada grupo empatado.
    if len(escore) == 1:
        cortes = np.array([0])
    else:
        muda = np.where(np.diff(escore) != 0)[0]
        cortes = np.r_[muda, len(escore) - 1]
    tpr = tp[cortes] / n_pos
    fpr = fp[cortes] / n_neg
    limiares = escore[cortes]
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]
    limiares = np.r_[limiares[0] + 1.0, limiares]
    return fpr, tpr, limiares
