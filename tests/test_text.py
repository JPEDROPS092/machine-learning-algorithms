"""Saco de palavras e TF-IDF alinhados ao CountVectorizer e ao TfidfVectorizer."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from mlalgos import BagOfWords, TfidfVectorizer as NossoTfidf
from mlalgos.text import tokenize


DOCUMENTOS = [
    "O gato subiu no telhado e o gato miou",
    "o cachorro latiu para o gato",
    "chuva fina em Manaus",
    "gato e cachorro dividem o telhado",
]


def test_tokenize_minusculas_e_acentos():
    assert tokenize("Não, o Gato!") == ["não", "gato"]
    assert tokenize("") == []


def test_saco_de_palavras_bate_com_sklearn():
    nosso = BagOfWords().fit_transform(DOCUMENTOS)
    deles = CountVectorizer().fit_transform(DOCUMENTOS).toarray()
    np.testing.assert_allclose(nosso, deles)


def test_tfidf_bate_com_sklearn():
    nosso = NossoTfidf().fit_transform(DOCUMENTOS)
    deles = TfidfVectorizer().fit_transform(DOCUMENTOS).toarray()
    np.testing.assert_allclose(nosso, deles, atol=1e-10)
    normas = np.linalg.norm(nosso, axis=1)
    np.testing.assert_allclose(normas, 1.0, atol=1e-10)


def test_max_features_limita_colunas():
    matriz = BagOfWords(max_features=3).fit_transform(DOCUMENTOS)
    assert matriz.shape[1] == 3


def test_palavra_desconhecida_vira_zero():
    modelo = BagOfWords().fit(DOCUMENTOS)
    linha = modelo.transform(["zumbido inedito"])
    assert linha.shape == (1, len(modelo.vocabulary_))
    assert linha.sum() == 0
