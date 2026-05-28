"""Структура BM25-индекса и формула ранжирования.

Что внутри:
  BM25Index — все данные индекса (vocab, doc_ids, kinds, tf-CSR, doc_lens, df).
  score()   — векторизованный расчёт BM25-скоров для запроса.

ФОРМУЛА BM25 Okapi (одно слагаемое для документа d, термина t):

                     idf(t) · tf(t,d) · (k1 + 1)
    score(t, d) = ─────────────────────────────────────
                  tf(t,d) + k1 · (1 − b + b · |d|/avgdl)

Полный score документа = сумма по терминам запроса.
  tf(t, d)   — сколько раз термин t в документе d
  idf(t)     = log((N − df + 0.5) / (df + 0.5) + 1)
  df(t)      — в скольких документах термин t
  |d|, avgdl — длина документа и средняя длина по корпусу
  k1, b      — гиперпараметры (1.5 и 0.75 по умолчанию)

ПОЧЕМУ scipy.sparse, А НЕ ОБЫЧНЫЙ numpy:
Матрица tf имеет форму [N_docs × V_vocab]. У нас обычно 99% нулей: один
документ упоминает крошечную долю всего словаря. Плотная матрица съела бы
сотни МБ; CSR — единицы МБ.

CSR — быстрая по строкам (документам), нужна при сборке.
CSC — быстрая по столбцам (терминам), нужна при поиске. Конвертация
ленивая, кэшируется в BM25Index._tf_csc.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


INDEX_VERSION = 5
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


@dataclass
class BM25Index:
    """Все структуры BM25-индекса, нужные для поиска."""
    # {лемма: индекс_столбца} в матрице tf.
    vocab: dict[str, int]
    # ID документов в порядке строк tf. doc_ids[i] = id i-го документа.
    doc_ids: list[str]
    # Тип каждого документа (publication / element); параллельно doc_ids.
    kinds: list[str]
    # pub_id-владелец каждого документа; параллельно doc_ids.
    # Нужен для масок типа "элементы только из других публикаций" в coverage,
    # чтобы не лазить за этим в documents (тяжело при ленивой загрузке).
    pub_ids: list[str]
    # tf[i, j] = частота термина j в документе i.
    tf: csr_matrix
    # Длины документов в токенах, int32[N].
    doc_lens: np.ndarray
    # df[j] = document frequency термина j, int32[V].
    df: np.ndarray
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    # Ленивый кэш CSC-версии tf. repr=False, чтобы не печатать при отладке.
    _tf_csc: csc_matrix | None = field(default=None, repr=False)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def avgdl(self) -> float:
        return float(self.doc_lens.mean()) if self.n_docs else 0.0

    @property
    def idf(self) -> np.ndarray:
        # Не сохраняем на диск; пересчёт быстрый и автоматически
        # консистентен с актуальными df/N. «+1» — стандартная поправка,
        # гарантирующая неотрицательный idf для очень частых терминов.
        N = self.n_docs
        return np.log((N - self.df + 0.5) / (self.df + 0.5) + 1.0).astype(np.float32)

    @property
    def tf_csc(self) -> csc_matrix:
        if self._tf_csc is None:
            self._tf_csc = self.tf.tocsc()
        return self._tf_csc


def score(index: BM25Index, query_tokens: list[str]) -> np.ndarray:
    """Оценить ВСЕ документы по списку лемм запроса. Возвращает np.array длиной N_docs.

    Векторизация: цикл по терминам запроса (их обычно единицы), внутри —
    одно матрично-векторное обновление scores[rows] += contrib.
    OOV-термины (нет в vocab) тихо пропускаются.
    """
    if index.n_docs == 0:
        return np.zeros(0, dtype=np.float32)

    term_ids = [index.vocab[t] for t in query_tokens if t in index.vocab]
    if not term_ids:
        return np.zeros(index.n_docs, dtype=np.float32)

    k1, b = index.k1, index.b
    avgdl = index.avgdl or 1.0
    # norm[i] = 1 − b + b · |d_i|/avgdl  — нормализатор по длине документа.
    norm = (1.0 - b) + b * (index.doc_lens.astype(np.float32) / avgdl)
    idf = index.idf

    scores = np.zeros(index.n_docs, dtype=np.float32)
    tf_csc = index.tf_csc
    for t_id in term_ids:
        col = tf_csc[:, t_id]
        if col.nnz == 0:
            continue
        rows = col.indices                       # документы, где термин есть
        tf_vals = col.data.astype(np.float32)    # их tf(t, d)
        denom = tf_vals + k1 * norm[rows]
        contrib = idf[t_id] * tf_vals * (k1 + 1.0) / denom
        scores[rows] += contrib
    return scores
