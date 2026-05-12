"""BM25 поиск на numpy + scipy.sparse.

ЧТО ТАКОЕ BM25 (в двух абзацах)
================================
BM25 — алгоритм ранжирования документов по релевантности к запросу-строке.
Идея: для каждого слова из запроса посчитать «вес этого слова в этом
документе» и просуммировать веса по всем словам. Чем сумма больше — тем
документ релевантнее. Никакой нейросети, никаких эмбеддингов; алгоритм
из 1994 года, до сих пор хорош как baseline и для классики веб-поиска.

Формула одного слагаемого (BM25 Okapi, для документа d и термина t):

                     idf(t) · tf(t,d) · (k1 + 1)
    score(t, d) = ─────────────────────────────────────
                  tf(t,d) + k1 · (1 − b + b · |d|/avgdl)

Полный score документа = сумма по всем терминам запроса.

Что значат компоненты:
  tf(t, d)   — сколько раз термин t встречается в документе d (term frequency).
               Например, «инфляция» в публикации про инфляцию встречается
               часто — score растёт. Но в формуле есть ограничение
               насыщения через k1: с какого-то момента «больше = больше»
               уже не работает (5 раз и 50 раз дают похожий вес).
  df(t)      — в скольких документах вообще встречается термин (document freq).
  idf(t)     = log((N − df + 0.5) / (df + 0.5) + 1).
               «редкие термины важнее частых». «инфляция» встречается в
               сотнях документов — низкий idf. «ОФЗ-ИН» — в единицах,
               высокий idf.
  |d|        — длина документа d в токенах.
  avgdl      — средняя длина документа по всему индексу.
  k1, b      — гиперпараметры. k1 (обычно 1.5) контролирует насыщение
               по tf. b (обычно 0.75) — насколько штрафовать длинные
               документы. b=0 = не нормализовать длину, b=1 = полная норма.

КАК ЭТО ХРАНИТСЯ
================
Главная структура — term-document матрица tf (см. ниже про CSR/CSC).
Дополнительно: словарь {термин: индекс_столбца}, длины документов,
df для каждого термина. idf и avgdl пересчитываются из них при загрузке.

ПОЧЕМУ scipy.sparse, А НЕ ОБЫЧНЫЙ numpy
========================================
Term-document матрица имеет размер [N_docs × V_vocab]. У нас:
N=16353 документа, V=4612 терминов. Это 75M ячеек.
Но 99% ячеек — нули (один документ упоминает малую часть всего словаря).

Если хранить плотно (np.array): 75M × 4 байта = 300 МБ только под tf.
Если разреженно (scipy.sparse.csr_matrix): хранятся только ненулевые
значения и их индексы — у нас это единицы МБ.

Внутри CSR-матрицы лежат три обычных numpy-массива (data/indices/indptr),
поэтому scipy.sparse — это не отдельная вселенная, а формальное
расширение numpy. Дублирования нет.

CSR vs CSC:
  CSR (Compressed Sparse Row) — быстрый доступ по строкам.
  CSC (Compressed Sparse Column) — быстрый доступ по столбцам.
Мы строим индекс как CSR (документы = строки), но при поиске нам
нужен быстрый доступ по столбцам (термины), поэтому при первом поиске
конвертируем в CSC (один раз, лениво, и кэшируем).

ТИПЫ ДОКУМЕНТОВ
===============
В одном индексе живут документы двух типов:
  publication  — целая публикация ЦБ (из pub_report.csv)
  element      — отдельный показатель публикации (из Показатели.csv)

Каждый документ имеет:
  doc_id   — глобально уникальный ключ ("pub:Аналитика/22" или "el:.../#42")
  kind     — "publication" или "element"
  pub_id   — связь с родительской публикацией (для элементов)
  + поля для индексации, специфичные для типа.

Какие поля индексировать для каждого типа — задаётся в INDEX_FIELDS_BY_KIND.
Это единственное место, которое меняется при смене схемы CSV.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
# scipy.sparse:
#   csr_matrix — формат «по строкам», быстро искать строку (документ).
#   csc_matrix — формат «по столбцам», быстро искать столбец (термин).
#   save_npz/load_npz — встроенная сериализация sparse-матриц в один файл.
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz

from bm25_ru import Tokenizer

# Версия формата индекса на диске. Меняется, если меняется схема файлов
# в data/index/. Используется в meta.json — при будущей миграции
# можно проверить и переломать совместимость осознанно.
INDEX_VERSION = 3

# Стандартные параметры BM25. Можно подкручивать на конкретном корпусе,
# но для большинства задач эти значения — здоровый дефолт.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

# Какие поля индексировать для каждого типа документа.
# При смене схемы CSV — правьте этот словарь и пересоберите индекс.
# Поля склеиваются через пробел в один текст, потом он токенизируется.
INDEX_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "publication": ("name", "business_description", "keywords"),
    "element": ("report_name", "name", "description"),
}


def index_fields_for(kind: str) -> tuple[str, ...]:
    if kind not in INDEX_FIELDS_BY_KIND:
        raise KeyError(f"unknown kind={kind!r}; known: {sorted(INDEX_FIELDS_BY_KIND)}")
    return INDEX_FIELDS_BY_KIND[kind]


@dataclass
class BM25Index:
    """Все структуры BM25-индекса, нужные для поиска.

    Размерности (для нашего корпуса в 16353 документа):
      vocab     ≈ 4612 пар {термин: column_idx}
      doc_ids   = 16353 строки (id документа в порядке строк tf)
      kinds     = 16353 строки (тип каждого документа)
      tf        = разреженная матрица [16353 × 4612], int32, ~единицы МБ
      doc_lens  = 16353 числа int32 (длина документа в токенах)
      df        = 4612 чисел int32 (в скольких документах встречается термин)
    """
    # term → индекс столбца в матрице tf. Используем обычный dict, не numpy
    # (это словарь строк, для них numpy не подходит).
    vocab: dict[str, int]
    # Идентификаторы документов в порядке строк tf. doc_ids[i] — id i-го документа.
    doc_ids: list[str]
    # Параллельный список kinds[i] — тип i-го документа (publication/element).
    # Нужен для фильтрации выдачи по типу.
    kinds: list[str]
    # tf[i, j] = сколько раз термин j встретился в документе i.
    # CSR — потому что при сборке мы заполняем построчно (документ за документом).
    tf: csr_matrix
    # doc_lens[i] — длина документа i в токенах. np.array нужен для
    # векторизованного вычисления нормализатора 1−b+b*|d|/avgdl.
    doc_lens: np.ndarray
    # df[j] — document frequency термина j (в скольких документах встречается).
    df: np.ndarray
    # Параметры BM25.
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
    # Кэш для CSC-версии матрицы. CSC быстро ходит по столбцам, что нужно
    # при поиске. Считается лениво на первом обращении (см. property).
    # repr=False, чтобы dataclass не пытался печатать матрицу при отладке.
    _tf_csc: csc_matrix | None = field(default=None, repr=False)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def avgdl(self) -> float:
        # Средняя длина документа в токенах. Нужна для BM25-нормализации.
        # При пустом индексе вернём 0, чтобы не делить на ноль выше.
        return float(self.doc_lens.mean()) if self.n_docs else 0.0

    @property
    def idf(self) -> np.ndarray:
        # IDF не сохраняем на диск — пересчитываем из df и N.
        # Это и быстро, и автоматически консистентно: после добавления
        # документов df и N меняются → idf пересчитается на лету.
        # Формула: log((N − df + 0.5) / (df + 0.5) + 1).
        # «+1» гарантирует положительный результат (классический BM25
        # допускал отрицательные idf для очень частых терминов; +1 это правит).
        N = self.n_docs
        return np.log((N - self.df + 0.5) / (self.df + 0.5) + 1.0).astype(np.float32)

    @property
    def tf_csc(self) -> csc_matrix:
        # Ленивая конвертация CSR → CSC при первом обращении.
        # Сама конвертация — O(nnz) и довольно быстрая.
        if self._tf_csc is None:
            self._tf_csc = self.tf.tocsc()
        return self._tf_csc


def _row_from_counts(counts: Counter[str], vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Превращает Counter лемм в пару (column_indices, tf_values) для одной строки tf-матрицы.

    Если термин впервые встретился во всём корпусе — он автоматически
    добавляется в vocab с очередным индексом. Так vocab растёт по мере
    обхода документов.
    """
    cols, vals = [], []
    for term, cnt in counts.items():
        idx = vocab.get(term)
        if idx is None:
            idx = len(vocab)
            vocab[term] = idx
        cols.append(idx)
        vals.append(cnt)
    # int32 хватает: 16к документов и десятки тыс. лемм — ничего не переполнится.
    return np.asarray(cols, dtype=np.int32), np.asarray(vals, dtype=np.int32)


def _doc_text(doc: dict) -> str:
    """Склейка индексируемых полей документа в один текст (через пробел).

    Какие поля брать — зависит от kind документа. См. INDEX_FIELDS_BY_KIND.
    """
    fields = index_fields_for(doc["kind"])
    return " ".join(str(doc.get(f, "") or "") for f in fields)


def build_from_documents(
    documents: list[dict],
    tokenizer: Tokenizer,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25Index:
    """Главная функция сборки индекса.

    Алгоритм:
      1. Идём по документам по порядку.
      2. Для каждого: склеиваем индексируемые поля → токенизируем → считаем
         частоты лемм через collections.Counter.
      3. Эти частоты дают одну строку матрицы tf.
      4. Параллельно ведём vocab (словарь термин→индекс), doc_lens, doc_ids.
      5. После всех документов собираем CSR-матрицу из накопленных строк.
      6. Считаем df через (tf > 0).sum(axis=0): по каждому столбцу — сколько
         документов содержат этот термин.
    """
    vocab: dict[str, int] = {}
    doc_ids: list[str] = []
    kinds: list[str] = []
    doc_lens: list[int] = []
    # Накапливаем будущие строки матрицы как пары numpy-массивов.
    # Так можно быстро собрать CSR одним вызовом, без пошагового изменения матрицы.
    rows_data: list[np.ndarray] = []
    rows_cols: list[np.ndarray] = []

    for doc in documents:
        tokens = tokenizer(_doc_text(doc))
        # Counter({"банк": 5, "кредит": 3, ...})
        counts = Counter(tokens)
        cols, vals = _row_from_counts(counts, vocab)
        doc_ids.append(doc["doc_id"])
        kinds.append(doc["kind"])
        doc_lens.append(len(tokens))
        rows_cols.append(cols)
        rows_data.append(vals)

    # ---- Сборка CSR-матрицы из накопленных строк -------------------------
    # CSR хранит три массива:
    #   data    — все ненулевые значения подряд (по строкам).
    #   indices — соответствующие им номера столбцов.
    #   indptr  — где начинается каждая строка в data/indices.
    # У строки i элементы лежат в data[indptr[i]:indptr[i+1]]
    # с column-индексами indices[indptr[i]:indptr[i+1]].
    V = len(vocab)
    indptr = np.zeros(len(documents) + 1, dtype=np.int64)
    for i, cols in enumerate(rows_cols):
        indptr[i + 1] = indptr[i] + len(cols)
    indices = np.concatenate(rows_cols) if rows_cols else np.zeros(0, dtype=np.int32)
    data = np.concatenate(rows_data) if rows_data else np.zeros(0, dtype=np.int32)
    tf = csr_matrix((data, indices, indptr), shape=(len(documents), V), dtype=np.int32)

    # df[j] = сколько документов содержат термин j.
    # (tf > 0) — sparse boolean matrix. .sum(axis=0) по разреженной — суммирует
    # по столбцам. Возвращает np.matrix формы (1, V), .ravel() → одномерный массив.
    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.int32)
    return BM25Index(
        vocab=vocab,
        doc_ids=doc_ids,
        kinds=kinds,
        tf=tf,
        doc_lens=np.asarray(doc_lens, dtype=np.int32),
        df=df,
        k1=k1,
        b=b,
    )


def score(index: BM25Index, query_tokens: list[str]) -> np.ndarray:
    """Оценить ВСЕ документы по запросу. Возвращает np.array длиной N_docs.

    Реализуем формулу BM25 векторизованно: вместо цикла по документам
    идём в цикле по терминам запроса (их обычно мало — единицы).
    Для каждого термина за один проход обновляем оценки тех документов,
    где этот термин встречается.
    """
    if index.n_docs == 0:
        return np.zeros(0, dtype=np.float32)

    # Леммы запроса → их колоночные индексы. OOV (нет в словаре) пропускаем.
    term_ids = [index.vocab[t] for t in query_tokens if t in index.vocab]
    if not term_ids:
        # Все слова запроса — OOV. Бессмысленно считать, всё будет 0.
        return np.zeros(index.n_docs, dtype=np.float32)

    k1, b = index.k1, index.b
    avgdl = index.avgdl or 1.0
    # norm[i] = 1 − b + b · |d_i| / avgdl. Это нормализатор по длине из формулы.
    # Длинные документы получат norm > 1 → в знаменателе больше → штраф.
    norm = (1.0 - b) + b * (index.doc_lens.astype(np.float32) / avgdl)
    idf = index.idf  # np.array длиной V

    # Накопитель скоров по каждому документу.
    scores = np.zeros(index.n_docs, dtype=np.float32)
    tf_csc = index.tf_csc  # CSC: быстрый доступ по столбцу=термину

    for t_id in term_ids:
        # tf_csc[:, t_id] — столбец-вектор для одного термина.
        # У него .indices — номера документов, где термин есть,
        #         .data    — соответствующие частоты tf.
        # Нули мы и так не обходим (sparse).
        col = tf_csc[:, t_id]
        if col.nnz == 0:
            # Термин в словаре есть, но ни в одном документе сейчас не встречается.
            # Бывает после удалений; для нашего пайплайна — почти никогда.
            continue
        rows = col.indices                                # документы, где термин есть
        tf_vals = col.data.astype(np.float32)            # tf(t, d) для этих документов
        # Знаменатель: tf + k1 * norm.
        # rows — индексы тех же документов, где tf_vals; норму берём по этим индексам.
        denom = tf_vals + k1 * norm[rows]
        # Числитель: idf(t) * tf * (k1 + 1) — всё векторизованно.
        contrib = idf[t_id] * tf_vals * (k1 + 1.0) / denom
        # scores[rows] += contrib — векторное прибавление в нужные позиции.
        scores[rows] += contrib

    return scores


@dataclass
class Searcher:
    """Тонкая обёртка: индекс + отображение doc_id→doc + токенизатор.

    Сам поиск — один вызов score(...) и сортировка топ-k.
    """
    index: BM25Index
    documents: dict[str, dict]
    tokenizer: Tokenizer

    def search(
        self,
        query: str,
        top_k: int = 5,
        kind: str | None = None,
    ) -> list[tuple[float, dict]]:
        q_tokens = self.tokenizer(query)
        if not q_tokens:
            # Запрос состоит только из стоп-слов / пустой.
            return []
        s = score(self.index, q_tokens)
        if s.size == 0:
            return []

        # Опциональный фильтр по типу документа: «только publication» / «только element».
        # Тут numpy-ловкость: маска булевых значений → np.where зануляет
        # скоры тех документов, чей kind не подходит. Сами скоры не пересчитываем.
        if kind is not None:
            mask = np.asarray([k == kind for k in self.index.kinds], dtype=bool)
            s = np.where(mask, s, 0.0)

        # Берём только документы с положительным скором. Если все нулевые — пусто.
        nz = s > 0
        if not nz.any():
            return []

        # Хитрый top-k без полной сортировки всего массива:
        #   1. Берём индексы ненулевых: candidate_idx (их обычно сильно меньше N).
        #   2. argpartition выбирает k наибольших значений за O(C),
        #      где C = размер candidate_idx. Не сортирует, но гарантирует,
        #      что первые k — самые большие (порядок внутри неважен).
        #   3. argsort уже среди этих k — быстро, k маленькое.
        k = min(top_k, int(nz.sum()))
        candidate_idx = np.flatnonzero(nz)
        if candidate_idx.size > k:
            partition = np.argpartition(-s[candidate_idx], kth=k - 1)[:k]
            candidate_idx = candidate_idx[partition]
        order = candidate_idx[np.argsort(-s[candidate_idx])]
        return [(float(s[i]), self.documents[self.index.doc_ids[i]]) for i in order]


# --- persistence: save/load в файлы ------------------------------------
# Источник правды (data/docs.jsonl) — отдельно. Здесь — только индекс.

def save_index(index_dir: Path, index: BM25Index) -> None:
    """Сохраняет индекс в директорию: 6 файлов, см. README."""
    index_dir.mkdir(parents=True, exist_ok=True)

    # save_npz — сохранение sparse-матрицы. Внутри это zip-архив с тремя
    # numpy-массивами (data, indices, indptr) и метаданными о форме.
    save_npz(index_dir / "tf.npz", index.tf)

    # np.save — бинарный формат numpy для одного массива (.npy).
    # Быстро читается, type-safe, кроссплатформенно.
    np.save(index_dir / "doc_lens.npy", index.doc_lens)
    np.save(index_dir / "df.npy", index.df)

    # vocab/doc_ids/kinds — структуры из строк, JSON удобнее: текстовый,
    # читаемый, не привязан к версии Python.
    (index_dir / "vocab.json").write_text(
        json.dumps(index.vocab, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "doc_ids.json").write_text(
        json.dumps(index.doc_ids, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "kinds.json").write_text(
        json.dumps(index.kinds, ensure_ascii=False), encoding="utf-8"
    )

    # meta.json — диагностика и параметры. Полезно при отладке: можно
    # просто открыть глазами и проверить, что индекс собран как ожидали.
    meta = {
        "version": INDEX_VERSION,
        "n_docs": index.n_docs,
        "vocab_size": index.vocab_size,
        "avgdl": index.avgdl,
        "k1": index.k1,
        "b": index.b,
        "fields_by_kind": {k: list(v) for k, v in INDEX_FIELDS_BY_KIND.items()},
        "kind_counts": {k: index.kinds.count(k) for k in set(index.kinds)},
    }
    (index_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_dir: Path) -> BM25Index:
    """Зеркальная процедура: читает 6 файлов и собирает BM25Index."""
    # load_npz возвращает coo_matrix; .tocsr() — приводим к нужному формату.
    tf = load_npz(index_dir / "tf.npz").tocsr()
    doc_lens = np.load(index_dir / "doc_lens.npy")
    df = np.load(index_dir / "df.npy")
    vocab = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
    doc_ids = json.loads((index_dir / "doc_ids.json").read_text(encoding="utf-8"))
    kinds = json.loads((index_dir / "kinds.json").read_text(encoding="utf-8"))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    return BM25Index(
        vocab=vocab,
        doc_ids=doc_ids,
        kinds=kinds,
        tf=tf,
        doc_lens=doc_lens,
        df=df,
        k1=float(meta.get("k1", DEFAULT_K1)),
        b=float(meta.get("b", DEFAULT_B)),
    )


# --- jsonl helpers ------------------------------------------------------
# JSONL = «JSON Lines»: одна валидная JSON-запись на строку.
# Удобный append-only формат: дописать новый документ = добавить строку.

def read_jsonl(path: Path) -> Iterable[dict]:
    """Генератор: читает JSONL-файл построчно, выдаёт dict-и.

    Генератор, а не list — экономит память на больших файлах.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    """Дописывает записи в JSONL. Возвращает их число.

    Открытие в режиме 'a' — append: ничего не перезаписываем, просто
    добавляем в конец. Это и есть наш «append-only» инвариант.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            # ensure_ascii=False — чтобы кириллица сохранялась как есть,
            # а не превращалась в ин... — занимает в 6 раз меньше места.
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_documents_map(jsonl_path: Path) -> dict[str, dict]:
    """Все документы из JSONL → {doc_id: doc}. Используется для дедупликации
    и для подтягивания полных данных документа в выдачу поиска."""
    return {doc["doc_id"]: doc for doc in read_jsonl(jsonl_path)}


# --- entry points -------------------------------------------------------

def open_searcher(data_dir: Path) -> Searcher:
    """Точка входа для query.py: загружает индекс с диска и собирает Searcher."""
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    if not (index_dir / "meta.json").exists():
        raise FileNotFoundError(f"index not found in {index_dir}; run add_doc.py first")
    index = load_index(index_dir)
    documents = load_documents_map(docs_path)
    return Searcher(index=index, documents=documents, tokenizer=Tokenizer())


def rebuild_and_save(data_dir: Path, tokenizer: Tokenizer | None = None) -> BM25Index:
    """Точка входа для add_doc.py: пересобирает индекс из docs.jsonl и сохраняет.

    Полная пересборка с нуля. На наших объёмах — несколько секунд.
    Зато гарантирует, что индекс всегда консистентен с docs.jsonl, никаких
    рассинхронизаций после неудачного инкремента.
    """
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    documents = list(read_jsonl(docs_path))
    tok = tokenizer or Tokenizer()
    index = build_from_documents(documents, tok)
    save_index(index_dir, index)
    return index
