"""Универсальная нормализация и сборка BM25Index.

После того как `bm25.sources` отдаёт поток строк-словарей с каноническими
колонками (CSV и SQL ведут себя одинаково), здесь живёт только нормализация
строки/группы в документ и сборка индекса. Никаких CSV-vs-SQL веток.

ТЕРМИНОЛОГИЯ:
  Решение (decision)  — kind="decision", doc_id="dec:<pub_id>"
  Атрибут (attribute) — kind="attribute", doc_id="attr:<pub_id>#g<group_idx>"
                                              или "attr:<pub_id>#<row_idx>" без группировки

ПРАВИЛО ИНДЕКСАЦИИ:
  В _doc_text идут ВСЕ строковые поля документа, кроме INDEX_BLACKLIST.
  Новая колонка в источнике (CSV или SQL) автоматически попадает в индекс.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
from scipy.sparse import csr_matrix

from .index import BM25Index, DEFAULT_B, DEFAULT_K1
from .tokenizer import Tokenizer


# ─── Что НЕ индексируется ───────────────────────────────────────────────────
INDEX_BLACKLIST = frozenset({
    # Служебные поля документа (выставляются нашим кодом)
    "doc_id", "kind", "pub_id", "n_elements", "element_names",
    # Идентификаторы и категориальные, общие для всех источников
    "id", "owner", "type",
    # Служебные метаданные decision (не индексируются)
    "parent_dataset", "frequency", "first_timestamp", "basis_document", "comments",
})

# Поля, у которых значение одинаково для всех атрибутов одного решения
# (это атрибуты родителя, а не самого атрибута). В группе храним их как
# скаляры из первой строки — иначе один и тот же токен дублируется N раз
# и смещает BM25-веса.
DECISION_LEVEL_FIELDS_IN_GROUP = frozenset({"report_name"})


# ─── Универсальные строитель документов ─────────────────────────────────────

def _pub_id_of(row: dict) -> str:
    """Достать pub_id из строки источника. Источник уже привёл колонку к
    канонической форме, но для SQL-результата pub_id лежит в поле `id`."""
    raw = row.get("pub_id") or row.get("id")
    if raw is None:
        raise ValueError(f"row missing pub_id/id: {row}")
    return str(raw)


def _doc_id(kind: str, pub_id: str, *, row_idx: int | None = None, group_idx: int | None = None) -> str:
    if kind == "decision":
        return f"dec:{pub_id}"
    if kind == "attribute":
        if group_idx is not None:
            return f"attr:{pub_id}#g{group_idx}"
        return f"attr:{pub_id}#{row_idx}"
    raise ValueError(f"unknown kind: {kind!r}")


def normalize_row(row: dict, kind: str, row_idx: int) -> dict:
    """Одна строка источника → один документ (без группировки).

    Все поля row сохраняются как есть (источник уже отфильтровал/переименовал
    под канонические имена). Сверху выставляем служебные doc_id/kind/pub_id.
    """
    pub_id = _pub_id_of(row)
    out: dict = {k: (v if v is not None else "") for k, v in row.items() if k != "id"}
    out["pub_id"] = pub_id
    out["kind"] = kind
    out["doc_id"] = _doc_id(kind, pub_id, row_idx=row_idx)
    return out


def normalize_group(group_rows: list[dict], kind: str, pub_id: str, group_idx: int) -> dict:
    """N подряд идущих строк одного pub_id → один документ-группа.

    Каждая колонка group_rows сохраняется как list[str] длины n_elements —
    по одному значению на каждую строку в группе. Это даёт гомогенный формат
    хранения для CSV и SQL — _doc_text для индексации и фильтр type-масок
    одинаково работают со списочными полями.

    Особый случай поля name: для attribute-групп удобно иметь и склейку через
    " | " (для отображения), и list оригинальных имён (для маски/фильтров).
    Делаем оба: name = "A | B | C", element_names = ["A", "B", "C"].
    """
    if not group_rows:
        raise ValueError("normalize_group: empty group")
    out: dict = {
        "pub_id": pub_id,
        "kind": kind,
        "doc_id": _doc_id(kind, pub_id, group_idx=group_idx),
        "n_elements": len(group_rows),
    }
    # Собираем все колонки из всех строк (на случай разных схем; обычно одинаковые).
    columns = set()
    for r in group_rows:
        columns.update(r.keys())
    SKIP = {"doc_id", "kind", "pub_id", "n_elements", "id"}
    for col in columns:
        if col in SKIP:
            continue
        if col in DECISION_LEVEL_FIELDS_IN_GROUP:
            # Скаляр от родителя; берём из первой строки группы.
            out[col] = group_rows[0].get(col, "") or ""
        else:
            out[col] = [
                "" if r.get(col) is None else str(r.get(col))
                for r in group_rows
            ]

    # Для kind=attribute name дублируем как "A | B | C" — это удобнее для
    # отображения в выдаче, чем чистый list. _doc_text всё равно индексирует
    # обе формы (через `element_names` в SKIP'е — only one of them).
    if kind == "attribute" and "name" in out:
        names_list = out["name"]
        out["element_names"] = names_list
        out["name"] = " | ".join(n for n in names_list if n)
    return out


def iter_grouped(rows: Iterable[dict], kind: str, group_size: int) -> Iterator[dict]:
    """Группирует строки по pub_id в чанки по group_size.

    Работает одинаково для любого источника — нужно лишь, чтобы строки
    одного pub_id шли подряд (CSV нативно, SQL — ORDER BY).
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    buffer: list[dict] = []
    buffer_pub_id: str | None = None
    group_idx_by_pub: dict[str, int] = {}

    def flush() -> dict | None:
        nonlocal buffer, buffer_pub_id
        if not buffer:
            return None
        gi = group_idx_by_pub.get(buffer_pub_id, 0)
        group_idx_by_pub[buffer_pub_id] = gi + 1
        doc = normalize_group(buffer, kind, buffer_pub_id, gi)
        buffer = []
        buffer_pub_id = None
        return doc

    for row in rows:
        pub_id = _pub_id_of(row)
        if buffer_pub_id is not None and pub_id != buffer_pub_id:
            doc = flush()
            if doc is not None:
                yield doc
        buffer.append(row)
        buffer_pub_id = pub_id
        if len(buffer) >= group_size:
            doc = flush()
            if doc is not None:
                yield doc

    doc = flush()
    if doc is not None:
        yield doc


# ─── _doc_text: единое правило индексации для всех kind и источников ────────

def _doc_text(doc: dict) -> str:
    """Склейка индексируемых полей документа в один текст.

    Индексируется ВСЁ строковое содержимое документа, кроме INDEX_BLACKLIST.
    Это работает и для CSV, и для SQL — никаких per-kind веток.
    """
    parts: list[str] = []
    for field_name, value in doc.items():
        if field_name in INDEX_BLACKLIST:
            continue
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(str(v) for v in value if v)
    return " ".join(parts)


# ─── Сборка индекса (in-memory и streaming) ─────────────────────────────────

def build_from_documents(
    documents: list[dict],
    tokenizer: Tokenizer,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25Index:
    return _build_core(iter(documents), tokenizer, k1, b)


def _build_core(
    docs_iter: Iterator[dict],
    tokenizer: Tokenizer,
    k1: float,
    b: float,
) -> BM25Index:
    """Один проход по итератору документов; COO-накопление → CSR в конце."""
    vocab: dict[str, int] = {}
    doc_ids: list[str] = []
    kinds: list[str] = []
    pub_ids: list[str] = []
    doc_lens: list[int] = []
    rows_buf: list[int] = []
    cols_buf: list[int] = []
    vals_buf: list[int] = []

    for row_idx, doc in enumerate(docs_iter):
        tokens = tokenizer(_doc_text(doc))
        counts = Counter(tokens)
        for term, cnt in counts.items():
            idx = vocab.get(term)
            if idx is None:
                idx = len(vocab)
                vocab[term] = idx
            rows_buf.append(row_idx)
            cols_buf.append(idx)
            vals_buf.append(cnt)
        doc_ids.append(doc["doc_id"])
        kinds.append(doc["kind"])
        pub_ids.append(str(doc.get("pub_id", "")))
        doc_lens.append(len(tokens))

    N = len(doc_ids)
    V = len(vocab)
    rows = np.asarray(rows_buf, dtype=np.int64)
    cols = np.asarray(cols_buf, dtype=np.int32)
    vals = np.asarray(vals_buf, dtype=np.int32)
    tf = csr_matrix((vals, (rows, cols)), shape=(N, V), dtype=np.int32)

    df = np.asarray((tf > 0).sum(axis=0)).ravel().astype(np.int32)
    return BM25Index(
        vocab=vocab,
        doc_ids=doc_ids,
        kinds=kinds,
        pub_ids=pub_ids,
        tf=tf,
        doc_lens=np.asarray(doc_lens, dtype=np.int32),
        df=df,
        k1=k1,
        b=b,
    )


def build_streaming_from_jsonl(
    jsonl_path: Path,
    tokenizer: Tokenizer,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> tuple[BM25Index, np.ndarray]:
    """Стрим из JSONL + параллельный расчёт offsets для lazy-load."""
    offsets: list[int] = []

    def doc_stream() -> Iterator[dict]:
        with jsonl_path.open("rb") as f:
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                offsets.append(pos)
                yield json.loads(stripped)

    index = _build_core(doc_stream(), tokenizer, k1, b)
    return index, np.asarray(offsets, dtype=np.int64)
