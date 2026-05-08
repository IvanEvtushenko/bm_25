from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, save_npz, load_npz

from bm25_ru import Tokenizer

INDEX_VERSION = 3
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

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
    vocab: dict[str, int]
    doc_ids: list[str]
    kinds: list[str]
    tf: csr_matrix
    doc_lens: np.ndarray
    df: np.ndarray
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B
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
        N = self.n_docs
        return np.log((N - self.df + 0.5) / (self.df + 0.5) + 1.0).astype(np.float32)

    @property
    def tf_csc(self) -> csc_matrix:
        if self._tf_csc is None:
            self._tf_csc = self.tf.tocsc()
        return self._tf_csc


def _row_from_counts(counts: Counter[str], vocab: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    cols, vals = [], []
    for term, cnt in counts.items():
        idx = vocab.get(term)
        if idx is None:
            idx = len(vocab)
            vocab[term] = idx
        cols.append(idx)
        vals.append(cnt)
    return np.asarray(cols, dtype=np.int32), np.asarray(vals, dtype=np.int32)


def _doc_text(doc: dict) -> str:
    fields = index_fields_for(doc["kind"])
    return " ".join(str(doc.get(f, "") or "") for f in fields)


def build_from_documents(
    documents: list[dict],
    tokenizer: Tokenizer,
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25Index:
    vocab: dict[str, int] = {}
    doc_ids: list[str] = []
    kinds: list[str] = []
    doc_lens: list[int] = []
    rows_data: list[np.ndarray] = []
    rows_cols: list[np.ndarray] = []

    for doc in documents:
        tokens = tokenizer(_doc_text(doc))
        counts = Counter(tokens)
        cols, vals = _row_from_counts(counts, vocab)
        doc_ids.append(doc["doc_id"])
        kinds.append(doc["kind"])
        doc_lens.append(len(tokens))
        rows_cols.append(cols)
        rows_data.append(vals)

    V = len(vocab)
    indptr = np.zeros(len(documents) + 1, dtype=np.int64)
    for i, cols in enumerate(rows_cols):
        indptr[i + 1] = indptr[i] + len(cols)
    indices = np.concatenate(rows_cols) if rows_cols else np.zeros(0, dtype=np.int32)
    data = np.concatenate(rows_data) if rows_data else np.zeros(0, dtype=np.int32)
    tf = csr_matrix((data, indices, indptr), shape=(len(documents), V), dtype=np.int32)

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
    if index.n_docs == 0:
        return np.zeros(0, dtype=np.float32)
    term_ids = [index.vocab[t] for t in query_tokens if t in index.vocab]
    if not term_ids:
        return np.zeros(index.n_docs, dtype=np.float32)

    k1, b = index.k1, index.b
    avgdl = index.avgdl or 1.0
    norm = (1.0 - b) + b * (index.doc_lens.astype(np.float32) / avgdl)
    idf = index.idf

    scores = np.zeros(index.n_docs, dtype=np.float32)
    tf_csc = index.tf_csc
    for t_id in term_ids:
        col = tf_csc[:, t_id]
        if col.nnz == 0:
            continue
        rows = col.indices
        tf_vals = col.data.astype(np.float32)
        denom = tf_vals + k1 * norm[rows]
        contrib = idf[t_id] * tf_vals * (k1 + 1.0) / denom
        scores[rows] += contrib
    return scores


@dataclass
class Searcher:
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
            return []
        s = score(self.index, q_tokens)
        if s.size == 0:
            return []
        if kind is not None:
            mask = np.asarray([k == kind for k in self.index.kinds], dtype=bool)
            s = np.where(mask, s, 0.0)
        nz = s > 0
        if not nz.any():
            return []
        k = min(top_k, int(nz.sum()))
        candidate_idx = np.flatnonzero(nz)
        if candidate_idx.size > k:
            partition = np.argpartition(-s[candidate_idx], kth=k - 1)[:k]
            candidate_idx = candidate_idx[partition]
        order = candidate_idx[np.argsort(-s[candidate_idx])]
        return [(float(s[i]), self.documents[self.index.doc_ids[i]]) for i in order]


# --- persistence ---------------------------------------------------------

def save_index(index_dir: Path, index: BM25Index) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    save_npz(index_dir / "tf.npz", index.tf)
    np.save(index_dir / "doc_lens.npy", index.doc_lens)
    np.save(index_dir / "df.npy", index.df)
    (index_dir / "vocab.json").write_text(
        json.dumps(index.vocab, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "doc_ids.json").write_text(
        json.dumps(index.doc_ids, ensure_ascii=False), encoding="utf-8"
    )
    (index_dir / "kinds.json").write_text(
        json.dumps(index.kinds, ensure_ascii=False), encoding="utf-8"
    )
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


# --- jsonl helpers -------------------------------------------------------

def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_documents_map(jsonl_path: Path) -> dict[str, dict]:
    return {doc["doc_id"]: doc for doc in read_jsonl(jsonl_path)}


def open_searcher(data_dir: Path) -> Searcher:
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    if not (index_dir / "meta.json").exists():
        raise FileNotFoundError(f"index not found in {index_dir}; run add_doc.py first")
    index = load_index(index_dir)
    documents = load_documents_map(docs_path)
    return Searcher(index=index, documents=documents, tokenizer=Tokenizer())


def rebuild_and_save(data_dir: Path, tokenizer: Tokenizer | None = None) -> BM25Index:
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    documents = list(read_jsonl(docs_path))
    tok = tokenizer or Tokenizer()
    index = build_from_documents(documents, tok)
    save_index(index_dir, index)
    return index
