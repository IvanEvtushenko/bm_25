"""Поисковый интерфейс: Searcher + точки входа для CLI.

Searcher = индекс + LazyDocuments + токенизатор. Документы НЕ грузятся
в RAM целиком — подтягиваются с диска по offset при обращении (typically
5–10 документов на запрос).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .builder import build_streaming_from_jsonl
from .index import BM25Index, score
from .storage import (
    LazyDocuments,
    load_index,
    load_offsets,
    save_index,
)
from .tokenizer import Tokenizer


@dataclass
class Searcher:
    index: BM25Index
    documents: LazyDocuments
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


def open_searcher(data_dir: Path) -> Searcher:
    """Точка входа для query.py / coverage.py.

    Грузит индекс целиком в RAM, но docs.jsonl НЕ грузит — открывает только
    дескриптор файла, поднимая в память лишь offsets и doc_id→row маппинг.
    """
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    if not (index_dir / "meta.json").exists():
        raise FileNotFoundError(f"index not found in {index_dir}; run add_doc.py first")
    index = load_index(index_dir)
    offsets = load_offsets(index_dir)
    documents = LazyDocuments(docs_path, index.doc_ids, offsets)
    return Searcher(index=index, documents=documents, tokenizer=Tokenizer())


def rebuild_and_save(data_dir: Path, tokenizer: Tokenizer | None = None) -> BM25Index:
    """Точка входа для add_doc.py: пересборка индекса из docs.jsonl + offsets.

    Стриминговая сборка (один проход по файлу), параллельно вычисляются
    offsets каждой строки и сохраняются как offsets.npy для lazy-load.
    """
    docs_path = data_dir / "docs.jsonl"
    index_dir = data_dir / "index"
    tok = tokenizer or Tokenizer()
    index, offsets = build_streaming_from_jsonl(docs_path, tok)
    save_index(index_dir, index, offsets=offsets)
    return index
