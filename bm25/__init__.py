"""Публичный API пакета bm25.

Терминология:
  Решение (decision)  — набор данных каталога (kind="decision").
  Атрибут (attribute) — поле/показатель решения (kind="attribute").

Слои:
  sources/       — источники: CSVSource, SQLSource (через Spark), create_source.
                   После Source различия источников заканчиваются.
  tokenizer.py   — Tokenizer (razdel + pymorphy3 + стоп-слова).
  index.py       — BM25Index (структура) + score() (формула BM25).
  builder.py     — универсальная нормализация (normalize_row / iter_grouped)
                   + единая логика индексации по INDEX_BLACKLIST + сборка.
  storage.py     — save/load индекса + LazyDocuments для JSONL.
  searcher.py    — Searcher + open_searcher / rebuild_and_save.
"""
from .builder import (
    INDEX_BLACKLIST,
    build_from_documents,
    build_streaming_from_jsonl,
    iter_grouped,
    normalize_group,
    normalize_row,
)
from .index import BM25Index, DEFAULT_B, DEFAULT_K1, INDEX_VERSION, score
from .searcher import Searcher, open_searcher, rebuild_and_save
from .sources import CSVSource, SQLSource, Source, create_source
from .storage import (
    LazyDocuments,
    append_jsonl,
    load_documents_map,
    load_index,
    load_offsets,
    read_jsonl,
    save_index,
)
from .tokenizer import RU_STOPWORDS, Tokenizer

__all__ = [
    "Tokenizer", "RU_STOPWORDS",
    "BM25Index", "score", "DEFAULT_K1", "DEFAULT_B", "INDEX_VERSION",
    "INDEX_BLACKLIST",
    "normalize_row", "normalize_group", "iter_grouped",
    "build_from_documents", "build_streaming_from_jsonl",
    "Source", "CSVSource", "SQLSource", "create_source",
    "save_index", "load_index", "load_offsets",
    "read_jsonl", "append_jsonl", "load_documents_map",
    "LazyDocuments",
    "Searcher", "open_searcher", "rebuild_and_save",
]
