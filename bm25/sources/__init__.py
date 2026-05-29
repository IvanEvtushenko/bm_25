"""Источники документов: CSV (локально) и SQL/Spark (прод).

Любой Source предоставляет два метода:
  kind()      -> str           # 'decision' или 'attribute'
  iter_rows() -> Iterator[dict] # строки с каноническими именами колонок

После Source различия CSV/SQL заканчиваются: вышестоящий код (add_doc.py,
builder) работает с одним общим потоком.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

from .csv_source import CSVSource
from .sql_source import ID_PLACEHOLDER, SQLSource

Source = Union[CSVSource, SQLSource]


def create_source(
    path: Path,
    kind_override: str | None = None,
    batch_ids: list | None = None,
    batch_size: int = 500,
) -> Source:
    """Подобрать источник по расширению файла.

    batch_ids / batch_size актуальны только для .sql-источников с
    плейсхолдером {id_filter} в WHERE. Для CSV игнорируются.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return CSVSource(path, kind_override=kind_override)
    if suffix == ".sql":
        return SQLSource(
            path,
            kind_override=kind_override,
            batch_ids=batch_ids,
            batch_size=batch_size,
        )
    raise ValueError(
        f"unsupported source extension: {path.suffix}. "
        f"Supported: .csv, .sql (use JSONL pre-normalized via separate path)."
    )


__all__ = ["Source", "CSVSource", "SQLSource", "create_source", "ID_PLACEHOLDER"]
