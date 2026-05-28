"""Аппендер: любой источник → docs.jsonl + пересборка numpy-индекса.

Источник определяется по расширению:
  *.csv  — локальный CSV (CSVSource).
  *.sql  — SQL через Spark/Hive (SQLSource); pyspark подгружается лениво.

После Source весь пайплайн одинаков:
  source.iter_rows() → (опционально) iter_grouped → normalize_row → append → rebuild.

Чтобы добавить третий тип источника — добавьте Source-класс в bm25/sources/
и зарегистрируйте его в create_source().
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from bm25 import (
    append_jsonl,
    create_source,
    iter_grouped,
    normalize_row,
    read_jsonl,
    rebuild_and_save,
)

DATA_DIR = Path("data")
DOCS_PATH = DATA_DIR / "docs.jsonl"
INDEX_DIR = DATA_DIR / "index"


def load_existing_doc_ids(jsonl_path: Path) -> set[str]:
    """Грузим только doc_id (не полные документы) — для дедупликации.

    На больших корпусах это экономит память: set строк вместо dict-of-dicts.
    """
    if not jsonl_path.exists():
        return set()
    return {doc["doc_id"] for doc in read_jsonl(jsonl_path)}


def filter_new(records: list[dict], existing_ids: set[str]) -> tuple[list[dict], list[str]]:
    new, dupes = [], []
    seen: set[str] = set()
    for rec in records:
        doc_id = rec["doc_id"]
        if doc_id in existing_ids or doc_id in seen:
            dupes.append(doc_id)
            continue
        seen.add(doc_id)
        new.append(rec)
    return new, dupes


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Append documents and rebuild numpy BM25 index")
    parser.add_argument("source", type=Path, help="input CSV or SQL file")
    parser.add_argument(
        "--kind",
        choices=("decision", "attribute"),
        default=None,
        help="document kind (autodetected from source columns if omitted)",
    )
    parser.add_argument(
        "--group-size", type=int, default=1,
        help="для kind=attribute: склеивать N подряд идущих атрибутов одного "
             "решения в один документ-группу (default: 1 = без группировки)",
    )
    args = parser.parse_args(argv[1:])

    if not args.source.exists():
        print(f"missing input: {args.source}", file=sys.stderr)
        return 1

    source = create_source(args.source, kind_override=args.kind)
    kind = source.kind()
    print(f"source: {args.source} (kind={kind})")

    # Один путь нормализации для всех источников.
    rows = source.iter_rows()
    if kind == "attribute" and args.group_size > 1:
        records = list(iter_grouped(rows, kind, args.group_size))
        print(f"normalized {len(records)} attribute-group documents (group_size={args.group_size})")
    else:
        records = [normalize_row(row, kind, i) for i, row in enumerate(rows)]
        print(f"normalized {len(records)} {kind} documents")

    existing = load_existing_doc_ids(DOCS_PATH)
    new_records, dupes = filter_new(records, existing)
    if dupes:
        head = dupes[:5]
        more = "..." if len(dupes) > 5 else ""
        print(f"skipping {len(dupes)} duplicate doc_id(s): {head}{more}")
    if not new_records:
        print("no new records to append; index left untouched")
        return 0

    appended = append_jsonl(DOCS_PATH, new_records)
    print(f"appended {appended} records to {DOCS_PATH}")

    t0 = time.perf_counter()
    index = rebuild_and_save(DATA_DIR)
    elapsed = time.perf_counter() - t0
    counts = {k: index.kinds.count(k) for k in set(index.kinds)}
    print(
        f"rebuilt index: {index.n_docs} docs ({counts}), "
        f"vocab {index.vocab_size}, avgdl {index.avgdl:.1f} ({elapsed:.2f}s)"
    )
    print(f"index written to {INDEX_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
