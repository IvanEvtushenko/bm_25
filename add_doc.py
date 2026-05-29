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
from bm25.sources import ID_PLACEHOLDER

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
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help=f"для SQL с плейсхолдером {ID_PLACEHOLDER!r}: сколько id решений "
             "подставляется в один запрос (default: 500). ids берутся из "
             "decision_id уже загруженных в data/docs.jsonl документов kind=decision. "
             "Атрибуты одного решения целиком попадают в один батч.",
    )
    args = parser.parse_args(argv[1:])

    if not args.source.exists():
        print(f"missing input: {args.source}", file=sys.stderr)
        return 1

    # Если SQL содержит {id_filter} — собираем список decision_id уже загруженных
    # решений из docs.jsonl и передаём в Source как batch_ids. Тогда SQLSource
    # переключится на _spark_iter_batched и будет гонять по batch_size за раз.
    batch_ids = None
    if args.source.suffix.lower() == ".sql":
        sql_text = args.source.read_text(encoding="utf-8")
        if ID_PLACEHOLDER in sql_text:
            if not DOCS_PATH.exists():
                print(
                    f"SQL содержит {ID_PLACEHOLDER!r}, но {DOCS_PATH} ещё нет.\n"
                    f"Сначала загрузите решения: python add_doc.py db/decisions.sql",
                    file=sys.stderr,
                )
                return 1
            ids_raw = sorted({
                doc["decision_id"] for doc in read_jsonl(DOCS_PATH)
                if doc.get("kind") == "decision"
            })
            if not ids_raw:
                print(
                    f"в {DOCS_PATH} нет документов kind=decision — нечем заполнить "
                    f"{ID_PLACEHOLDER!r}",
                    file=sys.stderr,
                )
                return 1
            # id в БД обычно числовое (bigint). Попробуем привести; если не выйдет —
            # оставим строками, _format_sql_value их обернёт в кавычки.
            try:
                batch_ids = [int(x) for x in ids_raw]
            except (ValueError, TypeError):
                batch_ids = ids_raw
            print(
                f"detected {ID_PLACEHOLDER!r} in SQL → batched ingest: "
                f"{len(batch_ids)} ids × batch_size={args.batch_size}"
            )

    source = create_source(
        args.source,
        kind_override=args.kind,
        batch_ids=batch_ids,
        batch_size=args.batch_size,
    )
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
