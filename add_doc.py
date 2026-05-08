from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from bm25_numpy import (
    INDEX_FIELDS_BY_KIND,
    append_jsonl,
    load_documents_map,
    rebuild_and_save,
)

DATA_DIR = Path("data")
DOCS_PATH = DATA_DIR / "docs.jsonl"
INDEX_DIR = DATA_DIR / "index"

PUBLICATION_KEEP_FIELDS = (
    "pub_id", "name", "business_description", "keywords",
    "parent_dataset", "frequency", "first_timestamp", "basis_document", "comments",
)
ELEMENT_KEEP_FIELDS = ("pub_id", "report_name", "name", "description")


def detect_kind(fieldnames: list[str]) -> str:
    fields = set(fieldnames)
    if "business_description" in fields:
        return "publication"
    if "description" in fields and "report_name" in fields:
        return "element"
    raise SystemExit(
        f"cannot autodetect kind from columns {sorted(fields)}; pass --kind explicitly"
    )


def normalize_publication(row: dict, _row_idx: int) -> dict:
    pub_id = row.get("pub_id") or row.get("﻿pub_id")
    if not pub_id:
        raise SystemExit(f"publication row missing pub_id: {row}")
    out = {k: row.get(k, "") for k in PUBLICATION_KEEP_FIELDS}
    out["pub_id"] = pub_id
    out["doc_id"] = f"pub:{pub_id}"
    out["kind"] = "publication"
    return out


def normalize_element(row: dict, row_idx: int) -> dict:
    pub_id = row.get("pub_id") or row.get("﻿pub_id")
    if not pub_id:
        raise SystemExit(f"element row missing pub_id (idx={row_idx}): {row}")
    out = {k: row.get(k, "") for k in ELEMENT_KEEP_FIELDS}
    out["pub_id"] = pub_id
    out["doc_id"] = f"el:{pub_id}#{row_idx}"
    out["kind"] = "element"
    return out


NORMALIZERS = {
    "publication": normalize_publication,
    "element": normalize_element,
}


def read_input(path: Path) -> tuple[list[dict], list[str]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader), list(reader.fieldnames or [])
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        out: list[dict] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        fields = sorted({k for d in out for k in d.keys()})
        return out, fields
    raise SystemExit(f"unsupported input format: {path.suffix}")


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
    parser.add_argument("source", type=Path, help="input CSV or JSONL")
    parser.add_argument(
        "--kind",
        choices=sorted(INDEX_FIELDS_BY_KIND),
        default=None,
        help="document kind (autodetected from CSV schema if omitted)",
    )
    args = parser.parse_args(argv[1:])

    if not args.source.exists():
        print(f"missing input: {args.source}", file=sys.stderr)
        return 1

    rows, fieldnames = read_input(args.source)
    kind = args.kind or detect_kind(fieldnames)
    normalizer = NORMALIZERS[kind]
    print(f"read {len(rows)} rows from {args.source}; kind={kind}")

    records = [normalizer(row, i) for i, row in enumerate(rows)]

    existing = load_documents_map(DOCS_PATH) if DOCS_PATH.exists() else {}
    new_records, dupes = filter_new(records, set(existing))
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
