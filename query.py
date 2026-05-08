from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bm25_numpy import open_searcher

DATA_DIR = Path("data")

SMOKE_QUERIES = [
    "инфляционные ожидания населения",
    "ликвидация банков и страховых компаний",
    "кредит экономике денежная масса",
    "брокеры розничные инвесторы",
    "ОФЗ-ИН вмененная инфляция",
]


def show(query: str, results) -> None:
    print(f"\n=== {query!r} ===")
    if not results:
        print("  (no matches)")
        return
    for score_, doc in results:
        kind_tag = doc.get("kind", "?")[:3]
        title = doc.get("name", "")
        if doc.get("kind") == "element":
            title = f"{doc.get('report_name', '')} :: {title}"
        print(f"  {score_:6.2f}  [{kind_tag}|{doc['pub_id']}] {title}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="BM25 search over publications and elements")
    parser.add_argument("queries", nargs="*", help="queries (default: smoke set)")
    parser.add_argument(
        "--kind",
        choices=("publication", "element"),
        default=None,
        help="filter results by document kind",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv[1:])

    try:
        searcher = open_searcher(DATA_DIR)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    counts = {k: searcher.index.kinds.count(k) for k in set(searcher.index.kinds)}
    print(
        f"loaded index: {searcher.index.n_docs} docs ({counts}), "
        f"vocab {searcher.index.vocab_size}, avgdl {searcher.index.avgdl:.1f}"
    )
    queries = args.queries or SMOKE_QUERIES
    for q in queries:
        show(q, searcher.search(q, top_k=args.top_k, kind=args.kind))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
