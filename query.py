"""CLI поиска по индексу.

Использование:
  python query.py "запрос"                       — top-5 по обоим типам
  python query.py --kind publication "запрос"    — только публикации
  python query.py --kind element --top-k 10 "q"  — только элементы, топ-10
  python query.py                                 — встроенный смоук-набор

Что происходит при запуске:
  1. Загружаем индекс из data/index/ (см. open_searcher в bm25_numpy.py).
  2. Прогоняем запросы через BM25.search.
  3. Печатаем top-k результатов с типом и заголовком.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bm25_numpy import open_searcher

DATA_DIR = Path("data")

# Запросы по умолчанию — для быстрой проверки, что всё работает.
SMOKE_QUERIES = [
    "инфляционные ожидания населения",
    "ликвидация банков и страховых компаний",
    "кредит экономике денежная масса",
    "брокеры розничные инвесторы",
    "ОФЗ-ИН вмененная инфляция",
]


def show(query: str, results) -> None:
    """Печать одного запроса и его top-k. results — список (score, doc)."""
    print(f"\n=== {query!r} ===")
    if not results:
        print("  (no matches)")
        return
    for score_, doc in results:
        # Префикс типа: "pub" или "ele" — чтобы глазами различать
        # публикации и элементы в перемешанной выдаче.
        kind_tag = doc.get("kind", "?")[:3]
        title = doc.get("name", "")
        # Для элементов добавляем имя родительской публикации через "::",
        # чтобы было понятно, в каком контексте этот показатель.
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

    # open_searcher грузит индекс (numpy/scipy-файлы в data/index/) и
    # маппинг doc_id→doc из data/docs.jsonl. Если индекса нет — выходим
    # с подсказкой запустить аппендер.
    try:
        searcher = open_searcher(DATA_DIR)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    # Заголовок с диагностикой: сколько документов, какого типа,
    # размер словаря, средняя длина. Полезно для отладки.
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
