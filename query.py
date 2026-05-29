"""CLI поиска по индексу.

Терминология:
  Решение (decision)  — набор данных каталога.
  Атрибут (attribute) — поле/показатель решения.

Режимы поиска:

  По тексту запроса:
    python query.py "запрос"                          top-5 по всем kind
    python query.py --kind decision "запрос"          только среди решений
    python query.py --kind attribute --top-k 10 "q"   только среди атрибутов

  По решению (sanity-check «найди похожие на 3571»):
    python query.py --like-decision 3571              все атрибуты решения 3571
                                                      склеиваются в один запрос;
                                                      решение маскируется;
                                                      выдача группируется по decision_id.

  С фильтром по типу атрибута (поле `type` в SQL):
    python query.py "инфляция" --search-among Витрина Форма
                                                      Берёт только документы,
                                                      у которых поле `type`
                                                      содержит хотя бы одно из
                                                      перечисленных значений
                                                      (case-insensitive).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from bm25 import open_searcher, read_jsonl, score as bm25_score
from bm25.builder import _doc_text

DATA_DIR = Path("data")
DOCS_PATH = DATA_DIR / "docs.jsonl"

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
        if isinstance(title, list):
            title = " | ".join(str(x) for x in title)
        if doc.get("kind") == "attribute":
            report = doc.get("report_name", "")
            if report:
                title = f"{report} :: {title}"
        print(f"  {score_:6.2f}  [{kind_tag}|{doc.get('decision_id', '?')}] {title}")


def build_type_mask(searcher, allowed_types: list[str]) -> np.ndarray:
    """Маска документов, у которых поле `type` пересекается с allowed_types.

    Сравнение case-insensitive. Поле `type` может быть str (одиночный атрибут)
    или list[str] (группа). Документы без `type` отсекаются.
    """
    wanted = {t.strip().lower() for t in allowed_types if t.strip()}
    n = searcher.index.n_docs
    id_to_row = {did: i for i, did in enumerate(searcher.index.doc_ids)}
    mask = np.zeros(n, dtype=bool)
    for doc in read_jsonl(DOCS_PATH):
        row = id_to_row.get(doc["doc_id"])
        if row is None:
            continue
        t_value = doc.get("type")
        if t_value is None:
            continue
        types = t_value if isinstance(t_value, list) else [t_value]
        if any(isinstance(t, str) and t.lower() in wanted for t in types):
            mask[row] = True
    return mask


def search_like_decision(
    searcher, target_decision_id: str, top_k: int, type_mask=None,
) -> list[tuple[float, dict]]:
    """Поиск похожих решений «как X»: все атрибуты X — один большой запрос."""
    decision_ids = np.asarray(searcher.index.decision_ids)
    target_rows = np.flatnonzero(decision_ids == target_decision_id)
    if target_rows.size == 0:
        print(f"no documents with decision_id={target_decision_id!r}", file=sys.stderr)
        return []

    parts = []
    for i in target_rows:
        doc = searcher.documents[searcher.index.doc_ids[i]]
        parts.append(_doc_text(doc))
    big_query = " ".join(parts)

    q_tokens = searcher.tokenizer(big_query)
    if not q_tokens:
        return []
    scores = bm25_score(searcher.index, q_tokens)
    # Маскируем target и (опционально) применяем type-фильтр.
    keep = (decision_ids != target_decision_id)
    if type_mask is not None:
        keep = keep & type_mask
    scores = np.where(keep, scores, 0.0)

    nz = np.flatnonzero(scores > 0)
    if nz.size == 0:
        return []

    # Группировка по decision_id: один лучший документ на каждое соседнее решение.
    best_per_decision: dict[str, tuple[float, int]] = {}
    for i in nz:
        did = decision_ids[i]
        cur = best_per_decision.get(did)
        if cur is None or scores[i] > cur[0]:
            best_per_decision[did] = (float(scores[i]), int(i))

    ranked = sorted(best_per_decision.values(), key=lambda x: x[0], reverse=True)[:top_k]
    return [(s, searcher.documents[searcher.index.doc_ids[i]]) for s, i in ranked]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="BM25 search over decisions and attributes")
    parser.add_argument("queries", nargs="*", help="queries (default: smoke set)")
    parser.add_argument(
        "--kind",
        choices=("decision", "attribute"),
        default=None,
        help="filter results by document kind",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--like-decision", type=str, default=None, metavar="DECISION_ID",
        help="режим поиска похожих: использует все атрибуты decision_id=DECISION_ID "
             "как большой запрос, маскирует само решение, группирует выдачу по decision_id",
    )
    parser.add_argument(
        "--search-among", nargs="+", default=None, metavar="TYPE",
        help="ограничить выдачу документами, у которых поле 'type' "
             "содержит одно из перечисленных значений (case-insensitive)",
    )
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

    type_mask = None
    if args.search_among:
        type_mask = build_type_mask(searcher, args.search_among)
        print(f"--search-among {args.search_among}: {int(type_mask.sum())} documents pass filter")

    if args.like_decision is not None:
        results = search_like_decision(
            searcher, args.like_decision, top_k=args.top_k, type_mask=type_mask,
        )
        show(f"like-decision:{args.like_decision}", results)
        return 0

    queries = args.queries or SMOKE_QUERIES
    for q in queries:
        if type_mask is None:
            results = searcher.search(q, top_k=args.top_k, kind=args.kind)
        else:
            q_tokens = searcher.tokenizer(q)
            if not q_tokens:
                results = []
            else:
                scores = bm25_score(searcher.index, q_tokens)
                if args.kind is not None:
                    kmask = np.asarray([k == args.kind for k in searcher.index.kinds], dtype=bool)
                    scores = np.where(kmask & type_mask, scores, 0.0)
                else:
                    scores = np.where(type_mask, scores, 0.0)
                nz = np.flatnonzero(scores > 0)
                if nz.size == 0:
                    results = []
                else:
                    k = min(args.top_k, nz.size)
                    if nz.size > k:
                        part = np.argpartition(-scores[nz], kth=k - 1)[:k]
                        nz = nz[part]
                    order = nz[np.argsort(-scores[nz])]
                    results = [
                        (float(scores[i]), searcher.documents[searcher.index.doc_ids[i]])
                        for i in order
                    ]
        show(q, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
