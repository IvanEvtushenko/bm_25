"""Coverage analysis: какие существующие публикации «покрывают» атрибуты target.

Идея:
  1. Атрибут target-публикации (один элемент или группа элементов) →
     BM25-запрос по корпусу элементов других публикаций.
  2. Из всех матчей оставляем тех, чей скор > threshold, далее top-K.
  3. Каждый из top-K получает долю голоса = score / sum(scores).
     Атрибут отдаёт суммарно 1.0; если ни один не прошёл threshold —
     атрибут не голосует.
  4. Голоса агрегируются по pub_id владельцев → финальный ранкинг.

Параллелизм: 50 атрибутов × BM25 — независимы между собой и хорошо ложатся
в multiprocessing.Pool. На Linux fork() даёт CoW: BM25Index в воркерах
не копируется, шарится с родителем. Tokenizer тоже наследуется.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from bm25 import open_searcher, score as bm25_score

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 1.0
DEFAULT_WORKERS = 20

# Документы-атрибуты (поля/показатели решения). Их используем для голосования.
ATTRIBUTE_KIND = "attribute"


# Глобальное состояние воркера — наследуется через fork() из main-процесса.
# Заполняется в main() перед спавном пула.
_WORKER_INDEX = None
_WORKER_MASK = None
_WORKER_PUB_IDS = None
_WORKER_TOP_K = DEFAULT_TOP_K
_WORKER_THRESHOLD = DEFAULT_THRESHOLD


def _worker_init(index, mask, pub_ids, top_k, threshold):
    """Вызывается в каждом воркере перед стартом — только если spawn (не fork).

    На Linux с fork воркеры уже наследуют глобальные переменные из main.
    Эта функция нужна как страховка для платформ, где fork недоступен.
    """
    global _WORKER_INDEX, _WORKER_MASK, _WORKER_PUB_IDS, _WORKER_TOP_K, _WORKER_THRESHOLD
    _WORKER_INDEX = index
    _WORKER_MASK = mask
    _WORKER_PUB_IDS = pub_ids
    _WORKER_TOP_K = top_k
    _WORKER_THRESHOLD = threshold


def _score_attr(q_tokens: list[str]) -> dict[str, float]:
    """Один BM25 + распределение голоса по top-K. Запускается в worker'е.

    Принимает уже токенизированный запрос (токенизатор живёт в main и
    результаты передаются — так избегаем дублирующего init MorphAnalyzer
    в каждом процессе).
    """
    if not q_tokens or _WORKER_INDEX is None:
        return {}
    scores = bm25_score(_WORKER_INDEX, q_tokens)
    scores = np.where(_WORKER_MASK, scores, 0.0)

    above = scores > _WORKER_THRESHOLD
    if not above.any():
        return {}
    candidate_idx = np.flatnonzero(above)
    if candidate_idx.size > _WORKER_TOP_K:
        partition = np.argpartition(-scores[candidate_idx], kth=_WORKER_TOP_K - 1)[:_WORKER_TOP_K]
        candidate_idx = candidate_idx[partition]

    chosen_scores = scores[candidate_idx]
    total = float(chosen_scores.sum())
    if total <= 0:
        return {}

    votes: dict[str, float] = {}
    for i, sc in zip(candidate_idx, chosen_scores):
        owner_pub = _WORKER_PUB_IDS[i]
        if not owner_pub:
            continue
        votes[owner_pub] = votes.get(owner_pub, 0.0) + float(sc) / total
    return votes


def coverage(
    searcher,
    target_pub_id: str,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    # Атрибуты target — все документы kind=attribute с этим pub_id.
    pub_ids = np.asarray(searcher.index.pub_ids)
    kinds = np.asarray(searcher.index.kinds)
    is_attribute = (kinds == ATTRIBUTE_KIND)
    target_rows = np.flatnonzero((pub_ids == target_pub_id) & is_attribute)
    if target_rows.size == 0:
        raise SystemExit(f"no attribute documents found for pub_id={target_pub_id!r}")

    # Маска кандидатов: атрибуты ДРУГИХ публикаций.
    candidate_mask = is_attribute & (pub_ids != target_pub_id)

    # Токенизируем все атрибуты в main-процессе (тяжёлый MorphAnalyzer
    # инициализирован только здесь, в воркерах его создавать не нужно).
    target_attrs = [searcher.documents[searcher.index.doc_ids[i]] for i in target_rows]
    q_tokens_list: list[list[str]] = [
        searcher.tokenizer(f"{a.get('name', '')} {a.get('description', '')}")
        for a in target_attrs
    ]

    # Глобальное состояние для воркеров (на Linux наследуется через fork).
    _worker_init(searcher.index, candidate_mask, list(pub_ids), top_k, threshold)

    if workers <= 1 or len(q_tokens_list) < workers:
        per_attr_votes = [_score_attr(qt) for qt in q_tokens_list]
    else:
        from multiprocessing import get_context
        ctx = get_context("fork") if os.name != "nt" else get_context("spawn")
        with ctx.Pool(
            workers,
            initializer=_worker_init,
            initargs=(searcher.index, candidate_mask, list(pub_ids), top_k, threshold),
        ) as pool:
            per_attr_votes = pool.map(_score_attr, q_tokens_list)

    voting_attrs = sum(1 for v in per_attr_votes if v)
    pub_votes: dict[str, float] = defaultdict(float)
    for v in per_attr_votes:
        for pub_id, w in v.items():
            pub_votes[pub_id] += w

    ranked = sorted(pub_votes.items(), key=lambda kv: kv[1], reverse=True)

    # Если в индексе есть документ-решение (kind="decision") — берём его name
    # для отображения. Если в индексе только атрибуты — name остаётся пустым.
    target_doc = searcher.documents.get(f"dec:{target_pub_id}", {}) or {}
    rankings = []
    for pub_id, vote_sum in ranked:
        pub_doc = searcher.documents.get(f"dec:{pub_id}", {}) or {}
        rankings.append({
            "pub_id": pub_id,
            "name": pub_doc.get("name", ""),
            "score": round(vote_sum, 4),
        })

    return {
        "target_pub_id": target_pub_id,
        "target_pub_name": target_doc.get("name", ""),
        "n_attributes": len(target_attrs),
        "voting_attributes": voting_attrs,
        "threshold": threshold,
        "top_k": top_k,
        "workers": workers,
        "rankings": rankings,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Coverage analysis")
    parser.add_argument("target_pub_id", help="например: Аналитика/10")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"параллельных процессов (default: {DEFAULT_WORKERS}; 1 = последовательно)",
    )
    args = parser.parse_args(argv[1:])

    searcher = open_searcher(DATA_DIR)
    result = coverage(
        searcher, args.target_pub_id,
        top_k=args.top_k, threshold=args.threshold, workers=args.workers,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = args.target_pub_id.replace("/", "-")
    out_path = OUTPUT_DIR / f"coverage_{safe_name}.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"target: {result['target_pub_id']} — {result['target_pub_name']}")
    print(
        f"attributes: {result['n_attributes']} total, "
        f"{result['voting_attributes']} voted "
        f"(threshold={result['threshold']}, top_k={result['top_k']}, workers={result['workers']})"
    )
    print("\ntop публикации по покрытию:")
    for r in result["rankings"][:10]:
        print(f"  {r['score']:6.3f}  [{r['pub_id']}] {r['name']}")
    if len(result["rankings"]) > 10:
        print(f"  ... ещё {len(result['rankings']) - 10}")
    print(f"\nfull result: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
