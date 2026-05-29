"""Диагностика частот токенов в собранном индексе.

Помогает увидеть, какие леммы чрезмерно частые (кандидаты в стоп-слова или
INDEX_BLACKLIST), а какие редкие. На вывод смотреть глазами; решения о
правках стоп-листа принимаются вручную.

Запуск (требует уже собранный индекс в data/):
  python inspect_tokens.py                       сводка + top-50 по df
  python inspect_tokens.py --top 100             больше строк топа
  python inspect_tokens.py --kind attribute      статистика только по атрибутам
  python inspect_tokens.py --kind decision       только по решениям
  python inspect_tokens.py --min-df 100          фильтр: только частые леммы
  python inspect_tokens.py --save                + дамп полной статистики в JSON

Понятия:
  df  (document frequency)     — в скольких документах встречается лемма.
                                   Чем выше, тем чаще лемма «во всём» (плохой
                                   сигнал в BM25).
  cf  (collection frequency)   — сколько раз лемма встречается во всём корпусе.
                                   df и cf близки, но cf учитывает повторы.
  idf                          — log((N - df + 0.5)/(df + 0.5) + 1).
                                   Чем меньше idf, тем «мусорнее» лемма в BM25.
  %docs                        — df / N, доля документов. Кандидаты в стоп-слова
                                   обычно имеют %docs > 30–50%.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from bm25 import open_searcher

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")


def _print_row(rank, term, df_v, cf_v, idf_v, n):
    pct = df_v / n * 100 if n else 0
    print(f"  {rank:>4} {df_v:>7} {pct:>5.1f}% {cf_v:>9} {idf_v:>6.2f}  {term}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Диагностика частот токенов в индексе")
    parser.add_argument("--top", type=int, default=50, help="сколько строк топа печатать")
    parser.add_argument(
        "--kind", choices=("decision", "attribute"), default=None,
        help="считать df только по документам этого kind (cf и idf — по всему корпусу)",
    )
    parser.add_argument(
        "--min-df", type=int, default=1,
        help="не показывать леммы с df меньше этого числа",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="сохранить полную статистику в output/token_stats.json",
    )
    args = parser.parse_args(argv[1:])

    try:
        searcher = open_searcher(DATA_DIR)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1

    index = searcher.index
    N = index.n_docs
    V = index.vocab_size

    # Обратный vocab: column_idx → лемма.
    idx_to_term = {v: k for k, v in index.vocab.items()}

    print(f"индекс: {N} документов, {V} терминов, avgdl={index.avgdl:.1f}")
    print(f"kinds: {dict(Counter(index.kinds))}")

    # ─── общие распределения ──────────────────────────────────────────────
    df = index.df.astype(np.int64)
    # cf — collection frequency, сумма tf по столбцу.
    cf = np.asarray(index.tf_csc.sum(axis=0)).ravel().astype(np.int64)
    idf = index.idf

    print("\n=== df (document frequency) ===")
    print(
        f"  min={int(df.min())}  max={int(df.max())}  "
        f"mean={float(df.mean()):.1f}  median={int(np.median(df))}"
    )
    print(
        f"  p90={int(np.percentile(df, 90))}  "
        f"p95={int(np.percentile(df, 95))}  "
        f"p99={int(np.percentile(df, 99))}"
    )
    print(f"  лемм с df=1: {int((df == 1).sum())} ({(df == 1).sum() / V * 100:.1f}% словаря)")
    half = N // 2
    print(f"  лемм с df > N/2 ({half}): {int((df > half).sum())}  ← очевидные кандидаты в стоп-слова")

    print("\n=== cf (collection frequency) ===")
    print(
        f"  min={int(cf.min())}  max={int(cf.max())}  "
        f"sum={int(cf.sum())} (всего токенов в корпусе после лемматизации)"
    )

    print("\n=== idf (вклад термина в BM25) ===")
    print(f"  min={float(idf.min()):.3f}  max={float(idf.max()):.3f}  mean={float(idf.mean()):.3f}")
    print(f"  лемм с idf < 0.5 (низкая полезность): {int((idf < 0.5).sum())}")

    # ─── kind-фильтр для df ───────────────────────────────────────────────
    if args.kind:
        kinds_arr = np.asarray(index.kinds)
        mask = (kinds_arr == args.kind)
        n_subset = int(mask.sum())
        print(f"\n--- фильтр kind={args.kind!r}: {n_subset} документов ---")
        if n_subset == 0:
            return 0
        sub_tf = index.tf[mask, :]
        df_view = np.asarray((sub_tf > 0).sum(axis=0)).ravel().astype(np.int64)
        n_for_pct = n_subset
    else:
        df_view = df
        n_for_pct = N

    # ─── Top по df ────────────────────────────────────────────────────────
    above = df_view >= args.min_df
    candidates = np.flatnonzero(above)
    if candidates.size == 0:
        print("\nнет лемм с df >= --min-df")
        return 0

    top_df = candidates[np.argsort(-df_view[candidates])][: args.top]
    title = f"Top-{args.top} по df"
    if args.kind:
        title += f" (внутри kind={args.kind})"
    if args.min_df > 1:
        title += f", min-df={args.min_df}"
    print(f"\n=== {title} ===")
    print(f"  {'#':>4} {'df':>7} {'%docs':>6} {'cf':>9} {'idf':>6}  term")
    for rank, col in enumerate(top_df, 1):
        _print_row(rank, idx_to_term[int(col)], int(df_view[col]), int(cf[col]),
                   float(idf[col]), n_for_pct)

    # ─── Top по cf ────────────────────────────────────────────────────────
    print(f"\n=== Top-{args.top} по cf (число вхождений) ===")
    top_cf = np.argsort(-cf)[: args.top]
    print(f"  {'#':>4} {'df':>7} {'%docs':>6} {'cf':>9} {'idf':>6}  term")
    for rank, col in enumerate(top_cf, 1):
        _print_row(rank, idx_to_term[int(col)], int(df[col]), int(cf[col]),
                   float(idf[col]), N)

    # ─── Полный дамп ──────────────────────────────────────────────────────
    if args.save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / "token_stats.json"
        # Сортировка по cf убыванию — удобно листать сверху.
        order = np.argsort(-cf)
        payload = {
            "n_docs": N,
            "vocab_size": V,
            "avgdl": index.avgdl,
            "kind_filter": args.kind,
            "tokens": [
                {
                    "term": idx_to_term[int(col)],
                    "df": int(df[col]),
                    "cf": int(cf[col]),
                    "idf": round(float(idf[col]), 4),
                }
                for col in order
            ],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nfull stats → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
