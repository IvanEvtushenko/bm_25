"""Аппендер: добавляет новые документы и пересобирает индекс.

Что делает:
  1. Читает входной CSV или JSONL (каждая строка → dict).
  2. По схеме (наличию определённых колонок) понимает, что это:
     публикации (pub_report-стиль) или элементы (Показатели-стиль).
     Можно явно указать через --kind.
  3. Нормализует записи: выставляет doc_id, kind, оставляет только
     полезные поля. Ненужные колонки CSV (служебные, длинные basis_document)
     не сохраняются — экономим место в docs.jsonl.
  4. Дедуплицирует по doc_id: если такой документ уже есть в data/docs.jsonl
     или дублируется в самом источнике — пропускаем. Это делает скрипт
     ИДЕМПОТЕНТНЫМ: повторный запуск на том же файле ничего не сломает.
  5. Дописывает новые записи в data/docs.jsonl.
  6. Пересобирает numpy-индекс заново из всего docs.jsonl.

Почему пересборка, а не инкремент:
  * На наших объёмах (десятки тысяч документов) — секунды.
  * Простой и надёжный путь: индекс всегда консистентен с источником правды.
  * Инкрементальное обновление CSR-матрицы возможно (vstack новых строк,
    расширение vocab), но добавляет хрупкости и кода. Сделаем при
    необходимости.
"""
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

# Какие колонки исходного CSV сохранять в docs.jsonl. Это и индексируемые
# поля, и «полезные» для отображения в выдаче (frequency, parent_dataset).
# Всё, что НЕ перечислено здесь, отбрасывается при нормализации.
PUBLICATION_KEEP_FIELDS = (
    "pub_id", "name", "business_description", "keywords",
    "parent_dataset", "frequency", "first_timestamp", "basis_document", "comments",
)
ELEMENT_KEEP_FIELDS = ("pub_id", "report_name", "name", "description")


def detect_kind(fieldnames: list[str]) -> str:
    """Определить тип документов по схеме CSV.

    Эвристика простая, но надёжная для текущих файлов:
      * есть колонка business_description → это pub_report.csv → publication.
      * есть description + report_name (и нет business_description) → element.
    Если непонятно — выходим с подсказкой пользователю.
    """
    fields = set(fieldnames)
    if "business_description" in fields:
        return "publication"
    if "description" in fields and "report_name" in fields:
        return "element"
    raise SystemExit(
        f"cannot autodetect kind from columns {sorted(fields)}; pass --kind explicitly"
    )


def normalize_publication(row: dict, _row_idx: int) -> dict:
    """CSV-строка публикации → нормализованный документ.

    Что добавляем:
      doc_id = "pub:<pub_id>" — глобально уникальный ключ.
      kind   = "publication"  — тип для INDEX_FIELDS_BY_KIND и фильтрации.

    Про "﻿pub_id" с символом BOM (﻿): pub_report.csv сохранён как
    UTF-8-with-BOM. csv.DictReader через encoding='utf-8-sig' это лечит,
    но на всякий случай ловим оба варианта — устойчивее к чужим CSV.
    """
    pub_id = row.get("pub_id") or row.get("﻿pub_id")
    if not pub_id:
        raise SystemExit(f"publication row missing pub_id: {row}")
    out = {k: row.get(k, "") for k in PUBLICATION_KEEP_FIELDS}
    out["pub_id"] = pub_id
    out["doc_id"] = f"pub:{pub_id}"
    out["kind"] = "publication"
    return out


def normalize_element(row: dict, row_idx: int) -> dict:
    """CSV-строка элемента → нормализованный документ.

    Уникальный ключ: "el:<pub_id>#<row_idx>". pub_id у элементов НЕ уникален
    (одна публикация = много элементов), поэтому в ключ добавляем номер
    строки в исходном CSV. Это даёт стабильный детерминированный id:
    тот же CSV → те же id.
    """
    pub_id = row.get("pub_id") or row.get("﻿pub_id")
    if not pub_id:
        raise SystemExit(f"element row missing pub_id (idx={row_idx}): {row}")
    out = {k: row.get(k, "") for k in ELEMENT_KEEP_FIELDS}
    out["pub_id"] = pub_id
    out["doc_id"] = f"el:{pub_id}#{row_idx}"
    out["kind"] = "element"
    return out


# Регистр нормализаторов: kind → функция (row, row_idx) → doc.
# Чтобы добавить новый тип документа, дописываем сюда нового нормализатора
# и (опционально) ветку в detect_kind.
NORMALIZERS = {
    "publication": normalize_publication,
    "element": normalize_element,
}


def read_input(path: Path) -> tuple[list[dict], list[str]]:
    """Прочитать CSV или JSONL → (список словарей, список имён колонок).

    encoding='utf-8-sig' — чтобы автоматически снимать BOM (Byte Order Mark)
    в начале UTF-8 файлов (его любит ставить Excel). Без этого первая
    колонка получит ключ "﻿pub_id" вместо "pub_id" и отвалится.
    """
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
        # Для JSONL "колонки" = объединение ключей всех записей.
        fields = sorted({k for d in out for k in d.keys()})
        return out, fields
    raise SystemExit(f"unsupported input format: {path.suffix}")


def filter_new(records: list[dict], existing_ids: set[str]) -> tuple[list[dict], list[str]]:
    """Отфильтровать дубликаты: уже существующие в data/ или дублирующиеся в источнике.

    Возвращает (записи_к_добавлению, список_id_дубликатов).
    """
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

    # Превращаем CSV-строки в нормализованные документы. row_idx нужен
    # для построения уникальных id у элементов.
    records = [normalizer(row, i) for i, row in enumerate(rows)]

    # Если data/docs.jsonl уже есть — собираем уже использованные doc_id.
    # На холодном старте (data/ ещё нет) — пустой set, все записи новые.
    existing = load_documents_map(DOCS_PATH) if DOCS_PATH.exists() else {}
    new_records, dupes = filter_new(records, set(existing))
    if dupes:
        head = dupes[:5]
        more = "..." if len(dupes) > 5 else ""
        print(f"skipping {len(dupes)} duplicate doc_id(s): {head}{more}")
    if not new_records:
        # Идемпотентность: если новых записей нет, индекс не трогаем.
        # Это важно — пересборка хоть и быстрая, но бесполезный disk I/O ни к чему.
        print("no new records to append; index left untouched")
        return 0

    # Append-only: только дописываем в конец. Существующие записи никогда
    # не переписываем (источник правды).
    appended = append_jsonl(DOCS_PATH, new_records)
    print(f"appended {appended} records to {DOCS_PATH}")

    # Полная пересборка индекса из data/docs.jsonl.
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
