"""CSV-источник документов: читает локальный CSV, переименовывает колонки в
канонические имена, фильтрует мусорные поля, выдаёт строки в общем формате.

После CSVSource.iter_rows() все источники (CSV и SQL) выглядят одинаково для
вышестоящего кода — единый поток строк-словарей с каноническими полями.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# Канонический набор колонок для каждого kind (для CSV-источника).
# Что не в этом списке — отбрасывается при чтении CSV (например, технические
# колонки Показатели.csv типа "Является ли элемент информативным (служ.)").
DECISION_CSV_FIELDS = (
    "pub_id", "name", "business_description", "keywords",
    "parent_dataset", "frequency", "first_timestamp", "basis_document", "comments",
)
ATTRIBUTE_CSV_FIELDS = ("pub_id", "report_name", "name", "description", "type")

# Маппинг исходных CSV-колонок (с длинными русскими именами) в канонические.
# Применяется ДО фильтрации по CSV_FIELDS.
CSV_COLUMN_ALIAS = {
    "Тип элемента состава набора данных": "type",
}


def _detect_kind_from_csv_columns(columns: list[str]) -> str:
    """Угадать kind по составу заголовка CSV."""
    cols = set(columns)
    if "business_description" in cols:
        return "decision"
    if "description" in cols and "report_name" in cols:
        return "attribute"
    raise ValueError(
        f"cannot autodetect kind from CSV columns {sorted(cols)}; "
        f"pass --kind explicitly"
    )


@dataclass
class CSVSource:
    """Источник «локальный CSV».

    Использование:
        src = CSVSource(Path("pub_report.csv"))
        for row in src.iter_rows():        # row уже с pub_id и canonical именами
            ...
        print(src.kind())                  # 'decision' / 'attribute'
    """
    path: Path
    kind_override: str | None = None
    _kind_cache: str | None = field(default=None, repr=False)

    def kind(self) -> str:
        if self.kind_override:
            return self.kind_override
        if self._kind_cache is None:
            with self.path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)
                except StopIteration:
                    raise ValueError(f"empty CSV: {self.path}")
            self._kind_cache = _detect_kind_from_csv_columns(header)
        return self._kind_cache

    def iter_rows(self) -> Iterator[dict]:
        kind = self.kind()
        canonical_fields = (
            DECISION_CSV_FIELDS if kind == "decision" else ATTRIBUTE_CSV_FIELDS
        )
        with self.path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for raw_row in reader:
                # Применяем aliases (длинная русская колонка → canonical).
                row = dict(raw_row)
                for csv_name, canonical in CSV_COLUMN_ALIAS.items():
                    if csv_name in row and canonical not in row:
                        row[canonical] = row.pop(csv_name)
                # pub_id может прийти с BOM-префиксом, если encoding оплошал.
                if "pub_id" not in row and "﻿pub_id" in row:
                    row["pub_id"] = row.pop("﻿pub_id")
                # Фильтруем только канонические колонки. Прочие выкидываются.
                yield {k: row.get(k, "") for k in canonical_fields}
