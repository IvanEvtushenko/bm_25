"""SQL-источник документов: читает SQL-файл, выполняет через Spark/Hive,
стримит результат как поток словарей с каноническими именами колонок.

После SQLSource.iter_rows() все источники (CSV и SQL) выглядят одинаково для
вышестоящего кода — единый поток строк-словарей.

pyspark импортируется ЛЕНИВО — модуль грузится на машинах без Spark
без ошибок; pyspark подгрузится только при реальном `iter_rows`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


def _detect_kind_from_sql_columns(columns: list[str]) -> str:
    """Угадать kind по составу колонок SQL-результата."""
    cols = set(columns)
    if "col_descr" in cols or "type" in cols:
        return "attribute"
    if "descr" in cols or "keywords" in cols:
        return "decision"
    raise ValueError(
        f"cannot autodetect kind from SQL columns {sorted(cols)}; "
        f"pass --kind explicitly"
    )


def _spark_iter(sql: str, app_name: str = "bm25_ingest") -> Iterator[dict]:
    """Выполнить SQL в Spark, выдавать строки результата как dict-ы.

    Используется toLocalIterator(): партиции тянутся в driver по одной,
    весь датафрейм в RAM не загружается. Подходит для прод-объёмов.
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise SystemExit(
            "pyspark недоступен. SQL-источник работает только в прод-окружении "
            "со Spark. Локально используйте CSV-источники."
        ) from e

    spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.sql(sql)
    columns = df.columns
    for row in df.toLocalIterator():
        yield {col: row[col] for col in columns}


@dataclass
class SQLSource:
    """Источник «SQL через Spark».

    Использование (только в прод-окружении):
        src = SQLSource(Path("db/attributes.sql"))
        kind = src.kind()                  # auto-detect, peek первой строки
        for row in src.iter_rows():        # стрим строк
            ...
    """
    sql_path: Path
    kind_override: str | None = None
    _first_row: dict | None = field(default=None, repr=False)
    _rest: Iterator[dict] | None = field(default=None, repr=False)
    _primed: bool = field(default=False, repr=False)

    def _prime(self) -> None:
        """Запустить запрос и подсмотреть первую строку (для detect_kind).

        Идемпотентно: повторные вызовы ничего не делают.
        """
        if self._primed:
            return
        sql = self.sql_path.read_text(encoding="utf-8").strip()
        it = _spark_iter(sql)
        try:
            self._first_row = next(it)
        except StopIteration:
            self._first_row = None
        self._rest = it
        self._primed = True

    def kind(self) -> str:
        if self.kind_override:
            return self.kind_override
        self._prime()
        if self._first_row is None:
            raise ValueError(f"SQL returned 0 rows from {self.sql_path}")
        return _detect_kind_from_sql_columns(list(self._first_row.keys()))

    def iter_rows(self) -> Iterator[dict]:
        self._prime()
        if self._first_row is not None:
            yield self._first_row
        if self._rest is not None:
            yield from self._rest
