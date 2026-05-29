"""SQL-источник документов: читает SQL-файл, выполняет через Spark/Hive,
забирает результат в driver и отдает словари с каноническими именами колонок.

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


ID_PLACEHOLDER = "{id_filter}"


def _format_sql_value(v) -> str:
    """Безопасная подстановка одного значения в SQL IN-список."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    # Строковое значение — обернуть в одинарные кавычки с экранированием.
    return "'" + str(v).replace("'", "''") + "'"


def _spark_session(app_name: str):
    """Создать/получить Spark-сессию с настройками для нашего инжеста."""
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise SystemExit(
            "pyspark недоступен. SQL-источник работает только в прод-окружении "
            "со Spark. Локально используйте CSV-источники."
        ) from e
    spark = SparkSession.builder.appName(app_name).enableHiveSupport().getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def _spark_iter(sql: str, app_name: str = "bm25_ingest") -> Iterator[dict]:
    """Выполнить SQL в Spark одним запросом и выдавать dict-ы.

    Используется collect(): результат запроса целиком материализуется в памяти
    driver-процесса. После collect() сессия Spark останавливается.

    Если в SQL есть плейсхолдер {id_filter} — это батчевый запрос, выполнять
    его одним вызовом нельзя; используйте _spark_iter_batched.
    """
    if ID_PLACEHOLDER in sql:
        raise ValueError(
            f"SQL содержит плейсхолдер {ID_PLACEHOLDER!r}, но вызван _spark_iter "
            f"без списка id. Используйте _spark_iter_batched(sql, ids, batch_size)."
        )
    spark = _spark_session(app_name)
    try:
        df = spark.sql(sql)
        columns = df.columns
        spark_rows = df.collect()
        rows = [{col: row[col] for col in columns} for row in spark_rows]
        print(f"[sql] collected {len(rows)} rows into driver", flush=True)
    finally:
        spark.stop()
    yield from rows


def _spark_iter_batched(
    sql_template: str,
    ids: list,
    batch_size: int = 500,
    app_name: str = "bm25_ingest",
    placeholder: str = ID_PLACEHOLDER,
) -> Iterator[dict]:
    """Выполнить SQL-template батчами по batch_size id, подставляя их в placeholder.

    sql_template обязан содержать placeholder (по умолчанию {id_filter}),
    в нужном месте WHERE: например `WHERE id IN ({id_filter})`. На каждой
    итерации placeholder заменяется на 'a, b, c, ...' — список id текущего батча.

    Это позволяет обойти лимит Hive на размер результата одного запроса:
    вместо одного SQL на N×K строк гоняем N запросов по K строк.
    """
    if placeholder not in sql_template:
        raise ValueError(
            f"SQL не содержит плейсхолдера {placeholder!r}; добавьте его в WHERE, "
            f"например: WHERE id IN ({placeholder})"
        )
    if not ids:
        return

    spark = _spark_session(app_name)
    try:
        all_rows: list[dict] = []
        n = len(ids)
        n_batches = (n + batch_size - 1) // batch_size
        for bi, start in enumerate(range(0, n, batch_size), 1):
            chunk = ids[start:start + batch_size]
            ids_str = ", ".join(_format_sql_value(x) for x in chunk)
            sql = sql_template.replace(placeholder, ids_str)
            df = spark.sql(sql)
            columns = df.columns
            spark_rows = df.collect()
            batch_rows = [{col: r[col] for col in columns} for r in spark_rows]
            all_rows.extend(batch_rows)
            print(
                f"[sql-batch] {bi}/{n_batches}: ids={len(chunk)} "
                f"→ +{len(batch_rows)} rows (total {len(all_rows)})",
                flush=True,
            )
    finally:
        spark.stop()
    yield from all_rows


@dataclass
class SQLSource:
    """Источник «SQL через Spark».

    Использование (только в прод-окружении):
        # Обычный режим — один SQL-запрос:
        src = SQLSource(Path("db/decisions.sql"))

        # Батчевый режим — SQL содержит {id_filter}, ids подставляются партиями:
        src = SQLSource(
            Path("db/attributes.sql"),
            batch_ids=[1001, 1002, ...],   # обычно — decision_id уже загруженных решений
            batch_size=10,
        )

        kind = src.kind()                  # auto-detect, peek первой строки
        for row in src.iter_rows():
            ...
    """
    sql_path: Path
    kind_override: str | None = None
    # batch_ids != None → выбираем _spark_iter_batched и подставляем ids в {id_filter}.
    batch_ids: list | None = None
    batch_size: int = 500
    _first_row: dict | None = field(default=None, repr=False)
    _rest: Iterator[dict] | None = field(default=None, repr=False)
    _primed: bool = field(default=False, repr=False)

    def _prime(self) -> None:
        """Запустить запрос (или первую партию батча) и подсмотреть первую строку.

        Идемпотентно: повторные вызовы ничего не делают.
        """
        if self._primed:
            return
        sql = self.sql_path.read_text(encoding="utf-8").strip()
        if self.batch_ids is None:
            it = _spark_iter(sql)
        else:
            it = _spark_iter_batched(sql, self.batch_ids, batch_size=self.batch_size)
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
