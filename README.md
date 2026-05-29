# BM25-поиск по каталогу данных ЦБ РФ

Полнотекстовый поиск с лемматизацией русского языка. Без СУБД — источник правды в `data/docs.jsonl` (append-only), индекс в `data/index/` (numpy/scipy-массивы). Перенос в закрытый контур — копированием четырёх `.py` файлов и SQL.

## Терминология

| Термин | Что это | Источник |
|---|---|---|
| **Решение** (`decision`) | Набор данных каталога (один документ верхнего уровня) | `pub_report.csv` (локально) или `db/decisions.sql` (прод) |
| **Атрибут** (`attribute`) | Поле/показатель внутри решения; в индексе хранится как группа из N подряд идущих атрибутов одного решения | `Показатели.csv` (локально) или `db/attributes.sql` (прод) |

В индексе у каждого документа есть `kind` (`"decision"` либо `"attribute"`) и `decision_id` — идентификатор решения-владельца.

## Холодный старт в контуре

Что нужно физически: Python 3.9+, и в системе должны быть `numpy`, `scipy`, `razdel`, `pymorphy3` (с подключённым словарём `pymorphy3-dicts-ru`), `pyspark`.

```bash
# 1) Проверка окружения — реально пробуем тот функционал, который нужен коду.
#    Если печатает «parse: кот OK» — всё на месте.
python3 - <<'PY'
import razdel, numpy, scipy, pyspark, pymorphy3, pymorphy3_dicts_ru
from pymorphy3 import MorphAnalyzer
m = MorphAnalyzer(path=pymorphy3_dicts_ru.get_path())
print("parse:", m.parse("коты")[0].normal_form, "OK")
PY

# 2) Сборка индекса с нуля. Порядок не важен; решения и атрибуты живут в одном индексе.
rm -rf data
python3 add_doc.py db/decisions.sql                  # решения каталога
python3 add_doc.py db/attributes.sql --group-size 5  # атрибуты решений (группы по 5)

# 3) Проверка поиска.
python3 query.py "инфляция"
python3 query.py --like-decision 3571 --top-k 10

# 4) (опционально) Coverage-анализ.
python3 coverage.py 3571
```

Если шаг 1 упал — увидите конкретную причину: либо отсутствует один из перечисленных импортов, либо `pymorphy3` не может прочитать словарь. Отдельный пакет `dawg-python` явно проверять не нужно: это транзитивная зависимость `pymorphy3`, и если `MorphAnalyzer.parse(...)` отрабатывает — pymorphy3 разберётся со своими внутренностями сам (там есть несколько fallback'ов).

### Что создастся в процессе

| Каталог/файл | Когда появляется | Что это |
|---|---|---|
| `data/docs.jsonl` | при первом `add_doc.py` | Append-only источник правды |
| `data/index/` | при первом `add_doc.py` | Numpy/scipy-кэш индекса |
| `output/coverage_*.json` | при `coverage.py` | Отчёт ранжирования |

Все три каталога — в `.gitignore`, в репо не уезжают; пересобираются на той стороне с нуля.

## Холодный старт локально (для разработки, на CSV-данных)

В корне репо лежат тестовые CSV (`pub_report.csv`, `Показатели.csv`) — на них можно проверить весь пайплайн без Spark.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

rm -rf data
.venv/bin/python add_doc.py pub_report.csv                      # 285 решений
.venv/bin/python add_doc.py Показатели.csv --group-size 5       # ~3 300 групп атрибутов
.venv/bin/python query.py "инфляционные ожидания"
.venv/bin/python query.py --like-decision "Аналитика/10"
.venv/bin/python coverage.py "Аналитика/10"
```

`kind` автоопределяется по составу колонок источника (CSV — по `business_description` vs `description`+`report_name`; SQL — по `descr`/`keywords` vs `col_descr`/`type`). При необходимости можно явно через `--kind decision|attribute`.

## Поиск

### По тексту

```bash
.venv/bin/python query.py "инфляционные ожидания населения"
.venv/bin/python query.py --kind decision  "брокеры розничные инвесторы"
.venv/bin/python query.py --kind attribute --top-k 10 "ОФЗ-ИН"
.venv/bin/python query.py                                     # встроенный смоук-набор
```

В выдаче префикс kind (`dec` / `att`), `decision_id` и заголовок. Для атрибутов через `::` дописывается `report_name` (имя родительского решения).

### По решению (найти похожие)

```bash
.venv/bin/python query.py --like-decision 3571 --top-k 10
```

Берёт все атрибуты решения `3571`, склеивает в один большой запрос, маскирует само решение в выдаче, группирует результаты по `decision_id` — выдаёт ОДИН лучший документ на каждое соседнее решение. Используется как sanity-check «найди похожих».

### Фильтр по типу атрибута

```bash
.venv/bin/python query.py "инфляция" --search-among Показатель Измерение
```

Берёт только документы, у которых поле `type` содержит хотя бы одно из перечисленных значений (case-insensitive). Можно комбинировать с `--like-decision` и `--kind`:

```bash
.venv/bin/python query.py --like-decision 3571 --search-among Показатель
```

### Coverage-анализ

«Какие решения покрывают атрибуты целевого?»

```bash
.venv/bin/python coverage.py 3571                    # дефолты: top-K=5, threshold=1.0, workers=20
.venv/bin/python coverage.py 3571 --top-k 3 --threshold 1.5 --workers 8
```

Каждый атрибут решения `3571` голосует за решения, в которых нашлись похожие атрибуты. Голос распределяется пропорционально BM25-скору внутри top-K. Результат пишется в `output/coverage_<safe_decision_id>.json`.

## Диагностика частот токенов

Отдельный инспектор для понимания, что лежит в индексе и какие леммы стоит добавить в стоп-слова. Ничего в индексе не меняет — только читает.

```bash
.venv/bin/python inspect_tokens.py                           # сводка + top-50 по df
.venv/bin/python inspect_tokens.py --top 100                 # больше строк топа
.venv/bin/python inspect_tokens.py --kind decision           # df только по решениям
.venv/bin/python inspect_tokens.py --kind attribute          # df только по атрибутам
.venv/bin/python inspect_tokens.py --min-df 200              # только частые леммы
.venv/bin/python inspect_tokens.py --save                    # + дамп в output/token_stats.json
```

Печатает:
- общую статистику индекса (число документов, словарь, avgdl),
- распределение `df` (min/max/percentiles, доля лемм с `df=1` и `df > N/2`),
- топ-N по `df` (в скольких документах термин) и по `cf` (общее число вхождений),
- для каждой леммы — `df`, `%docs`, `cf`, `idf`.

Что с этим делать:
- Леммы с `df > N/2` или `%docs > 30%` — кандидаты в стоп-слова (они «во всём корпусе» и BM25 на них не различает).
- Леммы с `df=1` — уникальные термины, на них поиск опирается сильнее всего, удалять не нужно.
- Если кандидат найден — добавьте его в `RU_STOPWORDS` в [bm25/tokenizer.py](bm25/tokenizer.py), `rm -rf data/index && python add_doc.py ...` (jsonl не трогаем), повторно прогнать `inspect_tokens.py` и проверить, что поиск/coverage не сильно поплыли.

## Добавление новой колонки в индекс (расширение схемы)

Главная инвариантность: **новая колонка в SQL/CSV автоматически попадает в индекс**.

1. Дописываем колонку в `SELECT` (db/decisions.sql или db/attributes.sql) либо в CSV.
2. `rm -rf data && python add_doc.py <источник>` — пересборка.
3. Готово, новое поле теперь индексируется.

Если новую колонку **не нужно** индексировать (например, числовой identifier или служебное поле) — допишите её имя в `INDEX_BLACKLIST` в [bm25/builder.py](bm25/builder.py). Это единственное место в Python, где может потребоваться правка.

Сейчас в `INDEX_BLACKLIST` лежат: `id`, `owner`, `type`, `parent_dataset`, `frequency`, `first_timestamp`, `basis_document`, `comments`, плюс служебные поля документа (`doc_id`, `kind`, `decision_id`, `n_elements`, `element_names`).

## Структура репозитория

```
bm25/                         ядро (пакет, импортируется как `import bm25`)
  tokenizer.py                Tokenizer (razdel + pymorphy3 + стоп-слова)
  index.py                    BM25Index + формула score()
  builder.py                  Схема + нормализаторы + сборка индекса
  storage.py                  Save/load индекса + LazyDocuments (lazy JSONL)
  searcher.py                 Searcher + open_searcher / rebuild_and_save
  sources/                    Источники документов:
    csv_source.py             CSVSource (локальные CSV + alias-маппинг колонок)
    sql_source.py             SQLSource (Spark/Hive, lazy import pyspark, батчинг)

add_doc.py                    CLI: CSV/SQL → инжест и пересборка индекса
query.py                      CLI: поиск (text / --like-decision / --search-among)
coverage.py                   CLI: анализ покрытия атрибутов
inspect_tokens.py             CLI: диагностика частот токенов (df / cf / idf)

db/
  decisions.sql               Решения из БД (прод-аналог pub_report.csv)
  attributes.sql              Атрибуты из БД (прод-аналог Показатели.csv)
  database_description.txt    Схема таблиц источника, для справки

data/                         (создаётся скриптами)
  docs.jsonl                  Источник правды — append-only
  index/                      Numpy/scipy-кэш индекса

docs/
  index_anatomy.md            Подробно: как устроен индекс изнутри

pub_report.csv, Показатели.csv, pub_elements.csv  — локальные тестовые данные
```

## Параметры и тонкости

| Что | Где задаётся | По умолчанию |
|---|---|---|
| Гиперпараметры BM25 (`k1`, `b`) | `DEFAULT_K1`, `DEFAULT_B` в `bm25/index.py` | 1.5 / 0.75 |
| Минимальная длина токена | `Tokenizer.min_len` в `bm25/tokenizer.py` | 2 |
| Стоп-слова | `RU_STOPWORDS` в `bm25/tokenizer.py` | базовый русский список |
| Что не индексируется | `INDEX_BLACKLIST` в `bm25/builder.py` | служебные + id/owner/type/… |
| Размер группы атрибутов | `--group-size N` в `add_doc.py` | 1 |
| Threshold для coverage | `--threshold` в `coverage.py` | 1.0 |
| Воркеры coverage | `--workers` в `coverage.py` | 20 (Linux fork) |

## Где смотреть детали

- **Архитектура индекса** (что в каждом файле `data/index/`, как устроены параллельные массивы, lazy-load документов, формула BM25): [docs/index_anatomy.md](docs/index_anatomy.md).
- **Перенос в закрытый контур**: pyspark обычно уже есть; нужно убедиться, что доступны `razdel`, `pymorphy3`, `pymorphy3-dicts-ru`, `dawg-python`, `numpy`, `scipy`. Если чего-то нет — собрать оффлайн-bundle, см. историю проекта.
