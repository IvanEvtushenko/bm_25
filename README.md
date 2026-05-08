# BM25 поиск по публикациям ЦБ РФ

Прототип полнотекстового поиска с лемматизацией русского языка. Без СУБД — источник правды лежит в `data/docs.jsonl` (append-only), а индекс держится в `data/index/` в виде numpy/scipy-массивов.

## Архитектура

```
pub_report.csv  ──┐
Показатели.csv  ──┤  add_doc.py  ──▶  data/docs.jsonl  (append-only, источник правды)
                                              │
                                              ▼
                                  [add_doc.py пересобирает]
                                              │
                                              ▼
                                     data/index/  (numpy-кэш)
                                              │
                                              ▼
                                [query.py загружает в RAM]
                                              │
                                              ▼
                                           поиск
```

- **Источник правды** — текстовый JSONL, в него только дописываем (одна запись = одна строка). Существующие записи никогда не меняются.
- **Индекс** — numpy/scipy-кэш на диске. После каждого добавления документов аппендер пересобирает индекс с нуля по `docs.jsonl` и записывает в `data/index/`. Для десятков тысяч документов это секунды; рассинхронизации с источником правды не возникает по построению.
- **Поиск** — `query.py` грузит numpy-массивы в память, формула BM25 — векторизованная по scipy.sparse.

## Типы документов

В одном индексе живут документы двух типов (поле `kind`):

| `kind` | Источник | Уникальный ключ `doc_id` | Индексируемые поля |
|---|---|---|---|
| `publication` | `pub_report.csv` | `pub:<pub_id>` | `name + business_description + keywords` |
| `element` | `Показатели.csv` | `el:<pub_id>#<row_idx>` | `report_name + name + description` |

`pub_id` сохраняется в каждом документе как поле для связи с публикацией, но **не индексируется как текст** (это идентификатор, а не семантика). Список индексируемых полей по типу задан в `INDEX_FIELDS_BY_KIND` в [bm25_numpy.py](bm25_numpy.py) — это точка для правок при смене схемы CSV.

При поиске можно фильтровать по типу: `--kind publication` или `--kind element`.

## Стек

| Компонент | Версия | Лицензия | Назначение |
|---|---|---|---|
| [razdel](https://github.com/natasha/razdel) | 0.5.0 | MIT | Токенизация русского текста |
| [pymorphy3](https://github.com/no-plagiarism/pymorphy3) | 2.0.2 | MIT | Словарная лемматизация (OpenCorpora) |
| numpy | ≥1.26 | BSD | Численные массивы |
| scipy | ≥1.11 | BSD | Разреженная term-doc матрица (CSR/CSC) |

Все библиотеки чистый Python, без нейросетей и сетевых вызовов после установки.

## Структура репозитория

### Исходники

| Файл | Назначение |
|---|---|
| [bm25_numpy.py](bm25_numpy.py) | Ядро. `BM25Index` (vocab, doc_ids, kinds, tf-CSR, doc_lens, df), `Searcher` с фильтром по `kind`, функции `build_from_documents` / `save_index` / `load_index` / `rebuild_and_save` / `open_searcher`, JSONL-хелперы. |
| [add_doc.py](add_doc.py) | Аппендер. Принимает CSV (или JSONL), автоопределяет `kind` по схеме, нормализует записи (выставляет `doc_id`/`kind`/`pub_id`), дедуплицирует по `doc_id`, дописывает в `data/docs.jsonl`, пересобирает и сохраняет numpy-индекс. |
| [query.py](query.py) | Поиск. Грузит индекс из `data/index/`, исполняет запросы. Поддерживает `--kind` и `--top-k`. |
| [bm25_ru.py](bm25_ru.py) | Класс `Tokenizer` (razdel + pymorphy3 + кэш лемм), список `RU_STOPWORDS` и набор знаков пунктуации. |
| [requirements.txt](requirements.txt) | Зависимости. |

### Данные

| Файл | Что внутри |
|---|---|
| [pub_report.csv](pub_report.csv) | 285 публикаций ЦБ РФ. Поля: `pub_id, name, business_description, keywords, ...`. UTF-8 с BOM. |
| [Показатели.csv](Показатели.csv) | 16 068 показателей/измерений публикаций. Из них индексируем `pub_id, report_name, name, description`; остальные колонки игнорируются (схема может меняться). |
| [pub_elements.csv](pub_elements.csv) | Старая версия `Показатели.csv`. Не используется. |
| `data/docs.jsonl` | Источник правды. Одна запись = одна JSON-строка с полями `doc_id`, `kind`, `pub_id` и полезными для отображения данными. |

### Артефакты numpy-индекса (`data/index/`, создаются `add_doc.py`)

| Файл | Содержимое |
|---|---|
| `tf.npz` | `scipy.sparse.csr_matrix` формы `[N × V]` — частоты лемм по документам. |
| `doc_lens.npy` | Длины документов в токенах, `int32[N]`. |
| `df.npy` | Document frequency на термин, `int32[V]`. |
| `vocab.json` | `{лемма: column_index}`. |
| `doc_ids.json` | Список `doc_id` в порядке строк матрицы. |
| `kinds.json` | Список `kind` параллельно `doc_ids` (для фильтрации в выдаче). |
| `meta.json` | Версия индекса, N, размер словаря, avgdl, параметры BM25, поля по `kind`, разбивка по типам. |

IDF не сохраняется — пересчитывается при загрузке (миллисекунды).

## Установка

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Использование

### Полная сборка с нуля

```bash
rm -rf data/                                      # если есть несовместимый старый индекс
.venv/bin/python add_doc.py pub_report.csv        # 285 публикаций
.venv/bin/python add_doc.py Показатели.csv        # 16 068 элементов
```

После этого `data/docs.jsonl` содержит 16 353 записи, индекс — в `data/index/`.

### Поиск

Без фильтра (показатели и публикации в общем пуле):
```bash
.venv/bin/python query.py "инфляционные ожидания населения"
```

Только публикации:
```bash
.venv/bin/python query.py --kind publication "брокеры розничные инвесторы"
```

Только конкретные показатели:
```bash
.venv/bin/python query.py --kind element --top-k 10 "ОФЗ-ИН"
```

Без аргументов — прогон встроенного смоук-набора.

В выдаче префикс `[pub|...]` — публикация, `[ele|...]` — элемент; для элементов в заголовке через `::` дописывается `report_name :: name`.

### Добавление новых данных

```bash
.venv/bin/python add_doc.py new_publications.csv          # автоопределит kind=publication
.venv/bin/python add_doc.py new_indicators.csv --kind element  # явное указание
```

Скрипт идемпотентен: повторно добавленные `doc_id` отфильтровываются, индекс не трогается, если новых записей нет.

### Программный доступ

```python
from pathlib import Path
from bm25_numpy import open_searcher

searcher = open_searcher(Path("data"))
for score, doc in searcher.search("ОФЗ-ИН", top_k=5, kind="element"):
    print(score, doc["doc_id"], doc.get("report_name"), "::", doc["name"])
```

## Параметры и точки настройки

| Что | Где | По умолчанию |
|---|---|---|
| Поля по типу документа | `INDEX_FIELDS_BY_KIND` в [bm25_numpy.py](bm25_numpy.py) | `publication: name+business_description+keywords`, `element: report_name+name+description` |
| Какие колонки CSV сохранять в JSONL | `PUBLICATION_KEEP_FIELDS`, `ELEMENT_KEEP_FIELDS` в [add_doc.py](add_doc.py) | См. файл |
| Стоп-слова | `RU_STOPWORDS` в [bm25_ru.py](bm25_ru.py) | Базовый русский служебный список |
| Минимальная длина токена | `Tokenizer.min_len` | `2` |
| Параметры BM25 | `DEFAULT_K1`, `DEFAULT_B` в [bm25_numpy.py](bm25_numpy.py) | `k1=1.5`, `b=0.75` |

После любого изменения индексируемых полей или стоп-слов нужно удалить `data/index/` и снова запустить аппендер на тех же CSV — индекс пересоберётся.

## Как добавить новый тип документа

Пользовательский кейс: придёт ещё один CSV с другой схемой (например, «Документы»).

1. В [bm25_numpy.py](bm25_numpy.py) добавить запись в `INDEX_FIELDS_BY_KIND`:
   ```python
   "document": ("title", "summary", "tags"),
   ```
2. В [add_doc.py](add_doc.py) дописать `normalize_document` и зарегистрировать в `NORMALIZERS`. По желанию — добавить ветку в `detect_kind` для автоопределения.
3. Запустить `python add_doc.py new_file.csv --kind document`.

Никаких изменений в `bm25_numpy.py` ниже верхней константы не нужно — пайплайн полей по `kind` сделан обобщённо.

## Ограничения

- Индекс полностью держится в памяти. На текущих 16 353 документах словарь 4 612 терминов — единицы МБ. Запас на десятки тысяч элементов есть.
- Аппендер пересобирает индекс целиком с каждой партией. На текущих объёмах — ~2 с; если станет узким местом, добавим инкрементальное расширение vocab + `vstack` новых строк.
- В общем пуле длинные публикации и короткие элементы — `avgdl ≈ 17` (тянут вниз 16 тыс. коротких элементов). BM25 нормализует это через `b`, но при необходимости тонкого контроля имеет смысл BM25F или раздельные индексы.
- Pymorphy3 не знает совсем свежих неологизмов. Для лексики ЦБ покрытие хорошее, но при странной выдаче проверьте OOV.
