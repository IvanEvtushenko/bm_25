-- Канонический запрос АТРИБУТЫ (показатели/поля решений). Аналог Показатели.csv.
--
-- Каждая строка результата = один атрибут решения.
-- Колонка `id` — идентификатор решения-владельца (ds_id); внутри индекса = decision_id.
--
-- ПРАВИЛО ИНДЕКСАЦИИ:
--   Все колонки этого запроса попадают в JSONL как поля документа-группы
--   (каждая колонка — list[str] длины group_size, по одному значению на атрибут
--   в группе). В BM25-индекс идут ВСЕ строковые колонки, КРОМЕ полей из
--   INDEX_BLACKLIST в bm25/builder.py (служебные + id/owner/type).
--
--   Чтобы добавить новую индексируемую колонку — допишите её в SELECT.
--   Чтобы новая колонка НЕ индексировалась — добавьте её имя в INDEX_BLACKLIST.
--
-- ORDER BY id критичен: группировка атрибутов по group_size требует, чтобы
-- атрибуты одного решения шли подряд.
--
-- {id_filter} — плейсхолдер для батчевого выполнения через add_doc.py:
--   при запуске список id текущего батча подставляется автоматически
--   (см. _spark_iter_batched в bm25/sources/sql_source.py).
-- Если запускать SQL руками (DBeaver/Hive CLI) — замените `({id_filter})`
-- на конкретный список id, например (1000, 1001, 1002).

SELECT
  ds.ds_id AS id,
  ds.podr_owner AS owner,
  ds.ds_type AS type,
  st.col_name AS name,
  st.col_descr AS col_descr
FROM zp_dm_aso_ddpp_kdb.arch_datamap_stg_catalog_datasets AS ds
  JOIN zp_dm_aso_ddpp_kdb.arch_datamap_stg_iportal_dm_struct AS st
    ON ds.ds_name = st.dm_name
WHERE TRUE
  AND ds.data_status = 'Активный'
  AND ds.ds_id IN ({id_filter})
ORDER BY ds.ds_id
