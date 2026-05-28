-- Канонический запрос РЕШЕНИЯ (наборы данных каталога). Аналог pub_report.csv.
--
-- Каждая строка результата = одно решение (один dataset).
-- Колонка `id` — идентификатор решения; используется как pub_id в индексе.
--
-- ПРАВИЛО ИНДЕКСАЦИИ:
--   Все колонки этого запроса попадают в JSONL как поля документа.
--   Из них в BM25-индекс идут ВСЕ строковые колонки, КРОМЕ полей из
--   INDEX_BLACKLIST в bm25/builder.py (служебные + id/owner/type).
--
--   Чтобы добавить новую индексируемую колонку — допишите её в SELECT.
--   Чтобы новая колонка НЕ индексировалась — добавьте её имя в INDEX_BLACKLIST.

SELECT
  ds_id AS id,
  ds_name AS name,
  ds_descr AS descr,
  keywords,
  prod_facr_owner AS owner
FROM zp_dm_aso_ddpp_kdb.arch_datamap_stg_catalog_datasets
WHERE data_status = 'Активный'
