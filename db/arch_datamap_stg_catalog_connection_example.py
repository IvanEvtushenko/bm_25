import pandas as pd
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import datetime


spark = SparkSession.builder.appName("ARCHITECTURE").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
sc = spark.sparkContext
print(sc)

# коннект в spark-контекст в переменной spark, Spark-контекст в переменной sc
# работа со спарк-кластером

print(spark.version)

# db_name - схема в которой формируется инфо-сервис
db_name = 'zp_dm_aso_ddpp_kdb'

# переменные для логирования
etl_month=datetime.date.today().strftime("%Y_%m")
etl_columns="etl_month string, etl_date timestamp, report_date string, form_code string, chapter_code string, mse string, etl_count int, flag_error int"

# переменные для логирования
etl_form="DM"


# Общие переменные
ds_type_concept = """
'3792;#Поставка данных Банка России',
'3757;#Внешние данные',
'3382;#Форма отчетности',
'3665;#Отчет',
'3818;#Выгрузка',
'3795;#Реестры Банка России',
'3787;#Публикация БР',
'3334;#Витрина',
'3805;#Дэшборд',
'3795;#Результат обследования',
'3799;#Результат запроса',
'3756;#Результат бизнес-процесса СП БР',
'3828;#Справочник централизованный'
"""

ds_type_concept_group = """
'3328;#Домен',
'3808;#Группа витрин',
'3838;#Группа объектов данных',
'3693;#АСО. Предметная область',
'3833;#Данные прикладной компоненты ИТС',
'3687;#НИКА. Модель данных'
"""

ds_type_logic = """
'3615;#Раздел/Подраздел',
'3796;#Справочник локальный'
"""

ds_type_concept_not_need = """
'3526;#Таблица',
'3810;#Иное',
'3683;#Ресурс доступа'
"""

data_status_need = """
'Активный'
"""

data_status_project = """
'Проектируется'
"""


# Наборы данных
# Формируем STG по определенным типам наборов
sql_exp=f"""
WITH mp AS (
SELECT m.ds_id, m.iportal_code, i.domen_code, i.dm_sources
FROM {db_name}.arch_datamap_pstg_iportal_mapping m
INNER JOIN {db_name}.arch_datamap_stg_iportal_dm i ON i.dm_code=m.iportal_code
)
SELECT
    cast(d.ds_id as bigint) as ds_id,
    ds_name,
    ds_type as ds_type_code,
    substr(d.ds_type,7,100) as ds_type,
    /*
    case
        when ds_type in ({ds_type_concept}) then 'CONCEPT'
        when ds_type in ({ds_type_concept_group}) then 'CONCEPT_GROUP'
        when ds_type in ({ds_type_logic}) then 'LOGIC'
        when ds_type in ({ds_type_concept_not_need}) then 'OTHER'
        else 'NEW_TYPE' end as ds_type_group_code,
    */
    case
        when ds_type in ({ds_type_concept}) then 'Концептуальный уровень'
        when ds_type in ({ds_type_concept_group}) then 'Группировки концептуального уровня'
        when ds_type in ({ds_type_logic}) then 'Логический уровень'
        when ds_type in ({ds_type_concept_not_need}) then 'Прочие'
        else 'Новый тип (не определен уровень)' end as ds_type_group,
    ds_system as ds_system_code,
    replace(regexp_replace(ds_system,';#[0-9]+',''),';#',';') as ds_system,
    data_status as data_status_code,
    substr(data_status,7,100) as data_status,
    podr_owner as podr_owner_code,
    replace(regexp_replace(substr(podr_owner,7,100),';#[0-9]+',''),';#',';') as podr_owner,
    podr_owner_contacts,
    podr_fact_owner as podr_fact_owner_code,
    replace(regexp_replace(substr(podr_fact_owner,7,100),';#[0-9]+',''),';#',';') as podr_fact_owner,
    podr_fact_owner_contacts,
    data_category as data_category_code,
    update_fio,
    mp.iportal_code,
    concat('https://iportal.cbr.ru/data-store/showcase/',mp.domen_code,'/',mp.iportal_code) as iportal_link,
    mp.dm_sources
FROM {db_name}.arch_datamap_pstg_catalog_datasets d
LEFT JOIN mp ON d.ds_id=mp.ds_id
"""

tbl=spark.sql(sql_exp)

cnt=tbl.count()

print('Наборы данных. Строк:',cnt)

#Сохраняем в Hive
tbl.write.mode("overwrite").saveAsTable(db_name+".arch_datamap_stg_catalog_datasets")

tbl=tbl.unpersist()

del tbl
