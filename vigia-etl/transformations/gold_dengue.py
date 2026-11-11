# Gold - casos de dengue por provincia y SE: la serie final, con cada gaceta en su versión más reciente
from pyspark import pipelines as dp
from pyspark.sql import functions as F

from utilities import features as ft


@dp.materialized_view(
    name="gold.dengue_semana",
    comment="Casos de dengue por provincia y SE: lo que crece el acumulado del año entre gacetas consecutivas, "
            "con cada gaceta en su versión más reciente. Si faltan gacetas, el crecimiento se reparte entre las "
            "semanas del hueco (imputado). Incluye la incidencia por 100 000 habitantes. Es la serie que se "
            "pronostica",
    cluster_by=["dpa_provin", "inicio_semana"])
@dp.expect_or_fail("casos_no_negativos", "casos >= 0")
@dp.expect("acumulado_no_baja", "casos_nuevos >= 0")
@dp.expect("hueco_corto", "semanas_cubiertas <= 4")
@dp.expect("con_poblacion", "poblacion IS NOT NULL")
def dengue_semana():
    semanas = ft.semanas(spark.read.table("gold.dim_calendario_epi"))
    provincias = spark.read.table("gold.dim_geografia").select("dpa_provin", "provincia").distinct()
    return ft.casos_por_semana(ft.gacetas(spark.read.table("silver.dengue_casos"), semanas), semanas,
                               spark.read.table("silver.poblacion_provincia_semana"), provincias)


@dp.materialized_view(
    name="chequeo_dengue_semana",
    private=True,
    comment="Control: una sola fila por provincia y semana.")
@dp.expect_or_fail("una_fila_por_semana", "filas = semanas_distintas")
def chequeo_dengue_semana():
    return spark.read.table("gold.dengue_semana").agg(
        F.count("*").alias("filas"), F.countDistinct("dpa_provin", "inicio_semana").alias("semanas_distintas"))