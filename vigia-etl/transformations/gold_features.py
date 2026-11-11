# Gold - features por provincia y SE de emisión, con los casos de las 4 semanas siguientes
from pyspark import pipelines as dp
from pyspark.sql import functions as F

from utilities import features as ft


@dp.materialized_view(
    name="gold.features_provincia_semana",
    comment="Una fila por provincia y SE t, emitida 7 días después de su cierre, solo con lo que se sabía ese "
            "día: las gacetas de la SE t o anteriores publicadas hasta entonces, el clima y el Niño hasta la SE t, "
            "y el canal y lo normal del clima hechos con años anteriores. y_h1 a y_h4 son los casos finales de "
            "t+1 a t+4: lo que se pronostica.",
    cluster_by=["dpa_provin", "inicio_semana"])
@dp.expect("con_casos_recientes", "semanas_con_dato >= 8")
@dp.expect("con_clima", "anomalia_lluvia_0_3 IS NOT NULL")
@dp.expect("con_nino", "nino12_lag0 IS NOT NULL")
@dp.expect_or_fail("objetivos_no_negativos", "coalesce(least(y_h1, y_h2, y_h3, y_h4), 0) >= 0")
def features_provincia_semana():
    semanas = ft.semanas(spark.read.table("gold.dim_calendario_epi"))
    return ft.features(
        gacetas_df=ft.gacetas(spark.read.table("silver.dengue_casos"), semanas),
        semanas_df=semanas,
        clima=spark.read.table("silver.clima_provincia_semana"),
        nino=spark.read.table("silver.nino12_semana"),
        poblacion=spark.read.table("silver.poblacion_provincia_semana"),
        canal=spark.read.table("gold.canal_endemico"),
        casos_finales=spark.read.table("gold.dengue_semana"),
        provincias=spark.read.table("gold.dim_geografia").select("dpa_provin", "provincia").distinct())


@dp.materialized_view(
    name="chequeo_features",
    private=True,
    comment="Control: una sola fila por provincia y semana de emisión.")
@dp.expect_or_fail("una_fila_por_emision", "filas = emisiones_distintas")
def chequeo_features():
    return spark.read.table("gold.features_provincia_semana").agg(
        F.count("*").alias("filas"), F.countDistinct("dpa_provin", "inicio_semana").alias("emisiones_distintas"))