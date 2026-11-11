# Silver - clima por provincia y semana, ponderado por la población de cada cantón
from pyspark import pipelines as dp
from pyspark.sql import functions as F

MEDIDAS = ["lluvia_mm", "dias_con_lluvia", "temp_max", "temp_min", "temp_media", "humedad"]


def ponderado(columna):
    """Promedio de los cantones que tienen el dato, cada uno con el peso de su población"""
    peso = F.when(F.col(columna).isNotNull(), F.col("poblacion"))
    return F.round(F.sum(F.col(columna) * F.col("poblacion")) / F.sum(peso), 2).alias(columna)


@dp.materialized_view(
    name="silver.clima_provincia_semana",
    comment="Clima por provincia y semana epidemiológica: el promedio de sus cantones ponderado por la población "
            "del cantón ese año, porque el dengue ocurre donde vive la gente")
@dp.expect_or_fail("con_peso", "poblacion > 0")
@dp.expect("todos_los_cantones", "cantones = cantones_de_la_provincia")
@dp.expect("semana_completa", "dias_min = 7")
def clima_provincia_semana():
    geografia = spark.read.table("gold.dim_geografia").select("dpa_canton", "dpa_provin", "provincia")
    cantones_por_provincia = geografia.groupBy("dpa_provin").agg(F.count("*").alias("cantones_de_la_provincia"))
    poblacion = spark.read.table("silver.poblacion_canton").select(
        "dpa_canton", F.col("anio").alias("anio_epi"), "poblacion")
    # Las dos tablas de la derecha son pequeñas (221 cantones; 221 × 26 años): van con broadcast
    cantones = (spark.read.table("silver.clima_semana")
                .join(F.broadcast(geografia), "dpa_canton")
                .join(F.broadcast(poblacion), ["dpa_canton", "anio_epi"]))
    return (
        cantones.groupBy("dpa_provin", "provincia", "anio_epi", "semana_epi", "inicio_semana")
        .agg(*[ponderado(c) for c in MEDIDAS],
             F.count("*").alias("cantones"),
             F.min("dias").alias("dias_min"),
             F.sum("poblacion").alias("poblacion"))
        .join(F.broadcast(cantones_por_provincia), "dpa_provin")
    )