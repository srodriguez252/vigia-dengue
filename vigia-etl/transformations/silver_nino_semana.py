# Silver - índice Niño por semana epidemiológica
from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.materialized_view(
    name="silver.nino12_semana",
    comment="Temperatura del mar y anomalía de Niño 1+2 (y de Niño 3.4) por semana epidemiológica. "
            "La semana de NOAA va de domingo a sábado y se fecha por su miércoles: es la misma SE")
@dp.expect_or_fail("fechada_en_miercoles", "dayofweek(semana_centro) = 4")
@dp.expect_or_fail("con_valor", "nino12_anomalia IS NOT NULL")
@dp.expect("anomalia_razonable", "abs(nino12_anomalia) < 6")
def nino12_semana():
    # El calendario empieza en 2010: las semanas anteriores de NOAA quedan fuera con el join
    calendario = spark.read.table("gold.dim_calendario_epi").select(
        F.col("fecha").alias("semana_centro"), "anio_epi", "semana_epi", "inicio_semana", "fin_semana")
    return (
        spark.read.table("silver.nino_oisst")
        .join(calendario, "semana_centro")
        .select("anio_epi", "semana_epi", "inicio_semana", "fin_semana", "semana_centro",
                "nino12_sst", "nino12_anomalia", "nino34_anomalia", "fecha_descarga")
    )


@dp.materialized_view(
    name="nino12_huecos",
    private=True,
    comment="Control: semanas sin dato de Niño entre la primera y la última semana con dato")
@dp.expect("sin_huecos", "semanas_sin_dato = 0")
def nino12_huecos():
    nino = spark.read.table("silver.nino12_semana")
    rango = nino.agg(F.min("inicio_semana").alias("desde"), F.max("inicio_semana").alias("hasta"))
    semanas = (spark.read.table("gold.dim_calendario_epi")
               .select("inicio_semana").distinct()
               .join(rango, F.col("inicio_semana").between(F.col("desde"), F.col("hasta"))))
    return (semanas.join(nino, "inicio_semana", "left_anti")
            .agg(F.count("*").alias("semanas_sin_dato")))