# Silver - Índice Niño semanal, con columnas y controles de calidad
from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

REGIONES = ["nino12", "nino3", "nino34", "nino4"]

#Validaciones
@dp.materialized_view(
    name="silver.nino_oisst",
    comment="Índice Niño semanal: temperatura del mar y anomalía de 4 regiones del Pacífico.",
)
@dp.expect_or_fail("fecha_valida", "semana_centro IS NOT NULL")
@dp.expect_or_fail("ocho_valores", "n_valores = 8")
@dp.expect("anomalia_razonable", "abs(nino12_anomalia) < 6")


def nino_oisst():
    lineas = (
        spark.read.table("bronze.nino_crudo")
        # Las líneas de datos empiezan con una fecha como 04JAN1990; el resto son encabezados
        .where(F.col("linea").rlike(r"^\s*\d{2}[A-Z]{3}\d{4}\s"))
        # Todos los números con un decimal y signo opcional, aunque el signo venga pegado: 23.4-0.4
        .withColumn("valores", F.regexp_extract_all("linea", F.lit(r"-?\d+\.\d"), 0))
        .withColumn(
            "semana_centro",
            F.to_date(F.regexp_extract("linea", r"(\d{2}[A-Z]{3}\d{4})", 1), "ddMMMyyyy"),
        )
    )

    # Cada descarga trae la serie completa: si una semana aparece en varias, gana la más reciente
    mas_reciente = Window.partitionBy("semana_centro").orderBy(
        F.col("fecha_descarga").desc(), F.col("cargado_en").desc()
    )
    lineas = lineas.withColumn("orden", F.row_number().over(mas_reciente)).where("orden = 1")

    # Cada línea trae SST y anomalía para cada región, en ese orden
    columnas = []
    for i, region in enumerate(REGIONES):
        columnas.append(F.get("valores", 2 * i).cast("double").alias(f"{region}_sst"))
        columnas.append(F.get("valores", 2 * i + 1).cast("double").alias(f"{region}_anomalia"))

    return lineas.select(
        "semana_centro",
        *columnas,
        F.size("valores").alias("n_valores"),
        "fecha_descarga",
        "archivo_origen",
    )
