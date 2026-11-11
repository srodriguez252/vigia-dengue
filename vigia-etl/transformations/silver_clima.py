# Silver - clima por cantón: diario y por semana epidemiológica
from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

# Variable de Open-Meteo -> columna de la tabla, por fuente. En Open-Meteo, ERA5-Land no trae lluvia: la
# temperatura y la humedad salen de ERA5-Land (celdas de 0,1°) y la lluvia, de ERA5 (celdas de 0,25°)
TEMPERATURA = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "relative_humidity_2m_mean": "humedad",
}
LLUVIA = {"precipitation_sum": "lluvia_mm"}


def dias_de(tabla, variables):
    """Un cantón y un día por fila, desde una tabla bronze de Open-Meteo. Si un día se bajó dos veces, gana la
    descarga más reciente"""
    # Auto Loader, sin inferir tipos, guarda cada campo del JSON como texto, también el objeto daily:
    # from_json lo convierte otra vez en una estructura, con una lista por variable
    esquema = "time ARRAY<STRING>, " + ", ".join(f"{v} ARRAY<DOUBLE>" for v in variables)
    diario = F.from_json("daily", esquema)
    listas = [diario["time"].alias("fechas")] + [diario[v].alias(c) for v, c in variables.items()]
    dias = (
        spark.read.table(tabla)
        .select("dpa_canton", "descargado_en", "elevacion", *listas)
        # Listas paralelas (una fecha y un valor por día): arrays_zip las une por posición y explode deja un
        # día por fila
        .select("dpa_canton", "descargado_en", "elevacion",
                F.explode(F.arrays_zip("fechas", *variables.values())).alias("d"))
        .select("dpa_canton",
                F.to_date("d.fechas").alias("fecha"),
                *[F.col(f"d.{c}").cast("double").alias(c) for c in variables.values()],
                F.col("elevacion").cast("double").alias("elevacion_m"),
                F.to_timestamp("descargado_en").alias("descargado_en"))
    )
    mas_reciente = Window.partitionBy("dpa_canton", "fecha").orderBy(F.col("descargado_en").desc())
    return dias.withColumn("n", F.row_number().over(mas_reciente)).where("n = 1").drop("n")


@dp.materialized_view(
    name="silver.clima_diario",
    comment="Clima diario por cantón: temperatura máxima y mínima (Grados C) y humedad relativa media (%) de ERA5-Land, "
            "y lluvia (mm) de ERA5, ambos vía Open-Meteo. Si un día se bajó dos veces, gana la descarga más reciente",
    cluster_by=["dpa_canton", "fecha"])
@dp.expect_or_drop("con_temperatura", "temp_max IS NOT NULL AND temp_min IS NOT NULL")
@dp.expect("con_lluvia", "lluvia_mm IS NOT NULL")
@dp.expect("temperaturas_coherentes", "temp_max >= temp_min")
@dp.expect("temperatura_razonable", "temp_min > -15 AND temp_max < 45")
@dp.expect("lluvia_razonable", "lluvia_mm IS NULL OR lluvia_mm BETWEEN 0 AND 500")
@dp.expect("humedad_razonable", "humedad IS NULL OR humedad BETWEEN 0 AND 100")
def clima_diario():
    temperatura = dias_de("bronze.clima_crudo", TEMPERATURA)
    lluvia = (dias_de("bronze.lluvia_crudo", LLUVIA)
              .drop("elevacion_m")  # queda la de la celda de ERA5-Land
              .withColumnRenamed("descargado_en", "lluvia_descargada_en"))
    # left: un día que todavía no tiene lluvia queda con lluvia_mm nula, y con_lluvia lo cuenta.
    return temperatura.join(lluvia, ["dpa_canton", "fecha"], "left")


@dp.materialized_view(
    name="chequeo_clima",
    private=True,
    comment="Control: el clima diario no puede quedar vacío, y casi todos sus días deben traer lluvia.")
@dp.expect_or_fail("con_filas", "dias > 0")
@dp.expect_or_fail("lluvia_casi_completa", "dias_sin_lluvia <= 0.01 * dias")
def chequeo_clima():
    # Una expectativa mira fila por fila: no ve que la tabla quedó vacía. Este conteo sí
    return (spark.read.table("silver.clima_diario")
            .agg(F.count("*").alias("dias"), F.count_if(F.col("lluvia_mm").isNull()).alias("dias_sin_lluvia")))


@dp.materialized_view(
    name="silver.clima_semana",
    comment="Clima por cantón y semana epidemiológica: lluvia acumulada, días con lluvia y promedios de "
            "temperatura y humedad.")
@dp.expect("semana_completa", "dias = 7")
@dp.expect("lluvia_completa", "dias_con_dato_lluvia = dias")
def clima_semana():
    calendario = spark.read.table("gold.dim_calendario_epi").select("fecha", "anio_epi", "semana_epi", "inicio_semana")
    return (
        spark.read.table("silver.clima_diario")
        # El calendario es pequeño (un día por fila): con broadcast viaja entero a cada tarea y el join no
        # necesita mezclar datos entre máquinas
        .join(F.broadcast(calendario), "fecha")
        .groupBy("dpa_canton", "anio_epi", "semana_epi", "inicio_semana")
        .agg(F.count("*").alias("dias"),
             F.count("lluvia_mm").alias("dias_con_dato_lluvia"),
             F.round(F.sum("lluvia_mm"), 1).alias("lluvia_mm"),
             F.count_if(F.col("lluvia_mm") >= 1).alias("dias_con_lluvia"),
             F.round(F.avg("temp_max"), 2).alias("temp_max"),
             F.round(F.avg("temp_min"), 2).alias("temp_min"),
             F.round(F.avg((F.col("temp_max") + F.col("temp_min")) / 2), 2).alias("temp_media"),
             F.round(F.avg("humedad"), 1).alias("humedad"))
    )