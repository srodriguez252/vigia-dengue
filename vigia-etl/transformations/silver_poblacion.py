# Silver - población por cantón y año, y por provincia y semana epidemiológica
from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F


@dp.materialized_view(
    name="silver.poblacion_canton",
    comment="Población al 30 de junio por cantón (DPA del Censo 2022) y año, de 2010 a 2035. "
            "Si el INEC publica otra versión, gana el archivo cargado más tarde.")
@dp.expect_or_fail("codigo_dpa", "dpa_canton RLIKE '^[0-9]{4}$' AND left(dpa_canton, 2) = dpa_provin")
@dp.expect_or_fail("poblacion_positiva", "poblacion > 0")
@dp.expect_or_fail("anio_de_la_proyeccion", "anio BETWEEN 2010 AND 2035")
def poblacion_canton():
    mas_reciente = Window.partitionBy("dpa_canton", "anio").orderBy(
        F.col("cargado_en").desc(), F.col("archivo_origen").desc())
    return (
        spark.read.table("bronze.poblacion_inec")
        .withColumn("n", F.row_number().over(mas_reciente))
        .where("n = 1")
        .select("dpa_provin", "dpa_canton", "anio", "poblacion", "sha256", "archivo_origen")
    )


def treinta_de_junio(anio):
    return F.make_date(anio, F.lit(6), F.lit(30))


@dp.materialized_view(
    name="silver.poblacion_provincia_semana",
    comment="Población de cada provincia en cada SE: interpolación lineal entre el 30 de junio anterior y el "
            "siguiente al miércoles de la semana. Sirve para tasas por 100 000 habitantes")
@dp.expect_or_fail("poblacion_positiva", "poblacion > 0")
@dp.expect_or_fail("fraccion_entre_0_y_1", "fraccion >= 0 AND fraccion < 1")
def poblacion_provincia_semana():
    anual = (spark.read.table("silver.poblacion_canton")
             .groupBy("dpa_provin", "anio")
             .agg(F.sum("poblacion").alias("poblacion")))
    semanas = (spark.read.table("gold.dim_calendario_epi")
               .select("anio_epi", "semana_epi", "inicio_semana").distinct()
               .withColumn("miercoles", F.date_add("inicio_semana", 3)))
    # El 30 de junio anterior al miércoles (anio0) y qué parte del año va desde ahí
    anio_miercoles = F.year("miercoles")
    semanas = (semanas
               .withColumn("anio0", F.when(F.col("miercoles") >= treinta_de_junio(anio_miercoles), anio_miercoles)
                           .otherwise(anio_miercoles - 1))
               .withColumn("fraccion",
                           F.datediff("miercoles", treinta_de_junio(F.col("anio0")))
                           / F.datediff(treinta_de_junio(F.col("anio0") + 1), treinta_de_junio(F.col("anio0")))))
    antes = anual.select("dpa_provin", F.col("anio").alias("anio0"), F.col("poblacion").alias("p0"))
    despues = anual.select("dpa_provin", (F.col("anio") - 1).alias("anio0"), F.col("poblacion").alias("p1"))
    # Las semanas sin un 30 de junio a cada lado (antes de mediados de 2010) quedan fuera con el join
    return (
        semanas.join(antes, "anio0")
        .join(despues, ["dpa_provin", "anio0"])
        .select("dpa_provin", "anio_epi", "semana_epi", "inicio_semana", "fraccion",
                F.round(F.col("p0") + (F.col("p1") - F.col("p0")) * F.col("fraccion")).cast("long").alias("poblacion"))
    )