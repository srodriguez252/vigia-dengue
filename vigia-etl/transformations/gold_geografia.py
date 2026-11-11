# Gold - los cantones del INEC con su provincia, su cabecera y el punto donde se pide el clima
from pyspark import pipelines as dp
from pyspark.sql import functions as F

CANTONES = "/Volumes/vigia/bronze/landing/geografia/cantones.csv"
ESQUEMA = ("dpa_canton STRING, canton STRING, dpa_provin STRING, provincia STRING, cabecera STRING, "
           "latitud DOUBLE, longitud DOUBLE, fuente_punto STRING, lugar_geonames STRING")


@dp.materialized_view(
    name="gold.dim_geografia",
    comment="Los 221 cantones del INEC (DPA del Censo 2022) con su provincia, su cabecera cantonal y un punto "
            "para el clima: el lugar de GeoNames más poblado de la cabecera o, si no hay, su centroide")
@dp.expect_or_fail("codigo_dpa", "dpa_canton RLIKE '^[0-9]{4}$' AND left(dpa_canton, 2) = dpa_provin")
@dp.expect_or_fail("punto_en_ecuador", "latitud BETWEEN -5.1 AND 1.7 AND longitud BETWEEN -92.1 AND -75.1")
def dim_geografia():
    # Un CSV pequeño de referencia: se lee entero en cada actualización, sin Auto Loader
    return spark.read.option("header", "true").schema(ESQUEMA).csv(CANTONES)


@dp.materialized_view(
    name="chequeo_cantones",
    private=True,
    comment="Control: los mismos 221 cantones en la geografía y en la población del INEC.")
@dp.expect_or_fail("221_cantones", "cantones = 221")
@dp.expect_or_fail("mismos_cantones", "sin_poblacion = 0 AND sin_punto = 0")
def chequeo_cantones():
    geografia = spark.read.table("gold.dim_geografia").select("dpa_canton", F.lit(True).alias("con_punto"))
    poblacion = (spark.read.table("silver.poblacion_canton")
                 .select("dpa_canton").distinct().withColumn("con_poblacion", F.lit(True)))
    return (
        geografia.join(poblacion, "dpa_canton", "full")
        .agg(F.count("*").alias("cantones"),
             F.count_if(F.col("con_poblacion").isNull()).alias("sin_poblacion"),
             F.count_if(F.col("con_punto").isNull()).alias("sin_punto"))
    )