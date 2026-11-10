# Bronze - población cantonal del INEC
from pyspark import pipelines as dp
from pyspark.sql import functions as F

LANDING_POBLACION = "/Volumes/vigia/bronze/landing/poblacion/"


@dp.table(
    name="bronze.poblacion_inec",
    comment="Población por cantón y año del INEC (Estimaciones y Proyecciones, revisión 2024 con áreas): "
            "una fila por cantón y año de cada archivo CSV")
def poblacion_inec():
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("header", "true")
        # Solo los CSV: el Excel original queda al lado, en la carpeta original/
        .option("pathGlobFilter", "*.csv")
        # Los códigos como texto, para no perder el cero de la izquierda (0901)
        .option("cloudFiles.schemaHints",
                "dpa_provin STRING, dpa_canton STRING, anio INT, poblacion BIGINT, sha256 STRING")
        .load(LANDING_POBLACION)
        .withColumn("archivo_origen", F.col("_metadata.file_path"))
        .withColumn("cargado_en", F.current_timestamp())
    )