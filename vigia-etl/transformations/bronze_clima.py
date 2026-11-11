# Bronze - clima diario y lluvia de Open-Meteo.
from pyspark import pipelines as dp
from pyspark.sql import functions as F

LANDING = "/Volumes/vigia/bronze/landing/"


def respuestas_de(carpeta):
    # Sin inferir tipos, Auto Loader deja cada valor como texto: los números se convierten en silver
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("pathGlobFilter", "*.json")
        .load(LANDING + carpeta)
        .withColumn("archivo_origen", F.col("_metadata.file_path"))
        .withColumn("cargado_en", F.current_timestamp())
    )


@dp.table(
    name="bronze.clima_crudo",
    comment="Respuestas de Open-Meteo con ERA5-Land (temperatura y humedad): una fila por cantón y archivo, con "
            "las series diarias como listas, tal como llegan.")
def clima_crudo():
    return respuestas_de("clima/")


@dp.table(
    name="bronze.lluvia_crudo",
    comment="Respuestas de Open-Meteo con ERA5 (lluvia): una fila por cantón y archivo, con la serie diaria como "
            "lista, tal como llega.")
def lluvia_crudo():
    return respuestas_de("lluvia/")