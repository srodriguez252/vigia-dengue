# Bronze - Índice Niño de NOAA: una fila por línea de cada archivo
from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Carpeta del Volume donde llegan las descargas del índice Niño
LANDING_NINO = "/Volumes/vigia/bronze/landing/nino/"

@dp.table(
    name="bronze.nino_crudo",
    comment="Índice Niño semanal de NOAA CPC: una fila por línea de cada archivo descargado.",
)
def nino_crudo():
    # Auto Loader (cloudFiles) lee solo los archivos nuevos en cada corrida, el pipeline recuerda cuáles ya procesó.
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("pathGlobFilter", "wksst9120.for")
        .load(LANDING_NINO)
        .select(
            F.col("value").alias("linea"),
            F.col("_metadata.file_path").alias("archivo_origen"),
            # La fecha sale de la carpeta fecha_descarga=AAAA-MM-DD, si no está, usa la fecha de subida
            F.coalesce(
                F.try_to_timestamp(
                    F.regexp_extract("_metadata.file_path", r"fecha_descarga=(\d{4}-\d{2}-\d{2})", 1),
                    F.lit("yyyy-MM-dd"),
                ).cast("date"),
                F.to_date("_metadata.file_modification_time"),
            ).alias("fecha_descarga"),
            F.current_timestamp().alias("cargado_en"),
        )
    )