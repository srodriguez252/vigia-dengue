# Bronze - egresos hospitalarios del INEC: una tabla por año, con las columnas tal cual llega
from pyspark import pipelines as dp
from pyspark.sql import functions as F

LANDING_EGRESOS = "/Volumes/vigia/bronze/landing/egresos"

# Todos los años usan ";" como separador, lo que cambia es la codificación
CODIFICACION = {
    2015: "ISO-8859-1",
    2016: "ISO-8859-1",
    2017: "ISO-8859-1",
    2018: "UTF-8",
    2019: "ISO-8859-1",
    2020: "UTF-8",
    2021: "UTF-8",
    2022: "UTF-8",
    2023: "UTF-8",
    2024: "UTF-8",
}


def limpiar(nombre):
    """Quita la marca BOM (U+FEFF) y los espacios del nombre, y lo pasa a minúsculas"""
    return nombre.replace(chr(0xFEFF), "").strip().lower()


def crear_bronze(anio, codificacion):
    @dp.table(
        name=f"bronze.egresos_{anio}",
        comment=f"Egresos hospitalarios {anio} del INEC: una fila por egreso, todas las columnas como texto.",
    )
    def egresos():
        crudo = (
            spark.readStream.format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("header", "true")
            .option("sep", ";")
            .option("encoding", codificacion)
            .option("pathGlobFilter", "*.csv")
            .load(f"{LANDING_EGRESOS}/anio={anio}/")
        )
        return crudo.select(
            *[F.col(f"`{c}`").alias(limpiar(c)) for c in crudo.columns],
            F.col("_metadata.file_path").alias("archivo_origen"),
            F.current_timestamp().alias("cargado_en"),
        )


for anio, codificacion in CODIFICACION.items():
    crear_bronze(anio, codificacion)