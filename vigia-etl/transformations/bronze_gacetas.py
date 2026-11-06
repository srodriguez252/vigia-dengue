# Bronze - gacetas vectoriales del MSP: un PDF por fila, con su contenido en binario
from pyspark import pipelines as dp
from pyspark.sql import functions as F

LANDING_GACETAS = "/Volumes/vigia/bronze/landing/gacetas"


def entero_de(texto):
    """'2017' → 2017; '' (la expresión regular no encontró nada) → nulo, sin error."""
    return F.when(texto != "", texto.cast("int"))


@dp.table(name="bronze.gacetas", comment="Gacetas vectoriales del MSP: un PDF por fila, con su contenido en binario.")
def gacetas():
    anio = F.regexp_extract("path", r"anio=(\d{4})", 1)
    # SE42__ETV_Gaceta_42.pdf -> 42, la semana que decía la página del MSP; SE00 quiere decir que no decía.
    semana = F.regexp_extract("path", r"/SE(\d{2})__[^/]*$", 1)
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "binaryFile")
        .option("pathGlobFilter", "*.[pP][dD][fF]")  # deja fuera los .parcial de una descarga a medias
        .load(LANDING_GACETAS)
        .select(
            entero_de(anio).alias("anio"),
            F.regexp_extract("path", r"([^/]+)$", 1).alias("archivo"),
            F.when(semana != "00", entero_de(semana)).alias("se_archivo"),
            "path",
            F.col("length").alias("bytes"),
            F.col("modificationTime").alias("modificado_en"),
            F.sha2("content", 256).alias("sha256"),
            "content",
            F.current_timestamp().alias("cargado_en"),
        )
    )