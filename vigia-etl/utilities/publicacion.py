"""Fecha de publicación de cada gaceta: desde qué día se puede decir que estaba en línea

El MSP no publica cuándo sube cada PDF, y la fecha de corte dice hasta cuándo cuentan los datos, no cuándo se
publicaron: muchas gacetas se suben semanas después, varias a la vez. El notebook descargar_gacetas deja en
bronze.gacetas_indice tres rastros de la subida:

- la carpeta de la URL (/uploads/AAAA/MM/): WordPress guarda cada archivo en la carpeta del mes en que se sube
- modificado_servidor: la cabecera Last-Modified, la última vez que el archivo cambió en el servidor. Es el día de
  la subida, salvo que el MSP haya reemplazado el archivo después
- visto_desde: la primera corrida semanal en que el enlace apareció en la página

fecha_publicacion es el día más tardío en que la gaceta pudo publicarse según esos rastros. Las features la usan
como la fecha desde la que se conoce la gaceta: así ninguna emisión usa un PDF que todavía no estaba en línea

Como features.py, recibe DataFrames y devuelve uno: lo usan el pipeline y las pruebas
"""
from pyspark.sql import functions as F

ZONA = "America/Guayaquil"   # la cabecera viene en UTC; el día de la subida es el de Ecuador
DIAS_CERCA = 60              # sin carpeta de subida, la fecha del servidor vale si cae hasta 60 días tras el cierre
CARPETA = r"/uploads/(20\d{2})/(0[1-9]|1[0-2])/"


def rastros(indice):
    """Una fila por archivo del índice, con el mes de su carpeta, la fecha del servidor y desde cuándo se ve

    Si dos enlaces llevan al mismo archivo, vale el rastro más antiguo de cada tipo
    """
    for columna, tipo in (("modificado_servidor", "timestamp"), ("visto_desde", "date")):
        if columna not in indice.columns:   # un índice que escribió la versión anterior del notebook
            indice = indice.withColumn(columna, F.lit(None).cast(tipo))
    mes = F.concat_ws("-", F.regexp_extract("url", CARPETA, 1), F.regexp_extract("url", CARPETA, 2), F.lit("01"))
    return (indice.where(F.col("archivo").isNotNull())
            .groupBy("anio", "archivo")
            .agg(F.min(F.try_to_timestamp(mes, F.lit("yyyy-MM-dd")).cast("date")).alias("mes_subida"),
                 F.min(F.to_date(F.from_utc_timestamp("modificado_servidor", ZONA))).alias("modificado"),
                 F.min("visto_desde").alias("visto_desde")))


def con_publicacion(gacetas, indice):
    """Agrega a cada gaceta su fecha de publicación; `gacetas` trae anio, archivo, fin_semana y fecha_corte

    - fecha_publicacion: el día más temprano entre los que prueban que la gaceta ya estaba en línea
        · la fecha del servidor, si cae en el mes de la carpeta (es la subida original) o, sin carpeta, hasta
          DIAS_CERCA días después del cierre de la semana. Una fecha posterior al mes de la carpeta dice que el
          archivo se reemplazó, no cuándo se publicó
        · visto_desde, cuando la corrida semanal vio aparecer el enlace
        · el último día del mes de la carpeta: se subió ese mes, no se sabe qué día
      Sin ningún rastro, queda la fecha de corte, que es lo que se suponía antes
    - publicacion_desde: el día más temprano en que pudo publicarse (la fecha de corte o el primer día del mes de
      la carpeta). Entre publicacion_desde y fecha_publicacion está la fecha real
    - publicacion_origen: de dónde sale fecha_publicacion: servidor, visto, carpeta o corte (sin rastro)

    La carpeta solo cuenta si su mes no es anterior al del cierre de la semana: en los primeros años hay archivos
    en carpetas de meses anteriores, que no dicen nada de la subida
    """
    g = gacetas.join(F.broadcast(rastros(indice)), ["anio", "archivo"], "left")
    con_carpeta = F.coalesce(F.col("mes_subida") >= F.trunc("fin_semana", "month"), F.lit(False))
    fin_carpeta = F.when(con_carpeta, F.last_day("mes_subida"))
    original = con_carpeta & F.col("modificado").between(F.col("mes_subida"), F.last_day("mes_subida"))
    cerca = ~con_carpeta & (F.col("modificado") <= F.date_add("fin_semana", DIAS_CERCA))
    del_servidor = F.when(original | cerca, F.col("modificado"))
    probado = F.least(del_servidor, F.col("visto_desde"), fin_carpeta)   # least salta los nulos
    desde = F.greatest(F.col("fecha_corte"), F.when(con_carpeta, F.col("mes_subida")))
    return (g
            .withColumn("publicacion_origen",
                        F.when(probado.isNull(), "corte").when(probado == del_servidor, "servidor")
                        .when(probado == F.col("visto_desde"), "visto").otherwise("carpeta"))
            .withColumn("fecha_publicacion", F.coalesce(probado, F.col("fecha_corte")))
            .withColumn("publicacion_desde", F.least(desde, F.col("fecha_publicacion")))
            .drop("mes_subida", "modificado", "visto_desde"))
