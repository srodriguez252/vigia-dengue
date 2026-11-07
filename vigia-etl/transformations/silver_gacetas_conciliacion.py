# Silver - gacetas del MSP : conciliaciones y semanas que faltan.
from pyspark import pipelines as dp
from pyspark.sql import functions as F


def semanas_del_anio(anio):
    """52 o 53: los días entre el comienzo de la SE 1 de este año y la del siguiente, divididos por 7"""
    def inicio(a):
        cuatro_de_enero = F.make_date(a, F.lit(1), F.lit(4))
        return F.date_sub(cuatro_de_enero, F.dayofweek(cuatro_de_enero) - 1)
    return (F.datediff(inicio(anio + 1), inicio(anio)) / 7).cast("int")


@dp.materialized_view(
    name="silver.gacetas_conciliacion",
    comment="Una fila por gaceta: la suma de las provincias contra el total del PDF, "
            "y el acumulado contra el de la semana anterior más los casos de la semana")
@dp.expect("provincias_suman_el_total", "total_pdf IS NULL OR suma_acumulados = total_pdf")
@dp.expect("acumulado_no_baja", "acumulado_anterior IS NULL OR suma_acumulados >= acumulado_anterior")
def gacetas_conciliacion():
    totales = (spark.read.table("silver.gacetas_extraidas").where("es_total")
               .select("anio", "archivo", F.col("casos_acumulados").alias("total_pdf")))
    gacetas = (spark.read.table("silver.gacetas_dengue")
               .groupBy("anio", "semana_epi", "version", "es_ultima_version", "fecha_corte", "familia", "archivo")
               .agg(F.count("*").alias("provincias"),
                    F.sum("casos_semana").alias("casos_semana"),
                    F.sum("casos_acumulados").alias("suma_acumulados"))
               .join(totales, ["anio", "archivo"], "left"))
    # El acumulado de la semana anterior, según su última versión. En la SE 1 no hay: el acumulado empieza de cero
    anterior = (gacetas.where("es_ultima_version")
                .select("anio", (F.col("semana_epi") + 1).alias("semana_epi"),
                        F.col("suma_acumulados").alias("acumulado_anterior")))
    return (
        gacetas.join(anterior, ["anio", "semana_epi"], "left")
        .withColumn("diferencia_con_total", F.col("suma_acumulados") - F.col("total_pdf"))
        # Lo que creció el acumulado y no explican los casos de la semana: casos que llegaron tarde
        # a semanas anteriores (si es negativo, casos que se descartaron).
        .withColumn("agregados_tarde",
                    F.col("suma_acumulados") - F.col("acumulado_anterior") - F.col("casos_semana"))
    )


@dp.materialized_view(
    name="silver.gacetas_semanas_faltantes",
    comment="Semanas epidemiológicas sin casos de dengue en silver.gacetas_dengue, con la razón.")
def gacetas_semanas_faltantes():
    # Los boletines excluidos en la descarga no son gacetas: aquí no cuentan
    indice = spark.read.table("bronze.gacetas_indice").where(~F.col("estado").startswith("excluido"))

    # Las semanas esperadas: todas las de los años cerrados; en el año en curso, hasta la última publicada
    esperadas = (indice.where("se IS NOT NULL").groupBy("anio").agg(F.max("se").alias("ultima"))
                 .withColumn("hasta", F.when(F.col("anio") < F.year(F.current_date()), semanas_del_anio(F.col("anio")))
                             .otherwise(F.col("ultima")))
                 .select("anio", F.explode(F.sequence(F.lit(1), F.col("hasta"))).alias("semana_epi")))
    con_datos = spark.read.table("silver.gacetas_dengue").select("anio", "semana_epi").distinct()
    faltan = esperadas.join(con_datos, ["anio", "semana_epi"], "left_anti")

    # Qué pasó con el enlace de esa semana: si se bajó y qué leyó el parser en el PDF
    leidas = (spark.read.table("silver.gacetas_extraidas")
              .groupBy("anio", "archivo")
              .agg(F.max("semana_epi").alias("semana_pdf"), F.max("motivo").alias("motivo"),
                   F.lit(True).alias("leido")))
    enlaces = (indice.select("anio", F.col("se").alias("semana_epi"), "archivo", "estado")
               .join(leidas, ["anio", "archivo"], "left"))
    razon = (F.when(F.col("estado").isNull(), F.lit("la página del MSP no tiene esta semana"))
             .when(F.col("estado").startswith("ERROR"), F.concat(F.lit("no se pudo bajar: "), F.col("estado")))
             .when(F.col("leido").isNull(), F.lit("el pipeline todavía no lee este PDF"))
             .when(F.col("motivo").isNotNull(), F.concat(F.lit("en cuarentena: "), F.col("motivo")))
             .when(F.col("semana_pdf") != F.col("semana_epi"),
                   F.concat(F.lit("el enlace abre la gaceta de la SE "), F.col("semana_pdf").cast("string")))
             .otherwise(F.lit("se leyó, pero no quedó en silver.gacetas_dengue")))
    return faltan.join(enlaces, ["anio", "semana_epi"], "left").select("anio", "semana_epi", "archivo",
                                                                      razon.alias("razon"))