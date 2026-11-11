# Silver - casos de dengue por provincia y semana, también donde la gaceta solo trae tasas
from pyspark import pipelines as dp
from pyspark.sql import functions as F


@dp.materialized_view(
    name="silver.poblacion_msp",
    comment="Población que usa el MSP en sus tasas, por provincia y año: casos acumulados ÷ tasa acumulada × "
            "100 000 en la gaceta del año con la tasa más alta, que es la de menor error de redondeo. "
            "Al lado, la del INEC del mismo año")
@dp.expect_or_fail("poblacion_positiva", "poblacion_msp > 0")
@dp.expect("cerca_del_inec", "abs(diferencia_inec) < 0.25")
def poblacion_msp():
    con_las_dos = spark.read.table("silver.gacetas_dengue").where("casos_acumulados > 0 AND tasa_acumulada > 0")
    implicita = F.col("casos_acumulados") / F.col("tasa_acumulada") * 100000
    msp = (con_las_dos.groupBy("provincia", "anio")
           .agg(F.round(F.max_by(implicita, "tasa_acumulada")).cast("long").alias("poblacion_msp"),
                F.max("tasa_acumulada").alias("tasa_usada"),
                F.countDistinct("semana_epi").alias("semanas_con_casos_y_tasa")))
    inec = (spark.read.table("silver.poblacion_canton")
            .groupBy(F.col("dpa_provin").alias("provincia"), "anio")
            .agg(F.sum("poblacion").alias("poblacion_inec")))
    return (msp.join(inec, ["provincia", "anio"], "left")
            .withColumn("diferencia_inec", F.round(F.col("poblacion_msp") / F.col("poblacion_inec") - 1, 4)))


@dp.materialized_view(
    name="silver.dengue_casos",
    comment="Casos de dengue por provincia, SE y versión de la gaceta. Donde la gaceta solo trae tasas "
            "(SE 23 de 2025 a SE 30 de 2026), los casos se estiman con la población del MSP de ese año, o con "
            "la del INEC si no hay; casos_origen dice cuáles")
@dp.expect_or_fail("casos_no_negativos", "coalesce(casos_acumulados, 0) >= 0 AND coalesce(casos_semana, 0) >= 0")
@dp.expect("con_casos_acumulados", "casos_acumulados IS NOT NULL")
def dengue_casos():
    msp = spark.read.table("silver.poblacion_msp").select("provincia", "anio", "poblacion_msp", "poblacion_inec")
    inec = (spark.read.table("silver.poblacion_canton")
            .groupBy(F.col("dpa_provin").alias("provincia"), "anio")
            .agg(F.sum("poblacion").alias("poblacion_inec_anio")))
    gacetas = (spark.read.table("silver.gacetas_dengue")
               .join(F.broadcast(msp.drop("poblacion_inec")), ["provincia", "anio"], "left")
               .join(F.broadcast(inec), ["provincia", "anio"], "left"))

    def a_casos(tasa):
        return F.round(tasa * F.col("poblacion_usada") / 100000).cast("long")

    # Las tablas de solo tasas dejan la celda vacía cuando no hubo casos: ahí el acumulado es 0. La tasa de la
    # semana falta en la tabla acumulada (SE 2 de 2026): esos casos de la semana quedan sin dato
    # Primero se marca el origen: después, casos_acumulados ya no está vacío
    solo_tasas = F.col("casos_acumulados").isNull() & F.col("familia").isin("tasas", "acumulada")
    es_tasa = F.col("casos_origen") == "tasa"
    return (
        gacetas
        .withColumn("casos_origen", F.when(solo_tasas, "tasa").otherwise("pdf"))
        .withColumn("poblacion_usada", F.when(es_tasa, F.coalesce("poblacion_msp", "poblacion_inec_anio")))
        .withColumn("fuente_poblacion", F.when(es_tasa, F.when(F.col("poblacion_msp").isNotNull(), "MSP")
                                               .otherwise("INEC")))
        .withColumn("casos_acumulados", F.when(es_tasa, a_casos(F.coalesce(F.col("tasa_acumulada"), F.lit(0.0))))
                    .otherwise(F.col("casos_acumulados")))
        .withColumn("casos_semana", F.when(es_tasa & F.col("tasa_semana").isNotNull(), a_casos(F.col("tasa_semana")))
                    .when(es_tasa & (F.col("semana_epi") == 1), F.col("casos_acumulados"))
                    .when(es_tasa, F.lit(None).cast("long"))
                    .otherwise(F.col("casos_semana")))
        # Los previos salen por resta, para que previos + semana = acumulados también en las filas estimadas.
        .withColumn("casos_previos", F.when(F.col("casos_origen") == "tasa",
                                            F.col("casos_acumulados") - F.col("casos_semana"))
                    .otherwise(F.col("casos_previos")))
        .drop("poblacion_msp", "poblacion_inec_anio")
    )