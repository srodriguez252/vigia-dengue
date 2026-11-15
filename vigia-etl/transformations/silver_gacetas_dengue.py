# Silver - gacetas del MSP: casos de dengue por provincia y semana, con una versión por fecha de corte y la fecha
# desde la que cada gaceta estaba publicada.
from pyspark import pipelines as dp
from pyspark.sql import Window
from pyspark.sql import functions as F

from utilities import publicacion as pb


def inicio_semana(anio, semana):
    """Domingo en que empieza la semana epidemiológica: la SE 1 es la que contiene el 4 de enero"""
    cuatro_de_enero = F.make_date(anio, F.lit(1), F.lit(4))
    primer_domingo = F.date_sub(cuatro_de_enero, F.dayofweek(cuatro_de_enero) - 1)
    return F.date_add(primer_domingo, (semana - 1) * 7)


@dp.materialized_view(
    name="silver.gacetas_dengue",
    comment="Casos de dengue por provincia y semana epidemiológica según cada gaceta del MSP. "
            "Cada gaceta es una versión de su semana, identificada por la fecha de corte. fecha_publicacion es el "
            "día desde el que consta que estaba en línea, según el servidor del MSP, la carpeta de su URL o la "
            "corrida semanal que la vio aparecer; publicacion_origen dice cuál")
@dp.expect_or_drop("semana_valida", "semana_epi IS NOT NULL AND semana_epi BETWEEN 1 AND 53")
@dp.expect("trae_casos_o_tasa", "casos_acumulados IS NOT NULL OR tasa_acumulada IS NOT NULL")
@dp.expect("fila_cuadra", "casos_previos IS NULL OR casos_semana IS NULL OR casos_acumulados IS NULL "
                          "OR casos_previos + casos_semana = casos_acumulados")
@dp.expect("semana_igual_a_la_columna", "semana_columna IS NULL OR semana_columna = semana_epi")
@dp.expect("corte_despues_de_la_semana", "fecha_corte BETWEEN fin_semana AND date_add(fin_semana, 30)")
@dp.expect("publicacion_con_rastro", "publicacion_origen <> 'corte'")
@dp.expect("publicada_desde_el_corte", "fecha_publicacion >= fecha_corte")
def gacetas_dengue():
    leidas = spark.read.table("silver.gacetas_extraidas").where("motivo IS NULL")

    # Una fila por gaceta. Si ni el PDF ni la tabla dicen la semana, queda la del nombre del archivo
    gacetas = (leidas.select("anio", "archivo", "se_archivo", "sha256", "semana_epi", "fecha_corte_pdf").distinct()
               .withColumn("semana_epi", F.coalesce("semana_epi", "se_archivo")))

    # El mismo PDF enlazado en dos semanas cuenta una vez: queda el archivo cuya semana coincide con la del PDF
    mismo_pdf = Window.partitionBy("sha256").orderBy(
        F.col("se_archivo").eqNullSafe(F.col("semana_epi")).desc(), "archivo")
    gacetas = gacetas.withColumn("n", F.row_number().over(mismo_pdf)).where("n = 1").drop("n")

    # La versión de cada gaceta es su fecha de corte; si el PDF no la trae, el sábado en que cierra la semana
    gacetas = (gacetas
               .withColumn("inicio_semana", inicio_semana(F.col("anio"), F.col("semana_epi")))
               .withColumn("fin_semana", F.date_add("inicio_semana", 6))
               .withColumn("corte_estimado", F.col("fecha_corte_pdf").isNull())
               .withColumn("fecha_corte", F.coalesce("fecha_corte_pdf", "fin_semana")))
    semana = Window.partitionBy("anio", "semana_epi")
    gacetas = (gacetas
               .withColumn("version", F.row_number().over(semana.orderBy("fecha_corte", "archivo")))
               .withColumn("es_ultima_version", F.col("version") == F.count("*").over(semana)))
    # Desde cuándo estaba en línea: el índice de la descarga guarda los rastros de la subida de cada PDF
    gacetas = pb.con_publicacion(gacetas, spark.read.table("bronze.gacetas_indice"))

    return (
        leidas.where("NOT es_total")
        .drop("semana_epi", "se_archivo", "sha256", "fecha_corte_pdf")
        .join(gacetas, ["anio", "archivo"])
        .select("anio", "semana_epi", "inicio_semana", "fin_semana", "fecha_corte", "corte_estimado",
                "fecha_publicacion", "publicacion_desde", "publicacion_origen",
                "version", "es_ultima_version", "provincia", "provincia_pdf",
                "casos_previos", "casos_semana", "casos_acumulados", "tasa_previa", "tasa_semana", "tasa_acumulada",
                "familia", "semana_columna", "pagina", "archivo", "se_archivo")
    )