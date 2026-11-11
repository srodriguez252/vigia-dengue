# Gold - calendario epidemiológico: un día por fila, con su semana epidemiológica (SE) de domingo a sábado
from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Del primer año de población del INEC (2010) a varios años de pronóstico
DESDE, HASTA = "2010-01-01", "2030-12-31"


def inicio_semana_1(anio):
    """Domingo en que empieza la SE 1 de un año: el de la semana que contiene el 4 de enero"""
    cuatro_de_enero = F.make_date(anio, F.lit(1), F.lit(4))
    return F.date_sub(cuatro_de_enero, F.dayofweek(cuatro_de_enero) - 1)


@dp.materialized_view(
    name="gold.dim_calendario_epi",
    comment="Un día por fila con su año y semana epidemiológica: de domingo a sábado, "
            "y la SE 1 es la que contiene el 4 de enero")
@dp.expect_or_fail("semana_valida", "semana_epi BETWEEN 1 AND semanas_del_anio")
@dp.expect_or_fail("anio_de_52_o_53_semanas", "semanas_del_anio IN (52, 53)")
@dp.expect_or_fail("de_domingo_a_sabado", "dayofweek(inicio_semana) = 1 AND fin_semana = date_add(inicio_semana, 6)")
def dim_calendario_epi():
    dias = spark.sql(f"SELECT explode(sequence(DATE'{DESDE}', DATE'{HASTA}', INTERVAL 1 DAY)) AS fecha")
    # dayofweek: 1 = domingo 7 = sábado. La semana es del año de su miércoles: así la SE 1 tiene al
    # menos cuatro días del año nuevo y contiene el 4 de enero
    domingo = F.date_sub("fecha", F.dayofweek("fecha") - 1)
    return (
        dias
        .withColumn("inicio_semana", domingo)
        .withColumn("fin_semana", F.date_add("inicio_semana", 6))
        .withColumn("anio_epi", F.year(F.date_add("inicio_semana", 3)))
        .withColumn("semana_epi",
                    (F.datediff("inicio_semana", inicio_semana_1(F.col("anio_epi"))) / 7 + 1).cast("int"))
        .withColumn("semanas_del_anio",
                    (F.datediff(inicio_semana_1(F.col("anio_epi") + 1), inicio_semana_1(F.col("anio_epi"))) / 7)
                    .cast("int"))
        .select("fecha", "anio_epi", "semana_epi", "inicio_semana", "fin_semana", "semanas_del_anio",
                F.dayofweek("fecha").alias("dia_de_la_semana"))
    )