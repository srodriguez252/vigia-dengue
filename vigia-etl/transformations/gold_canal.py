# Gold - canal endémico por provincia, año y SE
from pyspark import pipelines as dp

from utilities import features as ft


@dp.materialized_view(
    name="gold.canal_endemico",
    comment="Canal endémico por provincia, año y SE (cuartiles): resume hasta 7 años previos, sin los epidémicos, "
            "con lo publicado antes de que empiece el año. Zonas: éxito (< q1), seguridad (q1 a mediana), "
            "alerta (mediana a q3) y epidemia (> q3).")
@dp.expect_or_fail("cuartiles_ordenados", "q1 <= mediana AND mediana <= q3")
@dp.expect_or_fail("anios_suficientes", "anios_usados >= 5")
def canal_endemico():
    semanas = ft.semanas(spark.read.table("gold.dim_calendario_epi"))
    return ft.canal_endemico(ft.gacetas(spark.read.table("silver.dengue_casos"), semanas), semanas)