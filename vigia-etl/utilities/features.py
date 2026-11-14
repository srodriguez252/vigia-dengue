"""Fase 4 - casos semanales, canal endémico y features por provincia y semana epidemiológica (SE)

Cada función recibe DataFrames y devuelve uno. Así el pipeline (transformations/gold_*.py) y las pruebas
(tests/) usan exactamente el mismo código

La regla que atraviesa todo: la fila de la SE t se emite el sábado siguiente a su cierre (fecha_emision) y solo
usa lo que ya se sabía ese día:
- las gacetas de la SE t o anteriores, cada una en la versión publicada hasta esa fecha
- el clima, el Niño y la población de la SE t o anteriores
- el canal endémico y lo normal del clima, hechos con años anteriores
"""
import math

from pyspark.sql import Window
from pyspark.sql import functions as F

PROVINCIAS = [f"{i:02d}" for i in range(1, 25)]   # la 90 (zonas no delimitadas) no tiene clima ni población
DIAS_PARA_EMITIR = 7          # la fila de la SE t se emite 7 días después de que la SE t cierra
SEMANAS_ATRAS = 12            # ventana de cada emisión: la SE t y las 11 anteriores
TRAMOS = ((0, 3), (4, 7), (8, 11))   # tramos de 4 semanas: t−3 a t, t−7 a t−4 y t−11 a t−8
HORIZONTES = (1, 2, 3, 4)
ANIOS_CANAL = 7               # años previos que resume el canal endémico
MIN_ANIOS_CANAL = 5           # con menos años sin epidemia, no hay canal
MIN_SEMANAS_ANIO = 40         # un año entra al canal si tiene al menos estas semanas con dato
REZAGOS_NINO = (0, 4, 8, 12)
SEMANAS_INICIO = 4            # una gaceta de las primeras semanas del año...
FRACCION_CIERRE = 0.5         # ...con al menos esta parte del acumulado del año anterior trae ese año


def semanas(calendario):
    """Una fila por SE, con su inicio, su fin y la fecha en que se emite su fila de features"""
    return (calendario.select("anio_epi", "semana_epi", "inicio_semana", "fin_semana").distinct()
            .withColumn("fecha_emision", F.date_add("fin_semana", DIAS_PARA_EMITIR)))


def gacetas(dengue_casos, semanas_df):
    """Una fila por gaceta, versión y provincia, con el acumulado del año hasta la semana de la gaceta

    Descarta la gaceta de las primeras SEMANAS_INICIO semanas del año cuyo acumulado del país llega a FRACCION_CIERRE
    del acumulado más alto del año anterior que se conocía a su fecha de corte: trae los casos de ese año. La de la
    SE 1 de 2025 suma en su Total los 61 352 casos de 2024 y los 547 de su semana. Su semana queda como un hueco,
    que se reparte con la gaceta siguiente
    """
    por_provincia = (dengue_casos
                     .where(F.col("provincia").isin(PROVINCIAS) & F.col("casos_acumulados").isNotNull())
                     .groupBy(F.col("provincia").alias("dpa_provin"), F.col("anio").alias("anio_epi"), "semana_epi",
                              "version", "fecha_corte")
                     .agg(F.max("casos_acumulados").cast("double").alias("acumulado")))
    gaceta = ["anio_epi", "semana_epi", "version", "fecha_corte"]
    pais = por_provincia.groupBy(*gaceta).agg(F.sum("acumulado").alias("pais"))
    anio_anterior = pais.select((F.col("anio_epi") + 1).alias("anio_siguiente"),
                                F.col("fecha_corte").alias("corte_anterior"), F.col("pais").alias("pais_anterior"))
    con_el_anio_anterior = (pais.where(F.col("semana_epi") <= SEMANAS_INICIO)
                            .join(anio_anterior, (F.col("anio_epi") == F.col("anio_siguiente"))
                                  & (F.col("corte_anterior") <= F.col("fecha_corte")))
                            .groupBy(*gaceta, "pais").agg(F.max("pais_anterior").alias("maximo_anterior"))
                            .where(F.col("pais") >= FRACCION_CIERRE * F.col("maximo_anterior"))
                            .select(*gaceta))
    misma_gaceta = [F.col(f"p.{c}").eqNullSafe(F.col(f"q.{c}")) for c in gaceta]
    return (por_provincia.alias("p").join(con_el_anio_anterior.alias("q"), misma_gaceta, "left_anti")
            .join(semanas_df.select("anio_epi", "semana_epi", "inicio_semana"), ["anio_epi", "semana_epi"])
            .select("dpa_provin", "anio_epi", F.col("semana_epi").alias("gaceta_semana"),
                    F.col("inicio_semana").alias("inicio_gaceta"), "version", "fecha_corte", "acumulado"))


def vigentes(gacetas_df, claves):
    """La versión que vale de cada gaceta: la de fecha de corte más reciente; si empatan, la última versión"""
    orden = Window.partitionBy(*claves, "dpa_provin", "anio_epi", "gaceta_semana").orderBy(
        F.col("fecha_corte").desc(), F.col("version").desc())
    return gacetas_df.withColumn("_n", F.row_number().over(orden)).where("_n = 1").drop("_n")


def casos_semanales(vigentes_df, claves):
    """Los casos de cada semana: lo que crece el acumulado desde la gaceta anterior del mismo año

    Si faltan gacetas entre las dos, el crecimiento se reparte en partes iguales entre las semanas del hueco, que
    quedan como imputadas. Si el acumulado baja (el MSP corrigió hacia abajo), esa semana queda en 0 y las
    siguientes cuentan desde el máximo anterior: así el año suma su acumulado más alto
    """
    anterior = Window.partitionBy(*claves, "dpa_provin", "anio_epi").orderBy("gaceta_semana")
    hasta_aqui = anterior.rowsBetween(Window.unboundedPreceding, Window.currentRow)
    return (vigentes_df
            .withColumn("maximo", F.max("acumulado").over(hasta_aqui))
            .withColumn("semana_previa", F.coalesce(F.lag("gaceta_semana").over(anterior), F.lit(0)))
            .withColumn("maximo_previo", F.coalesce(F.lag("maximo").over(anterior), F.lit(0.0)))
            .withColumn("casos_nuevos",
                        F.col("acumulado") - F.coalesce(F.lag("acumulado").over(anterior), F.lit(0.0)))
            .withColumn("semanas_cubiertas", F.col("gaceta_semana") - F.col("semana_previa"))
            .withColumn("semana_epi", F.explode(F.sequence(F.col("semana_previa") + 1, F.col("gaceta_semana"))))
            .withColumn("casos", (F.col("maximo") - F.col("maximo_previo")) / F.col("semanas_cubiertas"))
            .withColumn("imputado", F.col("semanas_cubiertas") > 1))


def casos_por_semana(gacetas_df, semanas_df, poblacion, provincias):
    """La serie final: los casos de cada provincia y SE con cada gaceta en su versión más reciente"""
    return (casos_semanales(vigentes(gacetas_df, []), [])
            .join(semanas_df.select("anio_epi", "semana_epi", "inicio_semana"), ["anio_epi", "semana_epi"])
            .join(poblacion.select("dpa_provin", "anio_epi", "semana_epi", "poblacion"),
                  ["dpa_provin", "anio_epi", "semana_epi"], "left")
            .join(provincias, "dpa_provin", "left")
            .select("dpa_provin", "provincia", "anio_epi", "semana_epi", "inicio_semana",
                    F.round("casos", 2).alias("casos"), "imputado", "semanas_cubiertas", "casos_nuevos",
                    "gaceta_semana", "version", "fecha_corte", "poblacion",
                    F.round(F.col("casos") / F.col("poblacion") * 100000, 3).alias("incidencia_100k")))


def canal_endemico(gacetas_df, semanas_df):
    """Canal endémico por provincia, año y SE, hecho con lo que se sabía al empezar ese año

    Para el año Y resume hasta ANIOS_CANAL años previos con al menos MIN_SEMANAS_ANIO semanas con dato, sin los
    epidémicos: los que pasan Q3 + 1,5 x RIC de los totales anuales. Cada SE usa sus valores y los de sus dos
    vecinas. Con menos de MIN_ANIOS_CANAL años, no hay canal
    """
    anios = (semanas_df.where("semana_epi = 1")
             .select(F.col("anio_epi").alias("anio_canal"), F.col("inicio_semana").alias("inicio_anio"))
             .crossJoin(gacetas_df.agg(F.max("anio_epi").alias("ultimo_anio")))
             .where(F.col("anio_canal") <= F.col("ultimo_anio") + 1).drop("ultimo_anio"))
    # Para cada año del canal: las gacetas de los años previos, publicadas antes de que empiece
    previas = (anios
               .withColumn("anio_epi", F.explode(F.sequence(F.col("anio_canal") - ANIOS_CANAL,
                                                            F.col("anio_canal") - 1)))
               .join(gacetas_df, "anio_epi")
               .where(F.col("fecha_corte") < F.col("inicio_anio")))
    semanales = casos_semanales(vigentes(previas, ["anio_canal"]), ["anio_canal"])
    totales = (semanales.groupBy("anio_canal", "dpa_provin", "anio_epi")
               .agg(F.count("*").alias("semanas"), F.sum("casos").alias("total"))
               .where(F.col("semanas") >= MIN_SEMANAS_ANIO))
    cuartiles = (totales.groupBy("anio_canal", "dpa_provin")
                 .agg(F.expr("percentile(total, array(0.25, 0.75))").alias("q")))
    limite = F.col("q")[1] + 1.5 * (F.col("q")[1] - F.col("q")[0])
    marcados = (totales.join(cuartiles, ["anio_canal", "dpa_provin"])
                .withColumn("epidemico", F.col("total") > limite))
    resumen = (marcados.groupBy("anio_canal", "dpa_provin")
               .agg(F.count_if(~F.col("epidemico")).alias("anios_usados"),
                    F.sort_array(F.collect_list(F.when(F.col("epidemico"), F.col("anio_epi"))))
                    .alias("anios_epidemicos")))
    usados = marcados.where(~F.col("epidemico")).select("anio_canal", "dpa_provin", "anio_epi")
    # El valor de la SE w de cada año cuenta para las SE w−1, w y w+1 del canal
    valores = (semanales.join(usados, ["anio_canal", "dpa_provin", "anio_epi"])
               .withColumn("semana_canal", F.explode(F.array(F.col("semana_epi") - 1, F.col("semana_epi"),
                                                             F.col("semana_epi") + 1))))
    canal = (valores.groupBy("anio_canal", "dpa_provin", "semana_canal")
             .agg(F.expr("percentile(casos, array(0.25, 0.5, 0.75))").alias("p")))
    del_anio = semanas_df.select(F.col("anio_epi").alias("anio_canal"), F.col("semana_epi").alias("semana_canal"),
                                 "inicio_semana")
    return (canal.join(resumen, ["anio_canal", "dpa_provin"])
            .where(F.col("anios_usados") >= MIN_ANIOS_CANAL)
            .join(del_anio, ["anio_canal", "semana_canal"])  # quita la SE 0, y la SE 53 de los años de 52
            .select("dpa_provin", F.col("anio_canal").alias("anio_epi"), F.col("semana_canal").alias("semana_epi"),
                    "inicio_semana",
                    F.round(F.col("p")[0], 2).alias("q1"), F.round(F.col("p")[1], 2).alias("mediana"),
                    F.round(F.col("p")[2], 2).alias("q3"), "anios_usados", "anios_epidemicos"))


def normales_clima(clima):
    """Lo normal de cada provincia y SE del año Y: el promedio de los años anteriores en esa SE y sus vecinas"""
    base = clima.select("dpa_provin", F.col("anio_epi").alias("anio_base"), F.col("semana_epi").alias("semana_base"),
                        "lluvia_mm", "temp_min", "humedad")
    anios = clima.select("anio_epi").distinct()
    return (base.join(anios, F.col("anio_base") < F.col("anio_epi"))
            .withColumn("semana_epi", F.explode(F.array(F.col("semana_base") - 1, F.col("semana_base"),
                                                        F.col("semana_base") + 1)))
            .where(F.col("semana_epi").between(1, 53))
            .groupBy("dpa_provin", "anio_epi", "semana_epi")
            .agg(F.avg("lluvia_mm").alias("lluvia_normal"), F.avg("temp_min").alias("temp_min_normal"),
                 F.avg("humedad").alias("humedad_normal"), F.countDistinct("anio_base").alias("anios_normal")))


def features(gacetas_df, semanas_df, clima, nino, poblacion, canal, casos_finales, provincias):
    """Una fila por provincia y SE de emisión t: lo que se sabía al emitir y, aparte, los casos de t+1 a t+4"""
    emisiones = (semanas_df.select("anio_epi", "semana_epi", F.col("inicio_semana").alias("inicio_emision"),
                                   "fecha_emision")
                 .crossJoin(gacetas_df.agg(F.min("inicio_gaceta").alias("desde")))
                 .crossJoin(clima.agg(F.max("inicio_semana").alias("hasta")))
                 .where(F.col("inicio_emision").between(F.col("desde"), F.col("hasta")))
                 .drop("desde", "hasta"))
    base = emisiones.crossJoin(provincias.select("dpa_provin"))
    # Las 12 semanas de la ventana de cada emisión; k = 0 es la SE t
    ventana = (emisiones.select("inicio_emision")
               .withColumn("k", F.explode(F.sequence(F.lit(0), F.lit(SEMANAS_ATRAS - 1))))
               .withColumn("inicio_semana", F.date_sub("inicio_emision", F.col("k") * 7)))

    # Casos: para cada emisión, la serie que dan las gacetas de su año y del anterior que ya se conocían
    conocidas = (emisiones
                 .select("inicio_emision", "fecha_emision",
                         F.explode(F.array(F.col("anio_epi") - 1, F.col("anio_epi"))).alias("anio_epi"))
                 .join(gacetas_df, "anio_epi")
                 .where((F.col("inicio_gaceta") <= F.col("inicio_emision"))     # de la SE t o anteriores
                        & (F.col("fecha_corte") <= F.col("fecha_emision"))))    # y ya publicadas
    serie = (casos_semanales(vigentes(conocidas, ["inicio_emision"]), ["inicio_emision"])
             .join(semanas_df.select("anio_epi", "semana_epi", "inicio_semana"), ["anio_epi", "semana_epi"])
             .join(ventana, ["inicio_emision", "inicio_semana"]))
    casos = (serie.groupBy("dpa_provin", "inicio_emision")
             .agg(*[F.max(F.when(F.col("k") == k, F.col("casos"))).alias(f"casos_lag{k}") for k in range(4)],
                  F.avg(F.when(F.col("k") <= 3, F.col("casos"))).alias("media_4s"),
                  F.avg(F.when(F.col("k").between(4, 7), F.col("casos"))).alias("media_4s_previas"),
                  F.avg("casos").alias("media_12s"),
                  F.count("*").alias("semanas_con_dato"),
                  F.min("k").alias("rezago_dato")))

    # Clima: anomalías frente a lo normal de cada semana, en los tres tramos de 4 semanas
    clima_normal = (clima.select("dpa_provin", "anio_epi", "semana_epi", "inicio_semana", "lluvia_mm", "temp_min",
                                 "humedad")
                    .join(normales_clima(clima), ["dpa_provin", "anio_epi", "semana_epi"], "left"))
    agregados = []
    for desde, hasta in TRAMOS:
        en_tramo = F.col("k").between(desde, hasta)
        con_lluvia = en_tramo & F.col("lluvia_mm").isNotNull() & F.col("lluvia_normal").isNotNull()
        lluvia = F.sum(F.when(con_lluvia, F.col("lluvia_mm")))
        normal = F.sum(F.when(con_lluvia, F.col("lluvia_normal")))
        agregados += [
            # Lluvia como razón logarítmica: un sesgo que multiplica (el de ERA5 en la Sierra) se cancela
            F.log((lluvia + 1) / (normal + 1)).alias(f"anomalia_lluvia_{desde}_{hasta}"),
            F.avg(F.when(en_tramo, F.col("temp_min") - F.col("temp_min_normal")))
            .alias(f"anomalia_temp_min_{desde}_{hasta}")]
    ultimas_4 = F.col("k") <= 3
    agregados += [F.avg(F.when(ultimas_4, F.col("humedad") - F.col("humedad_normal"))).alias("anomalia_humedad_0_3"),
                  F.avg(F.when(ultimas_4, F.col("temp_min"))).alias("temp_min_0_3")]
    clima_f = (ventana.join(clima_normal, "inicio_semana")
               .groupBy("dpa_provin", "inicio_emision").agg(*agregados))

    # Niño 1+2: la anomalía de la SE t y de 4, 8 y 12 semanas antes
    nino_f = (emisiones.select("inicio_emision")
              .withColumn("rezago", F.explode(F.array(*[F.lit(r) for r in REZAGOS_NINO])))
              .withColumn("inicio_semana", F.date_sub("inicio_emision", F.col("rezago") * 7))
              .join(nino.select("inicio_semana", "nino12_anomalia"), "inicio_semana", "left")
              .groupBy("inicio_emision")
              .agg(*[F.max(F.when(F.col("rezago") == r, F.col("nino12_anomalia"))).alias(f"nino12_lag{r}")
                     for r in REZAGOS_NINO]))

    poblacion_t = poblacion.select("dpa_provin", F.col("inicio_semana").alias("inicio_emision"), "poblacion")
    canal_t = canal.select("dpa_provin", F.col("inicio_semana").alias("inicio_emision"),
                           F.col("mediana").alias("canal_mediana"), F.col("q3").alias("canal_q3"))

    # Objetivos: los casos finales de las SE t+1 a t+4. No son features: son lo que se quiere pronosticar
    objetivos = (base.select("dpa_provin", "inicio_emision")
                 .withColumn("h", F.explode(F.array(*[F.lit(h) for h in HORIZONTES])))
                 .withColumn("inicio_semana", F.date_add("inicio_emision", F.col("h") * 7))
                 .join(casos_finales.select("dpa_provin", "inicio_semana", "casos"),
                       ["dpa_provin", "inicio_semana"], "left")
                 .groupBy("dpa_provin", "inicio_emision")
                 .agg(*[F.max(F.when(F.col("h") == h, F.col("casos"))).alias(f"y_h{h}") for h in HORIZONTES]))

    tabla = (base
             .join(casos, ["dpa_provin", "inicio_emision"], "left")
             .join(clima_f, ["dpa_provin", "inicio_emision"], "left")
             .join(nino_f, "inicio_emision", "left")
             .join(poblacion_t, ["dpa_provin", "inicio_emision"], "left")
             .join(canal_t, ["dpa_provin", "inicio_emision"], "left")
             .join(objetivos, ["dpa_provin", "inicio_emision"], "left")
             .join(provincias, "dpa_provin", "left"))
    angulo = 2 * math.pi * F.col("semana_epi") / 52.18
    return tabla.select(
        "dpa_provin", "provincia", "anio_epi", "semana_epi", F.col("inicio_emision").alias("inicio_semana"),
        "fecha_emision",
        *[F.round(f"casos_lag{k}", 2).alias(f"casos_lag{k}") for k in range(4)],
        F.round("media_4s", 2).alias("casos_media_4s"),
        F.round("media_12s", 2).alias("casos_media_12s"),
        F.round(F.log((F.col("media_4s") + 1) / (F.col("media_4s_previas") + 1)), 3).alias("crecimiento_4s"),
        F.round(F.col("media_4s") / F.col("poblacion") * 100000, 3).alias("incidencia_4s_100k"),
        "semanas_con_dato", "rezago_dato",
        "canal_mediana", "canal_q3",
        F.round(F.log((F.col("media_4s") + 1) / (F.col("canal_mediana") + 1)), 3).alias("casos_vs_canal"),
        *[F.round(f"anomalia_lluvia_{d}_{h}", 3).alias(f"anomalia_lluvia_{d}_{h}") for d, h in TRAMOS],
        *[F.round(f"anomalia_temp_min_{d}_{h}", 2).alias(f"anomalia_temp_min_{d}_{h}") for d, h in TRAMOS],
        F.round("anomalia_humedad_0_3", 1).alias("anomalia_humedad_0_3"),
        F.round("temp_min_0_3", 1).alias("temp_min_0_3"),
        *[f"nino12_lag{r}" for r in REZAGOS_NINO],
        "poblacion",
        F.round(F.sin(angulo), 4).alias("estacion_sin"), F.round(F.cos(angulo), 4).alias("estacion_cos"),
        *[f"y_h{h}" for h in HORIZONTES])