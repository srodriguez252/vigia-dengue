"""Pruebas de la Fase 4 con datos sintéticos de dos provincias, de 2015 a 2023

La principal demuestra que ninguna feature usa datos que no se conocían al emitir: calcula las features dos
veces, con los datos tal cual y con todo lo posterior a una SE cambiado, y exige que las filas emitidas hasta
esa SE salgan idénticas
"""
import datetime as dt
import math

import pytest
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.testing import assertDataFrameEqual

from utilities import features as ft

PROVINCIAS = {"09": "GUAYAS", "13": "MANABÍ"}
DESDE, HASTA = dt.date(2015, 1, 4), dt.date(2023, 12, 30)   # de la SE 1 de 2015 a la SE 52 de 2023
# Gacetas que no existen. La SE 30 de 2022 de Manabí es la semana de corte de la prueba principal
FALTAN = {("09", 2018, 10), ("13", 2018, 10), ("09", 2020, 20), ("09", 2020, 21), ("13", 2022, 30)}
CORREGIDA = ("09", 2021, 30)   # su versión 2 sale 14 días después, con 10 casos más
BAJA = ("13", 2017, 20)        # su acumulado sale 3 casos por debajo del de la semana anterior


def semana_epi(fecha):
    """Año y número de SE: la semana va de domingo a sábado y es del año de su miércoles"""
    domingo = fecha - dt.timedelta(days=fecha.isoweekday() % 7)
    anio = (domingo + dt.timedelta(days=3)).year
    cuatro = dt.date(anio, 1, 4)
    primera = cuatro - dt.timedelta(days=cuatro.isoweekday() % 7)
    return anio, (domingo - primera).days // 7 + 1


def todas_las_semanas():
    filas, dia = [], DESDE
    while dia <= HASTA:
        filas.append((*semana_epi(dia), dia, dia + dt.timedelta(days=6)))
        dia += dt.timedelta(days=7)
    return filas


def casos_reales(provincia, anio, semana):
    """Casos sintéticos: estacionales, con una epidemia en Manabí en 2019"""
    base = {"09": 30.0, "13": 12.0}[provincia]
    estacion = 1 + 0.9 * math.sin(2 * math.pi * (semana - 6) / 52)
    epidemia = 4.0 if (provincia == "13" and anio == 2019) else 1.0
    ruido = 1 + 0.2 * math.sin(semana * 1.7 + anio)
    return float(round(base * estacion * epidemia * ruido))


def construir_datos(spark):
    semanas = todas_las_semanas()
    fin_de = {(a, s): fin for a, s, _, fin in semanas}
    publicada = fin_de[CORREGIDA[1:]] + dt.timedelta(days=14)
    gacetas, acumulado = [], {}
    for anio, semana, _, fin in semanas:
        if anio < 2016:
            continue
        corte = fin + dt.timedelta(days=3 if anio <= 2020 else 0)   # como en los PDF: de 0 a 5 días
        for provincia in PROVINCIAS:
            clave = (provincia, anio)
            previo = acumulado.get(clave, 0)
            acumulado[clave] = previo + casos_reales(provincia, anio, semana)
            if (provincia, anio, semana) in FALTAN:
                continue
            reportado = int(previo) - 3 if (provincia, anio, semana) == BAJA else int(acumulado[clave])
            # Después de la corrección, las gacetas siguientes ya traen los 10 casos de más
            extra = 10 if clave == CORREGIDA[:2] and corte >= publicada else 0
            gacetas.append((provincia, anio, semana, 1, corte, reportado + extra))
    original = next(g for g in gacetas if g[:3] == CORREGIDA)
    gacetas.append((*CORREGIDA, 2, publicada, original[5] + 10))
    clima, nino, poblacion = [], [], []
    for i, (anio, semana, inicio, _) in enumerate(semanas):
        onda = math.sin(2 * math.pi * (semana - 8) / 52)
        nino.append((anio, semana, inicio, round(1.2 * math.sin(2 * math.pi * i / 170), 2)))
        for provincia, sesgo in (("09", 1.0), ("13", 2.5)):   # en Manabí, la lluvia viene inflada
            clima.append((provincia, anio, semana, inicio, round(40 * (1 + 0.8 * onda) * sesgo, 1),
                          round(22 + 1.5 * onda + 0.3 * math.sin(i), 2), round(80 + 5 * onda, 1)))
            poblacion.append((provincia, anio, semana, inicio, {"09": 4_500_000, "13": 1_600_000}[provincia]))
    return {
        "calendario": spark.createDataFrame(semanas, "anio_epi INT, semana_epi INT, inicio_semana DATE, "
                                                     "fin_semana DATE"),
        "gacetas": spark.createDataFrame(gacetas, "provincia STRING, anio INT, semana_epi INT, version INT, "
                                                  "fecha_corte DATE, casos_acumulados BIGINT"),
        "clima": spark.createDataFrame(clima, "dpa_provin STRING, anio_epi INT, semana_epi INT, inicio_semana DATE, "
                                              "lluvia_mm DOUBLE, temp_min DOUBLE, humedad DOUBLE"),
        "nino": spark.createDataFrame(nino, "anio_epi INT, semana_epi INT, inicio_semana DATE, nino12_anomalia DOUBLE"),
        "poblacion": spark.createDataFrame(poblacion, "dpa_provin STRING, anio_epi INT, semana_epi INT, "
                                                      "inicio_semana DATE, poblacion BIGINT"),
        "provincias": spark.createDataFrame(list(PROVINCIAS.items()), "dpa_provin STRING, provincia STRING"),
    }


def construir(d):
    """Lo mismo que hace el pipeline, en el mismo orden: serie final, canal y features"""
    semanas = ft.semanas(d["calendario"])
    gacetas = ft.gacetas(d["gacetas"], semanas)
    finales = ft.casos_por_semana(gacetas, semanas, d["poblacion"], d["provincias"])
    canal = ft.canal_endemico(gacetas, semanas)
    tabla = ft.features(gacetas, semanas, d["clima"], d["nino"], d["poblacion"], canal, finales, d["provincias"])
    return finales, canal, tabla


def con_el_futuro_cambiado(spark, d, corte):
    """Los mismos datos, con todo lo que no se sabía al emitir la SE de corte cambiado: las gacetas de semanas
    posteriores o publicadas después, una corrección tardía de una semana vieja, y el clima, el Niño y la
    población de las semanas posteriores"""
    semana_posterior = ((F.col("anio") > corte.anio_epi)
                        | ((F.col("anio") == corte.anio_epi) & (F.col("semana_epi") > corte.semana_epi)))
    no_se_sabia = semana_posterior | (F.col("fecha_corte") > F.lit(corte.fecha_emision))
    gacetas = d["gacetas"].withColumn(
        "casos_acumulados",
        F.when(no_se_sabia, F.col("casos_acumulados") * 7 + 13).otherwise(F.col("casos_acumulados")))
    tardia = spark.createDataFrame(
        [("09", corte.anio_epi, 10, 9, corte.fecha_emision + dt.timedelta(days=1), 999999)], d["gacetas"].schema)
    despues = F.col("inicio_semana") > F.lit(corte.fin_semana)

    def cambiar(df, columna, cambio):
        return df.withColumn(columna, F.when(despues, cambio).otherwise(F.col(columna)))

    cambiado = dict(d)
    cambiado["gacetas"] = gacetas.unionByName(tardia)
    cambiado["clima"] = cambiar(cambiar(d["clima"], "lluvia_mm", F.col("lluvia_mm") * 3),
                                "temp_min", F.col("temp_min") + 2)
    cambiado["nino"] = cambiar(d["nino"], "nino12_anomalia", F.col("nino12_anomalia") + 2)
    cambiado["poblacion"] = cambiar(d["poblacion"], "poblacion", F.col("poblacion") * 2)
    return cambiado


@pytest.fixture(scope="module")
def datos(spark):
    return construir_datos(spark)


@pytest.fixture(scope="module")
def resultado(datos):
    return construir(datos)   # serie final, canal y features


def test_ninguna_feature_usa_datos_que_no_se_conocian_al_emitir(spark, datos, resultado):
    corte = ft.semanas(datos["calendario"]).where("anio_epi = 2022 AND semana_epi = 30").first()
    original = resultado[2]
    cambiado = construir(con_el_futuro_cambiado(spark, datos, corte))[2]
    columnas = [c for c in original.columns if not c.startswith("y_h")]   # los objetivos sí miran al futuro
    emitidas = F.col("fecha_emision") <= F.lit(corte.fecha_emision)

    assert original.where(emitidas).count() > 0
    assertDataFrameEqual(cambiado.where(emitidas).select(columnas), original.where(emitidas).select(columnas))
    # La prueba solo vale si el cambio sí se nota en las filas emitidas después.
    distintas = original.where(~emitidas).select(columnas).exceptAll(cambiado.where(~emitidas).select(columnas))
    assert distintas.count() > 0


def test_cada_anio_suma_su_ultimo_acumulado(datos, resultado):
    finales = resultado[0]
    total = {(r.dpa_provin, r.anio_epi): r.total
             for r in finales.groupBy("dpa_provin", "anio_epi").agg(F.sum("casos").alias("total")).collect()}
    ultima = Window.partitionBy("provincia", "anio").orderBy(F.desc("semana_epi"), F.desc("fecha_corte"))
    ultimas = datos["gacetas"].withColumn("n", F.row_number().over(ultima)).where("n = 1").collect()
    assert len(ultimas) == len(total) == 16   # 2 provincias × 8 años
    for g in ultimas:
        assert total[(g.provincia, g.anio)] == pytest.approx(g.casos_acumulados, abs=0.05)


def test_las_semanas_sin_gaceta_se_reparten(resultado):
    fila = {r.semana_epi: r for r in resultado[0].where("dpa_provin = '09' AND anio_epi = 2020").collect()}
    reparto = round(sum(casos_reales("09", 2020, s) for s in (20, 21, 22)) / 3, 2)
    assert [fila[s].imputado for s in (19, 20, 21, 22, 23)] == [False, True, True, True, False]
    assert [fila[s].casos for s in (20, 21, 22)] == [reparto] * 3
    assert fila[23].casos == casos_reales("09", 2020, 23)


def test_un_acumulado_que_baja_no_resta(resultado):
    casos = {r.semana_epi: r.casos for r in
             resultado[0].where("dpa_provin = '13' AND anio_epi = 2017 AND semana_epi IN (20, 21)").collect()}
    assert casos[20] == 0
    assert casos[21] == casos_reales("13", 2017, 20) + casos_reales("13", 2017, 21)


def test_una_semana_sin_gaceta_queda_vacia_hasta_la_siguiente(resultado):
    fila = {r.semana_epi: r for r in
            resultado[2].where("dpa_provin = '13' AND anio_epi = 2022 AND semana_epi IN (30, 31)").collect()}
    assert fila[30].casos_lag0 is None and fila[30].rezago_dato == 1
    reparto = round(sum(casos_reales("13", 2022, s) for s in (30, 31)) / 2, 2)
    assert (fila[31].casos_lag0, fila[31].casos_lag1, fila[31].rezago_dato) == (reparto, reparto, 0)


def test_una_gaceta_corregida_cuenta_desde_que_se_publica(resultado):
    c30 = casos_reales(*CORREGIDA)
    fila = {r.semana_epi: r for r in
            resultado[2].where("dpa_provin = '09' AND anio_epi = 2021 AND semana_epi IN (30, 31)").collect()}
    assert fila[30].casos_lag0 == c30            # al emitir la SE 30, la versión 2 todavía no existía
    assert fila[31].casos_lag1 == c30 + 10       # al emitir la SE 31, ya se había publicado
    final = resultado[0].where("dpa_provin = '09' AND anio_epi = 2021 AND semana_epi = 30").first()
    assert (final.casos, final.version) == (c30 + 10, 2)


def test_el_canal_usa_anios_previos_sin_epidemias(resultado):
    canal = resultado[1]
    fila = canal.where("dpa_provin = '13' AND anio_epi = 2023 AND semana_epi = 20").first()
    assert fila.anios_epidemicos == [2019]
    assert fila.anios_usados == 6            # de 2016 a 2022, sin 2019
    assert fila.q1 <= fila.mediana <= fila.q3
    # Sin 5 años previos no hay canal: para 2020 solo están 2016 a 2019.
    assert canal.where("anio_epi = 2020").count() == 0


def test_los_objetivos_son_las_semanas_siguientes(resultado):
    finales, _, tabla = resultado
    fila = tabla.where("dpa_provin = '09' AND anio_epi = 2021 AND semana_epi = 15").first()
    siguientes = {r.semana_epi: r.casos for r in
                  finales.where("dpa_provin = '09' AND anio_epi = 2021 AND semana_epi BETWEEN 16 AND 19").collect()}
    assert [fila.y_h1, fila.y_h2, fila.y_h3, fila.y_h4] == [siguientes[s] for s in (16, 17, 18, 19)]