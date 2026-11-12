"""Pruebas del pronóstico y de las alertas (src/modelos/pronostico.py). No se usa Spark: solo pandas y scikit-learn

La principal repite la idea de la prueba sin fuga de la Fase 4, ahora con el backtest: corre el origen móvil dos
veces, con los datos tal cual y con todo lo que no se sabía en una fecha de corte cambiado, y exige que los
pronósticos emitidos hasta ese día salgan idénticos
"""
import numpy as np
import pandas as pd
import pytest

from modelos import pronostico as P

CORTE = pd.Timestamp("2022-06-26")      # una emisión en medio del backtest
RAPIDO = dict(max_iter=40)              # menos árboles: la prueba mira qué datos entran, no la precisión


def semana_epi(fechas):
    """Año y número de semana aproximados (basta con que todas las tablas de prueba usen los mismos)"""
    centro = fechas + pd.Timedelta(days=3)
    return centro.year, (centro.dayofyear - 1) // 7 + 1


def features_de_prueba():
    """Tres provincias de 2017 a 2023 con la forma de gold.features_provincia_semana"""
    rng = np.random.default_rng(5)
    inicios = pd.date_range("2017-01-01", "2023-12-31", freq="7D")
    anios, numeros = semana_epi(inicios)
    n = len(inicios)
    filas = []
    for dpa, nombre, nivel, poblacion in (("09", "GUAYAS", 40.0, 4.6e6), ("13", "MANABI", 25.0, 1.6e6),
                                          ("17", "PICHINCHA", 2.0, 3.3e6)):
        estacion = 1 + 0.8 * np.cos(2 * np.pi * (np.arange(n) - 14) / 52.18)
        casos = rng.poisson(nivel * estacion * np.where(anios == 2023, 4.0, 1.0)).astype(float)
        sabido = np.where(rng.random(n) < 0.1, np.maximum(casos + rng.integers(-3, 4, n), 0), casos)   # lo sabido
        for i in range(n):
            lags = [sabido[i - k] if i >= k else np.nan for k in range(4)]
            fila = dict(dpa_provin=dpa, provincia=nombre, anio_epi=int(anios[i]), semana_epi=int(numeros[i]),
                        inicio_semana=inicios[i], fecha_emision=inicios[i] + pd.Timedelta(days=13),
                        poblacion=poblacion, estacion_sin=np.sin(2 * np.pi * numeros[i] / 52.18),
                        estacion_cos=np.cos(2 * np.pi * numeros[i] / 52.18),
                        semanas_con_dato=12.0, rezago_dato=0.0,
                        casos_media_4s=np.nanmean(lags), casos_media_12s=np.mean(sabido[max(0, i - 11):i + 1]),
                        canal_mediana=nivel if anios[i] >= 2020 else np.nan,
                        canal_q3=1.5 * nivel if anios[i] >= 2020 else np.nan)
            fila.update({f"casos_lag{k}": lags[k] for k in range(4)})
            fila.update({v: rng.normal() for v in P.VARIABLES_CLIMA})
            fila.update(crecimiento_4s=rng.normal(), incidencia_4s_100k=fila["casos_media_4s"] / poblacion * 1e5,
                        casos_vs_canal=rng.normal())
            fila.update({f"y_h{h}": casos[i + h] if i + h < n else np.nan for h in P.HORIZONTES})
            filas.append(fila)
    return pd.DataFrame(filas)


def canal_de_prueba(features):
    """Un canal por provincia, año y SE: el del año 2023 es el doble que el de 2022, para distinguirlos"""
    c = features[["dpa_provin", "anio_epi", "semana_epi"]].drop_duplicates()
    nivel = c["dpa_provin"].map({"09": 40.0, "13": 25.0, "17": 2.0}) * np.where(c["anio_epi"] == 2023, 2.0, 1.0)
    return c.assign(mediana=nivel, q3=1.5 * nivel)


def cambiar_lo_que_no_se_sabia(features, corte):
    """Todo lo emitido después del corte, y los casos finales de todas las filas, cambiados al azar"""
    rng = np.random.default_rng(99)
    f = features.copy()
    despues = f["inicio_semana"] > corte
    numericas = [c for c in P.VARIABLES if c != "provincia_codigo"]
    f.loc[despues, numericas] = rng.uniform(0, 500, (despues.sum(), len(numericas)))
    f[P.OBJETIVOS] = rng.uniform(0, 500, (len(f), len(P.OBJETIVOS)))
    return f


@pytest.mark.parametrize("cada", [1, 4])
def test_el_backtest_no_usa_nada_que_no_se_conociera_al_emitir(cada):
    features = features_de_prueba()
    desde, hasta = CORTE - 12 * P.SEMANA, CORTE + 10 * P.SEMANA
    tal_cual = P.backtest(P.preparar(features), desde, hasta, cada=cada, **RAPIDO)
    cambiado = P.backtest(P.preparar(cambiar_lo_que_no_se_sabia(features, CORTE)), desde, hasta, cada=cada,
                          **RAPIDO)
    hasta_el_corte = tal_cual["emision"] <= CORTE
    assert hasta_el_corte.sum() == 3 * 13 * 4     # 3 provincias × 13 emisiones × 4 horizontes
    pd.testing.assert_frame_equal(tal_cual[hasta_el_corte], cambiado[cambiado["emision"] <= CORTE])
    assert not tal_cual[~hasta_el_corte].equals(cambiado[~hasta_el_corte])   # después del corte sí cambian


def test_las_etiquetas_son_lo_que_se_sabia_tres_semanas_despues():
    f = P.preparar(features_de_prueba())
    g = f[f["dpa_provin"] == "13"].set_index("inicio_semana")
    t = pd.Timestamp("2021-03-07")
    for h in P.HORIZONTES:
        assert g.loc[t, f"e_h{h}"] == g.loc[t + (h + 3) * P.SEMANA, "casos_lag3"]


def test_se_entrena_solo_con_etiquetas_que_ya_se_conocian():
    f = P.preparar(features_de_prueba())
    datos = P.entrenamiento(f, CORTE)
    assert datos["inicio_semana"].max() == CORTE - 7 * P.SEMANA   # la etiqueta a 4 semanas se conoce en t+7
    assert datos[P.ETIQUETAS].notna().all().all()


def test_la_linea_base_del_canal_usa_el_canal_que_ya_se_conocia():
    features = features_de_prueba()
    f = P.preparar(features)
    canal = canal_de_prueba(features)
    semanas = features[["anio_epi", "semana_epi", "inicio_semana"]].drop_duplicates()
    base = P.evaluacion(f, canal, semanas, "2022-12-18", "2022-12-18")   # SE 51 de 2022
    lineas = P.unir(base, [P.lineas_base(base)])
    canal_h4 = lineas[(lineas["modelo"] == "canal") & (lineas["dpa_provin"] == "09") & (lineas["horizonte"] == 4)]
    assert canal_h4["anio_objetivo"].item() == 2023# pronostica una semana de 2023...
    assert canal_h4["pronostico"].item() == 40.0 # ...con el canal de 2022 (el de 2023 es 80)
    assert canal_h4["q3_real"].item() == 120.0# para evaluar sí vale el canal de 2023


def pronosticos_de_prueba(emisiones):
    """Pronósticos de un modelo para una provincia: emisiones = {semana: (pronóstico, real)} a 2, 3 y 4 semanas

    El canal es fijo: mediana 10 y q3 20. El promedio de las últimas 4 semanas es 10
    """
    inicio = pd.Timestamp("2024-01-07")
    filas = []
    for semana, (pronostico, real) in emisiones.items():
        emision = inicio + (semana - 1) * P.SEMANA
        for h in (2, 3, 4):
            filas.append(dict(modelo="m", dpa_provin="13", provincia="MANABI", emision=emision, anio_epi=2024,
                              semana_epi=semana, fecha_emision=emision + pd.Timedelta(days=13), horizonte=h,
                              objetivo=emision + h * P.SEMANA,
                              pronostico=pronostico, real=real, canal_mediana=10.0, canal_q3=20.0, q3_real=20.0,
                              casos_media_4s=10.0))
    return pd.DataFrame(filas)


def test_la_roja_pide_dos_emisiones_seguidas_en_zona_epidemica():
    p = pronosticos_de_prueba({1: (8, 8), 2: (25, 8), 3: (25, 8), 4: (15, 8), 5: (25, 8), 6: (8, 8)})
    nivel = P.boletines(p, "m").set_index("semana_epi")["nivel"]
    assert nivel.to_dict() == {1: "verde", 2: "amarilla", 3: "roja", 4: "amarilla", 5: "amarilla", 6: "verde"}


def test_con_menos_de_5_casos_no_hay_alerta():
    p = pronosticos_de_prueba({1: (4, 0), 2: (4, 0), 3: (4, 0)}).assign(canal_mediana=0.0, canal_q3=0.0)
    assert set(P.boletines(p, "m")["nivel"]) == {"verde"}


def test_con_subida_la_roja_pide_que_el_pronostico_crezca():
    p = pronosticos_de_prueba({1: (25, 8), 2: (25, 8), 3: (25, 8)})   # 25 = 2,5 veces el promedio de 10
    assert P.boletines(p, "m", subida=2.0)["roja"].tolist() == [False, True, True]
    assert P.boletines(p, "m", subida=3.0)["roja"].tolist() == [False, False, False]


def test_sensibilidad_y_falsas_alarmas():
    # Señal en las emisiones 2-4 y 9-11: la roja empieza en la segunda emisión seguida (3-4 y 10-11)
    # Las semanas 6 a 9 son epidémicas: la racha 3-4 acierta y la 10-11 es falsa alarma
    senal = {2, 3, 4, 9, 10, 11}
    p = pronosticos_de_prueba({s: (25 if s in senal else 8, 0) for s in range(1, 13)})
    semana_pronosticada = p["semana_epi"] + p["horizonte"]
    p["real"] = np.where(semana_pronosticada.between(6, 9), 30.0, 8.0)
    b = P.boletines(p, "m")
    assert b.loc[b["roja"], "semana_epi"].tolist() == [3, 4, 10, 11]
    total = P.metricas_alerta(b, p).set_index("dpa_provin").loc["total"]
    assert total["semanas_epidemicas"] == 4    # SE 6 a 9; las anuncian las emisiones t−4, t−3 y t−2
    assert total["semanas_alertadas"] == 3  # la SE 9 solo la anuncian las emisiones 5 a 7, ninguna roja
    assert total["sensibilidad"] == 0.75
    assert total["rachas_rojas"] == 2 and total["falsas_alarmas"] == 1


def serie_y_rojas(provincias, rojas_desde):
    """Una serie con valle y pico en 2024 por provincia, y boletines en rojo desde `rojas_desde` hasta el pico"""
    semanas = pd.date_range("2023-07-02", "2024-12-29", freq="7D")
    anios, numeros = semana_epi(semanas)
    casos = np.interp(np.arange(len(semanas)), [0, 20, 30, 50, 78], [40, 10, 10, 200, 20])
    serie = pd.concat([pd.DataFrame(dict(dpa_provin=dpa, provincia=nombre, inicio_semana=semanas, casos=casos,
                                         anio_epi=anios, semana_epi=numeros, q3_real=20.0))
                       for dpa, nombre in provincias.items()], ignore_index=True)
    pico = semanas[50]
    b = serie.rename(columns={"inicio_semana": "emision"})[["dpa_provin", "emision", "anio_epi", "semana_epi"]]
    b = b.assign(modelo="m", subida=0.0, fecha_emision=b["emision"] + pd.Timedelta(days=13),
                 roja=b["emision"].between(rojas_desde, pico))
    return serie, b.assign(nivel=np.where(b["roja"], "roja", "verde"))


def test_anticipacion_y_alerta_saturada():
    valle, pico = pd.Timestamp("2024-01-28"), pd.Timestamp("2024-06-16")   # semanas 30 y 50 de la serie
    serie, b = serie_y_rojas({"13": "MANABI"}, valle - 2 * P.SEMANA)
    a = P.anticipacion(b, serie).iloc[0]
    assert (a["valle"], a["pico"]) == (valle, pico)
    assert a["brote"] and a["avisada"] and not a["saturada"]
    # La roja de la emisión valle−2 se publica 13 días después de que empieza su semana: el sábado anterior a la
    # semana del valle. De ahí al inicio de la semana del pico hay tantas semanas completas como del valle al pico
    assert a["publicada"] == valle - pd.Timedelta(days=1)
    assert a["anticipacion"] == (pico - valle) / P.SEMANA
    serie, b = serie_y_rojas({"13": "MANABI"}, pd.Timestamp("2023-07-02"))   # en rojo desde la temporada anterior
    a = P.anticipacion(b, serie).iloc[0]
    assert a["saturada"] and not a["avisada"]


def test_el_grafico_tiene_un_panel_por_provincia_de_la_costa():
    pytest.importorskip("matplotlib")
    serie, b = serie_y_rojas(P.COSTA, pd.Timestamp("2024-01-14"))
    figura = P.grafico_costa(serie, b, P.anticipacion(b, serie))
    titulos = [eje.get_title(loc="left") for eje in figura.axes[:7]]
    assert titulos[4] == "Manabí · roja 20 semanas antes del pico"
    assert all(t.split(" · ")[0] == nombre for t, nombre in zip(titulos, P.COSTA.values()))