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
            # En estos datos cada gaceta llega a tiempo: el último dato conocido es el de la semana emitida
            fila.update({f"casos_ult{k}": lags[k] for k in range(4)})
            fila["casos_ventana"] = [sabido[i - k] if i >= k else np.nan for k in range(P.SEMANAS_VENTANA)]
            fila.update({v: rng.normal() for v in P.VARIABLES_CLIMA})
            fila.update(crecimiento_4s=rng.normal(), incidencia_4s_100k=fila["casos_media_4s"] / poblacion * 1e5,
                        casos_vs_canal=rng.normal())
            fila.update(casos_media_ult4=fila["casos_media_4s"], crecimiento_ult4=fila["crecimiento_4s"],
                        incidencia_ult4_100k=fila["incidencia_4s_100k"], casos_ult_vs_canal=fila["casos_vs_canal"])
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
    # También lo que esas emisiones sabían de sus últimas semanas: de ahí salen las etiquetas
    f.loc[despues, "casos_ventana"] = pd.Series(
        [list(rng.uniform(0, 500, P.SEMANAS_VENTANA)) for _ in range(despues.sum())], index=f.index[despues])
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


def test_las_etiquetas_son_lo_que_se_sabia_al_entrenar():
    features = features_de_prueba()
    datos = P.entrenamiento(P.preparar(features), CORTE)
    g = datos[datos["dpa_provin"] == "13"].set_index("inicio_semana")
    ventana = features[features["dpa_provin"] == "13"].set_index("inicio_semana")["casos_ventana"]
    reciente, antigua = CORTE - 6 * P.SEMANA, CORTE - 30 * P.SEMANA
    for h in P.HORIZONTES:
        # Fila reciente: su SE t+h, como la conocía la emisión en que se entrena, 6 - h semanas después
        assert g.loc[reciente, f"e_h{h}"] == ventana[CORTE][6 - h]
        # Fila antigua: como la conocía la última emisión que todavía la tenía en su ventana, 11 semanas después
        assert g.loc[antigua, f"e_h{h}"] == ventana[antigua + (h + 11) * P.SEMANA][11]


def test_se_entrena_solo_con_lo_que_ya_se_conocia():
    features = features_de_prueba()
    datos = P.entrenamiento(P.preparar(features), CORTE)
    assert datos["inicio_semana"].max() == CORTE - 4 * P.SEMANA   # su SE t+4 es la del origen, que ya tenía dato
    assert datos[P.ETIQUETAS].notna().all().all()
    # Si al entrenar no han llegado las gacetas de las dos últimas semanas, las filas que las pronostican esperan
    tarde = features.copy()
    del_origen = tarde["inicio_semana"] == CORTE
    tarde.loc[del_origen, "casos_ventana"] = tarde.loc[del_origen, "casos_ventana"].map(
        lambda v: [np.nan, np.nan] + list(v[2:]))
    datos = P.entrenamiento(P.preparar(tarde), CORTE)
    assert datos["inicio_semana"].max() == CORTE - 6 * P.SEMANA
    # ...y entran en cuanto llegan: una semana después ya tienen su dato en la ventana de la emisión siguiente
    assert P.entrenamiento(P.preparar(tarde), CORTE + P.SEMANA)["inicio_semana"].max() == CORTE - 3 * P.SEMANA


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


def test_el_ultimo_dato_conocido_es_una_linea_base():
    features = features_de_prueba()
    features.loc[features["inicio_semana"] == CORTE, "casos_lag0"] = np.nan   # esa semana la gaceta no llegó
    f = P.preparar(features)
    semanas = features[["anio_epi", "semana_epi", "inicio_semana"]].drop_duplicates()
    lineas = P.lineas_base(P.evaluacion(f, canal_de_prueba(features), semanas, CORTE - P.SEMANA, CORTE))
    ultimo = (lineas[(lineas["modelo"] == "ultimo") & (lineas["dpa_provin"] == "13")]
              .sort_values(["emision", "horizonte"]).groupby("emision")["pronostico"].apply(list))
    sabido = f[f["dpa_provin"] == "13"].set_index("inicio_semana")
    # Los 4 horizontes repiten los casos de la última semana conocida
    assert ultimo[CORTE - P.SEMANA] == [sabido.loc[CORTE - P.SEMANA, "casos_lag0"]] * 4
    # Si la gaceta de la SE emitida no se había publicado, es la semana anterior
    assert ultimo[CORTE] == [sabido.loc[CORTE, "casos_lag1"]] * 4


def serie_con_casos(casos, q3=20.0):
    """La serie final de una provincia desde la SE 1 de 2023, con un canal fijo"""
    semanas = pd.date_range("2023-01-01", periods=len(casos), freq="7D")
    anios, numeros = semana_epi(semanas)
    return pd.DataFrame(dict(dpa_provin="13", provincia="MANABI", inicio_semana=semanas,
                             casos=np.asarray(casos, dtype=float), anio_epi=anios, semana_epi=numeros, q3_real=q3))


def boletines_en_rojo(serie, emisiones_rojas):
    """Un boletín por semana de la serie, en rojo en las emisiones (su posición en la serie) que se indican"""
    b = serie.rename(columns={"inicio_semana": "emision"})[["dpa_provin", "provincia", "emision", "anio_epi",
                                                            "semana_epi"]]
    b = b.assign(modelo="m", subida=0.0, fecha_emision=b["emision"] + pd.Timedelta(days=13),
                 roja=b.index.isin(emisiones_rojas))
    return b.assign(nivel=np.where(b["roja"], "roja", "verde"))


def test_un_brote_empieza_cuando_la_provincia_sale_de_la_calma():
    calma, brote = [10.0] * 12, [30.0] * 6
    # Calma con un salto de una sola semana, un brote, 3 semanas de calma y otro brote pegado al primero
    casos = calma[:5] + [90.0] + calma[6:] + brote + [10.0] * 3 + brote + [10.0] * 4
    inicios = P.inicios_de_brote(serie_con_casos(casos))
    # El salto no es un brote, y el segundo brote no es un inicio: la provincia no había vuelto a la calma
    assert inicios["inicio"].tolist() == [pd.Timestamp("2023-01-01") + 12 * P.SEMANA]
    assert (inicios.loc[0, "casos"], inicios.loc[0, "umbral"]) == (30.0, 20.0)
    assert P.inicios_de_brote(serie_con_casos([30.0] * 30)).empty                # siempre epidémica: sin inicio
    assert P.inicios_de_brote(serie_con_casos(calma + brote, q3=np.nan)).empty   # sin canal no hay zona epidémica


def test_avisos_de_inicio_y_falsas_alarmas():
    serie = serie_con_casos([10.0] * 12 + [30.0] * 10 + [10.0] * 30)   # el brote empieza en la semana 12
    inicio = pd.Timestamp("2023-01-01") + 12 * P.SEMANA

    def aviso(rojas):
        b = boletines_en_rojo(serie, rojas)
        return P.avisos_de_inicio(b, P.inicios_de_brote(serie)).iloc[0], P.metricas_inicio(b, serie)

    # La emisión 7 se publica 13 días después de empezar su semana: 22 días antes del inicio, 3 semanas completas
    a, m = aviso([7, 8])
    assert (a["estado"], a["anticipacion"], a["publicada"]) == ("a tiempo", 3.0, inicio - pd.Timedelta(days=22))
    assert (m["inicios"], m["inicios_a_tiempo"], m["anticipacion_inicio"], m["falsas_inicio"]) == (1, 1, 3.0, 0)
    # La roja de la emisión 12 se publica con el brote ya empezado: tarde
    a, m = aviso([12])
    assert a["estado"] == "tarde" and np.isnan(a["anticipacion"])
    assert (m["inicios_tarde"], m["falsas_inicio"]) == (1, 0)
    # Una roja en pleno brote, 6 semanas después, no lo avisó; tampoco es falsa alarma
    a, m = aviso([18])
    assert a["estado"] == "sin aviso" and (m["inicios_sin_aviso"], m["falsas_inicio"]) == (1, 0)
    # Una racha roja en calma, lejos de cualquier inicio, es falsa alarma; cuenta una vez aunque dure 2 semanas
    a, m = aviso([7, 30, 31])
    assert (m["inicios_a_tiempo"], m["falsas_inicio"]) == (1, 1)
    assert m["falsas_inicio_por_anio"] == pytest.approx(P.SEMANAS_POR_ANIO / len(serie))
    # Una racha demasiado reciente para saber si después empezó un brote no se cuenta
    assert aviso([49])[1]["falsas_inicio"] == 0


def test_la_regla_elegida_es_la_que_avisa_mas_inicios_de_brote():
    reglas = pd.DataFrame({"inicios_a_tiempo": [36, 16, 36, 40], "falsas_inicio_por_anio": [0.49, 0.36, 0.60, 1.20]},
                          index=["plan", "con subida", "plan con más falsas", "fuera del presupuesto"])
    assert P.elegir_regla(reglas, presupuesto=1.0) == "plan"            # si empatan, la de menos falsas alarmas
    assert P.elegir_regla(reglas, presupuesto=2.0) == "fuera del presupuesto"
    assert P.elegir_regla(reglas, presupuesto=0.1) == "con subida"      # ninguna cumple: la de menos falsas alarmas


def test_el_grafico_tiene_un_panel_por_provincia_de_la_costa():
    pytest.importorskip("matplotlib")
    serie, b = serie_y_rojas(P.COSTA, pd.Timestamp("2024-01-14"))
    figura = P.grafico_costa(serie, b, P.anticipacion(b, serie))
    titulos = [eje.get_title(loc="left") for eje in figura.axes[:7]]
    assert titulos[4] == "Manabí · roja 20 semanas antes del pico"
    assert all(t.split(" · ")[0] == nombre for t, nombre in zip(titulos, P.COSTA.values()))