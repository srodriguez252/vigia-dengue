"""Fase 6 - el boletín de cada semana: pronóstico con intervalo, alerta con su motivo y camas-día

Usa pandas, sin Spark ni MLflow, como pronostico.py: lo importan los notebooks del Job (pronostico_semanal y
alerta_semanal) y las pruebas de tests/

Cada sábado, el Job pronostica las emisiones ya publicadas que todavía no tienen pronóstico, con el modelo que
tiene el alias champion, y les aplica la regla de alerta elegida en la Fase 5. Una emisión se publica en su
fecha_emision: el sábado siguiente al cierre de su semana
"""
import numpy as np
import pandas as pd

from modelos import pronostico as P

COBERTURA = 0.8   # el intervalo: 8 de cada 10 errores del backtest caen dentro
MIN_HOSPITALIZACIONES = 30  # con menos, la provincia usa la tasa y la estada del país
SIN_CANAL = "sin canal"

COLUMNAS_PRONOSTICO = ["dpa_provin", "provincia", "emision", "anio_epi", "semana_epi", "fecha_emision",
                       "horizonte", "objetivo", "anio_objetivo", "semana_objetivo", "pronostico", "pronostico_bajo",
                       "pronostico_alto", "canal_mediana", "canal_q3", "casos_semana", "casos_media_4s",
                       "camas_dia", "modelo", "version"]
COLUMNAS_ALERTA = ["dpa_provin", "provincia", "emision", "anio_epi", "semana_epi", "fecha_emision", "nivel",
                   "motivo", "en_epidemia", "sube", "senal", "casos_semana", "casos_media_4s", "pronostico_max",
                   "semana_max", "umbral_epidemia", "casos_4s", "camas_dia_4s", "camas_promedio", "subida",
                   "version"]


def emisiones_pendientes(f, hechas, hoy, semanas=8):
    """Las emisiones publicadas hasta `hoy`, entre las últimas `semanas`, que todavía no están en `hechas`

    Si una semana el Job falla, la siguiente pronostica las dos. La primera vez llena las últimas `semanas`
    """
    publicadas = f.loc[f["fecha_emision"] <= pd.Timestamp(hoy), "inicio_semana"].drop_duplicates()
    hechas = set(pd.to_datetime(pd.Series(list(hechas), dtype="object")))
    return sorted(e for e in publicadas.nlargest(semanas) if e not in hechas)


def factores_intervalo(backtest, modelo, cobertura=COBERTURA):
    """Por horizonte, los extremos del error del backtest de `modelo`, medido como log((real + 1) / (pronóstico + 1))

    El error crece con los casos, así que se mide relativo. Con cobertura 0,8 son los cuantiles 0,1 y 0,9: en el
    backtest, 8 de cada 10 semanas cayeron entre los dos extremos
    """
    b = backtest[(backtest["modelo"] == modelo) & backtest["evaluable"].eq(True)]
    if b.empty:
        raise ValueError(f"El backtest no tiene filas evaluables de {modelo}")
    error = np.log((b["real"] + 1) / (b["pronostico"] + 1))
    extremos = error.groupby(b["horizonte"]).quantile([(1 - cobertura) / 2, (1 + cobertura) / 2]).unstack()
    extremos.columns = ["error_bajo", "error_alto"]
    return extremos.rename_axis("horizonte").reset_index()


def tasas_hospitalarias(egresos, casos, minimo=MIN_HOSPITALIZACIONES):
    """Por provincia, cuántas hospitalizaciones hay por caso confirmado y cuántos días dura cada una

    Compara los años que están en las dos fuentes: los egresos del INEC (silver.egresos_dengue, por provincia de
    residencia) y los casos de las gacetas (gold.dengue_semana). Una provincia con menos de `minimo`
    hospitalizaciones en esos años usa los valores del país
    """
    h = egresos.rename(columns={"provincia_res": "dpa_provin"})
    anios = sorted(set(h["anio_epi"]) & set(casos["anio_epi"]))
    if not anios:
        raise ValueError("Los egresos y los casos no tienen ningún año en común")
    h = h[h["anio_epi"].isin(anios)].groupby("dpa_provin")[["hospitalizaciones", "dias_estada"]].sum()
    c = casos[casos["anio_epi"].isin(anios)].groupby("dpa_provin")["casos"].sum()
    t = c.to_frame().join(h, how="left").fillna(0.0).astype("float64")
    pais = t.sum()
    propia = (t["hospitalizaciones"] >= minimo) & (t["casos"] > 0)
    t["hosp_por_caso"] = np.where(propia, t["hospitalizaciones"] / t["casos"].where(propia),
                                  pais["hospitalizaciones"] / pais["casos"])
    t["dias_por_hospitalizacion"] = np.where(propia, t["dias_estada"] / t["hospitalizaciones"].where(propia),
                                             pais["dias_estada"] / pais["hospitalizaciones"])
    t["fuente"] = np.where(propia, "provincia", "país")
    t["anios"] = f"{anios[0]}-{anios[-1]}"
    return t.rename_axis("dpa_provin").reset_index()


def pronosticar(modelo, f, canal, semanas, emisiones, intervalo, tasas, nombre, version):
    """Las filas de gold.pronosticos para `emisiones`: una por provincia, emisión y horizonte

    - pronostico: los casos que el modelo `nombre` (versión `version` del registro) espera en la SE objetivo
    - pronostico_bajo y pronostico_alto: el intervalo, con los errores del backtest en ese horizonte
    - canal_mediana y canal_q3: el canal endémico que ya se conocía al emitir, en la SE objetivo
    - camas_dia: las camas-día que ocuparían esos casos: pronóstico x hospitalizaciones por caso x días de estada
    """
    filas = f[f["inicio_semana"].isin(emisiones)]
    variables = list(modelo.feature_names_in_)   # las columnas con que se entrenó, en el mismo orden
    pronosticos = P.a_largo(filas, modelo.predict(filas[variables]), nombre).drop(columns="origen")
    base = P.evaluacion(filas, canal, semanas, min(emisiones), max(emisiones))
    casos_semana = filas[["dpa_provin", "inicio_semana", "casos_lag0"]].rename(
        columns={"inicio_semana": "emision", "casos_lag0": "casos_semana"})
    p = (P.unir(base, [pronosticos])
         .merge(intervalo, on="horizonte", how="left")
         .merge(tasas[["dpa_provin", "hosp_por_caso", "dias_por_hospitalizacion"]], on="dpa_provin", how="left")
         .merge(casos_semana, on=["dpa_provin", "emision"], how="left"))
    p["pronostico_bajo"] = np.maximum((p["pronostico"] + 1) * np.exp(p["error_bajo"]) - 1, 0)
    p["pronostico_alto"] = (p["pronostico"] + 1) * np.exp(p["error_alto"]) - 1
    p["camas_dia"] = p["pronostico"] * p["hosp_por_caso"] * p["dias_por_hospitalizacion"]
    redondeo = {c: 1 for c in ("pronostico", "pronostico_bajo", "pronostico_alto", "camas_dia")}
    p = p.round(redondeo).assign(version=str(version))
    return p[COLUMNAS_PRONOSTICO].sort_values(["emision", "dpa_provin", "horizonte"], ignore_index=True)


def _numero(x, decimales=0):
    """1234.5 como «1 234,5»: espacio para los miles y coma para los decimales"""
    return f"{x:,.{decimales}f}".replace(",", " ").replace(".", ",")


def _semana_mas_alta(p, canal):
    """Por provincia y emisión, la fila de 2 a 4 semanas con el pronóstico más alto frente a su umbral

    El umbral es la columna `canal` del canal endémico, con un mínimo de PISO casos, como en la regla de alerta
    """
    q = p[p["horizonte"].between(2, 4)].copy()
    q["umbral"] = np.maximum(q[canal], P.PISO)
    q["razon"] = q["pronostico"] / q["umbral"]
    q = q.sort_values(["razon", "horizonte"], ascending=[False, True], na_position="last")
    return q.drop_duplicates(["dpa_provin", "emision"]).set_index(["dpa_provin", "emision"])


def motivo(a, subida):
    """Una frase que explica el nivel de una fila de alertas

    El umbral epidémico es el q3 del canal y el de alerta, la mediana; los dos, con un mínimo de PISO casos
    """
    if a["nivel"] == SIN_CANAL:
        return "Sin canal endémico: la provincia no tiene 5 años sin epidemia con qué comparar"
    epidemia = (f"{_numero(a['pronostico_max'], 1)} casos en la SE {int(a['semana_max'])}, sobre el umbral "
                f"epidémico de {_numero(a['umbral_epidemia'], 1)}")
    if a["nivel"] == "roja":
        texto = f"Segunda semana seguida con el pronóstico en zona epidémica: {epidemia}"
        if subida > 0:
            texto += (f". A 4 semanas pronostica {_numero(a['pronostico_h4'])} casos, "
                      f"{_numero(a['pronostico_h4'] / max(a['casos_media_4s'], 1), 1)} veces el promedio de las "
                      f"últimas 4 semanas ({_numero(a['casos_media_4s'])})")
        return texto
    if a["nivel"] == "amarilla" and a["senal"]:
        return f"Primera semana con el pronóstico en zona epidémica: {epidemia}. Si se repite, pasa a roja"
    if a["nivel"] == "amarilla" and a["en_epidemia"]:
        return (f"Pronóstico en zona epidémica, pero sin subida: {_numero(a['pronostico_h4'])} casos a 4 semanas "
                f"frente a un promedio de {_numero(a['casos_media_4s'])} en las últimas 4")
    if a["nivel"] == "amarilla":
        return (f"Pronóstico sobre la mediana del canal: {_numero(a['pronostico_alerta'], 1)} casos en la SE "
                f"{int(a['semana_alerta'])}, sobre el umbral de alerta de {_numero(a['umbral_alerta'], 1)}")
    if a["pronostico_pico"] <= P.PISO:
        return f"Hasta {P.PISO} casos por semana en las próximas 2 a 4 semanas"
    return (f"Pronóstico bajo la mediana del canal: hasta {_numero(a['pronostico_pico'], 1)} casos por semana en "
            f"las próximas 2 a 4 semanas")


def alertas(pronosticos, subida, emisiones):
    """Las filas de gold.alertas de `emisiones`: el nivel de cada provincia, su motivo y las camas de 4 semanas

    Aplica a los pronósticos guardados la regla de la Fase 5 (pronostico.boletines), con el umbral `subida` del
    campeón. La roja pide dos emisiones seguidas en zona epidémica: `pronosticos` debe traer también la emisión
    anterior a la primera de `emisiones`
    """
    p = pronosticos.assign(modelo="campeon", real=np.nan, q3_real=np.nan)
    b = P.boletines(p, "campeon", subida).set_index(["dpa_provin", "emision"])
    epidemia = _semana_mas_alta(p, "canal_q3")
    alerta = _semana_mas_alta(p, "canal_mediana")
    grupos = p.groupby(["dpa_provin", "emision"])
    a = b.assign(
        pronostico_max=epidemia["pronostico"], semana_max=epidemia["semana_objetivo"],
        umbral_epidemia=epidemia["umbral"], pronostico_alerta=alerta["pronostico"],
        semana_alerta=alerta["semana_objetivo"], umbral_alerta=alerta["umbral"],
        pronostico_pico=p[p["horizonte"].between(2, 4)].groupby(["dpa_provin", "emision"])["pronostico"].max(),
        casos_semana=grupos["casos_semana"].first(), casos_4s=grupos["pronostico"].sum(),
        camas_dia_4s=grupos["camas_dia"].sum(), version=grupos["version"].first()).reset_index()
    a = a[a["emision"].isin(pd.to_datetime(pd.Series(list(emisiones), dtype="object")))].copy()
    a["nivel"] = a["nivel"].fillna(SIN_CANAL)
    a["motivo"] = [motivo(fila, subida) for _, fila in a.iterrows()]
    a["camas_promedio"] = a["camas_dia_4s"] / 28    # camas ocupadas en un día cualquiera de las 4 semanas
    a = a.round({"pronostico_max": 1, "umbral_epidemia": 1, "casos_4s": 1, "camas_dia_4s": 1, "camas_promedio": 1})
    return a[COLUMNAS_ALERTA].sort_values(["emision", "dpa_provin"], ignore_index=True)