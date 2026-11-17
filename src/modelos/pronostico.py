"""Fase 5 - pronóstico a 1–4 semanas, backtest de origen móvil, alertas y métricas

Usa pandas y scikit-learn, sin Spark ni MLflow: lo importan los notebooks de la fase y las pruebas de tests/

La regla de la Fase 4 sigue mandando: para pronosticar la emisión t solo se usan filas de features emitidas hasta t.
Por eso el modelo no aprende de los casos finales (y_h1 a y_h4), que se conocen mucho después, sino de lo que se
sabía de cada semana el día en que se entrena. Los casos finales solo sirven para evaluar
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

HORIZONTES = (1, 2, 3, 4)
SEMANAS_VENTANA = 12     # cada emisión guarda lo que sabía de sus últimas 12 semanas (casos_ventana)
PISO = 5                 # con menos de 5 casos por semana no hay zona de alerta ni de epidemia
COSTA = {"07": "El Oro", "08": "Esmeraldas", "09": "Guayas", "12": "Los Ríos", "13": "Manabí",
         "23": "Santo Domingo", "24": "Santa Elena"}
MANABI = "13"
SEMANA = pd.Timedelta(weeks=1)
SEMANAS_POR_ANIO = 52.18

VARIABLES_FIJAS = ["provincia_codigo", "poblacion", "estacion_sin", "estacion_cos"]
# Los casos entran desde el último dato conocido (casos_ult0 es la semana más reciente con dato) y rezago_dato dice
# cuántas semanas tiene: así el modelo no se queda sin nivel de referencia cuando una gaceta se publica tarde
VARIABLES_CASOS = ["casos_ult0", "casos_ult1", "casos_ult2", "casos_ult3", "casos_media_ult4", "casos_media_12s",
                   "crecimiento_ult4", "incidencia_ult4_100k", "semanas_con_dato", "rezago_dato",
                   "canal_mediana", "canal_q3", "casos_ult_vs_canal"]
# Los casos por semana fija, que usan las líneas base y la regla de alerta, no el modelo
REFERENCIA = ["casos_lag0", "casos_lag1", "casos_lag2", "casos_lag3", "casos_media_4s"]
VARIABLES_CLIMA = ["anomalia_lluvia_0_3", "anomalia_lluvia_4_7", "anomalia_lluvia_8_11",
                   "anomalia_temp_min_0_3", "anomalia_temp_min_4_7", "anomalia_temp_min_8_11",
                   "anomalia_humedad_0_3", "temp_min_0_3",
                   "nino12_lag0", "nino12_lag4", "nino12_lag8", "nino12_lag12"]
VARIABLES = VARIABLES_FIJAS + VARIABLES_CASOS + VARIABLES_CLIMA
ETIQUETAS = [f"e_h{h}" for h in HORIZONTES]     # para entrenar: lo que se sabía el día del entrenamiento
OBJETIVOS = [f"y_h{h}" for h in HORIZONTES]     # para evaluar: los casos finales
FECHAS = ["inicio_semana", "fecha_emision", "emision", "objetivo", "origen", "pico", "valle", "primera_roja",
          "roja_desde", "inicio", "publicada"]

HIPERPARAMETROS = dict(loss="poisson", learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20,
                       l2_regularization=1.0, early_stopping=False, random_state=0)


# ---------------------------------------------------------------------------------------------------------------
# Datos y modelo
# ---------------------------------------------------------------------------------------------------------------

def de_spark(df):
    """Las fechas que llegan de Spark (datetime.date) como fechas de pandas, para poder sumarles semanas"""
    df = df.copy()
    for columna in FECHAS:
        if columna in df.columns:
            df[columna] = pd.to_datetime(df[columna])
    return df


def para_spark(df):
    """Las fechas como datetime.date, para que Spark las guarde como DATE y no como TIMESTAMP"""
    df = df.copy()
    for columna in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[columna]):
            df[columna] = df[columna].dt.date.astype(object).where(df[columna].notna(), None)
    return df


def calendario(semanas):
    """Una fila por SE: su inicio, su año y su número"""
    return (de_spark(semanas)[["inicio_semana", "anio_epi", "semana_epi"]]
            .drop_duplicates("inicio_semana").astype({"anio_epi": "int64", "semana_epi": "int64"}))


def preparar(features):
    """La tabla de features lista para el modelo: fechas, tipos, la provincia como código y lo que sirve para
    armar las etiquetas

    Las etiquetas dependen del día en que se entrena, así que las arma `entrenamiento`. Aquí queda, para la fila de
    la SE t y cada horizonte h, lo que supo de la SE t+h cada emisión que la tuvo en su ventana: la columna
    _s{h}_{k} son los casos de la SE t+h como los conocía la emisión t+h+k (su casos_ventana[k])
    """
    f = de_spark(features).sort_values(["dpa_provin", "inicio_semana"], ignore_index=True)
    f["provincia_codigo"] = f["dpa_provin"].astype("int64")
    numericas = [c for c in VARIABLES + REFERENCIA + OBJETIVOS if c != "provincia_codigo"]
    f[numericas] = f[numericas].astype("float64")
    f = f.astype({"anio_epi": "int64", "semana_epi": "int64"})
    sin_dato = np.full(SEMANAS_VENTANA, np.nan)
    ventana = np.array([sin_dato if v is None or np.ndim(v) == 0 else np.asarray(v, dtype="float64")
                        for v in f["casos_ventana"]])
    claves = f[["dpa_provin", "inicio_semana"]]
    sabido = claves.join(pd.DataFrame(ventana, index=f.index))      # columna k: lo que la emisión sabía de t-k
    partes = [f.drop(columns="casos_ventana")]
    for h in HORIZONTES:
        for k in range(SEMANAS_VENTANA):
            emision = sabido[["dpa_provin", "inicio_semana", k]].assign(
                inicio_semana=sabido["inicio_semana"] - (h + k) * SEMANA)
            partes.append(claves.merge(emision, on=["dpa_provin", "inicio_semana"], how="left")[k]
                          .rename(f"_s{h}_{k}"))
    return pd.concat(partes, axis=1)


def entrenamiento(f, origen):
    """Las filas para entrenar en la emisión `origen`, con sus etiquetas como se conocían ese día

    La etiqueta e_h de la fila de la SE t son los casos de la SE t+h según la emisión más reciente que la tenía en
    su ventana sin pasar de `origen`: la del origen para las semanas recientes, y la de SEMANAS_VENTANA - 1 semanas
    después para las antiguas. Una fila entra solo si sus 4 semanas ya tenían dato: si las gacetas llegan tarde,
    las filas más recientes esperan, pero ninguna queda fuera para siempre
    """
    origen = pd.Timestamp(origen)
    d = f[f["inicio_semana"] <= origen - max(HORIZONTES) * SEMANA].copy()
    semanas_antes = ((origen - d["inicio_semana"]) / SEMANA).round().astype("int64").to_numpy()
    filas = np.arange(len(d))
    for h in HORIZONTES:
        k = np.minimum(semanas_antes - h, SEMANAS_VENTANA - 1)
        d[f"e_h{h}"] = d[[f"_s{h}_{j}" for j in range(SEMANAS_VENTANA)]].to_numpy()[filas, k]
    return d.dropna(subset=ETIQUETAS)


def nuevo_modelo(variables=VARIABLES, **cambios):
    """Un gradient boosting con pérdida Poisson por horizonte: MultiOutputRegressor entrena uno por etiqueta"""
    categoricas = ["provincia_codigo"] if "provincia_codigo" in variables else None
    return MultiOutputRegressor(HistGradientBoostingRegressor(categorical_features=categoricas,
                                                              **{**HIPERPARAMETROS, **cambios}))


def a_largo(filas, valores, modelo, origen=None):
    """De una matriz de pronósticos (fila x horizonte) a una fila por provincia, emisión y horizonte"""
    ancho = pd.DataFrame(np.asarray(valores, dtype="float64"), columns=list(HORIZONTES), index=filas.index)
    ancho["dpa_provin"] = filas["dpa_provin"]
    ancho["emision"] = filas["inicio_semana"]
    largo = ancho.melt(id_vars=["dpa_provin", "emision"], var_name="horizonte", value_name="pronostico")
    largo["horizonte"] = largo["horizonte"].astype("int64")
    largo.insert(0, "modelo", modelo)
    largo["origen"] = pd.Timestamp(origen) if origen is not None else pd.NaT
    return largo


def backtest(f, desde, hasta=None, variables=VARIABLES, cada=4, modelo="gbm_global", **cambios):
    """Origen móvil: en cada origen entrena con lo que se sabía y pronostica esa emisión y las cada - 1 siguientes

    Con cada=1 reentrena todas las semanas; con 4, una vez por mes. El modelo de un origen solo pronostica
    ese origen y emisiones posteriores, así que nunca usa más de lo que se sabía al emitir
    """
    hasta = f["inicio_semana"].max() if hasta is None else pd.Timestamp(hasta)
    emisiones = np.sort(f.loc[f["inicio_semana"].between(pd.Timestamp(desde), hasta), "inicio_semana"].unique())
    partes = []
    for i in range(0, len(emisiones), cada):
        origen = pd.Timestamp(emisiones[i])
        datos = entrenamiento(f, origen)
        if datos.empty:
            raise ValueError(f"No hay filas para entrenar en el origen {origen.date()}")
        m = nuevo_modelo(variables, **cambios).fit(datos[variables], datos[ETIQUETAS])
        filas = f[f["inicio_semana"].isin(emisiones[i:i + cada])]
        partes.append(a_largo(filas, m.predict(filas[variables]), modelo, origen))
    return pd.concat(partes, ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------------------------------------------

def evaluacion(f, canal, semanas, desde, hasta=None):
    """La base para evaluar: una fila por provincia, emisión y horizonte del backtest, con lo que pasó después.

    - real: los casos finales de la SE pronosticada (objetivo)
    - canal_mediana y canal_q3: el canal del año de la emisión en la SE pronosticada. Ya se conocía al emitir, y
      con él se decide la alerta
    - q3_real: el q3 del canal del año de la SE pronosticada. Solo sirve para saber si esa semana fue epidémica
    - casos_ultimo: los casos de la semana más reciente que se conocía al emitir, entre la SE emitida y las 3
      anteriores. Si la gaceta de la SE emitida no se había publicado, es la de una semana antes
    - evaluable: tiene valor real, canal y casos recientes. Todos los modelos se comparan en estas mismas filas
    """
    hasta = f["inicio_semana"].max() if hasta is None else pd.Timestamp(hasta)
    claves = ["dpa_provin", "provincia", "inicio_semana", "anio_epi", "semana_epi", "fecha_emision",
              "casos_media_4s", "casos_media_12s", "casos_ultimo"]
    e = f[f["inicio_semana"].between(pd.Timestamp(desde), hasta)]
    e = e.assign(casos_ultimo=e[[f"casos_lag{k}" for k in range(4)]].bfill(axis=1).iloc[:, 0])
    b = e[claves + OBJETIVOS].melt(id_vars=claves, value_vars=OBJETIVOS, var_name="horizonte", value_name="real")
    b["horizonte"] = b["horizonte"].str.replace("y_h", "").astype("int64")
    b = b.rename(columns={"inicio_semana": "emision"})
    b["objetivo"] = b["emision"] + b["horizonte"] * SEMANA
    cal = calendario(semanas).rename(columns={"inicio_semana": "objetivo", "anio_epi": "anio_objetivo",
                                              "semana_epi": "semana_objetivo"})
    b = b.merge(cal, on="objetivo", how="left")
    c = canal[["dpa_provin", "anio_epi", "semana_epi", "mediana", "q3"]].astype({"anio_epi": "int64",
                                                                                   "semana_epi": "int64"})
    al_emitir = c.rename(columns={"semana_epi": "semana_objetivo", "mediana": "canal_mediana", "q3": "canal_q3"})
    b = b.merge(al_emitir, on=["dpa_provin", "anio_epi", "semana_objetivo"], how="left")
    de_su_anio = c.rename(columns={"anio_epi": "anio_objetivo", "semana_epi": "semana_objetivo", "q3": "q3_real"})
    b = b.merge(de_su_anio.drop(columns="mediana"), on=["dpa_provin", "anio_objetivo", "semana_objetivo"],
                how="left")
    b["evaluable"] = b["real"].notna() & b["canal_mediana"].notna() & b["casos_media_12s"].notna()
    return b.sort_values(["dpa_provin", "emision", "horizonte"], ignore_index=True)


def lineas_base(base):
    """Los tres pronósticos de referencia, para las mismas filas que el modelo

    - canal: la mediana del canal endémico de la SE pronosticada, lo que se espera en un año normal
    - ingenuo: el promedio de las últimas 4 semanas conocidas (o de 12, si faltan), igual para los 4 horizontes
    - ultimo: los casos de la última semana conocida, igual para los 4 horizontes. A diferencia del ingenuo, sí
      puede cumplir la condición de subida de la regla de alerta: sube cuando la última semana pasa del promedio.
      Es la regla sin modelo contra la que hay que medir las alertas
    """
    claves = base[["dpa_provin", "emision", "horizonte"]]
    canal = claves.assign(modelo="canal", pronostico=base["canal_mediana"])
    promedio = base["casos_media_4s"].fillna(base["casos_media_12s"])
    ingenuo = claves.assign(modelo="ingenuo", pronostico=promedio)
    ultimo = claves.assign(modelo="ultimo", pronostico=base["casos_ultimo"].fillna(promedio))
    partes = pd.concat([canal, ingenuo, ultimo], ignore_index=True).assign(origen=pd.NaT)
    return partes[["modelo", "dpa_provin", "emision", "horizonte", "pronostico", "origen"]]


def unir(base, partes):
    """Los pronósticos de varios modelos sobre la base: una fila por modelo, provincia, emisión y horizonte"""
    return base.merge(pd.concat(partes, ignore_index=True), on=["dpa_provin", "emision", "horizonte"])


def error_por_horizonte(pronosticos):
    """Error absoluto medio, en casos por semana, por modelo y horizonte, en las filas evaluables

    mae_relativo es el error medio dividido por el del canal: menos de 1 es mejor que el canal
    """
    p = pronosticos[pronosticos["evaluable"]]
    p = p.assign(error=(p["pronostico"] - p["real"]).abs())
    tabla = p.pivot_table(index="modelo", columns="horizonte", values="error", aggfunc="mean")
    tabla.columns = [f"mae_h{h}" for h in tabla.columns]
    tabla["mae_medio"] = tabla.mean(axis=1)
    if "canal" in tabla.index:
        tabla["mae_relativo"] = tabla["mae_medio"] / tabla.loc["canal", "mae_medio"]
    tabla["filas"] = p.groupby("modelo").size()
    return tabla.reset_index().sort_values("mae_medio", ignore_index=True)


# ---------------------------------------------------------------------------------------------------------------
# Alertas
# ---------------------------------------------------------------------------------------------------------------

def _anterior(df, columna):
    """El valor de `columna` en la emisión anterior de la misma provincia (False si no hay)."""
    previa = df[["dpa_provin", "emision", columna]].assign(emision=df["emision"] + SEMANA)
    unido = df[["dpa_provin", "emision"]].merge(previa, on=["dpa_provin", "emision"], how="left")
    return unido[columna].eq(True).to_numpy()


def boletines(pronosticos, modelo, subida=0.0):
    """La alerta de cada provincia y emisión con los pronósticos de `modelo` a 2, 3 y 4 semanas

    - roja: en dos emisiones seguidas, el pronóstico cae en zona epidémica (sobre el q3 del canal y sobre PISO) en
      alguna de las tres semanas. Si `subida` > 0, además el pronóstico a 4 semanas debe ser al menos `subida`
      veces el promedio de las últimas 4 semanas: tiene que anunciar un ascenso
    - amarilla: sin ser roja, el pronóstico pasa la mediana del canal (y PISO) en alguna de las tres semanas.
    - verde: lo demás. Sin canal para las tres semanas no hay alerta (nivel nulo)

    epidemia_real dice si alguna de esas tres semanas fue de verdad epidémica, con el canal de su propio año
    """
    p = pronosticos[(pronosticos["modelo"] == modelo) & pronosticos["horizonte"].between(2, 4)].copy()
    p["en_epidemia"] = p["pronostico"] > np.maximum(p["canal_q3"], PISO)
    p["en_alerta"] = p["pronostico"] > np.maximum(p["canal_mediana"], PISO)
    p["epidemica"] = p["real"] > np.maximum(p["q3_real"], PISO)
    p["con_canal"] = p["canal_q3"].notna()
    p["conocida"] = p["real"].notna() & p["q3_real"].notna()
    p["pronostico_h4"] = p["pronostico"].where(p["horizonte"] == 4)
    claves = ["dpa_provin", "provincia", "emision", "anio_epi", "semana_epi", "fecha_emision"]
    b = (p.groupby(claves, as_index=False)
         .agg(en_epidemia=("en_epidemia", "any"), en_alerta=("en_alerta", "any"), con_canal=("con_canal", "all"),
              epidemia_real=("epidemica", "any"), conocida=("conocida", "all"),
              casos_media_4s=("casos_media_4s", "first"), pronostico_h4=("pronostico_h4", "max")))
    b["sube"] = (b["pronostico_h4"] >= subida * b["casos_media_4s"]) if subida > 0 else True
    b["senal"] = b["en_epidemia"] & b["sube"] & b["con_canal"]
    b["roja"] = b["senal"] & _anterior(b, "senal")
    b["nivel"] = np.where(~b["con_canal"], None,
                          np.where(b["roja"], "roja", np.where(b["en_alerta"], "amarilla", "verde")))
    b.insert(0, "modelo", modelo)
    b.insert(1, "subida", float(subida))
    return b.sort_values(["dpa_provin", "emision"], ignore_index=True)


def metricas_alerta(b, base):
    """Sensibilidad y falsas alarmas de unos boletines (un modelo y una regla), por provincia y en total.

    - sensibilidad: de las semanas epidémicas, la parte que tuvo una alerta roja emitida 2 a 4 semanas antes. Solo
      cuentan las semanas cuyas tres emisiones (t-4, t-3 y t-2) tenían canal
    - racha roja: emisiones en rojo seguidas. Es falsa alarma si ninguna de ellas anunció una semana epidémica
    - falsas_por_anio: falsas alarmas por provincia y año de boletines evaluados
    """
    b = b[b["nivel"].notna()].copy()
    # Semanas epidémicas y si alguna de sus tres emisiones las anunció en rojo
    p = base[base["horizonte"].between(2, 4)].merge(b[["dpa_provin", "emision", "roja"]],
                                                     on=["dpa_provin", "emision"])
    s = p.groupby(["dpa_provin", "objetivo"], as_index=False).agg(
        emisiones=("roja", "size"), alertada=("roja", "any"), real=("real", "first"), q3_real=("q3_real", "first"))
    s = s[(s["emisiones"] == 3) & (s["real"] > np.maximum(s["q3_real"], PISO))]
    # Rachas rojas: cada una empieza donde la emisión anterior no era roja
    b["empieza"] = b["roja"] & ~_anterior(b.assign(r=b["roja"]), "r")
    b["racha"] = b.sort_values("emision").groupby("dpa_provin")["empieza"].cumsum()
    r = (b[b["roja"]].groupby(["dpa_provin", "racha"], as_index=False)
         .agg(acierta=("epidemia_real", "any"), conocida=("conocida", "all")))
    r["falsa"] = ~r["acierta"] & r["conocida"]
    tabla = pd.DataFrame({
        "semanas_epidemicas": s.groupby("dpa_provin").size(),
        "semanas_alertadas": s.groupby("dpa_provin")["alertada"].sum(),
        "rachas_rojas": r.groupby("dpa_provin").size(),
        "falsas_alarmas": r.groupby("dpa_provin")["falsa"].sum(),
        "anios_evaluados": b[b["conocida"]].groupby("dpa_provin").size() / SEMANAS_POR_ANIO,
    }).reindex(sorted(b["dpa_provin"].unique())).fillna(0)
    tabla.loc["total"] = tabla.sum()
    tabla["sensibilidad"] = tabla["semanas_alertadas"] / tabla["semanas_epidemicas"].replace(0, np.nan)
    tabla["falsas_por_anio"] = tabla["falsas_alarmas"] / tabla["anios_evaluados"].replace(0, np.nan)
    enteras = ["semanas_epidemicas", "semanas_alertadas", "rachas_rojas", "falsas_alarmas"]
    return tabla.astype({c: "int64" for c in enteras}).rename_axis("dpa_provin").reset_index()


def serie_final(f, canal, semanas):
    """Los casos finales de cada provincia y SE (el y_h1 de la emisión anterior), con el canal de su año."""
    s = f[["dpa_provin", "provincia", "inicio_semana", "y_h1"]].rename(columns={"y_h1": "casos"})
    s = s.assign(inicio_semana=s["inicio_semana"] + SEMANA).dropna(subset=["casos"])
    s = s.merge(calendario(semanas), on="inicio_semana", how="left")
    c = canal[["dpa_provin", "anio_epi", "semana_epi", "mediana", "q3"]].astype({"anio_epi": "int64",
                                                                                   "semana_epi": "int64"})
    s = s.merge(c.rename(columns={"mediana": "mediana_real", "q3": "q3_real"}),
                on=["dpa_provin", "anio_epi", "semana_epi"], how="left")
    return s.sort_values(["dpa_provin", "inicio_semana"], ignore_index=True)


def _se(fila):
    return f"SE {int(fila['semana_epi'])}/{int(fila['anio_epi'])}"


def anticipacion(b, serie, anio=2024, antes=26):
    """Por provincia: cuántas semanas antes del pico de `anio` salió la alerta roja de su última subida

    - pico: la SE de `anio` con más casos. Si no pasa el q3 de su canal (y PISO), ese año no hubo brote
    - valle: la SE con menos casos de las `antes` semanas previas al pico; ahí empieza la subida
    - primera_roja: la primera emisión en rojo entre 4 semanas antes del valle (lo anuncia a 4 semanas) y 2 antes
      del pico (lo anuncia a 2). Se publica en su fecha_emision, 7 días después de que cierra su semana
    - anticipacion: las semanas completas entre esa publicación y el inicio de la semana del pico
    - roja_desde: el inicio de la racha roja de esa emisión. Si empezó antes de la ventana, la alerta ya estaba
      encendida desde otra temporada (saturada): no avisó esta subida, y su anticipación no cuenta
    """
    emisiones = b.set_index(["dpa_provin", "emision"])
    filas = []
    for (dpa, provincia), s in serie.groupby(["dpa_provin", "provincia"]):
        del_anio = s[s["anio_epi"] == anio]
        if del_anio.empty:
            continue
        pico = del_anio.loc[del_anio["casos"].idxmax()]
        previas = s[s["inicio_semana"].between(pico["inicio_semana"] - antes * SEMANA, pico["inicio_semana"])]
        valle = previas.loc[previas["casos"][::-1].idxmin()]          # el último mínimo antes del pico
        fila = dict(dpa_provin=dpa, provincia=provincia, brote=bool(pico["casos"] > max(pico["q3_real"], PISO)),
                    pico=pico["inicio_semana"], pico_se=_se(pico), casos_pico=float(pico["casos"]),
                    valle=valle["inicio_semana"], valle_se=_se(valle), casos_valle=float(valle["casos"]),
                    primera_roja=pd.NaT, primera_roja_se=None, publicada=pd.NaT, anticipacion=np.nan,
                    roja_desde=pd.NaT, roja_desde_se=None, saturada=False)
        rojas = b[(b["dpa_provin"] == dpa) & b["roja"]]
        desde = valle["inicio_semana"] - 4 * SEMANA
        en_ventana = rojas[rojas["emision"].between(desde, pico["inicio_semana"] - 2 * SEMANA)]
        if fila["brote"] and not en_ventana.empty:
            primera = emisiones.loc[(dpa, en_ventana["emision"].min())]
            inicio = primera.name[1]
            while inicio - SEMANA in set(rojas["emision"]):
                inicio -= SEMANA
            fila.update(primera_roja=primera.name[1], primera_roja_se=_se(primera),
                        publicada=primera["fecha_emision"],
                        anticipacion=float((pico["inicio_semana"] - primera["fecha_emision"]).days // 7),
                        roja_desde=inicio, roja_desde_se=_se(emisiones.loc[(dpa, inicio)]),
                        saturada=bool(inicio < desde))
        filas.append(fila)
    a = pd.DataFrame(filas)
    a.insert(0, "modelo", b["modelo"].iloc[0])
    a.insert(1, "subida", float(b["subida"].iloc[0]))
    a["avisada"] = a["anticipacion"].notna() & ~a["saturada"]
    return a


# ---------------------------------------------------------------------------------------------------------------
# Inicios de brote
# ---------------------------------------------------------------------------------------------------------------

SEMANAS_BROTE = 4          # un brote empieza con 4 semanas seguidas en zona epidémica...
SEMANAS_CALMA = 8          # ...después de 8 semanas con canal en las que, a lo sumo,
EPIDEMICAS_EN_CALMA = 2    # 2 estuvieron en zona epidémica...
ENTRE_BROTES = 12          # ...y al menos 12 semanas después del inicio anterior
AVISO_ANTES = 8            # el aviso es a tiempo si la roja se publica hasta 8 semanas antes del inicio
AVISO_DESPUES = 4          # y tardío si se publica en las 4 semanas siguientes


def semanas_epidemicas(serie):
    """Por provincia y SE, si la semana está en zona epidémica según los casos finales

    Lo está si la mediana de 3 semanas (la anterior, ella y la siguiente) pasa el q3 del canal de su año y PISO.
    La mediana quita los saltos de una sola semana, como el cero que deja un acumulado que baja. Las semanas que
    faltan en la serie y las que no tienen canal no cuentan como epidémicas
    """
    partes = []
    for (dpa, provincia), s in serie.groupby(["dpa_provin", "provincia"]):
        s = s.set_index("inicio_semana").sort_index()
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="7D"))
        suave = s["casos"].rolling(3, center=True, min_periods=2).median()
        partes.append(pd.DataFrame({
            "dpa_provin": dpa, "provincia": provincia, "inicio_semana": s.index, "casos_suaves": suave.to_numpy(),
            "q3_real": s["q3_real"].to_numpy(),
            "epidemica": (suave > np.maximum(s["q3_real"], PISO)).to_numpy()}))
    return pd.concat(partes, ignore_index=True)


def inicios_de_brote(serie):
    """Los inicios de brote de cada provincia: el paso de la calma a la zona epidémica

    Un inicio es la primera de SEMANAS_BROTE semanas epidémicas seguidas, cuando las SEMANAS_CALMA anteriores
    tenían canal y a lo sumo EPIDEMICAS_EN_CALMA fueron epidémicas, y pasaron ENTRE_BROTES semanas desde el
    inicio anterior. Es lo que una alerta temprana tiene que anunciar. No depende del pico, que solo se conoce al
    terminar la temporada, y una provincia que ya está en zona epidémica no vuelve a tener inicio hasta que baja
    """
    filas = []
    for (dpa, provincia), e in semanas_epidemicas(serie).groupby(["dpa_provin", "provincia"]):
        e = e.reset_index(drop=True)
        epidemica, con_canal = e["epidemica"].to_numpy(), e["q3_real"].notna().to_numpy()
        anterior = None
        for i in range(SEMANAS_CALMA, len(e) - SEMANAS_BROTE + 1):
            previas = slice(i - SEMANAS_CALMA, i)
            en_calma = con_canal[previas].all() and epidemica[previas].sum() <= EPIDEMICAS_EN_CALMA
            if not (en_calma and epidemica[i:i + SEMANAS_BROTE].all()):
                continue
            if anterior is not None and i - anterior < ENTRE_BROTES:
                continue
            anterior = i
            filas.append(dict(dpa_provin=dpa, provincia=provincia, inicio=e.loc[i, "inicio_semana"],
                              casos=float(e.loc[i, "casos_suaves"]), umbral=float(max(e.loc[i, "q3_real"], PISO))))
    return pd.DataFrame(filas, columns=["dpa_provin", "provincia", "inicio", "casos", "umbral"])


def avisos_de_inicio(b, inicios):
    """Para cada inicio de brote, si los boletines `b` (un modelo y una regla) lo avisaron y con cuánta anticipación

    - a tiempo: hay una roja publicada entre AVISO_ANTES semanas antes del inicio y el día en que empieza su semana
    - tarde: la primera roja se publicó en las AVISO_DESPUES semanas siguientes
    - sin aviso: ninguna de las dos
    - anticipacion: las semanas completas entre la publicación de la primera roja a tiempo y el inicio

    Un inicio solo se evalúa si la provincia ya tenía boletines AVISO_ANTES semanas antes
    """
    primer_boletin = b[b["nivel"].notna()].groupby("dpa_provin")["fecha_emision"].min()
    rojas = b[b["roja"]]
    filas = []
    for i in inicios.itertuples(index=False):
        desde = i.inicio - AVISO_ANTES * SEMANA
        if i.dpa_provin not in primer_boletin.index or primer_boletin[i.dpa_provin] > desde:
            continue
        publicadas = rojas.loc[rojas["dpa_provin"] == i.dpa_provin, "fecha_emision"]
        antes = publicadas[publicadas.between(desde, i.inicio)]
        despues = publicadas[(publicadas > i.inicio) & (publicadas <= i.inicio + AVISO_DESPUES * SEMANA)]
        estado = "a tiempo" if len(antes) else "tarde" if len(despues) else "sin aviso"
        publicada = antes.min() if len(antes) else despues.min() if len(despues) else pd.NaT
        filas.append({**i._asdict(), "estado": estado, "publicada": publicada,
                      "anticipacion": float((i.inicio - publicada).days // 7) if len(antes) else np.nan})
    a = pd.DataFrame(filas, columns=[*inicios.columns, "estado", "publicada", "anticipacion"])
    a.insert(0, "modelo", b["modelo"].iloc[0])
    a.insert(1, "subida", float(b["subida"].iloc[0]))
    return a


def metricas_inicio(b, serie):
    """Cuántos inicios de brote avisaron los boletines `b` y cuántas de sus rachas rojas fueron falsas alarmas

    Una racha roja es falsa alarma si empieza en una semana que no es epidémica y ningún brote empieza entre
    AVISO_DESPUES semanas antes y AVISO_ANTES semanas después de su publicación. Una roja que se enciende con el
    brote ya en curso no avisa nada, pero tampoco es falsa. Las rachas demasiado recientes para saber si después
    empezó un brote no se cuentan
    """
    inicios = inicios_de_brote(serie)
    a = avisos_de_inicio(b, inicios)
    con_nivel = b[b["nivel"].notna()].copy()
    con_nivel["empieza"] = con_nivel["roja"] & ~_anterior(con_nivel.assign(r=con_nivel["roja"]), "r")
    epidemicas = semanas_epidemicas(serie).rename(columns={"inicio_semana": "emision"})
    rachas = con_nivel[con_nivel["empieza"]].merge(epidemicas[["dpa_provin", "emision", "epidemica"]],
                                                   on=["dpa_provin", "emision"], how="left")
    ultimo_inicio_visible = serie["inicio_semana"].max() - (SEMANAS_BROTE - 1) * SEMANA
    falsas = 0
    for r in rachas.itertuples(index=False):
        hasta = r.fecha_emision + AVISO_ANTES * SEMANA
        cerca = inicios[(inicios["dpa_provin"] == r.dpa_provin)
                        & inicios["inicio"].between(r.fecha_emision - AVISO_DESPUES * SEMANA, hasta)]
        falsas += int(cerca.empty and r.epidemica != True and hasta <= ultimo_inicio_visible)  # noqa: E712
    anios = len(con_nivel) / SEMANAS_POR_ANIO
    a_tiempo = a["estado"] == "a tiempo"
    return dict(inicios=int(len(a)), inicios_a_tiempo=int(a_tiempo.sum()),
                inicios_tarde=int((a["estado"] == "tarde").sum()),
                inicios_sin_aviso=int((a["estado"] == "sin aviso").sum()),
                anticipacion_inicio=float(a.loc[a_tiempo, "anticipacion"].median()) if a_tiempo.any() else np.nan,
                falsas_inicio=falsas, falsas_inicio_por_anio=falsas / anios if anios else np.nan)


def resumen_regla(b, base, serie, anio=2024):
    """Una fila con las métricas de una regla: sensibilidad, falsas alarmas, inicios de brote avisados y
    anticipación en la Costa y Manabi

    En la anticipación de la Costa, una provincia con brote que no fue avisada (o cuya alerta estaba saturada)
    cuenta como 0 semanas
    """
    total = metricas_alerta(b, base).set_index("dpa_provin").loc["total"]
    a = anticipacion(b, serie, anio)
    costa = a[a["dpa_provin"].isin(list(COSTA)) & a["brote"]]
    manabi = a[a["dpa_provin"] == MANABI]
    util = costa["anticipacion"].where(costa["avisada"], 0.0)
    return dict(modelo=b["modelo"].iloc[0], subida=float(b["subida"].iloc[0]),
                sensibilidad=float(total["sensibilidad"]), semanas_epidemicas=int(total["semanas_epidemicas"]),
                rachas_rojas=int(total["rachas_rojas"]), falsas_alarmas=int(total["falsas_alarmas"]),
                falsas_por_anio=float(total["falsas_por_anio"]),
                anticipacion_costa=float(util.median()) if len(util) else np.nan,
                costa_con_brote=int(len(costa)), costa_avisadas=int(costa["avisada"].sum()),
                anticipacion_manabi=float(manabi["anticipacion"].iloc[0])
                if len(manabi) and manabi["avisada"].iloc[0] else np.nan,
                manabi_roja_desde=manabi["roja_desde_se"].iloc[0] if len(manabi) else None,
                **metricas_inicio(b, serie))


def elegir_regla(reglas, presupuesto=1.0):
    """La regla que avisa a tiempo más inicios de brote entre las que no pasan `presupuesto` falsas alarmas por
    provincia y año; si empatan, la de menos falsas alarmas. Si ninguna cumple, la de menos falsas alarmas

    Antes elegía por la anticipación al pico de 2024 en la Costa, que mide una sola temporada en 7 provincias y da
    crédito a rojas que no duran. Los inicios de brote cubren todas las provincias y todos los años del backtest
    """
    dentro = reglas[reglas["falsas_inicio_por_anio"] <= presupuesto]
    if dentro.empty:
        return reglas.sort_values(["falsas_inicio_por_anio", "inicios_a_tiempo"], ascending=[True, False]).index[0]
    return dentro.sort_values(["inicios_a_tiempo", "falsas_inicio_por_anio"], ascending=[False, True]).index[0]


def tabla_resumen(error, reglas):
    """La tabla del backtest: el error por horizonte de cada modelo y, en los que tienen reglas, sus métricas"""
    t = error.merge(reglas, on="modelo", how="left")
    enteras = ["filas", "semanas_epidemicas", "rachas_rojas", "falsas_alarmas", "costa_con_brote", "costa_avisadas",
               "inicios", "inicios_a_tiempo", "inicios_tarde", "inicios_sin_aviso", "falsas_inicio"]
    t = t.astype({c: "Int64" for c in enteras})     # enteros que admiten nulos: los modelos sin reglas
    t["elegida"] = t["elegida"].eq(True)
    return t


# ---------------------------------------------------------------------------------------------------------------
# Gráfico
# ---------------------------------------------------------------------------------------------------------------

FONDO, TINTA, TINTA_2, TENUE, REJILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL = "#2a78d6"
COLOR_NIVEL = {"verde": "#0ca30c", "amarilla": "#fab219", "roja": "#d03b3b"}


def grafico_costa(serie, b, a, desde="2023-07-02", hasta="2024-12-29", titulo="Alerta de 2024 en la Costa"):
    """Un panel por provincia de la Costa: los casos de cada semana, el q3 de su canal, la alerta vigente cada semana
    (la franja: cada boletín rige desde el sábado en que se publica), la primera roja de la subida y el pico"""
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    fig = Figure(figsize=(16, 7.6), facecolor=FONDO)     # sin pyplot: el notebook la muestra con display()
    ejes = fig.subplots(2, 4)
    for eje, (dpa, nombre) in zip(ejes.flat, COSTA.items()):
        s = serie[(serie["dpa_provin"] == dpa) & serie["inicio_semana"].between(desde, hasta)]
        boletin = b[(b["dpa_provin"] == dpa) & b["fecha_emision"].between(desde, hasta) & b["nivel"].notna()]
        fila = a[a["dpa_provin"] == dpa].iloc[0] if (a["dpa_provin"] == dpa).any() else None
        tope = 1.18 * max(s["casos"].max(), s["q3_real"].max(skipna=True) if s["q3_real"].notna().any() else 0, 1)
        eje.set_facecolor(FONDO)
        eje.plot(s["inicio_semana"], s["q3_real"], color=TENUE, lw=1)
        eje.plot(s["inicio_semana"], s["casos"], color=AZUL, lw=2, solid_joinstyle="round", solid_capstyle="round")
        eje.bar(boletin["fecha_emision"] + pd.Timedelta(days=1), 0.05 * tope, width=7, bottom=-0.09 * tope,
                align="edge", color=boletin["nivel"].map(COLOR_NIVEL), linewidth=0)
        estado = "sin brote en 2024"
        if fila is not None and fila["brote"]:
            eje.plot(fila["pico"], fila["casos_pico"], "o", ms=8, color=AZUL, mec=FONDO, mew=2, zorder=5)
            eje.annotate(f"pico {fila['pico_se']}", (fila["pico"], fila["casos_pico"]), xytext=(7, 2),
                         textcoords="offset points", fontsize=8.5, color=TINTA_2)
            if fila["avisada"]:
                eje.axvline(fila["publicada"], color=COLOR_NIVEL["roja"], lw=1)
                eje.annotate(f"roja {fila['primera_roja_se']}", (fila["publicada"], 0.95 * tope), xytext=(4, 0),
                             textcoords="offset points", va="center", fontsize=8.5, color=TINTA_2)
                estado = f"roja {int(fila['anticipacion'])} semanas antes del pico"
            elif fila["saturada"]:
                estado = f"en roja desde la {fila['roja_desde_se']}"
            else:
                estado = "sin roja antes del pico"
        eje.set_title(f"{nombre} · {estado}", loc="left", fontsize=10, color=TINTA)
        eje.set_ylim(-0.1 * tope, tope)
        eje.yaxis.set_major_locator(MaxNLocator(4, integer=True))
        eje.set_yticks([v for v in eje.get_yticks() if 0 <= v <= tope])
        marcas = s[s["semana_epi"].isin([1, 14, 27, 40])]
        eje.set_xticks(marcas["inicio_semana"], [f"SE {n}\n{y}" for n, y in zip(marcas["semana_epi"],
                                                                                marcas["anio_epi"])])
        eje.set_xlim(desde, hasta + SEMANA)
        eje.grid(axis="y", color=REJILLA, lw=0.8)
        eje.set_axisbelow(True)
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color(EJE)
        eje.tick_params(colors=TENUE, labelcolor=TINTA_2, labelsize=8.5)
    leyenda = ejes.flat[-1]
    leyenda.axis("off")
    leyenda.legend(handles=[
        Line2D([], [], color=AZUL, lw=2, label="casos de la semana (cifra final)"),
        Line2D([], [], color=TENUE, lw=1, label="q3 del canal: encima, zona epidémica"),
        Patch(color=COLOR_NIVEL["verde"], label="alerta verde"),
        Patch(color=COLOR_NIVEL["amarilla"], label="alerta amarilla"),
        Patch(color=COLOR_NIVEL["roja"], label="alerta roja"),
        Line2D([], [], color=COLOR_NIVEL["roja"], lw=1, label="se publica la primera roja de la subida"),
        Line2D([], [], color=AZUL, marker="o", ms=8, mec=FONDO, mew=2, lw=0, label="pico de 2024"),
    ], loc="center left", frameon=False, fontsize=9.5, labelcolor=TINTA_2)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=14, color=TINTA)
    fig.text(0.01, 0.915, "Casos semanales frente al canal endémico. La franja de abajo es la alerta vigente cada "
             "semana, desde el sábado en que se publica. Cada panel tiene su propia escala.",
             fontsize=10, color=TINTA_2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig