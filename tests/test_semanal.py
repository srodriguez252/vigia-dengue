"""Pruebas del boletín semanal (src/modelos/semanal.py). No usan Spark: solo pandas y scikit-learn

Revisan lo que el Job hace cada sábado: qué emisiones pronostica, el intervalo, las camas-día, la regla de alerta
con la emisión anterior y las frases que explican cada nivel
"""
import numpy as np
import pandas as pd

from modelos import pronostico as P
from modelos import semanal as S
from test_pronostico import canal_de_prueba, features_de_prueba


def test_pronostica_solo_lo_publicado_que_falta():
    f = P.preparar(features_de_prueba())
    hoy = pd.Timestamp("2023-06-15")   # jueves: la última emisión publicada es la del sábado 10 de junio
    todas = S.emisiones_pendientes(f, [], hoy, semanas=3)
    assert todas == [pd.Timestamp("2023-05-14"), pd.Timestamp("2023-05-21"), pd.Timestamp("2023-05-28")]
    assert all(f.loc[f["inicio_semana"].isin(todas), "fecha_emision"] <= hoy)
    hechas = [pd.Timestamp("2023-05-21").date()]   # como llegan de Spark
    assert S.emisiones_pendientes(f, hechas, hoy, semanas=3) == [pd.Timestamp("2023-05-14"),
                                                                 pd.Timestamp("2023-05-28")]


def test_el_intervalo_cubre_lo_pedido():
    rng = np.random.default_rng(3)
    n = 4000
    pronostico = rng.uniform(0, 200, n)
    real = (pronostico + 1) * np.exp(rng.normal(0, 0.4, n)) - 1
    backtest = pd.DataFrame({"modelo": "m", "horizonte": np.tile([1, 2, 3, 4], n // 4), "pronostico": pronostico,
                             "real": real, "evaluable": True})
    factores = S.factores_intervalo(backtest, "m", cobertura=0.8)
    b = backtest.merge(factores, on="horizonte")
    bajo = np.maximum((b["pronostico"] + 1) * np.exp(b["error_bajo"]) - 1, 0)
    alto = (b["pronostico"] + 1) * np.exp(b["error_alto"]) - 1
    assert (bajo <= b["pronostico"]).all() and (b["pronostico"] <= alto).all()
    assert abs(((b["real"] >= bajo) & (b["real"] <= alto)).mean() - 0.8) < 0.01


def test_las_tasas_de_hospitalizacion_usan_el_pais_si_hay_pocos_datos():
    egresos = pd.DataFrame({"provincia_res": ["09", "09", "13", "17"], "anio_epi": [2016, 2018, 2018, 2018],
                            "hospitalizaciones": [999, 100, 10, 20], "dias_estada": [9999, 500, 30, 60]})
    casos = pd.DataFrame({"dpa_provin": ["09", "13", "17", "09"], "anio_epi": [2018, 2018, 2018, 2025],
                          "casos": [400.0, 100.0, 100.0, 999.0]})
    t = S.tasas_hospitalarias(egresos, casos).set_index("dpa_provin")
    assert t.loc["09", "anios"] == "2018-2018"    # 2016 no tiene gacetas y 2025 no tiene egresos
    assert t.loc["09", "hosp_por_caso"] == 0.25 and t.loc["09", "dias_por_hospitalizacion"] == 5
    assert t.loc["09", "fuente"] == "provincia"
    assert t.loc["13", "fuente"] == "país"    # 10 hospitalizaciones: menos de 30
    assert t.loc["13", "hosp_por_caso"] == 130 / 600 and t.loc["13", "dias_por_hospitalizacion"] == 590 / 130


class ModeloDePrueba:
    """Pronostica 10, 20, 30 y 40 casos a 1, 2, 3 y 4 semanas; solo mira las columnas con que se entrenó"""

    feature_names_in_ = np.array(["casos_lag0", "casos_media_4s"])

    def predict(self, x):
        assert list(x.columns) == list(self.feature_names_in_)
        return np.tile([10.0, 20.0, 30.0, 40.0], (len(x), 1))


def test_pronosticar_arma_las_filas_de_gold_pronosticos():
    features = features_de_prueba()
    f = P.preparar(features)
    semanas = features[["anio_epi", "semana_epi", "inicio_semana"]].drop_duplicates()
    emisiones = [pd.Timestamp("2022-12-11"), pd.Timestamp("2022-12-18")]
    intervalo = pd.DataFrame({"horizonte": [1, 2, 3, 4], "error_bajo": -0.5, "error_alto": 0.5})
    tasas = pd.DataFrame({"dpa_provin": ["09", "13", "17"], "hosp_por_caso": [0.2, 0.5, 0.1],
                          "dias_por_hospitalizacion": [4.0, 3.0, 2.0]})
    p = S.pronosticar(ModeloDePrueba(), f, canal_de_prueba(features), semanas, emisiones, intervalo, tasas,
                      "gbm_global", 7)
    assert list(p.columns) == S.COLUMNAS_PRONOSTICO
    assert len(p) == 3 * 2 * 4 and set(p["emision"]) == set(emisiones)
    assert (p["pronostico_bajo"] <= p["pronostico"]).all() and (p["pronostico"] <= p["pronostico_alto"]).all()
    manabi = p[(p["dpa_provin"] == "13") & (p["horizonte"] == 4)].set_index("emision")
    assert manabi.loc[emisiones[1], "camas_dia"] == 40 * 0.5 * 3
    # La emisión de la SE 51 de 2022 pronostica a 4 semanas una SE de 2023 con el canal de 2022 (el de 2023 es 50)
    assert manabi.loc[emisiones[1], "anio_objetivo"] == 2023 and manabi.loc[emisiones[1], "canal_mediana"] == 25
    assert set(p["version"]) == {"7"} and set(p["modelo"]) == {"gbm_global"}


def pronosticos_guardados(emisiones, canal_mediana=10.0, canal_q3=20.0, casos_media_4s=10.0):
    """Lo que guarda gold.pronosticos para una provincia: emisiones = {semana: pronóstico a 1-4 semanas}"""
    filas = []
    for semana, pronostico in emisiones.items():
        emision = pd.Timestamp("2024-01-07") + (semana - 1) * P.SEMANA
        for h in P.HORIZONTES:
            filas.append(dict(dpa_provin="13", provincia="MANABÍ", emision=emision, anio_epi=2024, semana_epi=semana,
                              fecha_emision=emision + pd.Timedelta(days=13), horizonte=h,
                              objetivo=emision + h * P.SEMANA, anio_objetivo=2024, semana_objetivo=semana + h,
                              pronostico=float(pronostico), canal_mediana=canal_mediana, canal_q3=canal_q3,
                              casos_semana=9.0, casos_media_4s=casos_media_4s, camas_dia=2.0, version="3"))
    return pd.DataFrame(filas)


def test_la_roja_mira_la_emision_anterior_y_cada_nivel_explica_su_motivo():
    p = pronosticos_guardados({1: 8, 2: 25, 3: 25, 4: 15})
    semanas = sorted(p["emision"].unique())
    a = S.alertas(p, 0.0, semanas[1:]).set_index("semana_epi")
    assert list(a.columns) == [c for c in S.COLUMNAS_ALERTA if c != "semana_epi"]
    assert a["nivel"].to_dict() == {2: "amarilla", 3: "roja", 4: "amarilla"}   # la SE 1 da contexto
    assert a.loc[2, "motivo"].startswith("Primera semana con el pronóstico en zona epidémica: 25,0 casos")
    assert a.loc[3, "motivo"].startswith("Segunda semana seguida")
    assert a.loc[4, "motivo"] == ("Pronóstico sobre la mediana del canal: 15,0 casos en la SE 6, sobre el umbral "
                                  "de alerta de 10,0")
    assert a.loc[3, "camas_dia_4s"] == 8.0 and a.loc[3, "casos_4s"] == 100.0


def test_la_subida_y_la_falta_de_canal_tambien_se_explican():
    p = pronosticos_guardados({1: 25, 2: 25})
    a = S.alertas(p, 3.0, sorted(p["emision"].unique())).set_index("semana_epi")
    assert set(a["nivel"]) == {"amarilla"}     # 25 casos no son 3 veces el promedio de 10
    assert a.loc[2, "motivo"].startswith("Pronóstico en zona epidémica, pero sin subida")
    verde = S.alertas(pronosticos_guardados({1: 3}), 0.0, [pd.Timestamp("2024-01-07")])
    assert verde.loc[0, "motivo"] == "Hasta 5 casos por semana en las próximas 2 a 4 semanas"
    sin_canal = S.alertas(pronosticos_guardados({1: 30}, np.nan, np.nan), 0.0, [pd.Timestamp("2024-01-07")])
    assert sin_canal.loc[0, "nivel"] == S.SIN_CANAL and sin_canal.loc[0, "motivo"].startswith("Sin canal")