"""Pruebas de la escritura de los boletines (src/modelos/tablas.py), con Spark local

El MERGE necesita una tabla Delta y solo corre en Databricks; aquí se prueba lo que decide qué filas entran y el
control que detiene la tarea si una clave queda repetida
"""
import datetime as dt

import pytest

from modelos import tablas as T

CLAVES = ["dpa_provin", "emision"]
ESQUEMA = "dpa_provin STRING, emision DATE, casos DOUBLE"
SE_1, SE_2 = dt.date(2026, 1, 4), dt.date(2026, 1, 11)


def test_solo_entran_las_filas_cuya_clave_no_esta(spark):
    spark.createDataFrame([("09", SE_1, 30.0), ("13", SE_1, 12.0)], ESQUEMA).createOrReplaceTempView("boletines")
    # Llegan con otro orden de columnas y otros tipos, una clave que ya está y una repetida dentro del mismo lote
    lote = spark.createDataFrame([(99, "09", SE_1), (40, "09", SE_2), (41, "09", SE_2), (15, "13", SE_2)],
                                 "casos INT, dpa_provin STRING, emision DATE")
    nuevas = T.filas_nuevas(spark, lote, "boletines", CLAVES)
    assert nuevas.schema.simpleString() == spark.table("boletines").schema.simpleString()
    filas = sorted((f.dpa_provin, f.emision) for f in nuevas.collect())
    assert filas == [("09", SE_2), ("13", SE_2)]       # la fila ya escrita de la SE 1 no se toca ni se repite
    assert T.filas_nuevas(spark, spark.table("boletines"), "boletines", CLAVES).count() == 0   # correr dos veces


def test_una_clave_repetida_detiene_la_tarea(spark):
    spark.createDataFrame([("09", SE_1, 30.0), ("13", SE_1, 12.0)], ESQUEMA).createOrReplaceTempView("boletines_bien")
    T.comprobar_clave(spark, "boletines_bien", CLAVES)
    spark.createDataFrame([("09", SE_1, 30.0), ("09", SE_1, 31.0)], ESQUEMA).createOrReplaceTempView("boletines_mal")
    with pytest.raises(ValueError, match="1 claves repetidas"):
        T.comprobar_clave(spark, "boletines_mal", CLAVES)
