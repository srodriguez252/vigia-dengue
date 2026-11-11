"""Configuración de pytest: las pruebas importan el mismo módulo que usa el pipeline

Corren igual en GitHub Actions (con un Spark local) y en un notebook de Databricks (con la sesión del notebook)
"""
import os
import pathlib
import sys

import pytest
from pyspark.sql import SparkSession

CARPETA_DEL_PIPELINE = "vigia-etl"   # la que tiene transformations/ y utilities/
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / CARPETA_DEL_PIPELINE))


@pytest.fixture(scope="session")
def spark():
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:   # dentro de Databricks: la sesión que ya existe
        yield SparkSession.builder.getOrCreate()
        return
    sesion = (SparkSession.builder.master("local[2]").appName("pruebas-vigia")
              .config("spark.sql.shuffle.partitions", "4")
              .config("spark.sql.session.timeZone", "UTC")
              .config("spark.ui.enabled", "false")
              .getOrCreate())
    yield sesion
    sesion.stop()