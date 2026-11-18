"""Escritura de las tablas que solo crecen: gold.pronosticos y gold.alertas

Es lo único de modelos/ que usa Spark: lo importan los notebooks del Job. Cada boletín tiene una clave (provincia,
emisión y, en los pronósticos, horizonte) y una fila ya escrita no se toca: si la tarea corre dos veces, la
segunda no agrega nada
"""
from pyspark.sql import functions as F


def filas_nuevas(spark, df, tabla, claves):
    """Las filas de `df` cuya clave todavía no está en `tabla`, con las columnas y los tipos de la tabla"""
    destino = spark.table(tabla)
    filas = df.select([F.col(c.name).cast(c.dataType) for c in destino.schema.fields]).dropDuplicates(claves)
    return filas.join(destino.select(*claves).distinct(), claves, "left_anti")


def comprobar_clave(spark, tabla, claves):
    """Falla si en `tabla` hay más de una fila con la misma clave"""
    repetidas = spark.table(tabla).groupBy(*claves).count().where("count > 1").count()
    if repetidas:
        raise ValueError(f"{tabla} tiene {repetidas} claves repetidas ({', '.join(claves)})")


def agregar_sin_repetir(spark, df, tabla, claves):
    """Agrega a `tabla` las filas de `df` cuya clave no está y comprueba que la clave siga siendo única

    El MERGE es atómico: si dos corridas escriben a la vez (un reintento, o una corrida a mano mientras corre la
    programada), la segunda encuentra las claves de la primera y no las repite
    """
    filas_nuevas(spark, df, tabla, claves).createOrReplaceTempView("_filas_nuevas")
    coinciden = " AND ".join(f"t.{c} = s.{c}" for c in claves)
    spark.sql(f"MERGE INTO {tabla} AS t USING _filas_nuevas AS s ON {coinciden} WHEN NOT MATCHED THEN INSERT *")
    comprobar_clave(spark, tabla, claves)
