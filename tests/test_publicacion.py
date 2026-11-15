"""Pruebas de la fecha de publicación de las gacetas (vigia-etl/utilities/publicacion.py)

Cada caso es una gaceta de la semana que cierra el sábado 16 de diciembre de 2023, sin fecha de corte en el PDF
(su corte es ese sábado), con un rastro distinto de su subida al servidor del MSP
"""
import datetime as dt

import pytest

from utilities import publicacion as pb

FIN = dt.date(2023, 12, 16)
SITIO = "https://www.salud.gob.ec/wp-content/"


def utc(anio, mes, dia, hora=15):
    return dt.datetime(anio, mes, dia, hora, tzinfo=dt.timezone.utc)


# archivo: (carpeta de la URL, Last-Modified, visto_desde) -> (fecha_publicacion, publicacion_desde, origen)
CASOS = {
    # La fecha del servidor cae en el mes de la carpeta: es la subida original
    "original.pdf": (("uploads/2023/12/", utc(2023, 12, 20), None),
                     (dt.date(2023, 12, 20), FIN, "servidor")),
    # 02:00 UTC del 21 son las 21:00 del 20 en Ecuador
    "de_noche.pdf": (("uploads/2023/12/", utc(2023, 12, 21, 2), None),
                     (dt.date(2023, 12, 20), FIN, "servidor")),
    # El MSP reemplazó el archivo en marzo: se sabe que se subió en diciembre, no qué día
    "reemplazada.pdf": (("uploads/2023/12/", utc(2024, 3, 2), None),
                        (dt.date(2023, 12, 31), FIN, "carpeta")),
    # Se subió en febrero: antes del 1 de febrero no estaba
    "dos_meses_despues.pdf": (("uploads/2024/02/", utc(2024, 3, 2), None),
                              (dt.date(2024, 2, 29), dt.date(2024, 2, 1), "carpeta")),
    # La corrida semanal del 23 vio aparecer el enlace; el archivo cambió después
    "vista.pdf": (("uploads/2023/12/", utc(2023, 12, 27), dt.date(2023, 12, 23)),
                  (dt.date(2023, 12, 23), FIN, "visto")),
    # Una carpeta de un mes anterior a la semana no dice nada; la fecha del servidor vale si está cerca
    "carpeta_vieja.pdf": (("uploads/2022/05/", utc(2023, 12, 21), None),
                          (dt.date(2023, 12, 21), FIN, "servidor")),
    # ...y si está lejos, no se sabe si se publicó tarde o se reemplazó: queda la fecha de corte
    "carpeta_vieja_y_lejos.pdf": (("uploads/2022/05/", utc(2024, 6, 1), None),
                                  (FIN, FIN, "corte")),
    "sin_carpeta.pdf": (("documentos/", None, None),
                        (FIN, FIN, "corte")),
}


def gacetas_de(spark, archivos):
    return spark.createDataFrame([(2023, archivo, FIN, FIN) for archivo in archivos],
                                 "anio INT, archivo STRING, fin_semana DATE, fecha_corte DATE")


@pytest.fixture(scope="module")
def publicadas(spark):
    indice = [(2023, f"{SITIO}{carpeta}{archivo}", archivo, modificado, visto)
              for archivo, ((carpeta, modificado, visto), _) in CASOS.items()]
    # El mismo archivo enlazado dos veces: vale la fecha más antigua
    indice.append((2023, f"{SITIO}uploads/2023/12/copia/original.pdf", "original.pdf", utc(2023, 12, 28), None))
    indice.append((2023, f"{SITIO}uploads/2023/12/roto.pdf", None, None, None))   # un enlace que no se pudo bajar
    indice = spark.createDataFrame(
        indice, "anio INT, url STRING, archivo STRING, modificado_servidor TIMESTAMP, visto_desde DATE")
    return pb.con_publicacion(gacetas_de(spark, [*CASOS, "fuera_del_indice.pdf"]), indice).collect()


@pytest.mark.parametrize("archivo", CASOS)
def test_la_fecha_de_publicacion_sale_del_rastro_mas_temprano(publicadas, archivo):
    fila = next(f for f in publicadas if f.archivo == archivo)
    assert (fila.fecha_publicacion, fila.publicacion_desde, fila.publicacion_origen) == CASOS[archivo][1]


def test_cada_gaceta_sale_una_vez_y_sin_rastro_queda_su_corte(publicadas):
    assert len(publicadas) == len(CASOS) + 1                # los enlaces repetidos no duplican la gaceta
    fila = next(f for f in publicadas if f.archivo == "fuera_del_indice.pdf")
    assert (fila.fecha_publicacion, fila.publicacion_origen) == (FIN, "corte")
    assert all(f.publicacion_desde <= f.fecha_publicacion for f in publicadas)


def test_un_indice_sin_las_columnas_nuevas_no_rompe_el_pipeline(spark):
    """El índice que dejó la versión anterior del notebook no trae modificado_servidor ni visto_desde"""
    viejo = spark.createDataFrame([(2023, f"{SITIO}uploads/2023/12/a.pdf", "a.pdf")],
                                  "anio INT, url STRING, archivo STRING")
    fila = pb.con_publicacion(gacetas_de(spark, ["a.pdf"]), viejo).first()
    assert (fila.fecha_publicacion, fila.publicacion_origen) == (dt.date(2023, 12, 31), "carpeta")
