"""Pruebas del parser de gacetas (vigia-etl/transformations/silver_gacetas.py)

Las gacetas reales no están en el repositorio. Cada prueba arma un PDF mínimo con la forma de una gaceta (portada con
la semana y la fecha de corte, y una tabla por provincia) y se lo da al mismo código que corre en el pipeline. El
parser lee por posición, así que el PDF coloca cada número bajo el rótulo de su columna, como los del MSP
"""
import datetime as dt
import importlib.util
import pathlib
import sys
import types

import pandas as pd
import pytest

pytest.importorskip("pdfplumber")   # sin pdfplumber (un notebook sin instalarlo), estas pruebas se saltan

RAIZ = pathlib.Path(__file__).resolve().parents[1]
ALTO = 792
PRIMERA_COLUMNA, ENTRE_COLUMNAS = 250, 75


@pytest.fixture(scope="module")
def parser():
    """El módulo del pipeline, importado fuera de él: `dp` se reemplaza por decoradores que no hacen nada"""
    import pyspark

    falso = types.ModuleType("pyspark.pipelines")
    falso.__getattr__ = lambda nombre: (lambda *args, **kwargs: (lambda funcion: funcion))
    ausente = object()
    antes = sys.modules.get("pyspark.pipelines", ausente), getattr(pyspark, "pipelines", ausente)
    sys.modules["pyspark.pipelines"] = pyspark.pipelines = falso
    try:
        spec = importlib.util.spec_from_file_location(
            "silver_gacetas", RAIZ / "vigia-etl" / "transformations" / "silver_gacetas.py")
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
    finally:
        for destino, nombre, valor in ((sys.modules, "pyspark.pipelines", antes[0]), (vars(pyspark), "pipelines",
                                                                                   antes[1])):
            if valor is ausente:
                destino.pop(nombre, None)
            else:
                destino[nombre] = valor
    return modulo


def pdf_de(paginas):
    """Un PDF de texto: cada página es una lista de (x, y desde arriba, texto[, tamaño]), en Helvetica"""
    objetos = [b"<< /Type /Catalog /Pages 2 0 R >>", None,
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    hijos = []
    for textos in paginas:
        trazos = []
        for x, y, texto, *tamano in textos:
            seguro = texto.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            trazos.append(f"BT /F1 {tamano[0] if tamano else 9} Tf {x:.2f} {ALTO - y:.2f} Td ({seguro}) Tj ET")
        contenido = "\n".join(trazos).encode("cp1252")
        n = len(objetos) + 1
        hijos.append(f"{n} 0 R")
        objetos.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 {ALTO}] "
                       f"/Resources << /Font << /F1 3 0 R >> >> /Contents {n + 1} 0 R >>".encode())
        objetos.append(b"<< /Length " + str(len(contenido)).encode() + b" >>\nstream\n" + contenido + b"\nendstream")
    objetos[1] = f"<< /Type /Pages /Kids [{' '.join(hijos)}] /Count {len(hijos)} >>".encode()
    salida, posiciones = b"%PDF-1.4\n", []
    for i, cuerpo in enumerate(objetos, start=1):
        posiciones.append(len(salida))
        salida += f"{i} 0 obj\n".encode() + cuerpo + b"\nendobj\n"
    inicio_xref = len(salida)
    salida += f"xref\n0 {len(objetos) + 1}\n0000000000 65535 f \n".encode()
    salida += b"".join(f"{p:010d} 00000 n \n".encode() for p in posiciones)
    salida += f"trailer\n<< /Size {len(objetos) + 1} /Root 1 0 R >>\nstartxref\n{inicio_xref}\n%%EOF\n".encode()
    return salida


def ancho(texto, tamano=9):
    """Ancho aproximado en Helvetica: los dígitos miden 0,556 del tamaño; la coma, el punto y el espacio, 0,278"""
    return sum(0.278 if c in ",. " else 0.556 if c.isdigit() or c == "-" else 0.667 for c in texto) * tamano


def gaceta(filas, rotulos=("SE 01-41", "SE 42", "TOTAL"), total=None, titulo="CASOS DE DENGUE POR PROVINCIA",
           semana=42, corte="19/10/2024"):
    """Una gaceta de dos páginas: la portada, con la semana y el corte, y la tabla PROVINCIA | rótulos

    filas: (nombre, valor por columna); None deja la celda vacía. total: los valores de la fila Total, si la hay
    """
    portada = [(72, 80, "MINISTERIO DE SALUD PUBLICA DEL ECUADOR", 12),
               (72, 100, "GACETA DE ENFERMEDADES TRANSMITIDAS POR VECTORES", 12),
               (72, 130, f"SEMANA EPIDEMIOLOGICA (SE) {semana} / 2024", 11), (72, 150, f"FECHA DE CORTE: {corte}", 11)]
    portada += [(72, 200 + 14 * i, "La vigilancia de las enfermedades vectoriales se hace por notificacion semanal.")
                for i in range(8)]
    columnas = [PRIMERA_COLUMNA + ENTRE_COLUMNAS * i for i in range(len(rotulos))]
    tabla = [(72, 90, titulo, 11), (72, 130, "PROVINCIA")] + [(x, 130, r) for x, r in zip(columnas, rotulos)]
    y = 146
    for nombre, *valores in [*filas, *([("TOTAL", *total)] if total else [])]:
        tabla.append((72, y, nombre))
        for x, rotulo, valor in zip(columnas, rotulos, valores):
            if valor is not None:   # centrado bajo su rótulo
                tabla.append((x + ancho(rotulo) / 2 - ancho(str(valor)) / 2, y, str(valor)))
        y += 14
    return pdf_de([portada, tabla])


def leer(parser, contenido, se_archivo=42):
    pdf = types.SimpleNamespace(anio=2024, archivo="SE42__gaceta.pdf", sha256="abc", se_archivo=se_archivo,
                                content=contenido)
    return pd.DataFrame(parser.leer_pdf(pdf))


CASOS = [("AZUAY", 17, None, 17), ("GUAYAS", "1.234", 56, "1.290"), ("LOS RIOS", 310, 12, 322),
         ("MANABI", 845, 40, 885), ("SANTO DOMINGO DE LOS TSACHILAS", 96, 4, 100)]
TOTAL_CASOS = ("2.502", 112, "2.614")


def test_lee_una_tabla_de_casos(parser):
    g = leer(parser, gaceta(CASOS, total=TOTAL_CASOS))
    assert "motivo" not in g or g["motivo"].isna().all()
    # La semana sale de la portada; el corte, de su fecha; el formato, de las columnas
    assert set(g["semana_epi"]) == {42} and set(g["fecha_corte_pdf"]) == {dt.date(2024, 10, 19)}
    assert set(g["familia"]) == {"semanal"} and set(g["pagina"]) == {2}
    provincias = g[~g["es_total"]].set_index("provincia")
    assert list(provincias.index) == ["01", "09", "12", "13", "23"]          # el código DPA de cada nombre
    # El punto separa miles, y una celda vacía es un cero: AZUAY 17 _ 17 son 17 previos y ninguno en la semana
    assert provincias.loc["09", ["casos_previos", "casos_semana", "casos_acumulados"]].tolist() == [1234, 56, 1290]
    assert provincias.loc["01", ["casos_previos", "casos_semana", "casos_acumulados"]].tolist() == [17, 0, 17]
    assert (provincias["casos_previos"] + provincias["casos_semana"] == provincias["casos_acumulados"]).all()
    # La fila Total queda aparte, para conciliar: las provincias suman lo que dice el PDF
    total = g[g["es_total"]].iloc[0]
    assert total["casos_acumulados"] == provincias["casos_acumulados"].sum() == 2614


def test_una_tabla_con_decimales_son_tasas_no_casos(parser):
    tasas = [("AZUAY", "2,10", "0,00", "2,10"), ("GUAYAS", "27,45", "1,25", "28,70"),
             ("MANABI", "52,80", "2,50", "55,30")]
    g = leer(parser, gaceta(tasas)).set_index("provincia")
    assert set(g["familia"]) == {"tasas"}
    assert g.loc["09", ["tasa_previa", "tasa_semana", "tasa_acumulada"]].tolist() == [27.45, 1.25, 28.7]
    assert g[["casos_previos", "casos_semana", "casos_acumulados"]].isna().all().all()


def test_la_tabla_que_solo_trae_el_acumulado(parser):
    """El formato de 2017: DCSA | DG | DSSA | Total, sin columnas por semana"""
    filas = [("GUAYAS", 30, 2, 400, 432), ("MANABI", 12, 1, 150, 163)]
    rotulos, total = ("DCSA", "DG", "DSSA", "TOTAL"), (42, 3, 550, 595)
    g = leer(parser, gaceta(filas, rotulos, total)).set_index("provincia")
    assert set(g["familia"]) == {"acumulada"}
    assert g.loc["13", "casos_acumulados"] == 163 and pd.isna(g.loc["13", "casos_semana"])
    # En la gaceta de la SE 1, el acumulado es todo de la semana
    se_1 = leer(parser, gaceta(filas, rotulos, total, semana=1), se_archivo=1).set_index("provincia")
    assert se_1.loc["13", ["casos_previos", "casos_semana", "casos_acumulados"]].tolist() == [0, 163, 163]


@pytest.mark.parametrize("contenido, motivo", [
    (gaceta(CASOS), "la tabla terminó antes de la fila Total"),                          # sin fila Total
    (gaceta([*CASOS, ("GUAYAS", 1, 1, 2)], total=TOTAL_CASOS), "provincias repetidas en la tabla: ['09']"),
    (gaceta(CASOS, total=TOTAL_CASOS, titulo="CASOS DE MALARIA POR PROVINCIA"), "no encontré la tabla de dengue"),
    (pdf_de([[]]), "el PDF no trae texto: es una imagen"),
], ids=["sin_total", "provincia_repetida", "otra_enfermedad", "sin_texto"])
def test_lo_que_no_se_lee_con_confianza_va_a_cuarentena(parser, contenido, motivo):
    g = leer(parser, contenido)
    assert len(g) == 1 and g.loc[0, "motivo"].startswith(motivo)        # una fila con el motivo, sin datos a medias
    assert g.loc[0, "archivo"] == "SE42__gaceta.pdf"


def test_un_pdf_danado_no_detiene_el_lote(parser):
    lote = pd.DataFrame({"anio": [2024, 2024], "archivo": ["SE41__rota.pdf", "SE42__gaceta.pdf"],
                         "se_archivo": [41, 42], "sha256": ["a", "b"],
                         "content": [b"esto no es un PDF", gaceta(CASOS, total=TOTAL_CASOS)]})
    salida = next(parser.leer_lote(iter([lote])))
    assert list(salida.columns) == parser.COLUMNAS
    rota = salida[salida["archivo"] == "SE41__rota.pdf"]
    assert len(rota) == 1 and rota["motivo"].iloc[0].startswith("error del parser")
    assert len(salida[salida["archivo"] == "SE42__gaceta.pdf"]) == len(CASOS) + 1     # las provincias y el Total


def test_numeros_y_nombres_como_los_escribe_el_msp(parser):
    assert [parser.numero(t) for t in ("1.234", "12,5", "0", "N/A", "12.5")] == [1234, 12.5, 0, None, None]
    assert parser.provincia("Los Ríos") == "12" and parser.provincia("SANTO DOMINGO") == "23"   # nombre cortado
    assert parser.provincia("ZONA NO DELIMITADA") == "90"
    assert parser.provincia("SANT") is None      # Santa Elena o Santo Domingo: no se adivina
    assert parser.provincia("TOTAL") is None


def test_la_coma_como_separador_de_miles(parser):
    """«3,098» son 3 098 casos solo si ningún número lleva punto y las provincias suman la fila Total"""
    fila = lambda codigo, *textos: {"codigo": codigo, "textos": dict(enumerate(textos))}   # noqa: E731
    assert parser.coma_de_miles([fila("09", "3,098"), fila("13", "902"), fila(None, "4,000")])
    assert not parser.coma_de_miles([fila("09", "3,098"), fila("13", "902"), fila(None, "9,999")])   # no suma
    assert not parser.coma_de_miles([fila("09", "30,98"), fila("13", "9,02"), fila(None, "40,00")])   # son tasas


def test_la_semana_de_la_gaceta(parser):
    # Manda la portada; sin ella votan el título, la portada vieja, la columna y el enlace, y en un empate gana la
    # columna de la tabla
    assert parser.elegir_semana(42, 41, 41, 41, 41) == 42
    assert parser.elegir_semana(None, 41, 42, 42, 41) == 42
    assert parser.elegir_semana(None, None, None, None, None) is None
    columnas = lambda *textos: [{"texto": t} for t in textos]   # noqa: E731
    # Una semana de más (SE 9 de 2017) solo cuenta en el total
    assert parser.columnas_de_la_semana(columnas("SE 08", "SE 09", "SE 10", "TOTAL"), 9) == (1, [0])
    # La semana primero y sin SE, porque la tabla dinámica ordenó el número antes que el texto (SE 46 de 2019)
    assert parser.columnas_de_la_semana(columnas("46", "SE01-44", "SE45", "TOTAL"), 46) == (0, [1, 2])
