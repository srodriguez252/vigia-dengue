# Silver - gacetas del MSP: la tabla de dengue por provincia de cada PDF.
# El parser corre dentro de mapInPandas; lo que no puede leer con confianza va a la cuarentena
import collections
import datetime as dt
import io
import itertools
import re
import statistics
import unicodedata

import pandas as pd
import pdfplumber
from pyspark import pipelines as dp
from pyspark.sql import functions as F

# Códigos DPA del INEC. Las claves van como las deja clave(): sin espacios, tildes ni puntos
PROVINCIAS = {
    "AZUAY": "01", "BOLIVAR": "02", "CANAR": "03", "CARCHI": "04", "COTOPAXI": "05", "CHIMBORAZO": "06",
    "ELORO": "07", "ESMERALDAS": "08", "GUAYAS": "09", "IMBABURA": "10", "LOJA": "11", "LOSRIOS": "12",
    "MANABI": "13", "MORONASANTIAGO": "14", "NAPO": "15", "PASTAZA": "16", "PICHINCHA": "17",
    "TUNGURAHUA": "18", "ZAMORACHINCHIPE": "19", "GALAPAGOS": "20", "SUCUMBIOS": "21", "ORELLANA": "22",
    "SANTODOMINGODELOSTSACHILAS": "23", "SANTAELENA": "24",
}
OTRAS_ENFERMEDADES = ("MALARIA", "PALUDISMO", "LEISHMANIASIS", "CHAGAS", "CHIKUNGUNYA", "ZIKA", "AMARILLA", "MAYARO")
CLASIFICACION = {"DSSA", "DCSA", "DG"}
NUMERO = re.compile(r"^(?:\d{1,3}(?:\.\d{3})+|\d+)(?:,\d+)?$")
PEGADO = re.compile(r"^(.*\.\.)(\d[\d.,]*)$")  #TSAC..1.040: nombre cortado y número sin espacio
ETIQUETA_SE = re.compile(r"^(?:SE\b.*|\d{1,2})$")
ANIO = re.compile(r"^20\d{2}$")
RANGO = re.compile(r"\d\s*(?:-|A|HASTA)\s*(?:LA\s+)?(?:SE\s*)?\d")
SEMANA_PORTADA = re.compile(r"SEMANA EPIDEMIOLOGICA\s*\(SE\)\s*(\d{1,2})\s*(?:,|/|$)", re.MULTILINE)
SEMANA_RANGO = re.compile(r"\bSE\)?\s*\d{1,2}\s*(?:-|A|HASTA)\s*(?:LA\s+)?(?:SE\s*)?(\d{1,2})\b")
SEMANA_SOLA = re.compile(r"\bSE\)?\s*(\d{1,2})\b")
CORTE = re.compile(r"(?:CORTE|ACTUALIZAD[OA])\D{0,80}?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")


class Cuarentena(Exception):
    """La gaceta no se puede leer con confianza: va a la cuarentena con este motivo"""


def simple(texto):
    """Mayúsculas y sin tildes"""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()


def clave(texto):
    return re.sub(r"[^A-Z]", "", simple(texto))


def provincia(nombre):
    """Código DPA de un nombre como lo escribe la gaceta, aunque venga cortado"""
    buscada = clave(nombre)
    if len(buscada) < 4:
        return None
    if buscada.startswith("ZONA"):  # «ZONA NO DELIMITADA» y variantes
        return "90"
    if buscada in PROVINCIAS:
        return PROVINCIAS[buscada]
    codigos = {codigo for completa, codigo in PROVINCIAS.items() if completa.startswith(buscada)}
    return codigos.pop() if len(codigos) == 1 else None


def es_continuacion(codigo, nombre):
    """TSACHILAS debajo de Santo Domingo, «HINCHIPE debajo de Zamora: el nombre siguió en la línea de abajo"""
    resto = clave(nombre)
    completa = next((k for k, v in PROVINCIAS.items() if v == codigo), "")
    return len(resto) >= 3 and completa.endswith(resto)


def numero(texto):
    """'Punto de miles y coma decimal"""
    if not NUMERO.match(texto):
        return None
    entero, _, decimales = texto.replace(".", "").partition(",")
    return float(f"{entero}.{decimales}") if decimales else int(entero)


def palabras_de(pagina):
    """Las palabras de una página con su posición, en mayúsculas y sin tildes"""
    palabras = []
    for w in pagina.extract_words():
        texto = simple(w["text"])
        pegado = PEGADO.match(texto)
        if pegado:  # se separa en dos; el ancho se reparte según los caracteres
            corte = w["x0"] + (w["x1"] - w["x0"]) * len(pegado.group(1)) / len(texto)
            palabras.append({**w, "texto": pegado.group(1), "x1": corte})
            palabras.append({**w, "texto": pegado.group(2), "x0": corte})
        else:
            palabras.append({**w, "texto": texto})
    return palabras


def lineas_de(palabras, tolerancia=3.0):
    """Agrupa por altura: las palabras cuyo borde superior difiere en menos de la tolerancia forman una línea."""
    lineas = []
    for w in sorted(palabras, key=lambda w: w["top"]):
        if lineas and w["top"] - lineas[-1]["top"] <= tolerancia:
            lineas[-1]["palabras"].append(w)
        else:
            lineas.append({"top": w["top"], "palabras": [w]})
    for linea in lineas:
        linea["palabras"].sort(key=lambda w: w["x0"])
        linea["texto"] = " ".join(w["texto"] for w in linea["palabras"])
    return lineas


def etiquetas(palabras, separacion=6.0):
    """Junta las palabras pegadas del encabezado"""
    grupos = []
    for w in palabras:
        if grupos and w["x0"] - grupos[-1]["x1"] <= separacion:
            grupos[-1]["texto"] += " " + w["texto"]
            grupos[-1]["x1"] = max(grupos[-1]["x1"], w["x1"])
        else:
            grupos.append({"texto": w["texto"], "x0": w["x0"], "x1": w["x1"]})
    return grupos


def semana_en(texto):
    """La semana de un título"""
    for patron in (SEMANA_RANGO, SEMANA_SOLA):
        encontrado = patron.search(texto)
        if encontrado and 1 <= int(encontrado.group(1)) <= 53:
            return int(encontrado.group(1))
    return None


def corte_en(lineas):
    """La fecha de corte de la portada"""
    encontrado = CORTE.search("\n".join(l["texto"] for l in lineas))
    if not encontrado:
        return None
    dia, mes, anio = (int(g) for g in encontrado.groups())
    try:
        return dt.date(anio + 2000 if anio < 100 else anio, mes, dia)
    except ValueError:
        return None


def familia_de(cols, arriba):
    """Qué formato de tabla es, según sus columnas y las líneas de encabezado que tiene encima.

    Una tabla con columnas SE trae casos o tasas por 100 000 habitantes: eso lo deciden sus números (leer_gaceta)."""
    textos = [c["texto"] for c in cols]
    totales = sum(t.startswith("TOTAL") for t in textos)
    datos = textos[:len(textos) - totales]
    if totales == 1 and datos and all(ETIQUETA_SE.match(t) for t in datos):
        return "semanal"
    if totales == 2 and len(datos) == 12 and set(datos) <= CLASIFICACION:
        medidas = [w["texto"] for l in arriba for w in l["palabras"] if w["texto"] in ("TASA", "CASOS")]
        if medidas == ["TASA", "CASOS"] * 3:
            return "clasificacion"
    return None


def total_de_arriba(cols, lineas_arriba, top):
    """En 2026 el Total del encabezado queda una línea más arriba que las columnas SE: se suma como la última"""
    semanas = list(itertools.takewhile(lambda c: ETIQUETA_SE.match(c["texto"]), cols))
    if not semanas:
        return cols
    for linea in reversed(lineas_arriba):
        if top - linea["top"] > 20:
            break
        for w in linea["palabras"]:
            if w["texto"].startswith("TOTAL") and 0 <= w["x0"] - semanas[-1]["x1"] <= 80:
                return semanas + [{"texto": "TOTAL", "x0": w["x0"], "x1": w["x1"]}]
    return cols


def encabezados(palabras):
    """Los encabezados de tablas de dengue por provincia de una página, de arriba abajo"""
    lineas = lineas_de(palabras)
    titulo_pagina = " ".join(l["texto"] for l in lineas[:6])
    for i, linea in enumerate(lineas):
        ws = linea["palabras"]
        anclas = [k for k, w in enumerate(ws) if w["texto"] == "PROVINCIA"
                  or (w["texto"] == "PROV" and k + 1 < len(ws) and ws[k + 1]["texto"] == "DOMIC")]
        if not anclas:
            continue
        k = anclas[0]
        ancla = ws[k]
        resto = [w for w in ws[k + (2 if ancla["texto"] == "PROV" else 1):] if not ANIO.match(w["texto"])]
        cols = []
        for c in etiquetas(resto):
            if cols and cols[-1]["texto"].startswith("TOTAL") and not c["texto"].startswith("TOTAL"):
                break  # lo que sigue a la derecha del último Total ya no es de la tabla
            cols.append(c)
        if cols and not cols[-1]["texto"].startswith("TOTAL"):
            cols = total_de_arriba(cols, lineas[:i], linea["top"])
        if not cols or not cols[-1]["texto"].startswith("TOTAL"):
            continue
        # Los nombres empiezan donde empieza el encabezado, lo que quede más
        # a la izquierda, como el recuadro de Galápagos del mapa, no es de la tabla.
        x0 = min([w["x0"] for w in ws[:k] if ancla["x0"] - w["x0"] <= 30] + [ancla["x0"]]) - 4
        x1 = cols[-1]["x1"] + 12
        # Las filas empiezan debajo del borde inferior del encabezado. La línea puede empezar más arriba,
        # en un rótulo del mapa que quedó a la misma altura
        fondo = max(w["bottom"] for w in ws if x0 <= w["x0"] and w["x1"] <= x1)
        cerca = lambda l: [w for w in l["palabras"] if x0 - 40 <= w["x0"] and w["x1"] <= x1 + 40]
        arriba = [{**l, "palabras": cerca(l)} for l in lineas[:i] if linea["top"] - 45 <= l["top"]]
        leyenda = " ".join(w["texto"] for l in lineas[:i] if linea["top"] - 110 <= l["top"] for w in cerca(l))
        familia = familia_de(cols, arriba)
        de_dengue = "DENGUE" in leyenda or (
            "DENGUE" in titulo_pagina and not any(e in leyenda for e in OTRAS_ENFERMEDADES))
        if familia and de_dengue:
            yield {"familia": familia, "top": linea["top"], "fondo": fondo, "x0": x0, "x1": x1, "cols": cols,
                   "titulos": [l["texto"] for l in lineas[:i] if "DENGUE" in l["texto"]]}


def columna(w, cols):
    """La columna de un número: la que lo tiene debajo o, si no, la de centro más cercano"""
    centro = (w["x0"] + w["x1"]) / 2
    debajo = [j for j, c in enumerate(cols) if c["x0"] - 3 <= centro <= c["x1"] + 3]
    if len(debajo) == 1:
        return debajo[0]
    return min(range(len(cols)), key=lambda j: abs(centro - (cols[j]["x0"] + cols[j]["x1"]) / 2))


def sin_rotulos(recorte, desde):
    """Quita las letras mucho más pequeñas que las de la tabla, como el rótulo GALAPAGOS del mapa.

    En algunas gacetas de 2021 y 2022 ese rótulo cae sobre una fila y sus letras se intercalan con
    el nombre: MORONA GSAALANPTAGIAOSGO. El tamaño de referencia sale de las primeras filas"""
    tamanos = [c["size"] for c in recorte.chars if c["top"] < desde + 60 and c["text"].strip()]
    if not tamanos:
        return recorte
    minimo = 0.75 * statistics.median(tamanos)
    return recorte.filter(lambda o: o.get("object_type") != "char" or o["size"] >= minimo)


def filas_de(pagina, tabla):
    """Las filas de la tabla, con el nombre a la izquierda y cada número en su columna, y dónde terminó

    La página se recorta al ancho de la tabla, desde el borde inferior del encabezado, antes de leer
    las palabras: así los rótulos del mapa de al lado no se mezclan con las filas."""
    cols = tabla["cols"]
    x0, _, x1, y1 = pagina.bbox
    recorte = pagina.within_bbox((max(tabla["x0"], x0), tabla["fondo"] + 0.5, min(tabla["x1"], x1), y1))
    recorte = sin_rotulos(recorte, tabla["fondo"])
    filas, ultima, saltos, fin, pendiente = [], tabla["fondo"], [], "el final de la página", None
    for linea in lineas_de(palabras_de(recorte)):
        paso = linea["top"] - ultima
        if (len(saltos) >= 2 and paso > 3 * statistics.median(saltos)) or (not saltos and paso > 40):
            fin = f"un espacio grande después de «{filas[-1]['nombre'] if filas else 'PROVINCIA'}»"
            break
        if not filas and any(w["texto"] in ("PROVINCIA", "PROV") for w in linea["palabras"]):
            continue  # otra línea del mismo encabezado
        nombre = " ".join(w["texto"] for w in linea["palabras"]
                          if numero(w["texto"]) is None and w["x0"] < cols[0]["x0"])
        valores = {}
        for w in linea["palabras"]:
            v = numero(w["texto"])
            if v is None or w["x1"] < cols[0]["x0"] - 15:
                continue
            j = columna(w, cols)
            if j in valores:
                raise Cuarentena(f"dos números en la misma columna en la fila «{linea['texto'][:60]}»")
            valores[j] = v
        if clave(nombre).startswith("TOTAL"):
            filas.append({"nombre": nombre, "codigo": None, "valores": valores})
            fin = "la fila Total"
            break
        codigo = provincia(nombre) if nombre else None
        if not valores:
            if codigo:
                # Una provincia sin números: puede que sus números vengan unos puntos más abajo.
                pendiente = {"nombre": nombre, "codigo": codigo, "top": linea["top"]}
            elif nombre and filas and es_continuacion(filas[-1]["codigo"], nombre):
                pass  # el resto del nombre de la fila de arriba
            elif nombre and pendiente and provincia(f"{pendiente['nombre']} {nombre}"):
                unido = f"{pendiente['nombre']} {nombre}"
                pendiente = {"nombre": unido, "codigo": provincia(unido), "top": linea["top"]}
            elif nombre:
                fin = f"«{linea['texto'][:40]}»"
                break  # otro texto: la tabla terminó
            continue
        if not nombre and pendiente and linea["top"] - pendiente["top"] <= 6:
            nombre, codigo = pendiente["nombre"], pendiente["codigo"]  # el nombre quedó un poco más arriba
        elif nombre and not codigo and pendiente and provincia(f"{pendiente['nombre']} {nombre}"):
            nombre = f"{pendiente['nombre']} {nombre}"  # «SANTO DOMINGO DE LOS» arriba, «TSACHILAS» aquí
            codigo = provincia(nombre)
        if not nombre:
            raise Cuarentena(f"números sin provincia en la fila «{linea['texto'][:60]}»")
        if not codigo:
            fin = f"«{linea['texto'][:40]}»"
            break  # otro texto: la tabla terminó
        filas.append({"nombre": nombre, "codigo": codigo, "valores": valores})
        pendiente = None
        saltos.append(paso)
        ultima = linea["top"]
    return filas, fin


def suma(valores, indices, decimales=None):
    total = sum(valores.get(j, 0) for j in indices)
    return round(total, decimales) if decimales is not None else total


def interpretar(familia, cols, valores, semana):
    """Casos (o tasas) previos, de la semana y acumulados de una fila, según el formato de la tabla

    Una celda vacía es un cero: AZUAY 17 17 quiere decir 17 previos, ninguno en la semana, 17 en total"""
    n = len(cols)
    r = dict.fromkeys(["casos_previos", "casos_semana", "casos_acumulados",
                       "tasa_previa", "tasa_semana", "tasa_acumulada"])
    if familia == "semanal":
        # La última columna SE es la semana de la gaceta; las anteriores suman lo previo
        r["casos_acumulados"] = valores.get(n - 1)
        if n >= 3:
            r["casos_semana"] = valores.get(n - 2, 0)
            r["casos_previos"] = suma(valores, range(n - 2))
        elif semana == 1:  # la gaceta de la SE 1 trae una sola columna
            r["casos_semana"], r["casos_previos"] = r["casos_acumulados"], 0
    elif familia == "tasas":
        # Las columnas 'SE 1 - 45' son lo previo; las 'SE 46', la semana (una por clasificación)
        rangos = [j for j in range(n - 1) if RANGO.search(cols[j]["texto"])]
        r["tasa_previa"] = suma(valores, rangos, 2)
        r["tasa_semana"] = suma(valores, [j for j in range(n - 1) if j not in rangos], 2)
        r["tasa_acumulada"] = valores.get(n - 1)
    elif familia == "clasificacion":
        # Tasa y casos de SE 1 a N-1, tasa y casos de la SE N (DCSA, DG y DSSA cada una), tasa y casos totales.
        r["tasa_previa"], r["casos_previos"] = suma(valores, range(0, 3), 2), suma(valores, range(3, 6))
        r["tasa_semana"], r["casos_semana"] = suma(valores, range(6, 9), 2), suma(valores, range(9, 12))
        r["tasa_acumulada"], r["casos_acumulados"] = valores.get(12), valores.get(13)
    return r


def elegir_semana(portada, titulo, titulo_portada, columna_se, enlace):
    """La semana de la gaceta. Manda la portada nueva (2025-2026). Si no está, votan el título de la página
    de dengue, la portada vieja, la última columna SE de la tabla y el enlace de la página del MSP; en un
    empate gana la columna, que es parte de la tabla"""
    if portada:
        return portada
    votos = collections.Counter(s for s in (titulo, titulo_portada, columna_se, enlace) if s)
    if not votos:
        return None
    mas = max(votos.values())
    return next(s for s in (columna_se, titulo_portada, titulo, enlace) if s and votos[s] == mas)


def leer_gaceta(contenido, se_enlace=None):
    """Lee un PDF y devuelve su semana, su corte y las filas de la tabla, o levanta Cuarentena.

    se_enlace es la semana que decía la página del MSP: solo pesa cuando el PDF se contradice."""
    with pdfplumber.open(io.BytesIO(contenido)) as pdf:
        portada = titulo_portada = corte = None
        caracteres = 0
        for numero_pagina, pagina in enumerate(pdf.pages, start=1):
            palabras = palabras_de(pagina)
            caracteres += sum(len(w["texto"]) for w in palabras)
            if numero_pagina == 1:
                lineas = lineas_de(palabras)
                corte = corte_en(lineas)
                encontrada = SEMANA_PORTADA.search("\n".join(l["texto"] for l in lineas))
                portada = int(encontrada.group(1)) if encontrada else None
                titulo_portada = next((semana_en(l["texto"]) for l in lineas[:12]
                                       if ("ECUADOR" in l["texto"] or "ANO 20" in l["texto"])
                                       and semana_en(l["texto"])), None)
            for tabla in encabezados(palabras):
                filas, fin = filas_de(pagina, tabla)
                codigos = [f["codigo"] for f in filas if f["codigo"]]
                if not codigos:
                    continue
                repetidas = sorted({c for c in codigos if codigos.count(c) > 1})
                if repetidas:
                    raise Cuarentena(f"provincias repetidas en la tabla: {repetidas}")
                familia = tabla["familia"]
                numeros_tabla = [v for f in filas for v in f["valores"].values()]
                decimales = sum(isinstance(v, float) for v in numeros_tabla)
                if familia == "semanal" and decimales > len(numeros_tabla) / 2:
                    # Casi todo con decimales: son tasas por 100 000 habitantes (2025 y 2026). La palabra tasa
                    # cerca de la tabla no basta: en 2021 aparece en el texto de encima de tablas de casos
                    familia = "tasas"
                if familia == "semanal" and fin != "la fila Total":
                    raise Cuarentena(f"la tabla terminó antes de la fila Total, en {fin}")
                se = [c["texto"] for c in tabla["cols"] if ETIQUETA_SE.match(c["texto"])]
                numeros = re.findall(r"\d{1,2}", re.sub(r"/\s*\d{4}", "", se[-1])) if se else []
                columna_se = int(numeros[-1]) if numeros else None
                titulo = next((semana_en(t) for t in tabla["titulos"] if semana_en(t)), None)
                return {
                    "semana": elegir_semana(portada, titulo, titulo_portada, columna_se, se_enlace),
                    # Con dos o más columnas SE, la última es la semana: sirve para verificar la elegida
                    "semana_columna": columna_se if familia == "semanal" and len(se) >= 2 else None,
                    "corte": corte, "familia": familia, "pagina": numero_pagina,
                    "cols": tabla["cols"], "filas": filas,
                }
        if caracteres < 500:
            raise Cuarentena("el PDF no trae texto: es una imagen")
        raise Cuarentena("no encontré la tabla de dengue por provincia")


ESQUEMA = ("anio int, archivo string, se_archivo int, sha256 string, semana_epi int, semana_columna int, "
           "fecha_corte_pdf date, familia string, pagina int, fila int, provincia string, provincia_pdf string, "
           "es_total boolean, casos_previos long, casos_semana long, casos_acumulados long, "
           "tasa_previa double, tasa_semana double, tasa_acumulada double, motivo string")
COLUMNAS = [c.split()[0] for c in ESQUEMA.split(", ")]


def leer_pdf(pdf):
    """Una gaceta -> sus filas, o una sola fila con el motivo de la cuarentena"""
    base = {"anio": int(pdf.anio), "archivo": pdf.archivo, "sha256": pdf.sha256,
            "se_archivo": None if pd.isna(pdf.se_archivo) else int(pdf.se_archivo)}
    try:
        g = leer_gaceta(pdf.content, base["se_archivo"])
        filas = []
        for orden, fila in enumerate(g["filas"], start=1):
            valores = interpretar(g["familia"], g["cols"], fila["valores"], g["semana"])
            if any(isinstance(valores[c], float) for c in ("casos_previos", "casos_semana", "casos_acumulados")):
                raise Cuarentena(f"decimales en una columna de casos, en la fila «{fila['nombre']}»")
            filas.append({**base, **valores, "semana_epi": g["semana"], "semana_columna": g["semana_columna"],
                          "fecha_corte_pdf": g["corte"], "familia": g["familia"], "pagina": g["pagina"],
                          "fila": orden, "provincia": fila["codigo"], "provincia_pdf": fila["nombre"],
                          "es_total": fila["codigo"] is None})
        return filas
    except Cuarentena as motivo:
        return [{**base, "motivo": str(motivo)}]
    except Exception as error:  # un PDF dañado no debe detener el pipeline
        return [{**base, "motivo": f"error del parser: {type(error).__name__}: {error}"[:300]}]


def leer_lote(lotes):
    """Para mapInPandas: recibe lotes de PDF y devuelve, por cada uno, las filas que se leyeron"""
    for lote in lotes:
        filas = [fila for pdf in lote.itertuples(index=False) for fila in leer_pdf(pdf)]
        yield pd.DataFrame(filas, columns=COLUMNAS, dtype=object)


@dp.table(name="silver.gacetas_extraidas",
          comment="Lo que el parser leyó de cada gaceta: una fila por provincia y el total, "
                  "o una sola fila con el motivo si el PDF va a cuarentena.")
def gacetas_extraidas():
    return (
        spark.readStream.table("bronze.gacetas")
        .select("anio", "archivo", "se_archivo", "sha256", "content")
        .repartition(32)  # pocos PDF por tarea: cada lote cabe holgado en la memoria
        .mapInPandas(leer_lote, ESQUEMA)
        .withColumn("leido_en", F.current_timestamp())
    )


@dp.materialized_view(name="silver.gacetas_cuarentena", comment="Gacetas que el parser no pudo leer, con el motivo.")
def gacetas_cuarentena():
    return (
        spark.read.table("silver.gacetas_extraidas")
        .where("motivo IS NOT NULL")
        .select("anio", "se_archivo", "archivo", "motivo", "leido_en")
    )