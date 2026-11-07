# Silver - gacetas del MSP: la tabla de dengue por provincia de cada PDF
# El parser corre dentro de mapInPandas; lo que no puede leer va a cuarentena
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

# Códigos DPA del INEC. Las claves van como las deja clave(): sin espacios, tildes ni puntos.
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
MILES_CON_COMA = re.compile(r"^\d{1,3}(?:,\d{3})+$")  #la coma como separador de miles
PEGADO = re.compile(r"^(.*\.\.)(\d[\d.,]*)$")  #nombre cortado y número sin espacio
ETIQUETA_SE = re.compile(r"^(?:SE(?:\b|(?=\d)).*|\d{1,2})$")
CONECTORES = {"-", "A", "AL", "HASTA"} 
ANIO = re.compile(r"^20\d{2}$")
RANGO = re.compile(r"\d\s*(?:-|A|HASTA)\s*(?:LA\s+)?(?:SE\s*)?\d")
SEMANA_PORTADA = re.compile(r"SEMANA EPIDEMIOLOGICA\s*\(SE\)\s*(\d{1,2})\s*(?:,|/|$)", re.MULTILINE)
SEMANA_RANGO = re.compile(r"\bSE\)?\s*\d{1,2}\s*(?:-|A|HASTA)\s*(?:LA\s+)?(?:SE\s*)?(\d{1,2})\b")
SEMANA_SOLA = re.compile(r"\bSE\)?\s*(\d{1,2})\b")
CORTE = re.compile(r"(?:CORTE|ACTUALIZAD[OA])\D{0,80}?(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
PERIODOS = ("previo", "semana", "total")
CAMPOS = {"casos": ("casos_previos", "casos_semana", "casos_acumulados"),
          "tasa": ("tasa_previa", "tasa_semana", "tasa_acumulada")}


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
    if buscada.startswith("ZONA"):  #'ZONA NO DELIMITADA' y variantes
        return "90"
    if buscada in PROVINCIAS:
        return PROVINCIAS[buscada]
    codigos = {codigo for completa, codigo in PROVINCIAS.items() if completa.startswith(buscada)}
    return codigos.pop() if len(codigos) == 1 else None


def es_continuacion(codigo, nombre):
    """TSACHILAS debajo de Santo Domingo, CHINCHIPE debajo de Zamora: el nombre siguió en la línea de abajo"""
    resto = clave(nombre)
    completa = next((k for k, v in PROVINCIAS.items() if v == codigo), "")
    return len(resto) >= 3 and completa.endswith(resto)


def numero(texto):
    """'punto de miles y coma decimal"""
    if not NUMERO.match(texto):
        return None
    entero, _, decimales = texto.replace(".", "").partition(",")
    return float(f"{entero}.{decimales}") if decimales else int(entero)


def palabras_de(pagina, tolerancia=3.0):
    """Las palabras de una página con su posición, en mayúsculas y sin tildes.

    tolerancia es cuánto pueden diferir en altura las letras que pdfplumber pone en una misma línea. Con los 3
    puntos de siempre, los rótulos de un mapa a media altura pueden unir dos filas de la tabla, y sus letras se
    intercalan: P R O V I N CIA. Por eso leer_pdf reintenta con 1 punto"""
    palabras = []
    for w in pagina.extract_words(y_tolerance=tolerancia):
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
    """Agrupa por altura: las palabras cuyo borde superior difiere en menos de la tolerancia forman una línea"""
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
    """Junta las palabras pegadas del encabezado

    Una palabra SE o TOTAL abre otra etiqueta aunque esté pegada a la anterior (SE 01-25 SE - 26, 2025),
    salvo que la anterior termine en un conector de rango (SE 1 A SE 52)."""
    grupos = []
    for w in palabras:
        texto = w["texto"]
        abre = texto.startswith("TOTAL") or (texto.startswith("SE") and ETIQUETA_SE.match(texto))
        pegada = grupos and w["x0"] - grupos[-1]["x1"] <= separacion
        anterior = grupos[-1]["texto"].split() if grupos else []  # puede venir vacía: «©» queda en ""
        tras_conector = bool(anterior) and anterior[-1] in CONECTORES
        if pegada and (not abre or tras_conector):
            grupos[-1]["texto"] += " " + texto
            grupos[-1]["x1"] = max(grupos[-1]["x1"], w["x1"])
        else:
            grupos.append({"texto": texto, "x0": w["x0"], "x1": w["x1"]})
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


def grupos_de_clasificacion(textos):
    """Cuántos grupos DCSA | DG | DSSA | Total seguidos tiene el encabezado, o 0 si no tiene esa forma"""
    if not textos or len(textos) % 4:
        return 0
    for i in range(0, len(textos), 4):
        if set(textos[i:i + 3]) != CLASIFICACION or not textos[i + 3].startswith("TOTAL"):
            return 0
    return len(textos) // 4


def por_rotulo(cols, rotulos):
    """A cada columna, la medida y el periodo del rótulo TASA» o «CASOS que tiene encima.

    Los rótulos van en tres pares: de SE 1 a N-1, de la SE N y del total. Con 12 columnas de datos hay tres
    debajo de cada rótulo. Si una clasificación no tuvo casos en la semana, su columna falta (SE 34 de 2026) y
    cada columna toma el rótulo más cercano. Devuelve False si eso no deja los dos Total en el último par."""
    datos = [c for c in cols if not c["texto"].startswith("TOTAL")]
    if len(datos) == 12:
        indices = [j // 3 for j in range(12)] + [4, 5]
    else:
        centro = lambda o: (o["x0"] + o["x1"]) / 2
        indices = [min(range(6), key=lambda k: abs(centro(c) - centro(rotulos[k]))) for c in cols]
    for c, i in zip(cols, indices):
        c["medida"], c["periodo"] = ("tasa", "casos")[i % 2], PERIODOS[i // 2]
    return all((c["periodo"] == "total") == c["texto"].startswith("TOTAL") for c in cols)


def familia_de(cols, arriba):
    """Qué formato de tabla es, según sus columnas y las líneas de encabezado que tiene encima.

    Cuando el encabezado no dice si una columna trae casos o tasas por 100 000 habitantes, lo deciden sus
    números (leer_tabla)"""
    textos = [c["texto"] for c in cols]
    if grupos_de_clasificacion(textos):
        # DCSA | DG | DSSA | Total, a veces repetido con el año anterior a la izquierda (2017, 2026):
        # solo el acumulado del año
        return "acumulada"
    totales = sum(t.startswith("TOTAL") for t in textos)
    datos = textos[:len(textos) - totales]
    if not datos or any(t.startswith("TOTAL") for t in datos):
        return None
    if all(ETIQUETA_SE.match(t) for t in datos):
        if totales == 1:
            # Provincia | SE 1-41 | SE 42 | Total, sola o repetida por clasificación (DCSA, DG, DSSA)
            return "semanal"
        if totales == 2:
            # 2025: casos y tasas de SE 1-20 y de la SE 21, y un total de cada uno
            for c in cols:
                c["periodo"] = ("total" if c["texto"].startswith("TOTAL")
                                else "previo" if RANGO.search(c["texto"]) else "semana")
            return "mixta"
    if totales == 2 and set(datos) <= CLASIFICACION:
        # 2026: tasa y casos por clasificación (DCSA, DG, DSSA) de SE 1-41, de la SE 42 y del total
        rotulos = sorted((w for l in arriba for w in l["palabras"] if w["texto"] in ("TASA", "CASOS")),
                         key=lambda w: w["x0"])
        if [w["texto"] for w in rotulos] == ["TASA", "CASOS"] * 3 and por_rotulo(cols, rotulos):
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


def columnas_de(palabras):
    """Las columnas de un encabezado, de izquierda a derecha, hasta el último Total"""
    cols = []
    for c in etiquetas(palabras):
        if cols and cols[-1]["texto"].startswith("TOTAL"):
            # Lo que sigue a la derecha del último Total ya no es de la tabla, salvo otro Total u otro grupo
            # DCSA | DG | DSSA | Total» (el año anterior y el actual, en la SE 1 de 2026).
            otro_grupo = c["texto"] in CLASIFICACION and grupos_de_clasificacion([x["texto"] for x in cols])
            if not (c["texto"].startswith("TOTAL") or otro_grupo):
                break
        cols.append(c)
    return cols


def encabezados(palabras):
    """Los encabezados de tablas de dengue por provincia de una página, de arriba abajo

    El ancla es PROVINCIA (o PROV DOMIC). Un encabezado sin ella (SE 6 de 2023) vale si sus columnas son
    semanas y un Total y la línea de abajo empieza con una provincia; queda marcado sin_ancla"""
    lineas = lineas_de(palabras)
    titulo_pagina = " ".join(l["texto"] for l in lineas[:6])
    for i, linea in enumerate(lineas):
        ws = linea["palabras"]
        anclas = [k for k, w in enumerate(ws) if w["texto"] == "PROVINCIA"
                  or (w["texto"] == "PROV" and k + 1 < len(ws) and ws[k + 1]["texto"].startswith("DOMIC"))]
        if anclas:
            k = anclas[0]
            ancla = ws[k]
            # En 2025 el año va suelto en el encabezado, sobre las columnas SE (PROVINCIA 2025 SE 01-12 SE 13
            # TOTAL): no es una columna.
            resto = [w for w in ws[k + (2 if ancla["texto"] == "PROV" else 1):] if not ANIO.match(w["texto"])]
            cols = columnas_de(resto)
            if cols and not cols[-1]["texto"].startswith("TOTAL"):
                cols = total_de_arriba(cols, lineas[:i], linea["top"])
            # Los nombres empiezan donde empieza el encabezado (Inf. Provincia incluido); lo que quede más
            # a la izquierda, como el recuadro de Galápagos del mapa, no es de la tabla
            x0 = min([w["x0"] for w in ws[:k] if ancla["x0"] - w["x0"] <= 30] + [ancla["x0"]]) - 4
        else:
            cols = columnas_de([w for w in ws if not ANIO.match(w["texto"])])
            debajo = lineas[i + 1]["palabras"] if i + 1 < len(lineas) else []
            nombre = " ".join(w["texto"] for w in debajo if cols and w["x1"] < cols[0]["x0"])
            if not (len(cols) >= 3 and all(ETIQUETA_SE.match(c["texto"]) for c in cols[:-1]) and provincia(nombre)):
                continue
            x0 = debajo[0]["x0"] - 4
        if not cols or not cols[-1]["texto"].startswith("TOTAL"):
            continue
        x1 = cols[-1]["x1"] + 12
        # Las filas empiezan debajo del borde inferior del encabezado. La línea puede empezar más arriba,
        # en un rótulo del mapa que quedó a la misma altura.
        fondo = max(w["bottom"] for w in ws if x0 <= w["x0"] and w["x1"] <= x1)
        cerca = lambda l: [w for w in l["palabras"] if x0 - 40 <= w["x0"] and w["x1"] <= x1 + 40]
        arriba = [{**l, "palabras": cerca(l)} for l in lineas[:i] if linea["top"] - 45 <= l["top"]]
        leyenda = " ".join(w["texto"] for l in lineas[:i] if linea["top"] - 110 <= l["top"] for w in cerca(l))
        familia = familia_de(cols, arriba)
        de_dengue = "DENGUE" in leyenda or (
            "DENGUE" in titulo_pagina and not any(e in leyenda for e in OTRAS_ENFERMEDADES))
        if familia and de_dengue:
            yield {"familia": familia, "top": linea["top"], "fondo": fondo, "x0": x0, "x1": x1, "cols": cols,
                   "sin_ancla": not anclas, "titulos": [l["texto"] for l in lineas[:i] if "DENGUE" in l["texto"]]}


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


def filas_de(pagina, tabla, tolerancia=3.0):
    """Las filas de la tabla, con el nombre a la izquierda y cada número en su columna, y dónde terminó.

    La página se recorta al ancho de la tabla, desde el borde inferior del encabezado, antes de leer
    las palabras: así los rótulos del mapa de al lado no se mezclan con las filas. Cada fila guarda
    también el texto de sus números, por si hay que leerlos con otro separador de miles"""
    cols = tabla["cols"]
    x0, _, x1, y1 = pagina.bbox
    recorte = pagina.within_bbox((max(tabla["x0"], x0), tabla["fondo"] + 0.5, min(tabla["x1"], x1), y1))
    recorte = sin_rotulos(recorte, tabla["fondo"])
    filas, ultima, saltos, fin, pendiente = [], tabla["fondo"], [], "el final de la página", None
    for linea in lineas_de(palabras_de(recorte, tolerancia)):
        paso = linea["top"] - ultima
        if (len(saltos) >= 2 and paso > 3 * statistics.median(saltos)) or (not saltos and paso > 40):
            fin = f"un espacio grande después de «{filas[-1]['nombre'] if filas else 'PROVINCIA'}»"
            break
        if not filas and any(w["texto"] in ("PROVINCIA", "PROV") for w in linea["palabras"]):
            continue  # otra línea del mismo encabezado
        nombre = " ".join(w["texto"] for w in linea["palabras"]
                          if numero(w["texto"]) is None and w["x0"] < cols[0]["x0"])
        valores, textos = {}, {}
        for w in linea["palabras"]:
            v = numero(w["texto"])
            if v is None or w["x1"] < cols[0]["x0"] - 15:
                continue
            j = columna(w, cols)
            if j in valores:
                raise Cuarentena(f"dos números en la misma columna en la fila «{linea['texto'][:60]}»")
            valores[j], textos[j] = v, w["texto"]
        if clave(nombre).startswith("TOTAL"):
            filas.append({"nombre": nombre, "codigo": None, "valores": valores, "textos": textos})
            fin = "la fila Total"
            break
        codigo = provincia(nombre) if nombre else None
        if not valores:
            if codigo:
                # Una provincia sin números: puede que sus números vengan unos puntos más abajo
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
            nombre = f"{pendiente['nombre']} {nombre}"  # SANTO DOMINGO DE LOS arriba, TSACHILAS aquí
            codigo = provincia(nombre)
        if not nombre:
            raise Cuarentena(f"números sin provincia en la fila «{linea['texto'][:60]}»")
        if not codigo:
            fin = f"«{linea['texto'][:40]}»"
            break  # otro texto: la tabla terminó
        filas.append({"nombre": nombre, "codigo": codigo, "valores": valores, "textos": textos})
        pendiente = None
        saltos.append(paso)
        ultima = linea["top"]
    return filas, fin


def suma(valores, indices, decimales=None):
    total = sum(valores.get(j, 0) for j in indices)
    return round(total, decimales) if decimales is not None else total


def columna_de_la_semana(cols, semana):
    """La columna SE de la semana de la gaceta: la que lleva su número o, si ninguna, la última.

    En 2017 algunas tablas traen una semana más, con los casos que ya llegaron de la siguiente
    («SE 01-07 | SE 08 | SE 09 | SE 10 | Total» en la gaceta de la SE 9): esos solo cuentan en el total"""
    ultima = len(cols) - 2
    for j in range(ultima, -1, -1):
        texto = re.sub(r"/\s*\d{2,4}\.*", "", cols[j]["texto"])
        numeros = re.findall(r"\d{1,2}", texto)
        if numeros and not RANGO.search(texto) and int(numeros[-1]) == semana:
            return j
    return ultima


def interpretar(familia, cols, valores, semana):
    """Casos (o tasas) previos, de la semana y acumulados de una fila, según el formato de la tabla

    Una celda vacía es un cero: AZUAY 17 17 quiere decir 17 previos, ninguno en la semana, 17 en total"""
    n = len(cols)
    r = dict.fromkeys(["casos_previos", "casos_semana", "casos_acumulados",
                       "tasa_previa", "tasa_semana", "tasa_acumulada"])
    if familia == "semanal":
        # La columna de la semana de la gaceta (casi siempre la última SE); las anteriores suman lo previo
        r["casos_acumulados"] = valores.get(n - 1)
        if n >= 3:
            k = columna_de_la_semana(cols, semana)
            r["casos_semana"] = valores.get(k, 0)
            r["casos_previos"] = suma(valores, range(k))
        elif semana == 1:  # la gaceta de la SE 1 trae una sola columna
            r["casos_semana"], r["casos_previos"] = r["casos_acumulados"], 0
    elif familia == "tasas":
        # Las columnas 'SE 1 - 45' son lo previo; las 'SE 46', la semana (una por clasificación)
        rangos = [j for j in range(n - 1) if RANGO.search(cols[j]["texto"])]
        r["tasa_previa"] = suma(valores, rangos, 2)
        r["tasa_semana"] = suma(valores, [j for j in range(n - 1) if j not in rangos], 2)
        r["tasa_acumulada"] = valores.get(n - 1)
    elif familia == "acumulada":
        # Solo el acumulado del año, en la última columna Total; con la gaceta de la SE 1, todo es de la semana
        previo, de_la_semana, acumulado = CAMPOS[cols[-1]["medida"]]
        r[acumulado] = valores.get(n - 1)
        if semana == 1 and r[acumulado] is not None:
            r[de_la_semana], r[previo] = r[acumulado], 0
    else:
        # clasificacion y mixta: cada columna ya sabe su medida (casos o tasa) y su periodo (previo, semana, total)
        for medida, campos in CAMPOS.items():
            for periodo, campo in zip(PERIODOS, campos):
                js = [j for j, c in enumerate(cols) if c.get("medida") == medida and c["periodo"] == periodo]
                if js and periodo == "total":
                    r[campo] = valores.get(js[-1])
                elif js:
                    r[campo] = suma(valores, js, 2 if medida == "tasa" else None)
    return r


def coma_de_miles(filas):
    """¿La coma separa miles en esta tabla? (3,098 en la SE 17 de 2025.) Solo si ningún número lleva punto,
    todos los que llevan coma tienen grupos de tres cifras y, leídos así, las provincias suman la fila Total."""
    textos = [t for f in filas for t in f["textos"].values()]
    con_coma = [t for t in textos if "," in t]
    if not con_coma or any("." in t for t in textos) or not all(MILES_CON_COMA.match(t) for t in con_coma):
        return False
    total = next((f for f in filas if f["codigo"] is None and f["textos"]), None)
    if total is None:
        return False
    j = max(total["textos"])
    entero = lambda t: int(t.replace(",", ""))
    return sum(entero(f["textos"][j]) for f in filas if f["codigo"] and j in f["textos"]) == entero(total["textos"][j])


def elegir_semana(portada, titulo, titulo_portada, columna_se, enlace):
    """La semana de la gaceta. Manda la portada nueva (2025-2026). Si no está, votan el título de la página
    de dengue, la portada vieja, la última columna SE de la tabla y el enlace de la página del MSP; en un
    empate gana la columna, que es parte de la tabla."""
    if portada:
        return portada
    votos = collections.Counter(s for s in (titulo, titulo_portada, columna_se, enlace) if s)
    if not votos:
        return None
    mas = max(votos.values())
    return next(s for s in (columna_se, titulo_portada, titulo, enlace) if s and votos[s] == mas)


def leer_gaceta(contenido, se_enlace=None, tolerancia=3.0):
    """Lee un PDF y devuelve su semana, su corte y las filas de la tabla, o levanta Cuarentena.

    se_enlace es la semana que decía la página del MSP: solo pesa cuando el PDF se contradice."""
    with pdfplumber.open(io.BytesIO(contenido)) as pdf:
        portada = titulo_portada = corte = reserva = None
        caracteres = 0
        for numero_pagina, pagina in enumerate(pdf.pages, start=1):
            palabras = palabras_de(pagina, tolerancia)
            caracteres += sum(len(w["texto"]) for w in palabras)
            if numero_pagina == 1:
                lineas = lineas_de(palabras)
                corte = corte_en(lineas)
                encontrada = SEMANA_PORTADA.search("\n".join(l["texto"] for l in lineas))
                portada = int(encontrada.group(1)) if encontrada else None
                titulo_portada = next((semana_en(l["texto"]) for l in lineas[:12]
                                       if ("ECUADOR" in l["texto"] or "ANO 20" in l["texto"])
                                       and semana_en(l["texto"])), None)
            contexto = {"pagina": numero_pagina, "corte": corte, "tolerancia": tolerancia,
                        "semanas": (portada, titulo_portada, se_enlace)}
            for tabla in encabezados(palabras):
                if tabla["familia"] == "acumulada" or tabla["sin_ancla"]:
                    # Una tabla con solo el acumulado del año, o sin PROVINCIA en el encabezado, queda de
                    # reserva: sirve si el PDF no trae otra (2017, la SE 1 y 2 de 2026, la SE 6 de 2023).
                    if reserva is None:
                        try:
                            reserva = leer_tabla(pagina, tabla, contexto)
                        except Exception:  # una reserva que no se deja leer no cambia nada
                            pass
                    continue
                g = leer_tabla(pagina, tabla, contexto)
                if g:
                    return g
        if reserva:
            return reserva
        if caracteres < 500:
            raise Cuarentena("el PDF no trae texto: es una imagen")
        raise Cuarentena("no encontré la tabla de dengue por provincia")


def leer_tabla(pagina, tabla, contexto):
    """Las filas de una tabla y su semana, o None si no trae provincias; Cuarentena si no se puede leer"""
    filas, fin = filas_de(pagina, tabla, contexto["tolerancia"])
    codigos = [f["codigo"] for f in filas if f["codigo"]]
    if not codigos:
        return None
    repetidas = sorted({c for c in codigos if codigos.count(c) > 1})
    if repetidas:
        raise Cuarentena(f"provincias repetidas en la tabla: {repetidas}")
    if coma_de_miles(filas):
        for f in filas:
            f["valores"] = {j: int(t.replace(",", "")) for j, t in f["textos"].items()}
    familia, cols = tabla["familia"], tabla["cols"]
    numeros_tabla = [v for f in filas for v in f["valores"].values()]
    con_decimales = sum(isinstance(v, float) for v in numeros_tabla) > len(numeros_tabla) / 2
    if familia == "semanal" and con_decimales:
        # Casi todo con decimales: son tasas por 100 000 habitantes (2025 y 2026). La palabra tasa
        # cerca de la tabla no basta: en 2021 aparece en el texto de encima de tablas de casos.
        familia = "tasas"
    elif familia == "acumulada":
        for c in cols:
            c["medida"] = "tasa" if con_decimales else "casos"
    elif familia == "mixta":
        # Cada columna es de casos o de tasas según sus propios números.
        for j, c in enumerate(cols):
            columna_j = [f["valores"][j] for f in filas if j in f["valores"]]
            c["medida"] = "tasa" if sum(isinstance(v, float) for v in columna_j) > len(columna_j) / 2 else "casos"
    con_casos = familia in ("semanal", "mixta") or (familia == "acumulada" and not con_decimales)
    if con_casos and fin != "la fila Total":
        raise Cuarentena(f"la tabla terminó antes de la fila Total, en {fin}")
    se = [c["texto"] for c in cols if ETIQUETA_SE.match(c["texto"])]
    numeros = re.findall(r"\d{1,2}", re.sub(r"/\s*\d{4}", "", se[-1])) if se else []
    columna_se = int(numeros[-1]) if numeros else None
    titulo = next((semana_en(t) for t in tabla["titulos"] if semana_en(t)), None)
    portada, titulo_portada, se_enlace = contexto["semanas"]
    return {
        "semana": elegir_semana(portada, titulo, titulo_portada, columna_se, se_enlace),
        # Con dos o más columnas SE, la última es la semana: sirve para verificar la elegida.
        "semana_columna": columna_se if familia == "semanal" and len(se) >= 2 else None,
        "corte": contexto["corte"], "familia": familia, "pagina": contexto["pagina"],
        "cols": cols, "filas": filas,
    }


ESQUEMA = ("anio int, archivo string, se_archivo int, sha256 string, semana_epi int, semana_columna int, "
           "fecha_corte_pdf date, familia string, pagina int, fila int, provincia string, provincia_pdf string, "
           "es_total boolean, casos_previos long, casos_semana long, casos_acumulados long, "
           "tasa_previa double, tasa_semana double, tasa_acumulada double, motivo string")
COLUMNAS = [c.split()[0] for c in ESQUEMA.split(", ")]


def filas_de_gaceta(contenido, se_enlace, tolerancia):
    """Las filas de una gaceta, listas para la tabla, o Cuarentena."""
    g = leer_gaceta(contenido, se_enlace, tolerancia)
    filas = []
    for orden, fila in enumerate(g["filas"], start=1):
        valores = interpretar(g["familia"], g["cols"], fila["valores"], g["semana"])
        if any(isinstance(valores[c], float) for c in CAMPOS["casos"]):
            raise Cuarentena(f"decimales en una columna de casos, en la fila «{fila['nombre']}»")
        filas.append({**valores, "semana_epi": g["semana"], "semana_columna": g["semana_columna"],
                      "fecha_corte_pdf": g["corte"], "familia": g["familia"], "pagina": g["pagina"],
                      "fila": orden, "provincia": fila["codigo"], "provincia_pdf": fila["nombre"],
                      "es_total": fila["codigo"] is None})
    return filas


def leer_pdf(pdf):
    """Una gaceta -> sus filas, o una sola fila con el motivo de la cuarentena.

    Si la primera lectura va a cuarentena, se intenta otra con 1 punto de tolerancia vertical (palabras_de).
    Una gaceta que ya se leía bien no cambia: el segundo intento solo corre cuando el primero falla"""
    base = {"anio": int(pdf.anio), "archivo": pdf.archivo, "sha256": pdf.sha256,
            "se_archivo": None if pd.isna(pdf.se_archivo) else int(pdf.se_archivo)}
    try:
        try:
            filas = filas_de_gaceta(pdf.content, base["se_archivo"], 3.0)
        except Cuarentena as primera:
            if "imagen" in str(primera):
                raise
            try:
                filas = filas_de_gaceta(pdf.content, base["se_archivo"], 1.0)
            except Exception:  # si el segundo intento tampoco sale, queda el motivo del primero
                raise primera from None
        return [{**base, **fila} for fila in filas]
    except Cuarentena as motivo:
        return [{**base, "motivo": str(motivo)}]
    except Exception as error:  # un PDF dañado no detiene el pipeline
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