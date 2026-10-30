"""Descarga fuentes públicas a una carpeta local con la misma estructura del Volume de landing.

Cada archivo queda en <destino>/<fuente>/fecha_descarga=AAAA-MM-DD/<archivo>, por ejemplo:
    landing/nino/fecha_descarga=2026-10-22/wksst9120.for

Después, GitHub Actions (o a mano) sube la carpeta al Volume con:
    databricks fs cp -r landing dbfs:/Volumes/vigia/bronze/landing --overwrite
Al llegar el archivo nuevo, el Job vigia-semanal de Databricks corre el ETL pipeline.

Uso:
    python scripts/descargar_fuentes.py --fuente nino --destino landing

Solo usa la biblioteca estándar de Python: no hace falta instalar nada.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
import time
import urllib.request

AGENTE = "vigia-dengue/0.1 (proyecto de portafolio sin fines comerciales)"

# Línea de datos del índice Niño: empieza con una fecha como 04JAN1990
LINEA_NINO = re.compile(r"^\s*\d{2}[A-Z]{3}\d{4}\s")


def validar_nino(contenido: bytes) -> int:
    """Revisa que el archivo de NOAA tenga la serie semanal completa y devuelve cuántas semanas trae."""
    lineas = contenido.decode("ascii", errors="replace").splitlines()
    semanas = sum(1 for linea in lineas if LINEA_NINO.match(linea))
    if semanas < 1000:
        raise ValueError(f"Se esperaban más de 1000 semanas desde 1990 y llegaron {semanas}")
    return semanas


# Para sumar una fuente nueva, agrega una entrada aquí con su URL, el nombre del archivo y su validación
FUENTES = {
    "nino": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for",
        "archivo": "wksst9120.for",
        "validar": validar_nino,
    },
}


def bajar(url: str, intentos: int = 3, espera_s: float = 10.0) -> bytes:
    """Descarga una URL con reintentos y una espera que crece en cada intento."""
    for intento in range(1, intentos + 1):
        try:
            pedido = urllib.request.Request(url, headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(pedido, timeout=60) as respuesta:
                return respuesta.read()
        except OSError as error:  # incluye errores HTTP, de red y de tiempo de espera
            if intento == intentos:
                raise
            print(f"Intento {intento} falló ({error}); reintento en {espera_s * intento:.0f} s", file=sys.stderr)
            time.sleep(espera_s * intento)
    raise RuntimeError("inalcanzable")


def descargar(fuente: str, destino: pathlib.Path, hoy: dt.date) -> pathlib.Path:
    config = FUENTES[fuente]
    contenido = bajar(config["url"])
    detalle = config["validar"](contenido)

    carpeta = destino / fuente / f"fecha_descarga={hoy.isoformat()}"
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / config["archivo"]
    ruta.write_bytes(contenido)

    print(f"{fuente}: {ruta} ({len(contenido) / 1024:.0f} KB, validación: {detalle})")
    return ruta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fuente", choices=sorted(FUENTES), action="append", required=True,
                        help="Fuente a descargar; repite la opción para varias.")
    parser.add_argument("--destino", type=pathlib.Path, default=pathlib.Path("landing"),
                        help="Carpeta local de salida (por defecto: landing).")
    args = parser.parse_args()

    hoy = dt.date.today()
    for fuente in args.fuente:
        descargar(fuente, args.destino, hoy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
