"""Genera cantones.csv: los 221 cantones del INEC con su cabecera y un punto para pedir el clima

Fuentes:
- Límites de cantones y parroquias del CONALI (Organización Territorial del Estado, versión del 17 jul 2026,
  CC BY), tal como los publica el paquete geoecu en GitHub
- Lugares poblados de GeoNames (CC BY), tal como vienen en el paquete geonamescache (cities500)

El punto de cada cantón es el lugar de GeoNames con más población dentro de su cabecera cantonal; si no
hay ninguno, el centroide de la cabecera (o un punto interior, si el centroide cae fuera).

Uso: pip install geopandas pyogrio geonamescache requests
     python generar_cantones.py [carpeta_de_salida]
"""
import json
import pathlib
import sys
import unicodedata

import geopandas as gpd
import requests

GPKG_URL = "https://github.com/f-slg/geoecu/releases/download/datos-v0.3.0/geoecu_dpa_alta.gpkg"
GPKG_MD5 = "59ddf5f007f4e6a73e70069a958bc08a"
# Cantones creados después del Censo 2022: el INEC los cuenta dentro del cantón del que salieron
NUEVOS = {"0809": "0802",  # Borbón, de Eloy Alfaro (Esmeraldas)
          "1413": "1401"}  # Sevilla Don Bosco, de Morona (Morona Santiago)


def bajar(url, destino):
    if not destino.exists():
        respuesta = requests.get(url, timeout=300)
        respuesta.raise_for_status()
        destino.write_bytes(respuesta.content)
    return destino


def sin_tildes(texto):
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().upper()


def lugares_geonames():
    import geonamescache  # trae los lugares de GeoNames con 500 habitantes o más
    ruta = pathlib.Path(geonamescache.__file__).parent / "data" / "cities500.json"
    lugares = json.loads(ruta.read_text(encoding="utf-8"))
    lugares = lugares.values() if isinstance(lugares, dict) else lugares
    ecuador = [l for l in lugares if l["countrycode"] == "EC"]
    return gpd.GeoDataFrame(
        {"lugar": [l["name"] for l in ecuador], "habitantes": [l["population"] for l in ecuador]},
        geometry=gpd.points_from_xy([l["longitude"] for l in ecuador], [l["latitude"] for l in ecuador]),
        crs="EPSG:4326")


def main(salida):
    import hashlib
    gpkg = bajar(GPKG_URL, salida / "geoecu_dpa_alta.gpkg")
    assert hashlib.md5(gpkg.read_bytes()).hexdigest() == GPKG_MD5, "el GeoPackage no es el esperado"
    parroquias = gpd.read_file(gpkg, layer="parroquias")
    cabeceras = parroquias[(parroquias["tipo"] == "cabecera cantonal")
                           & ~parroquias["dpa_canton"].isin(NUEVOS)].copy()
    assert len(cabeceras) == 221, len(cabeceras)

    # Lugares de GeoNames dentro de cada cabecera; gana el más poblado.
    lugares = lugares_geonames()
    dentro = gpd.sjoin(lugares, cabeceras[["dpa_canton", "geometry"]], predicate="within")
    elegido = (dentro.sort_values(["dpa_canton", "habitantes"], ascending=[True, False])
               .drop_duplicates("dpa_canton").set_index("dpa_canton"))

    filas = []
    for c in cabeceras.itertuples():
        if c.dpa_canton in elegido.index:
            punto, fuente, lugar = elegido.loc[c.dpa_canton, "geometry"], "geonames", elegido.loc[c.dpa_canton, "lugar"]
        else:
            centroide = c.geometry.centroid
            punto = centroide if c.geometry.contains(centroide) else c.geometry.representative_point()
            fuente, lugar = "centroide", None
        filas.append({"dpa_canton": c.dpa_canton, "canton": c.canton, "dpa_provin": c.dpa_provin,
                      "provincia": c.provincia, "cabecera": c.parroquia,
                      "latitud": round(punto.y, 5), "longitud": round(punto.x, 5),
                      "fuente_punto": fuente, "lugar_geonames": lugar})

    import pandas as pd
    tabla = pd.DataFrame(filas).sort_values("dpa_canton")
    tabla.to_csv(salida / "cantones.csv", index=False, encoding="utf-8")
    print(f"{len(tabla)} cantones; {(tabla.fuente_punto == 'geonames').sum()} con lugar de GeoNames")
    return tabla


if __name__ == "__main__":
    carpeta = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    carpeta.mkdir(parents=True, exist_ok=True)
    main(carpeta)