import socket

import requests

fuentes = {
    "MSP · gacetas": "https://www.salud.gob.ec/",
    "INEC · egresos": "https://www.ecuadorencifras.gob.ec/camas-y-egresos-hospitalarios/",
    "Open-Meteo · clima": (
        "https://archive-api.open-meteo.com/v1/archive?latitude=-1.05&longitude=-80.45"
        "&start_date=2024-03-01&end_date=2024-03-02&daily=precipitation_sum&timezone=America%2FGuayaquil"
    ),
    "NOAA CPC · Niño": "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for",
    "PyPI · librerías": "https://pypi.org/simple/pdfplumber/",
}

resultados = []
for nombre, url in fuentes.items():
    host = url.split("/")[2]

    # 1) ¿El nombre de dominio resuelve? Si Free Edition bloquea el dominio, falla aquí
    try:
        socket.getaddrinfo(host, 443)
        dns = "resuelve"
    except OSError as error:
        dns = f"no resuelve ({type(error).__name__})"

    # 2) ¿Responde el servidor? stream=True evita descargar el archivo completo
    try:
        with requests.get(url, timeout=20, stream=True) as respuesta:
            http = str(respuesta.status_code)
    except requests.RequestException as error:
        http = type(error).__name__

    resultados.append((nombre, host, dns, http))

display(spark.createDataFrame(resultados, "fuente string, host string, dns string, http string"))
