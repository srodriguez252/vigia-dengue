# Junta en un ZIP los PDF que el parser dejó en cuarentena
import shutil
import zipfile

spark.sql("CREATE VOLUME IF NOT EXISTS vigia.bronze.exportes "
          "COMMENT 'Archivos para revisar fuera de Databricks'")

pdfs = spark.sql("""
    SELECT g.anio, g.archivo, g.content
    FROM vigia.bronze.gacetas AS g
    LEFT SEMI JOIN vigia.silver.gacetas_cuarentena AS c
      ON g.anio = c.anio AND g.archivo = c.archivo
""").collect()

# Un ZIP no se puede escribir directo en un Volume: se arma en el disco local y luego se copia.
local = "/tmp/gacetas_cuarentena.zip"
with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as z:
    for pdf in pdfs:
        z.writestr(f"anio={pdf.anio}/{pdf.archivo}", bytes(pdf.content))
destino = "/Volumes/vigia/bronze/exportes/gacetas_cuarentena.zip"
shutil.copyfile(local, destino)
print(f"{len(pdfs)} PDF en {destino}")