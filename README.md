# Vigía Dengue

Dengue early-warning system for Ecuador, built on Databricks Free Edition. It will forecast confirmed dengue cases by province one to four weeks ahead and issue a green, yellow or red alert, using public data from Ecuador's Ministry of Health (MSP), INEC, Open-Meteo and NOAA.

> **Resumen en español.** Sistema de alerta temprana de dengue para Ecuador sobre Databricks Free Edition. Pronosticará los casos confirmados por provincia a 1–4 semanas y emitirá una alerta verde, amarilla o roja con datos públicos del MSP, el INEC, Open-Meteo y NOAA.

**Status:** Phase 0, foundations. Every week GitHub Actions downloads NOAA's Niño SST index and lands it in a Unity Catalog volume; a file-arrival trigger starts a Databricks job that runs the ETL pipeline, which builds a bronze streaming table and a silver materialized view.

## How it fits together

```text
GitHub Actions (Tuesdays)             Databricks Free Edition
─────────────────────────             ──────────────────────────────────────────────────
download NOAA Niño index   ──►  /Volumes/vigia/bronze/landing/nino/fecha_descarga=…/
                                  │ new file (file-arrival trigger)
                                  ▼
                                job vigia-semanal ──► ETL pipeline vigia-etl
                                                        ├─ bronze.nino_crudo   (streaming table, Auto Loader)
                                                        └─ silver.nino_oisst   (materialized view, expectations)
```

Downloads run on GitHub because Free Edition only allows outbound traffic to a short list of trusted domains.

## Repository layout

| Path | What it holds |
| --- | --- |
| `vigia-etl/transformations/` | ETL pipeline source, created from the Databricks pipeline editor |
| `src/setup/` | One-time SQL: catalog, schemas and volume |
| `src/notebooks/` | Connectivity check and exploration notebooks |
| `scripts/descargar_fuentes.py` | Downloads public sources into a local `landing/` folder |
| `.github/workflows/descargar_fuentes.yml` | Weekly download and upload to the volume |

## Run it yourself

1. Run `src/setup/00_crear_catalogo.sql` in the Databricks SQL editor.
2. Create the ETL pipeline `vigia-etl` from the files in `vigia-etl/transformations/`, catalog `vigia`, schema `bronze`.
3. Create the job `vigia-semanal` with one pipeline task and a file-arrival trigger on `/Volumes/vigia/bronze/landing/`.
4. Download and upload the first file:
   `python scripts/descargar_fuentes.py --fuente nino --destino landing`, then
   `databricks fs cp -r landing dbfs:/Volumes/vigia/bronze/landing --overwrite`.
5. For the weekly run, add the repository secrets `DATABRICKS_HOST` and `DATABRICKS_TOKEN`.

## Data sources

- NOAA Climate Prediction Center, weekly OISST v2.1 Niño indices, 1991–2020 base: <https://www.cpc.ncep.noaa.gov/data/indices/>

## Disclaimer

Portfolio prototype, not an official health alert. It uses aggregated public data only.
