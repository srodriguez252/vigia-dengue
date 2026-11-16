# Vigía Dengue: dengue early warning system for Ecuador on Databricks

[![CI](https://github.com/srodriguez252/vigia-dengue/actions/workflows/ci.yml/badge.svg)](https://github.com/srodriguez252/vigia-dengue/actions/workflows/ci.yml)

**Since March 2024, when the upload date of each new bulletin is known to the day, Vigía Dengue's red alert came before 23 of 29 outbreak onsets, a median of 6 weeks ahead. A 4-week average under the same rule caught 17.**

Every Saturday it forecasts confirmed dengue cases in each of Ecuador's 24 provinces one to four weeks ahead, issues a green, yellow or red alert with its reason, and estimates the hospital bed-days those cases would take. It runs end to end on Databricks Free Edition.

> **Resumen en español.** Sistema de alerta temprana de dengue para Ecuador sobre Databricks Free Edition. Cada sábado pronostica los casos confirmados de cada provincia a 1–4 semanas, emite una alerta verde, amarilla o roja con su motivo y estima las camas-día que ocuparían. Usa datos públicos del MSP, el INEC, Open-Meteo (ERA5 de Copernicus) y NOAA. Desde marzo de 2024, cuando la fecha de subida de cada gaceta nueva se conoce al día, la alerta roja se adelantó a 23 de 29 inicios de brote, con 6 semanas de mediana; el promedio de 4 semanas, con la misma regla, a 17.

## The problem

- Dengue cases in Ecuador more than doubled in one year: 61,400 confirmed cases in 2024 against 27,906 in 2023 ([MSP bulletin, epi week 42 of 2025](https://www.salud.gob.ec/wp-content/uploads/2025/10/Enfermedades-Trasmitidas-por-Vectores-DNVE-SE-42.pdf)).
- The Ministry of Health's weekly bulletins describe what already happened, in PDF tables with five different layouts since 2017. None projects the weeks ahead.
- Climate gives warning: El Niño forecasts anticipated the early 2016 peak in Machala ([Lowe et al., 2017](https://par.nsf.gov/servlets/purl/10039611)), and at the time Ecuador had no official dengue early warning system ([LSHTM](https://www.lshtm.ac.uk/expert-opinion/can-climate-forecasts-predict-outbreaks)).

The decision it supports: which provinces should get vector control and supplies before the peak, not after.

## How it works

```mermaid
flowchart TB
    subgraph SRC["Public sources"]
        direction LR
        msp["MSP bulletins<br/>weekly PDF, 5 layouts"]
        om["Open-Meteo<br/>ERA5 and ERA5-Land"]
        noaa["NOAA CPC<br/>Niño 1+2 index"]
        inec["INEC<br/>discharges, population"]
    end

    dl["Tasks Gacetas and Clima<br/>every Saturday, 2 retries"]
    gha["GitHub Actions<br/>Tuesdays"]
    nb["Notebooks<br/>once a year, by hand"]
    vol[("Unity Catalog volume<br/>vigia.bronze.landing")]

    subgraph ETL["Task ETL: Lakeflow pipeline vigia-etl"]
        direction LR
        bronze["bronze<br/>Auto Loader,<br/>streaming tables"] --> silver["silver<br/>PDF parser, one version<br/>per bulletin cut-off date"] --> feat["gold<br/>weekly cases, endemic channel,<br/>no-look-ahead features"]
        silver -.-> dq[/"quarantine and<br/>reconciliation tables"/]
    end

    subgraph ML["Model: re-run by hand"]
        direction LR
        bt["Rolling-origin backtest<br/>tracked in MLflow"] --> reg[("Unity Catalog registry<br/>pronostico_dengue@champion")]
    end

    subgraph WK["Weekly bulletin"]
        direction LR
        fc["Task Pronostico<br/>1–4 weeks, 80% interval,<br/>bed-days"] --> al["Task Alerta<br/>level and reason"] --> out[("gold.pronosticos<br/>gold.alertas<br/>append-only")]
    end

    subgraph USE["Consumption"]
        direction TB
        dash["Task Boletin<br/>AI/BI dashboard,<br/>emailed as PDF"]
        genie["Genie Agent<br/>questions in Spanish"]
    end

    msp --> dl
    om --> dl
    noaa --> gha
    inec --> nb
    dl --> vol
    gha --> vol
    nb --> vol
    vol --> ETL
    ETL -. "features" .-> ML
    ETL -- "features of the week" --> WK
    ML -. "champion and alert rule" .-> WK
    WK --> USE

    classDef job fill:#dbeafe,stroke:#2272b4,color:#0b2545
    classDef manual fill:#f3f4f6,stroke:#6b7280,color:#111827,stroke-dasharray: 5 4
    class dl,bronze,silver,feat,dq,fc,al,dash job
    class nb,bt manual
```

Blue boxes run inside the job `vigia-semanal`, every Saturday at 08:00, Ecuador time. Dashed gray boxes are run by hand, and dashed arrows are the offline path: the backtest reads the gold features and leaves the champion model and its alert rule in the registry. The dashboard and Genie also read the pipeline's gold tables (`dengue_semana`, `canal_endemico`).

The job's six tasks, and how a change reaches them:

```mermaid
flowchart LR
    subgraph CI["GitHub Actions, every push"]
        direction LR
        T["pytest<br/>41 tests"] --> D["bundle validate,<br/>plan and deploy<br/>(main only)"] --> W[("Pipeline, job and<br/>dashboard in prod")]
    end
    subgraph JOB["Job vigia-semanal, Saturdays 08:00 (Ecuador)"]
        direction LR
        G["Gacetas"] --> E["ETL<br/>runs even if a<br/>download failed"]
        C["Clima"] --> E
        E --> P["Pronostico"] --> A["Alerta"] --> B["Boletin"]
    end
    classDef job fill:#dbeafe,stroke:#2272b4,color:#0b2545
    classDef ci fill:#f3f4f6,stroke:#6b7280,color:#111827
    class G,C,E,P,A,B job
    class T,D ci
```

- **Ingestion.** Download notebooks and a GitHub Actions workflow land raw files in a Unity Catalog volume. Auto Loader reads only new files.
- **ETL.** One Lakeflow Spark Declarative Pipeline (`vigia-etl`, 20 source files) builds the bronze, silver and gold tables, with expectations that drop bad rows or stop the update.
- **Model.** A gradient boosting model, chosen by a rolling-origin backtest and registered in Unity Catalog through MLflow.
- **Weekly run.** A six-task job forecasts the weeks still missing, applies the alert rule, estimates bed-days and emails the dashboard as a PDF.
- **Operations.** The pipeline, the job and the dashboard are code in a Declarative Automation Bundle. GitHub Actions runs 41 tests on every push and deploys the bundle when they pass on `main`.

## Results

Rolling-origin backtest: one forecast per epidemiological week since week 1 of 2022, retraining every 4 weeks, with only the bulletins that were online at each issue date.

**What each issue knew.** Only 31% of the weekly issues since 2022 had the bulletin of their own week. None did in 2022 and 2023, when the upload month is the only trace and a bulletin counts from the last day of that month; 65% did in 2025 and 75% in 2026. Since week 10 of 2024, when the upload date of each new bulletin is known to the day, 55% had it and 84% had data at most one week old. An earlier version of this backtest took each bulletin's cut-off date as its publication date and put that share at 92%; the results below replace the ones it gave.

**Forecast error.** Mean absolute error in cases per province and week. Every forecast cuts the endemic channel's error by 20% to 46%. The two baselines without a model have the lowest error at every horizon: 14.0 and 14.1 cases on average, against 17.9 for the champion. With the cut-off date as publication date the gap was about one case, so the model depends on fresh bulletins more than the averages do.

| Model | 1 week | 2 weeks | 3 weeks | 4 weeks | Relative to the endemic channel |
| --- | --- | --- | --- | --- | --- |
| Last known week | 11.9 | 13.4 | 14.7 | 16.1 | 0.54 |
| Last 4 weeks (naive) | 11.9 | 13.3 | 14.8 | 16.3 | 0.54 |
| **Gradient boosting with weather and El Niño (champion)** | 17.1 | 17.4 | 17.8 | 19.4 | 0.69 |
| Gradient boosting, no weather | 17.6 | 18.2 | 19.1 | 20.2 | 0.72 |
| Gradient boosting, one model per province | 19.9 | 20.3 | 21.1 | 21.4 | 0.80 |
| Endemic channel, median | 25.8 | 25.9 | 26.0 | 26.1 | 1.00 |

**Alerts.** An alert turns red when the 2–4 week forecast is above the channel's third quartile in two consecutive issues. It is measured against outbreak onsets: the first of 4 consecutive weeks in the epidemic zone after 8 calm weeks. A warning is in time when a red issue is published up to 8 weeks before the onset, and late when it comes in the 4 weeks after. A run of red issues is a false alarm when it starts in a calm week and no outbreak begins from 4 weeks before to 8 weeks after its publication.

Issues since week 10 of 2024, with 29 onsets in 17 provinces between May 2024 and September 2026:

| Forecast behind the alert | Warned in time | Warned late | Not warned | Median lead (weeks) | False alarms per province-year |
| --- | --- | --- | --- | --- | --- |
| **Champion** | 23 | 5 | 1 | 6 | 0.48 |
| Last 4 weeks (naive) | 17 | 5 | 7 | 7 | 0.31 |
| Last known week | 14 | 11 | 4 | 7 | 0.33 |

The champion misses 1 onset where the naive forecast misses 7, and raises about 55% more false alarms. Over the whole backtest, with 56 onsets since 2022, the champion and the naive forecast each warned 25 in time, with 0.39 and 0.34 false alarms per province and year: the champion's advantage shows only where the upload dates are exact. Stricter rules, which also ask the 4-week forecast to reach 1, 1.25 or 1.5 times the last 4 weeks' average, warned 20, 11 and 9 of the 29 onsets.

**The 2024 epidemic on the coast.** The rise the project set out to catch would have been caught late, and the reason is in the bulletins. No bulletin sits in the upload folder of January 2024: those of the last two weeks of 2023 and the first eight of 2024 went up in February, two of them in November. From December to late February each issue worked with data 2 to 9 weeks old, and the provinces that were red in December fell back to yellow or green. Red came back with the bulletins: on 16 March in Santa Elena, 8 weeks before its peak, and in Los Ríos, after its peak; on 23 March in Santo Domingo, 3 weeks before; and on 13 April in Manabí, 6 weeks before, in Guayas, the day before its peak week began, and in El Oro, at its peak. Esmeraldas stayed red except for four issues in January and February.

![Weekly bulletin dashboard](docs/img/dashboard.png)

![The 2024 epidemic on the coast: weekly cases, epidemic threshold and the alert in force each week](docs/img/alerta_2024_costa.png)

The panel titles count the weeks from the first red issue of the rise to the peak, even when that red did not last: Manabí's 24 weeks are two red issues in December 2023.

## Design decisions

- **No look-ahead.** Each MSP bulletin is stored as a version with its cut-off date and the date it can be shown to have been online. The Ministry does not publish upload dates, and many bulletins went up weeks after their cut-off, several at a time, so the publication date comes from three traces: the server's `Last-Modified` header, the upload-month folder in the URL, and the first weekly run that saw the link. A bulletin counts from the latest date those traces allow. The features of week *t*, issued 7 days after the week closed, use only the bulletins already published that day, and the training labels are the case counts as known 3 weeks later. Two tests perturb everything that was not known on a given date and require identical features and forecasts.
- **A PDF parser that quarantines instead of guessing.** pdfplumber runs inside `mapInPandas` and reads five table layouts from 2017 to 2026 by the position of each number. A bulletin it cannot read with confidence goes to a quarantine table, and every bulletin with a total row is reconciled against it.
- **Rates back to cases.** From mid-2025 to mid-2026 the bulletins publish rates only. They are converted with the population the MSP itself used, deduced from its earlier bulletins, and every row says whether its cases come from the PDF or are estimated.
- **A bulletin that carries the previous year.** The total row of the bulletin for week 1 of 2025 adds all the cases of 2024 to the 547 of its own week. Weekly cases are the growth of each year's highest cumulative count, so that one bulletin gave Manabí 10,473 cases in the first week of 2025, about its whole 2024, and zero in the other 52 weeks. A bulletin from the first 4 weeks of a year is dropped when its national cumulative count reaches half of the previous year's highest count known at its cut-off date; its week is filled like any other gap, and a test reproduces the case.
- **The endemic channel as the yardstick.** For each province, year and week: the quartiles of up to 7 previous non-epidemic years ([Bortman, 1999](https://scielosp.org/pdf/rpsp/1999.v5n1/1-8/es)). It is both the baseline to beat and the threshold of the alert.
- **The model.** `HistGradientBoostingRegressor` with Poisson loss, one per horizon, shared by all provinces, with the province as a categorical feature. It uses lagged cases and their growth, incidence per 100,000, the endemic channel, data-quality flags, season, population, rainfall, temperature and humidity anomalies and the lagged Niño 1+2 index. The backtest compared it with the same model without weather and with one model per province; the champion is the global model with the lower mean error. Three baselines without a model go through the same backtest and the same alert rules: the endemic channel, the average of the last 4 weeks and the last known week.
- **The alert rule.** Red when, in two consecutive weekly issues, the 2–4 week forecast is above the channel's third quartile (and above 5 cases). Yellow when the forecast is above the channel's median. Asking also for a rising 4-week forecast warned fewer outbreak onsets and no earlier on the coast in 2024, so the plain rule stays.
- **Bed-days.** Forecast cases × hospitalizations per case × mean length of stay, from INEC discharges and MSP cases of 2017–2024. Provinces with fewer than 30 hospitalizations use national rates.
- **Append-only weekly issues.** Each run forecasts only the issues not yet in `gold.pronosticos`, so a failed Saturday is made up the next week and past issues are never rewritten. The 80% interval comes from the backtest errors of the champion.

## Limitations

- **The model does not beat the baselines on error.** The last known week and the 4-week average miss by 14.0 and 14.1 cases on average and the champion by 17.9, and it trails at every horizon. What it adds is in the alerts: since March 2024 it warned 23 of 29 outbreak onsets in time against 17, at the price of 0.48 false alarms per province and year against 0.31.
- **It learned from stale history.** Under the dates the backtest uses, no issue of 2022 or 2023 had the bulletin of its own week, and the lag features are tied to fixed weeks, so they are empty when a bulletin is late. That is the likely reason the model forecasts too low when cases are high: Manabí stayed green through March 2024 with 250 to 330 cases a week.
- **Weather adds little.** With rainfall, temperature, humidity and El Niño the error is 17.9 cases against 18.8 without, about 5% less. The rainfall bias below and a history that starts in 2017 may explain part of it.
- **Rainfall is biased in the Andes.** ERA5 cells of about 28 km mix valleys and slopes: Quito adds up to 3,518 mm in 2025 against a normal of about 1,100 mm. That is why rainfall enters the model as an anomaly against its own climatology.
- **The epidemic zone is the usual state.** 45% of the province-weeks since 2022 are above the channel's third quartile, 64% on the coast, and Manabí stayed above it for 123 weeks in a row, from week 21 of 2023 to week 39 of 2025. A red alert says cases are above the province's history, not that an outbreak is starting, and a province already in red cannot be warned again. That is why alerts are measured against outbreak onsets, the passage from calm to epidemic.
- **The onset measure is new and small.** Its thresholds were set before looking at its results but have not been validated on other data, and the 29 onsets since March 2024 are 2 of 2024, 9 of 2025 and 18 of 2026.
- **The bulletins have gaps.** From 2017 to 2026, 21 weeks were never posted, 3 links are broken and 4 PDFs cannot be read, and figures are revised in later bulletins. The growth across a gap is spread evenly over its weeks.
- **Publication dates are exact only since March 2024.** Bulletins uploaded since then keep their original server date, and it always falls in the month of their URL folder. Every earlier file shows the same bulk re-upload of 2 March 2024, so the folder is the only trace: those bulletins count from the last day of their upload month, up to a month after the real upload, and the ones of 2017–2019 with no usable trace still count from their cut-off date. `publicacion_origen` marks each case, and `publicacion_desde` keeps the earliest possible date, to rerun the backtest at the other end of the range.
- **Hospital data arrive late.** INEC discharges are published more than a year later, so they calibrate bed-days but do not feed the weekly forecast. Bed-days are an order of magnitude, not a capacity plan.
- **Short channel history.** The endemic channel needs at least 5 previous years, so it starts in 2022 or 2023 depending on the province.
- **Free Edition limits.** Serverless compute only, one pipeline, limited outbound internet, and Open-Meteo's free tier for non-commercial use, which spreads the weather history over several days.

## Repository layout

| Path | What it holds |
| --- | --- |
| `vigia-etl/transformations/` | Lakeflow pipeline: bronze, silver and gold tables |
| `vigia-etl/utilities/` | Weekly cases, endemic channel and features (`features.py`) and the publication date of each bulletin (`publicacion.py`), shared by the pipeline and the tests |
| `src/modelos/` | Backtest, baselines, alert rule and metrics (`pronostico.py`); weekly forecast, interval and bed-days (`semanal.py`) |
| `src/notebooks/` | Downloads, backtest, model registration, weekly forecast and alert |
| `src/dashboards/` | AI/BI dashboard definition |
| `src/setup/` | Catalog, schemas and volume; the 221 cantons with their weather point (`cantones.csv`) |
| `databricks.yml`, `resources/` | Declarative Automation Bundle: pipeline, job and dashboard |
| `tests/` | pytest suites: no-leakage tests, publication dates, backtest and weekly bulletin |
| `scripts/` | Niño download and the generator of `cantones.csv` |
| `.github/workflows/` | Weekly Niño download; CI that tests and deploys the bundle |

## Run it yourself

You need a Databricks workspace (Free Edition works), the Databricks CLI 1.22 or later logged in with the `DEFAULT` profile, and a fork of this repository.

1. Run `src/setup/00_crear_catalogo.sql` in the SQL editor: catalog `vigia`, schemas `bronze`, `silver`, `gold` and `ml`, and the volume `landing`.
2. Add your fork as a Git folder in the workspace.
3. Copy the cantons to the volume, then load the history once by running these notebooks from `src/notebooks/`: `descargar_egresos`, `descargar_poblacion`, `descargar_gacetas` and `descargar_clima`. The weather history needs several runs because of Open-Meteo's free quota; the notebook resumes where it stopped. The Niño index comes from the script that GitHub Actions runs every Tuesday:

   ```bash
   databricks fs cp src/setup/cantones.csv dbfs:/Volumes/vigia/bronze/landing/geografia/cantones.csv
   python scripts/descargar_fuentes.py --fuente nino --destino landing
   databricks fs cp -r landing dbfs:/Volumes/vigia/bronze/landing --overwrite
   ```

4. Put your SQL warehouse ID in `resources/vigia_dengue.dashboard.yml` and `resources/vigia_semanal.job.yml`, then deploy and build the tables:

   ```bash
   databricks bundle deploy -t prod
   databricks bundle run vigia_etl -t prod
   ```

5. Run the notebooks `backtest`, `por_provincia`, `alertas` and `registrar`, in that order. They write the backtest tables and register `vigia.ml.pronostico_dengue` with the alias `champion`.
6. Run the weekly job once: `databricks bundle run vigia_semanal -t prod`. After that it runs every Saturday on its own.
7. For CI, add the repository secrets `DATABRICKS_HOST` and `DATABRICKS_TOKEN`. To run the tests locally (Python 3.12 and Java 17):

   ```bash
   pip install pyspark==4.0.1 pytest==8.4.2 scikit-learn==1.7.2 pandas==2.3.3 numpy==2.3.4 matplotlib==3.10.6
   pytest tests
   ```

## Data sources and licenses

The repository does not redistribute raw data: the notebooks download each source into a Unity Catalog volume. The only derived dataset stored here is `src/setup/cantones.csv`.

| Source | Used for | License |
| --- | --- | --- |
| [MSP, weekly bulletins of vector-borne diseases](https://www.salud.gob.ec/gacetas-vectoriales-2026/), 2017–2026 | Confirmed dengue cases by province and week | Public bulletins of Ecuador's Ministry of Public Health |
| [INEC, hospital beds and discharges](https://www.ecuadorencifras.gob.ec/camas-y-egresos-hospitalarios/), 2015–2024 | Hospitalizations and length of stay for bed-days | Public data of Ecuador's national statistics institute |
| [INEC, population estimates and projections, 2024 revision](https://www.ecuadorencifras.gob.ec/estimaciones-y-proyecciones-de-poblacion-revision-2024/) | Population by canton and year | Public data of Ecuador's national statistics institute |
| [Open-Meteo Historical Weather API](https://open-meteo.com/) | Daily temperature, rainfall and humidity per canton | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Copernicus Climate Change Service, ERA5 and ERA5-Land, through Open-Meteo | The reanalysis behind the weather data | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| [NOAA Climate Prediction Center, weekly Niño indices](https://www.cpc.ncep.noaa.gov/data/indices/) | Niño 1+2 sea surface temperature anomaly | U.S. Government data |
| CONALI, Organización Territorial del Estado (17 July 2026), through [geoecu](https://cran.r-universe.dev/geoecu) | Canton seats in `cantones.csv` | CC BY |
| [GeoNames](https://www.geonames.org/) | Most populated place of each canton seat in `cantones.csv` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

### Attribution

- [Weather data by Open-Meteo.com](https://open-meteo.com/). Zippenfenig, P. (2023). *Open-Meteo.com Weather API* [Computer software]. Zenodo. <https://doi.org/10.5281/ZENODO.7970649>
- Contains modified Copernicus Climate Change Service information 2026. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
  - Hersbach, H., et al. (2023). *ERA5 hourly data on single levels from 1940 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). <https://doi.org/10.24381/cds.adbb2d47>
  - Muñoz Sabater, J. (2019). *ERA5-Land hourly data from 1950 to present*. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). <https://doi.org/10.24381/cds.e2161bac>
- Canton boundaries: Comité Nacional de Límites Internos (CONALI), Ministerio de Gobierno del Ecuador, *Organización Territorial del Estado*, version of 17 July 2026, CC BY, as published by the geoecu package.
- Places: [GeoNames](https://www.geonames.org/), CC BY 4.0, through the cities500 file of the geonamescache package.
- Niño 1+2 index: NOAA Climate Prediction Center, weekly OISST v2.1, 1991–2020 base period.
- Dengue cases: Ministerio de Salud Pública del Ecuador, *Gacetas epidemiológicas de enfermedades vectoriales*.
- Hospital discharges and population: Instituto Nacional de Estadística y Censos (INEC).

**Changes made.** Weather is aggregated to epidemiological weeks, weighted by canton population and expressed as anomalies against each province's own climatology. `cantones.csv` derives one point per canton: the GeoNames place with the most people inside the canton seat's CONALI polygon, or the polygon's centroid when there is none.

## Disclaimer

Portfolio project, not an official health alert. It uses aggregated public data only, with no personal information. Forecasts and bed-days carry the uncertainty described above and should not drive real decisions without the Ministry of Health's own surveillance.

## License

Code under the MIT License (see [LICENSE](LICENSE)). The data keep the licenses of their sources.
