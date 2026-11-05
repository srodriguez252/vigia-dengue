# Silver - egresos del INEC con los mismos nombres, códigos y tipos en todos los años,
# y las hospitalizaciones por dengue por provincia y semana epidemiológica
from pyspark import pipelines as dp
from pyspark.sql import functions as F

ANIOS = range(2015, 2025)

# Códigos DPA de provincia, para los años que traen el nombre en vez del código.
PROVINCIAS = {
    "azuay": "01", "bolivar": "02", "canar": "03", "carchi": "04", "cotopaxi": "05",
    "chimborazo": "06", "el oro": "07", "esmeraldas": "08", "guayas": "09", "imbabura": "10",
    "loja": "11", "los rios": "12", "manabi": "13", "morona santiago": "14", "napo": "15",
    "pastaza": "16", "pichincha": "17", "tungurahua": "18", "zamora chinchipe": "19",
    "galapagos": "20", "sucumbios": "21", "orellana": "22", "santo domingo de los tsachilas": "23",
    "santa elena": "24", "zonas no delimitadas": "90",
}

MESES = {
    "enero": "1", "febrero": "2", "marzo": "3", "abril": "4", "mayo": "5", "junio": "6",
    "julio": "7", "agosto": "8", "septiembre": "9", "octubre": "10", "noviembre": "11", "diciembre": "12",
}


def mapa(diccionario):
    """Convierte un diccionario de Python en un mapa de Spark para buscar valores"""
    return F.create_map(*[F.lit(x) for par in diccionario.items() for x in par])


def normalizar(columna):
    """Para quitar minúsculas, tildes y espacios de más"""
    sin_tildes = F.translate(F.lower(F.trim(columna)), "áéíóúüñ", "aeiouun")
    return F.regexp_replace(sin_tildes, r"\s+", " ")


def es_numero(columna):
    return F.trim(columna).rlike(r"^[0-9]+$")


def entero(columna):
    return F.when(es_numero(columna), F.trim(columna).cast("int"))


def fecha(df, sufijo):
    """Arma la fecha con anio_, mes_ y dia_. El mes llega como número o como nombre"""
    mes = F.when(es_numero(df[f"mes_{sufijo}"]), F.trim(df[f"mes_{sufijo}"])).otherwise(
        F.try_element_at(mapa(MESES), normalizar(df[f"mes_{sufijo}"]))
    )
    texto = F.concat_ws("-", F.trim(df[f"anio_{sufijo}"]), mes, F.trim(df[f"dia_{sufijo}"]))
    return F.try_to_timestamp(texto, F.lit("yyyy-M-d")).cast("date")


def provincia(columna):
    """'09' en los años con códigos y 'Guayas' en los años con nombres: siempre devuelve el código DPA"""
    return F.when(es_numero(columna), F.lpad(F.trim(columna), 2, "0")).otherwise(
        F.try_element_at(mapa(PROVINCIAS), normalizar(columna))
    )


def sexo(columna):
    """Formulario del INEC, 1 = hombre y 2 = mujer"""
    valor = normalizar(columna)
    return F.when(valor.isin("1", "hombre"), "hombre").when(valor.isin("2", "mujer"), "mujer")


def condicion(columna):
    """Formulario del INEC, 1 = vivo, 2 = fallecido antes de 48 horas y 3 = después de 48 horas"""
    valor = normalizar(columna)
    return F.when(valor.isin("1", "vivo"), "vivo").when(
        valor.isin("2", "3") | valor.startswith("fallec"), "fallecido"
    )


def cie10(columna):
    """'A970', 'A970   Dengue sin…' o 'K808 Otras colelitiasis' (piedras o cálculos) -> el código. Si solo trae descripción, nulo"""
    codigo = F.regexp_extract(F.upper(F.trim(columna)), r"^([A-Z][0-9]{2}[0-9A-Z]?)(\s|$)", 1)
    return F.nullif(codigo, F.lit(""))


def causa(columna):
    """La descripción: lo que sigue al código, o todo el texto si no trae código"""
    descripcion = F.regexp_replace(F.trim(columna), r"(?i)^[A-Z][0-9]{2}[0-9A-Z]?(\s+|$)", "")
    return F.nullif(F.trim(descripcion), F.lit(""))


def es_dengue(codigo, descripcion):
    """A90 y A91 en la CIE-10 clásica, A97 desde su actualización; sin código, se busca en la descripción."""
    return F.coalesce(codigo.rlike(r"^A9[017]") | F.lower(descripcion).contains("dengue"), F.lit(False))


def armonizar(df, anio):
    codigo, descripcion = cie10(df["cau_cie10"]), causa(df["cau_cie10"])
    return df.select(
        F.lit(anio).alias("anio"),
        fecha(df, "ingr").alias("fecha_ingreso"),
        fecha(df, "egr").alias("fecha_egreso"),
        entero(df["dia_estad"]).alias("dias_estada"),
        provincia(df["prov_res"]).alias("provincia_res"),
        F.trim(df["prov_res"]).alias("provincia_res_original"),
        F.trim(df["cant_res"]).alias("canton_res_original"),
        sexo(df["sexo"]).alias("sexo"),
        condicion(df["con_egrpa"]).alias("condicion_egreso"),
        codigo.alias("cie10"),
        descripcion.alias("causa"),
        es_dengue(codigo, descripcion).alias("es_dengue"),
        df["archivo_origen"],
    )


dp.create_streaming_table(
    name="silver.egresos",
    comment="Egresos hospitalarios 2015–2024 del INEC con nombres, códigos y tipos comunes.",
    expect_all={
        "fecha_ingreso_valida": "fecha_ingreso IS NOT NULL",
        "egreso_en_su_anio": "fecha_egreso IS NOT NULL AND year(fecha_egreso) = anio",
        "ingreso_antes_del_egreso": "fecha_ingreso IS NULL OR fecha_egreso IS NULL OR fecha_ingreso <= fecha_egreso",
        "provincia_conocida": "provincia_res IS NOT NULL AND provincia_res IN ({})".format(
            ", ".join(f"'{codigo}'" for codigo in PROVINCIAS.values())
        ),
        "condicion_conocida": "condicion_egreso IS NOT NULL",
    },
)


def crear_flujo(anio):
    # Un append flow por año: cada uno lee su tabla bronze y agrega sus filas a silver.egresos
    @dp.append_flow(target="silver.egresos", name=f"armonizar_{anio}")
    def flujo():
        return armonizar(spark.readStream.table(f"bronze.egresos_{anio}"), anio)


for anio in ANIOS:
    crear_flujo(anio)


def semana_epidemiologica(fecha):
    """Semana de domingo a sábado. La semana 1 es la que contiene el 4 de enero: algunos años tienen 53,
    y los primeros días de enero pueden caer en la última semana del año anterior"""
    domingo = F.date_sub(fecha, F.dayofweek(fecha) - 1)  # dayofweek: 1 = domingo … 7 = sábado
    anio_epi = F.year(F.date_add(domingo, 3))  # la semana es del año en que cae su miércoles
    cuatro_de_enero = F.make_date(anio_epi, F.lit(1), F.lit(4))
    primer_domingo = F.date_sub(cuatro_de_enero, F.dayofweek(cuatro_de_enero) - 1)
    semana = (F.datediff(domingo, primer_domingo) / 7).cast("int") + 1
    return anio_epi, semana, domingo


@dp.materialized_view(
    name="silver.egresos_dengue",
    comment="Hospitalizaciones por dengue por provincia de residencia y semana epidemiológica de ingreso.",
)
@dp.expect_or_fail("semana_valida", "semana_epi BETWEEN 1 AND 53")
def egresos_dengue():
    anio_epi, semana_epi, inicio_semana = semana_epidemiologica(F.col("fecha_ingreso"))
    return (
        spark.read.table("silver.egresos")
        .where("es_dengue AND fecha_ingreso IS NOT NULL")
        .withColumns({"anio_epi": anio_epi, "semana_epi": semana_epi, "inicio_semana": inicio_semana})
        .groupBy("provincia_res", "anio_epi", "semana_epi", "inicio_semana")
        .agg(
            F.count("*").alias("hospitalizaciones"),
            F.sum("dias_estada").alias("dias_estada"),
            F.count_if(F.col("condicion_egreso") == "fallecido").alias("fallecidos"),
        )
    )