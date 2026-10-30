-- Vigía Dengue
-- Crea el catálogo, las cuatro capas y el Volume donde aterrizan los archivos

CREATE CATALOG IF NOT EXISTS vigia
  COMMENT 'Vigía Dengue: alerta temprana de dengue para Ecuador';

CREATE SCHEMA IF NOT EXISTS vigia.bronze COMMENT 'Datos tal como llegan de cada fuente';
CREATE SCHEMA IF NOT EXISTS vigia.silver COMMENT 'Datos limpios y conformados';
CREATE SCHEMA IF NOT EXISTS vigia.gold   COMMENT 'Tablas para el modelo, el dashboard y Genie';
CREATE SCHEMA IF NOT EXISTS vigia.ml     COMMENT 'Modelos, features y predicciones';

CREATE VOLUME IF NOT EXISTS vigia.bronze.landing
  COMMENT 'Archivos crudos de todas las fuentes';
