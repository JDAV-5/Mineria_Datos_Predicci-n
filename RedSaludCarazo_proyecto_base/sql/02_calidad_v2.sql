USE RedSaludCarazoDW_V2;
GO
IF OBJECT_ID(N'dbo.vw_calidad_citas', N'V') IS NOT NULL DROP VIEW dbo.vw_calidad_citas;
GO
CREATE VIEW dbo.vw_calidad_citas AS
WITH base AS (
    SELECT r.*,
           COUNT(*) OVER (PARTITION BY r.id_cita) AS repeticiones_id,
           UPPER(LTRIM(RTRIM(r.centro_salud))) AS centro_normalizado,
           UPPER(LTRIM(RTRIM(r.especialidad))) AS especialidad_normalizada,
           UPPER(LTRIM(RTRIM(r.canal_agendamiento))) AS canal_normalizado
    FROM dbo.raw_cita AS r
    WHERE r.origen_dataset=N'ampliado'
      AND r.lote_carga=(
          SELECT MAX(lote_carga) FROM dbo.raw_cita WHERE origen_dataset=N'ampliado'
      )
), flags AS (
    SELECT b.*,
        CASE WHEN b.repeticiones_id>1 THEN 1 ELSE 0 END AS id_repetido,
        CASE WHEN NULLIF(LTRIM(RTRIM(b.id_cita)),N'') IS NULL
               OR b.fecha_cita IS NULL
               OR NULLIF(b.centro_normalizado,N'') IS NULL
               OR NULLIF(b.especialidad_normalizada,N'') IS NULL
               OR NULLIF(b.canal_normalizado,N'') IS NULL
               OR NULLIF(LTRIM(RTRIM(b.grupo_edad)),N'') IS NULL
               OR b.espera_dias IS NULL
               OR b.recordatorios_enviados IS NULL
               OR b.citas_previas_perdidas IS NULL
               OR b.no_asistio IS NULL
               OR LTRIM(RTRIM(b.no_asistio)) NOT IN (N'0',N'1')
             THEN 1 ELSE 0 END AS faltan_obligatorios,
        CASE WHEN b.espera_dias<0 OR b.recordatorios_enviados<0
               OR b.citas_previas_perdidas<0 OR b.costo_estimado_usd<0
             THEN 1 ELSE 0 END AS fuera_de_dominio,
        CASE WHEN b.centro_salud COLLATE Latin1_General_100_BIN2 <> b.centro_normalizado COLLATE Latin1_General_100_BIN2
               OR b.especialidad COLLATE Latin1_General_100_BIN2 <> b.especialidad_normalizada COLLATE Latin1_General_100_BIN2
               OR b.canal_agendamiento COLLATE Latin1_General_100_BIN2 <> b.canal_normalizado COLLATE Latin1_General_100_BIN2
             THEN 1 ELSE 0 END AS requirio_normalizacion
    FROM base AS b
)
SELECT f.*,
    CASE WHEN f.id_repetido=0 AND f.faltan_obligatorios=0
              AND f.fuera_de_dominio=0
         THEN 1 ELSE 0 END AS es_valido
FROM flags AS f;
GO
/* NOTA: Especificar rangos superiores únicamente si el caso documenta límites.
   La regla actual identifica negativos, no presume que espera_dias=90 es inválido.
   La dimensión temporal/fecha_carga NO se usa para predecir si no está disponible
   en el momento real de la predicción.
*/
SELECT COUNT(*) AS filas_originales,
       SUM(id_repetido) AS filas_con_id_repetido,
       SUM(faltan_obligatorios) AS filas_incompletas,
       SUM(fuera_de_dominio) AS filas_fuera_dominio,
       SUM(es_valido) AS filas_validas
FROM dbo.vw_calidad_citas;
GO
