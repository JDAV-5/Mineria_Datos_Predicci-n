USE RedSaludCarazoDW_V2;
GO
INSERT INTO dbo.dim_tiempo(fecha,anio,mes,trimestre,dia,dia_semana_lunes_1,es_fin_semana)
SELECT DISTINCT q.fecha_cita, YEAR(q.fecha_cita), MONTH(q.fecha_cita),
       DATEPART(QUARTER,q.fecha_cita), DAY(q.fecha_cita),
       (DATEDIFF(DAY,CONVERT(date,'19000101'),q.fecha_cita)%7)+1,
       CASE WHEN (DATEDIFF(DAY,CONVERT(date,'19000101'),q.fecha_cita)%7) IN (5,6)
            THEN 1 ELSE 0 END
FROM dbo.vw_calidad_citas q
WHERE q.es_valido=1
  AND NOT EXISTS(SELECT 1 FROM dbo.dim_tiempo d WHERE d.fecha=q.fecha_cita);
GO
INSERT INTO dbo.dim_centro_salud(centro_salud)
SELECT DISTINCT q.centro_normalizado FROM dbo.vw_calidad_citas q
WHERE q.es_valido=1 AND NOT EXISTS (
    SELECT 1 FROM dbo.dim_centro_salud d WHERE d.centro_salud=q.centro_normalizado);
GO
INSERT INTO dbo.dim_especialidad(especialidad)
SELECT DISTINCT q.especialidad_normalizada FROM dbo.vw_calidad_citas q
WHERE q.es_valido=1 AND NOT EXISTS (
    SELECT 1 FROM dbo.dim_especialidad d WHERE d.especialidad=q.especialidad_normalizada);
GO
INSERT INTO dbo.dim_canal(canal_agendamiento)
SELECT DISTINCT q.canal_normalizado FROM dbo.vw_calidad_citas q
WHERE q.es_valido=1 AND NOT EXISTS (
    SELECT 1 FROM dbo.dim_canal d WHERE d.canal_agendamiento=q.canal_normalizado);
GO
/* Recarga derivada. NO afecta raw_cita: conserva las filas de origen. */
DELETE FROM dbo.hecho_cita;
GO
INSERT INTO dbo.hecho_cita
(id_raw,id_cita,fecha_cita,id_centro,id_especialidad,id_canal,grupo_edad,
 tipo_cobertura,espera_dias,recordatorios_enviados,citas_previas_perdidas,
 costo_estimado_usd,sistema_origen,consentimiento_contacto,no_asistio,fecha_carga)
SELECT q.id_raw,q.id_cita,q.fecha_cita,dc.id_centro,de.id_especialidad,dca.id_canal,
       q.grupo_edad,q.tipo_cobertura,q.espera_dias,q.recordatorios_enviados,
       q.citas_previas_perdidas,q.costo_estimado_usd,q.sistema_origen,
       q.consentimiento_contacto,TRY_CONVERT(tinyint,q.no_asistio),q.fecha_carga
FROM dbo.vw_calidad_citas q
JOIN dbo.dim_centro_salud dc ON dc.centro_salud=q.centro_normalizado
JOIN dbo.dim_especialidad de ON de.especialidad=q.especialidad_normalizada
JOIN dbo.dim_canal dca ON dca.canal_agendamiento=q.canal_normalizado
WHERE q.es_valido=1;
GO
IF OBJECT_ID(N'dbo.dm_dataset_calidad', N'U') IS NOT NULL DROP TABLE dbo.dm_dataset_calidad;
GO
SELECT id_raw,id_cita,fecha_cita,centro_salud,canal_agendamiento,especialidad,
       no_asistio,fecha_carga,id_repetido,faltan_obligatorios,fuera_de_dominio,
       requirio_normalizacion,es_valido
INTO dbo.dm_dataset_calidad FROM dbo.vw_calidad_citas;
GO
IF OBJECT_ID(N'dbo.dm_dataset_patrones', N'U') IS NOT NULL DROP TABLE dbo.dm_dataset_patrones;
GO
SELECT h.id_cita,h.fecha_cita,t.anio,t.mes,
       dc.centro_salud,de.especialidad,dca.canal_agendamiento,
       h.grupo_edad,h.tipo_cobertura,h.espera_dias,h.recordatorios_enviados,
       h.citas_previas_perdidas,h.costo_estimado_usd,
       h.consentimiento_contacto,h.no_asistio
INTO dbo.dm_dataset_patrones
FROM dbo.hecho_cita h
JOIN dbo.dim_tiempo t ON h.fecha_cita=t.fecha
JOIN dbo.dim_centro_salud dc ON h.id_centro=dc.id_centro
JOIN dbo.dim_especialidad de ON h.id_especialidad=de.id_especialidad
JOIN dbo.dim_canal dca ON h.id_canal=dca.id_canal;
GO
IF OBJECT_ID(N'dbo.dm_dataset_prediccion', N'U') IS NOT NULL DROP TABLE dbo.dm_dataset_prediccion;
GO
SELECT id_cita,fecha_cita,anio,mes,centro_salud,especialidad,
       canal_agendamiento,grupo_edad,tipo_cobertura,espera_dias,
       citas_previas_perdidas,recordatorios_enviados,no_asistio
INTO dbo.dm_dataset_prediccion
FROM dbo.dm_dataset_patrones;
GO
/* recordatorios_enviados SOLO puede entrar al MODELO si se demuestra
   que su valor era conocido ANTES de la cita. El script Python lo excluye
   por defecto para evitar fuga de información temporal. */
SELECT 'calidad' dataset,COUNT(*) registros FROM dbo.dm_dataset_calidad
UNION ALL SELECT 'patrones',COUNT(*) FROM dbo.dm_dataset_patrones
UNION ALL SELECT 'prediccion',COUNT(*) FROM dbo.dm_dataset_prediccion;
GO
