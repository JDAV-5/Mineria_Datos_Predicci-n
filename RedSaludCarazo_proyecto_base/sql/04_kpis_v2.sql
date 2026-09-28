USE RedSaludCarazoDW_V2;
GO

SELECT
    COUNT(*) AS registros_evaluados,

    SUM(es_valido) AS registros_validados,

    CAST(
        100.0 * SUM(es_valido)
        / NULLIF(COUNT(*), 0)
        AS DECIMAL(7,2)
    ) AS PRV_porcentaje

FROM dbo.dm_dataset_calidad;
GO


/* ============================================================
   KPI 2
   TASA DE INASISTENCIA EN REGISTROS VALIDADOS (TIRV)
   ============================================================ */

SELECT
    COUNT(*) AS citas_validas,

    SUM(
        CASE
            WHEN no_asistio = 1
            THEN 1
            ELSE 0
        END
    ) AS inasistencias,

    CAST(
        100.0 *
        SUM(
            CASE
                WHEN no_asistio = 1
                THEN 1
                ELSE 0
            END
        )
        / NULLIF(COUNT(*), 0)
        AS DECIMAL(7,2)
    ) AS TIRV_porcentaje

FROM dbo.dm_dataset_patrones;
GO


/* ============================================================
   KPI 2 - DESGLOSE POR ESPECIALIDAD
   ============================================================ */

SELECT
    especialidad,

    COUNT(*) AS citas,

    CAST(
        100.0 *
        SUM(
            CASE
                WHEN no_asistio = 1
                THEN 1
                ELSE 0
            END
        )
        / NULLIF(COUNT(*), 0)
        AS DECIMAL(7,2)
    ) AS TIRV_porcentaje

FROM dbo.dm_dataset_patrones

GROUP BY especialidad

ORDER BY TIRV_porcentaje DESC;
GO


/* ============================================================
   KPI 3
   EXACTITUD DEL MODELO PREDICTIVO (ACCURACY)

   MODELO:
   RandomForest_v3
   ============================================================ */

SELECT
    nombre_modelo,

    COUNT(*) AS citas_prueba,

    SUM(
        CASE
            WHEN no_asistio = pred_inasistencia
            THEN 1
            ELSE 0
        END
    ) AS predicciones_correctas,

    SUM(
        CASE
            WHEN no_asistio <> pred_inasistencia
            THEN 1
            ELSE 0
        END
    ) AS predicciones_incorrectas,

    CAST(
        100.0 *
        SUM(
            CASE
                WHEN no_asistio = pred_inasistencia
                THEN 1
                ELSE 0
            END
        )
        / NULLIF(COUNT(*), 0)
        AS DECIMAL(7,2)
    ) AS accuracy_porcentaje,

    SUM(
        CASE
            WHEN no_asistio = 1
             AND pred_inasistencia = 1
            THEN 1
            ELSE 0
        END
    ) AS VP,

    SUM(
        CASE
            WHEN no_asistio = 0
             AND pred_inasistencia = 1
            THEN 1
            ELSE 0
        END
    ) AS FP,

    SUM(
        CASE
            WHEN no_asistio = 1
             AND pred_inasistencia = 0
            THEN 1
            ELSE 0
        END
    ) AS FN,

    SUM(
        CASE
            WHEN no_asistio = 0
             AND pred_inasistencia = 0
            THEN 1
            ELSE 0
        END
    ) AS VN,

    CAST(
        100.0 *
        SUM(
            CASE
                WHEN no_asistio = 1
                 AND pred_inasistencia = 1
                THEN 1
                ELSE 0
            END
        )
        /
        NULLIF(
            SUM(
                CASE
                    WHEN no_asistio = 1
                    THEN 1
                    ELSE 0
                END
            ),
            0
        )
        AS DECIMAL(7,2)
    ) AS recall_porcentaje,

    CAST(
        100.0 *
        SUM(
            CASE
                WHEN no_asistio = 1
                 AND pred_inasistencia = 1
                THEN 1
                ELSE 0
            END
        )
        /
        NULLIF(
            SUM(
                CASE
                    WHEN pred_inasistencia = 1
                    THEN 1
                    ELSE 0
                END
            ),
            0
        )
        AS DECIMAL(7,2)
    ) AS precision_porcentaje

FROM dbo.dm_resultados_prediccion

WHERE nombre_modelo = 'RandomForest_v3'
  AND conjunto = 'prueba'

GROUP BY nombre_modelo;
GO