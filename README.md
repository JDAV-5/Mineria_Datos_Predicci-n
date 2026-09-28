# Red Salud Integral Carazo · Proyecto base de Minería de Datos

**Adaptación académica de la estructura SkyTravel, NO de sus datos ni de sus indicadores.**
Requiere SQL Server 2016+, Python y el archivo `C02_Citas.xlsx` entregado/ampliado por el equipo.
Este paquete no crea citas nuevas ni inventa resultados del modelo. Tampoco modifica el Excel original.

## Objetivo
Validar datos de citas, identificar patrones asociados con la inasistencia y proponer/evaluar un modelo que estime el riesgo de ausencia en citas posteriores.

## Tres KPI y tres datasets

| KPI | Dataset SQL | Fórmula/resultado |
|---|---|---|
| PRV · Porcentaje de Registros Validados | `dm_dataset_calidad` | filas válidas / filas evaluadas × 100 |
| TIRV · Tasa de Inasistencia Validada | `dm_dataset_patrones` | citas sin asistencia / citas validadas con resultado conocido × 100 |
| Recall · Detección de Inasistencias | `dm_dataset_prediccion` + `dm_resultados_prediccion` | verdaderos positivos / (verdaderos positivos + falsos negativos) × 100 |

La tasa preliminar de 38,42 % (136/354) pertenece al resumen del dataset **v0** y NO se debe mezclar con la tasa del ampliado (2026). Un registro sintético no es evidencia externa de comportamiento clínico real.

## Orden exacto de ejecución

1. En SSMS ejecutar `sql/01_crear_dw.sql` (conserva tablas preexistentes; no elimina la BD).
2. Crear entorno: `python -m venv venv` y activarlo. Ejecutar `pip install -r requirements.txt`.
3. Copiar `.env.ejemplo` a `.env` y ajustar servidor, driver y modo de autenticación. NO incluir contraseñas en código.
4. Colocar `C02_Citas.xlsx` en la carpeta del proyecto. Cargar: `python src/01_cargar_excel.py C02_Citas.xlsx ampliado`. Si se tiene por separado el v0, también puede cargarse con la etiqueta `v0`; la vista selecciona SOLO el último lote ampliado.
5. En SSMS ejecutar `sql/02_calidad.sql` y revisar el conteo. Los IDs repetidos se **ponen en cuarentena** (no se depuran automáticamente, pues algunos registros repetidos difieren).
6. En SSMS ejecutar `sql/03_datasets.sql` y luego `sql/04_kpis.sql` para los KPI 1 y 2.
7. Ejecutar `python src/03_modelado.py` para entrenar y evaluar. Genera CSV de prueba, JSON de métricas, modelo `.joblib` y guarda sus predicciones en `dbo.dm_resultados_prediccion`.
8. Volver a ejecutar `sql/04_kpis.sql` para ver Recall del KPI 3.

## Límites importantes

- Se validan negativos, resultado 0/1, campos requeridos y claves duplicadas. Otros límites superiores (p. ej. máximo de espera o recordatorios) requieren reglas oficiales del caso; NO se inventan.
- Las 80 filas adicionales con id repetido NO implican automáticamente 80 duplicados exactos; comparar origen y atributos antes de resolver. La versión actual excluye **todos** los registros de IDs repetidos para preservar unicidad, a costa de reducir la muestra.
- No se incluye `fecha_carga` como predictor. Se excluye `recordatorios_enviados` del entrenamiento mientras no esté garantizado que se conocía antes de la cita. El valor `citas_previas_perdidas` existe como atributo, pero el archivo no tiene `id_paciente` para reconstruir historiales y comprobar esa cuenta.
- Se divide por tiempo (enero-abril entrenamiento, mayo validación, junio prueba): los conteos finales dependerán de la depuración.
- Los patrones y métricas resultantes son del **dataset sintético ampliado**, no estimaciones validadas de la población clínica de Carazo.
- Los cálculos no incluyen metas ni valores predictivos inventados: después de la primera evaluación se pueden establecer objetivos técnicos y umbrales institucionales.
- El generador de SkyTravel adjuntado es REFERENCIA ESTRUCTURAL, no se debe ejecutar ni adaptar copiando distribuciones aéreas para pacientes.

## Antes de entregar al docente

Documentar el método/semilla/distribuciones utilizados para generar 9.720 filas adicionales (si ese fue el número efectivamente creado), conservar `v0` inalterado, justificar las reglas de calidad y entregar los datasets SQL separados, el notebook/modelo, la evaluación en test y la ficha técnica SMART de cada KPI.
