
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from database import engine_sqlserver



CAMPOS_OBLIGATORIOS = [
    "id_cita",
    "fecha_cita",
    "centro_salud",
    "especialidad",
    "canal_agendamiento",
    "grupo_edad",
    "espera_dias",
    "citas_previas_perdidas",
    "recordatorios_enviados",
    "no_asistio"
]


def normalizar_ids(serie):
    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
    )


def validar_calidad(anio=None):

    print("=" * 65)
    print("RED SALUD INTEGRAL CARAZO")
    print("PIPELINE DE VALIDACIÓN DE CALIDAD")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. EXTRAER LOS DATOS DEL DATA WAREHOUSE
    # --------------------------------------------------------

    engine = engine_sqlserver()

    with engine.connect() as conexion:

        raw = pd.read_sql_query(
            text("SELECT * FROM dbo.raw_cita"),
            conexion
        )

        predictivo = pd.read_sql_query(
            text("""
                SELECT id_cita, fecha_cita
                FROM dbo.dm_dataset_prediccion
            """),
            conexion
        )

    # --------------------------------------------------------
    # 2. VALIDAR PERÍODO SOLICITADO
    # --------------------------------------------------------

    raw["fecha_cita"] = pd.to_datetime(
        raw["fecha_cita"], errors="coerce"
    )

    predictivo["fecha_cita"] = pd.to_datetime(
        predictivo["fecha_cita"], errors="coerce"
    )

    if anio is not None:

        raw = raw.loc[
            raw["fecha_cita"].dt.year == anio
        ].copy()

        predictivo = predictivo.loc[
            predictivo["fecha_cita"].dt.year == anio
        ].copy()

    if raw.empty:
        raise ValueError(
            f"No existen registros originales para "
            f"el período solicitado: {anio}. "
            "Se detiene la validación de ese período."
        )

    print("\nPeríodo solicitado:", anio or "Todos")
    print("Registros originales:", len(raw))
    print("Registros predictivos:", len(predictivo))

    # --------------------------------------------------------
    # 3. KPI 1: PORCENTAJE DE REGISTROS VALIDADOS
    # Correspondencia con el dataset predictivo
    # --------------------------------------------------------

    raw_ids = normalizar_ids(raw["id_cita"])
    dm_ids = normalizar_ids(predictivo["id_cita"])

    conjunto_dm = set(dm_ids.dropna())

    tiene_correspondencia = raw_ids.isin(conjunto_dm)

    registros_correspondientes = int(
        tiene_correspondencia.sum()
    )

    kpi_prv = (
        100 * registros_correspondientes / len(raw)
    )

    # --------------------------------------------------------
    # 4. KPI 2: PORCENTAJE DE COMPLETITUD
    # --------------------------------------------------------

    campos_presentes = pd.DataFrame(index=raw.index)

    for campo in CAMPOS_OBLIGATORIOS:

        presente = raw[campo].notna()

        if pd.api.types.is_object_dtype(raw[campo]):

            presente = presente & (
                raw[campo]
                .astype("string")
                .str.strip()
                .ne("")
                .fillna(False)
            )

        campos_presentes[campo] = presente

    registros_completos = campos_presentes.all(axis=1)

    kpi_completitud = (
        100 * registros_completos.sum() / len(raw)
    )

    # --------------------------------------------------------
    # 5. KPI 3: PORCENTAJE DE VALIDEZ LÓGICA
    # Se evalúa entre los registros completos.
    # --------------------------------------------------------

    espera = pd.to_numeric(
        raw["espera_dias"], errors="coerce"
    )

    previas = pd.to_numeric(
        raw["citas_previas_perdidas"], errors="coerce"
    )

    recordatorios = pd.to_numeric(
        raw["recordatorios_enviados"], errors="coerce"
    )

    resultado = (
        raw["no_asistio"]
        .astype("string")
        .str.strip()
    )

    # Reglas lógicas mínimas.
    # Los límites máximos deberán definirse
    # según las reglas oficiales del proyecto.
    reglas_validas = (
        raw_ids.notna()
        & raw["fecha_cita"].notna()
        & espera.ge(0)
        & previas.ge(0)
        & recordatorios.ge(0)
        & resultado.isin(["0", "1"])
    ).fillna(False)

    completos = int(registros_completos.sum())

    validos_entre_completos = int(
        (registros_completos & reglas_validas).sum()
    )

    kpi_validez = (
        100 * validos_entre_completos / completos
        if completos > 0
        else 0.0
    )

    # --------------------------------------------------------
    # 6. CONSTRUIR TABLA DE KPI
    # --------------------------------------------------------

    indicadores = pd.DataFrame([
        {
            "KPI": "Registros validados (PRV)",
            "Resultado (%)": round(kpi_prv, 2),
            "Meta": ">= 90%",
            "Estado": (
                "CUMPLE"
                if kpi_prv >= 90
                else "NO CUMPLE"
            )
        },
        {
            "KPI": "Completitud",
            "Resultado (%)": round(kpi_completitud, 2),
            "Meta": "Por definir",
            "Estado": "PROPUESTO"
        },
        {
            "KPI": "Validez lógica entre completos",
            "Resultado (%)": round(kpi_validez, 2),
            "Meta": "Por definir",
            "Estado": "PROPUESTO"
        }
    ])

    # --------------------------------------------------------
    # 7. REGISTRAR HALLAZGOS AGREGADOS
    # --------------------------------------------------------

    hallazgos = pd.DataFrame([
        {
            "Hallazgo": "Sin correspondencia predictiva",
            "Cantidad": int((~tiene_correspondencia).sum()),
            "Accion": "Investigar reglas y causas de exclusión"
        },
        {
            "Hallazgo": "Registros incompletos",
            "Cantidad": int((~registros_completos).sum()),
            "Accion": "Revisar los campos obligatorios"
        },
        {
            "Hallazgo": "Registros completos con reglas inválidas",
            "Cantidad": int(
                (registros_completos & ~reglas_validas).sum()
            ),
            "Accion": "Revisar rangos y valores permitidos"
        },
        {
            "Hallazgo": "Identificadores duplicados en origen",
            "Cantidad": int(
                raw_ids[
                    raw_ids.notna()
                ].duplicated(keep=False).sum()
            ),
            "Accion": "Revisar posibles duplicidades"
        }
    ])

    # --------------------------------------------------------
    # 8. GUARDAR EVIDENCIA DE LA VALIDACIÓN
    # --------------------------------------------------------

    carpeta = (
        Path(__file__).resolve().parent.parent
        / "resultados_jupyter"
    )

    carpeta.mkdir(parents=True, exist_ok=True)

    momento = datetime.now(timezone.utc)

    marca = momento.strftime("%Y%m%dT%H%M%SZ")

    indicadores.to_csv(
        carpeta / f"calidad_kpis_{marca}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    hallazgos.to_csv(
        carpeta / f"calidad_hallazgos_{marca}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    trazabilidad = pd.DataFrame([{
        "fecha_ejecucion_utc": momento.isoformat(),
        "tabla_origen": "dbo.raw_cita",
        "tabla_predictiva": "dbo.dm_dataset_prediccion",
        "periodo_solicitado": (
            str(anio) if anio is not None else "Todos"
        ),
        "registros_originales": len(raw),
        "registros_predictivos": len(predictivo),
        "registros_correspondientes": registros_correspondientes
    }])

    trazabilidad.to_csv(
        carpeta / f"calidad_trazabilidad_{marca}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nValidación ejecutada.")
    print("Evidencias guardadas en:", carpeta)

    return {
        "indicadores": indicadores,
        "hallazgos": hallazgos,
        "trazabilidad": trazabilidad
    }

# ============================================================
# PASO 22. CONTROL PREVIO A MINERÍA DE DATOS
# ============================================================

def verificar_periodo_mineria(anio=2026):

    print("=" * 65)
    print("CONTROL DE CALIDAD ANTES DEL ENTRENAMIENTO")
    print("=" * 65)

    # 1. Consultar registros del período solicitado
    consulta = text("""
        SELECT
            id_cita,
            fecha_cita,
            centro_salud,
            especialidad,
            canal_agendamiento,
            grupo_edad,
            espera_dias,
            citas_previas_perdidas,
            recordatorios_enviados,
            no_asistio
        FROM dbo.dm_dataset_prediccion
        WHERE YEAR(fecha_cita) = :anio
    """)

    engine = engine_sqlserver()

    with engine.connect() as conexion:
        datos = pd.read_sql_query(
            consulta,
            conexion,
            params={"anio": anio}
        )

    hallazgos = []

    print("\nAño solicitado:", anio)
    print("Registros disponibles:", len(datos))

    # 2. Comprobar disponibilidad
    if datos.empty:
        hallazgos.append(
            "No existen registros predictivos "
            "para el año solicitado."
        )

        entrenamiento = 0
        validacion = 0
        prueba = 0

    else:
        # 3. Comprobar campos obligatorios
        obligatorios = [
            "id_cita",
            "fecha_cita",
            "centro_salud",
            "especialidad",
            "canal_agendamiento",
            "grupo_edad",
            "espera_dias",
            "citas_previas_perdidas",
            "recordatorios_enviados",
            "no_asistio"
        ]

        faltantes = datos[obligatorios].isna().any(axis=1)

        for campo in [
            "id_cita",
            "centro_salud",
            "especialidad",
            "canal_agendamiento",
            "grupo_edad"
        ]:
            vacios = (
                datos[campo]
                .astype("string")
                .str.strip()
                .eq("")
                .fillna(False)
            )

            faltantes = faltantes | vacios

        if faltantes.any():
            hallazgos.append(
                f"{int(faltantes.sum())} registros "
                "presentan campos obligatorios vacíos."
            )

        # 4. Comprobar variable objetivo
        objetivo = pd.to_numeric(
            datos["no_asistio"],
            errors="coerce"
        )

        if not objetivo.isin([0, 1]).all():
            hallazgos.append(
                "Existen valores inválidos en no_asistio."
            )

        # 5. Comprobar rangos numéricos mínimos
        for campo in [
            "espera_dias",
            "citas_previas_perdidas",
            "recordatorios_enviados"
        ]:
            valores = pd.to_numeric(
                datos[campo],
                errors="coerce"
            )

            if valores.isna().any() or (valores < 0).any():
                hallazgos.append(
                    f"El campo {campo} contiene "
                    "valores faltantes, no numéricos "
                    "o negativos."
                )

        # 6. Comprobar identificadores duplicados
        ids = normalizar_ids(datos["id_cita"])

        if ids.duplicated(keep=False).any():
            hallazgos.append(
                "Existen identificadores repetidos "
                "en el dataset predictivo."
            )

        # 7. Comprobar períodos necesarios
        fechas = pd.to_datetime(
            datos["fecha_cita"],
            errors="coerce"
        )

        entrenamiento = int(
            fechas.dt.month.isin([1, 2, 3, 4]).sum()
        )

        validacion = int(
            (fechas.dt.month == 5).sum()
        )

        prueba = int(
            (fechas.dt.month == 6).sum()
        )

        if min(entrenamiento, validacion, prueba) == 0:
            hallazgos.append(
                "Faltan registros en entrenamiento, "
                "validación o prueba."
            )

        # Comprobar ambas clases en cada partición
        for nombre, mascara in [
            ("Entrenamiento", fechas.dt.month.isin([1, 2, 3, 4])),
            ("Validación", fechas.dt.month == 5),
            ("Prueba", fechas.dt.month == 6)
        ]:
            if objetivo[mascara].nunique() < 2:
                hallazgos.append(
                    f"{nombre} no contiene ambas clases."
                )

    # 8. Determinar aptitud del período
    apto = len(hallazgos) == 0

    print("\nPARTICIÓN TEMPORAL")
    print("Entrenamiento:", entrenamiento)
    print("Validación:", validacion)
    print("Prueba:", prueba)

    if apto:
        print("\nESTADO: APTO PARA ENTRENAMIENTO")
        print("Las comprobaciones esenciales se superaron.")
    else:
        print("\nESTADO: NO APTO PARA ENTRENAMIENTO")

        for hallazgo in hallazgos:
            print("-", hallazgo)

    # 9. Registrar evidencia de la comprobación
    carpeta = (
        Path(__file__).resolve().parent.parent
        / "resultados_jupyter"
    )

    carpeta.mkdir(parents=True, exist_ok=True)

    marca = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    evidencia = pd.DataFrame([{
        "fecha_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "anio_solicitado": anio,
        "tabla_origen": "dbo.dm_dataset_prediccion",
        "registros": len(datos),
        "entrenamiento": entrenamiento,
        "validacion": validacion,
        "prueba": prueba,
        "estado": "APTO" if apto else "NO APTO",
        "hallazgos": " | ".join(hallazgos)
    }])

    evidencia.to_csv(
        carpeta / f"control_mineria_{anio}_{marca}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return {
        "apto": apto,
        "anio": anio,
        "hallazgos": hallazgos,
        "evidencia": evidencia
    }
