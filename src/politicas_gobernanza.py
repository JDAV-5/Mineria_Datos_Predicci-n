import pandas as pd

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


# ============================================================
# 2. FUNCIÓN DE NORMALIZACIÓN
# ============================================================

def normalizar(serie):

    """
    Normaliza los valores de texto para facilitar
    las comparaciones.

    - Convierte a texto.
    - Elimina espacios al inicio y al final.
    - Convierte a mayúsculas.

    No modifica los datos originales.
    """

    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
    )


# ============================================================
# 3. FUNCIÓN PRINCIPAL DE VALIDACIÓN
# ============================================================

def evaluar_politicas(datos, catalogos=None):

    # Trabajar sobre una copia de los datos
    df = datos.copy()

    if df.empty:
        raise ValueError(
            "No hay registros disponibles para validar."
        )

    # Comprobar columnas necesarias
    faltan = (
        set(CAMPOS_OBLIGATORIOS)
        - set(df.columns)
    )

    if faltan:
        raise ValueError(
            f"Columnas faltantes: {sorted(faltan)}"
        )

    # Lista de resultados
    resultados = []

    # ========================================================
    # POL-02. POLÍTICA DE COMPLETITUD DE REGISTROS
    # ========================================================

    campos_validos = pd.DataFrame(
        index=df.index
    )

    for campo in CAMPOS_OBLIGATORIOS:

        serie = df[campo]

        # Detectar valores no nulos
        presente = serie.notna()

        # Para campos de texto, detectar cadenas vacías
        if (
            pd.api.types.is_object_dtype(serie)
            or pd.api.types.is_string_dtype(serie)
        ):

            presente = presente & (
                normalizar(serie)
                .ne("")
                .fillna(False)
            )

        campos_validos[campo] = presente

    # ----------------------Identificar registros incompletos---------------------------------------------
    incompletos = ~campos_validos.all(axis=1)

    cantidad_incompletos = int(
        incompletos.sum()
    )

    resultados.append({

        "Politica": "POL-02",

        "Control": "Completitud",

        "Incumplimientos": cantidad_incompletos,

        "Estado": (
            "SIN HALLAZGOS"
            if cantidad_incompletos == 0
            else "CON HALLAZGOS"
        )

    })

    # ========================================================
    # POL-03. POLÍTICA DE ESTANDARIZACIÓN
    # ========================================================
    #
    # Comprueba que los valores pertenezcan a los
    # catálogos proporcionados.
    #
    # Los valores nulos se consideran en POL-02.
    #
    # Los catálogos usados actualmente son provisionales
    # y requieren aprobación institucional.
    #
    # ========================================================

    if catalogos is None:

        resultados.append({

            "Politica": "POL-03",

            "Control": "Estandarización",

            "Incumplimientos": None,

            "Estado": "PENDIENTE DE CATÁLOGOS"

        })

    else:

        if not isinstance(catalogos, dict) or not catalogos:
            raise ValueError(
                "Debe proporcionar un diccionario "
                "de catálogos no vacío."
            )

        errores_catalogo = pd.Series(
            False,
            index=df.index,
            dtype=bool
        )

        # Evaluar cada catálogo
        for campo, valores_permitidos in catalogos.items():

            # Comprobar que exista el campo
            if campo not in df.columns:

                raise ValueError(
                    f"No existe el campo '{campo}'."
                )

            # Normalizar categorías permitidas
            permitidos = {
                str(valor).strip().upper()
                for valor in valores_permitidos
                if pd.notna(valor)
            }

            if not permitidos:

                raise ValueError(
                    f"El catálogo '{campo}' está vacío."
                )

            # Normalizar valores encontrados en SQL Server
            valores = normalizar(
                df[campo]
            )

            # Detectar valores fuera del catálogo
            invalidos = (
                valores.notna()
                & valores.ne("").fillna(False)
                & ~valores.isin(permitidos)
            )

            # Acumular incumplimientos
            errores_catalogo = (
                errores_catalogo
                | invalidos.fillna(False)
            )

        # Total de registros con categorías inválidas
        cantidad_errores = int(
            errores_catalogo.sum()
        )

        # Guardar resultado
        resultados.append({

            "Politica": "POL-03",

            "Control": "Estandarización",

            "Incumplimientos": cantidad_errores,

            "Estado": (
                "SIN HALLAZGOS - CATÁLOGO PROVISIONAL"
                if cantidad_errores == 0
                else "CON HALLAZGOS"
            )

        })

    # ========================================================
    # POL-04. POLÍTICA DE CONTROL DE DUPLICIDAD
    # ========================================================
    #
    # Identifica grupos de citas con el mismo id_cita.
    #
    # No elimina ningún registro duplicado.
    #
    # ========================================================

    ids = normalizar(
        df["id_cita"]
    )

    # Tratar cadenas vacías como valores faltantes
    ids = ids.replace(
        "",
        pd.NA
    )

    # Detectar identificadores repetidos
    duplicados = (
        ids.notna()
        & ids.duplicated(keep=False)
    )

    # Cantidad de identificadores distintos repetidos
    grupos_duplicados = int(
        ids[duplicados].nunique()
    )

    resultados.append({

        "Politica": "POL-04",

        "Control": "Duplicidad",

        "Incumplimientos": grupos_duplicados,

        "Estado": (
            "SIN HALLAZGOS"
            if grupos_duplicados == 0
            else "REQUIERE REVISIÓN"
        )

    })

    # ========================================================
    # POL-05. POLÍTICA DE CONTROL DE VALORES ANÓMALOS
    # ========================================================
    #
    # Comprueba:
    #
    # - Valores numéricos válidos.
    # - Números enteros.
    # - Ausencia de valores negativos.
    # - Variable no_asistio con valores 0 o 1.
    #
    # Los campos faltantes se reportan en POL-02.
    #
    # ========================================================

    errores_rango = pd.Series(
        False,
        index=df.index,
        dtype=bool
    )

    columnas_numericas = [
        "espera_dias",
        "citas_previas_perdidas",
        "recordatorios_enviados"
    ]

    for campo in columnas_numericas:

        # Convertir valores a números
        numero = pd.to_numeric(
            df[campo],
            errors="coerce"
        )

        # Detectar valores no numéricos, negativos
        # o que no sean enteros
        invalidos = (
            df[campo].notna()
            & (
                numero.isna()
                | numero.lt(0)
                | numero.ne(numero.round())
            )
        )

        # Acumular registros inválidos
        errores_rango = (
            errores_rango
            | invalidos.fillna(False)
        )

    # --------------------------------------------------------
    # Validar variable objetivo: no_asistio
    # --------------------------------------------------------

    resultado = normalizar(
        df["no_asistio"]
    )

    # Solo se permiten los valores 0 y 1
    resultado_invalido = (
        resultado.notna()
        & resultado.ne("").fillna(False)
        & ~resultado.isin(["0", "1"])
    )

    errores_rango = (
        errores_rango
        | resultado_invalido.fillna(False)
    )

    # Calcular incumplimientos
    cantidad_anomalos = int(
        errores_rango.sum()
    )

    resultados.append({

        "Politica": "POL-05",

        "Control": "Valores anómalos",

        "Incumplimientos": cantidad_anomalos,

        "Estado": (
            "SIN HALLAZGOS"
            if cantidad_anomalos == 0
            else "CON HALLAZGOS"
        )

    })

    # ========================================================
    # 4. CONSTRUIR TABLA FINAL
    # ========================================================

    tabla_resultados = pd.DataFrame(
        resultados
    )

    return tabla_resultados
