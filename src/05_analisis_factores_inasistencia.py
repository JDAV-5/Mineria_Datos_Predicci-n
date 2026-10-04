import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sqlalchemy import text

from database import engine_sqlserver


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

OBJETIVO = "no_asistio"

VARIABLE_CITAS_PREVIAS = "citas_previas_perdidas"

VARIABLE_ESPERA = "espera_dias"

VARIABLE_CANAL = "canal_agendamiento"


# ============================================================
# 2. FUNCIÓN AUXILIAR
# ============================================================

def porcentaje(parte, total):

    if total == 0:
        return 0.0

    return (
        parte
        / total
        * 100
    )


# ============================================================
# 3. ANÁLISIS DE CITAS PREVIAS PERDIDAS
# ============================================================

def analizar_citas_previas(
    df,
    salida
):

    print("\n")
    print("=" * 100)
    print("1. ANÁLISIS SEGÚN CITAS PREVIAS PERDIDAS")
    print("=" * 100)

    datos = df.loc[
        df[VARIABLE_CITAS_PREVIAS].notna()
    ].copy()

    datos[VARIABLE_CITAS_PREVIAS] = (
        pd.to_numeric(
            datos[VARIABLE_CITAS_PREVIAS],
            errors="coerce"
        )
    )

    datos = datos.loc[
        datos[VARIABLE_CITAS_PREVIAS].notna()
    ].copy()

    # --------------------------------------------------------
    # Agrupar
    # --------------------------------------------------------

    resumen = (

        datos
        .groupby(
            VARIABLE_CITAS_PREVIAS
        )
        .agg(

            total_citas=(
                OBJETIVO,
                "count"
            ),

            inasistencias=(
                OBJETIVO,
                "sum"
            ),

        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Calcular asistencias
    # --------------------------------------------------------

    resumen["asistencias"] = (

        resumen["total_citas"]
        - resumen["inasistencias"]
    )

    # --------------------------------------------------------
    # Tasa de inasistencia
    # --------------------------------------------------------

    resumen[
        "tasa_inasistencia_pct"
    ] = (

        100
        *
        resumen["inasistencias"]
        /
        resumen["total_citas"]

    ).round(2)

    resumen = (

        resumen
        .sort_values(
            VARIABLE_CITAS_PREVIAS
        )
        .reset_index(
            drop=True
        )
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ruta_csv = (
        salida
        / "analisis_citas_previas_perdidas.csv"
    )

    resumen.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        resumen[
            VARIABLE_CITAS_PREVIAS
        ].astype(str),

        resumen[
            "tasa_inasistencia_pct"
        ]
    )

    plt.xlabel(
        "Cantidad de citas previas perdidas"
    )

    plt.ylabel(
        "Tasa de inasistencia (%)"
    )

    plt.title(
        "Tasa de inasistencia según citas previas perdidas"
    )

    # Mostrar porcentaje encima de cada barra
    for i, valor in enumerate(
        resumen["tasa_inasistencia_pct"]
    ):

        plt.text(
            i,
            valor + 1,
            f"{valor:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.ylim(
        0,
        min(
            100,
            max(
                resumen[
                    "tasa_inasistencia_pct"
                ].max() + 10,
                50
            )
        )
    )

    plt.tight_layout()

    ruta_png = (
        salida
        / "analisis_citas_previas_perdidas.png"
    )

    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Identificar extremos
    # --------------------------------------------------------

    if not resumen.empty:

        menor = resumen.loc[
            resumen[
                "tasa_inasistencia_pct"
            ].idxmin()
        ]

        mayor = resumen.loc[
            resumen[
                "tasa_inasistencia_pct"
            ].idxmax()
        ]

        print("\nMenor tasa observada:")

        print(
            f"{menor[VARIABLE_CITAS_PREVIAS]:.0f} "
            f"citas previas perdidas -> "
            f"{menor['tasa_inasistencia_pct']:.2f}%"
        )

        print("\nMayor tasa observada:")

        print(
            f"{mayor[VARIABLE_CITAS_PREVIAS]:.0f} "
            f"citas previas perdidas -> "
            f"{mayor['tasa_inasistencia_pct']:.2f}%"
        )

    print(
        f"\nCSV guardado en:\n"
        f"{ruta_csv.resolve()}"
    )

    print(
        f"\nGráfico guardado en:\n"
        f"{ruta_png.resolve()}"
    )

    return resumen


# ============================================================
# 4. ANÁLISIS DE DÍAS DE ESPERA
# ============================================================

def analizar_dias_espera(
    df,
    salida
):

    print("\n")
    print("=" * 100)
    print("2. ANÁLISIS SEGÚN DÍAS DE ESPERA")
    print("=" * 100)

    datos = df.loc[
        df[VARIABLE_ESPERA].notna()
    ].copy()

    datos[VARIABLE_ESPERA] = (
        pd.to_numeric(
            datos[VARIABLE_ESPERA],
            errors="coerce"
        )
    )

    datos = datos.loc[
        datos[VARIABLE_ESPERA].notna()
    ].copy()

    # --------------------------------------------------------
    # Mostrar rango real
    # --------------------------------------------------------

    minimo = datos[
        VARIABLE_ESPERA
    ].min()

    maximo = datos[
        VARIABLE_ESPERA
    ].max()

    print(
        f"\nRango observado de días de espera: "
        f"{minimo:.0f} a {maximo:.0f} días"
    )

    # --------------------------------------------------------
    # Crear rangos
    #
    # 0-3
    # 4-7
    # 8-14
    # 15-30
    # 31+
    # --------------------------------------------------------

    condiciones = [

        datos[
            VARIABLE_ESPERA
        ].between(
            0,
            3,
            inclusive="both"
        ),

        datos[
            VARIABLE_ESPERA
        ].between(
            4,
            7,
            inclusive="both"
        ),

        datos[
            VARIABLE_ESPERA
        ].between(
            8,
            14,
            inclusive="both"
        ),

        datos[
            VARIABLE_ESPERA
        ].between(
            15,
            30,
            inclusive="both"
        ),

        datos[
            VARIABLE_ESPERA
        ] >= 31,

    ]

    etiquetas = [

        "0-3 días",

        "4-7 días",

        "8-14 días",

        "15-30 días",

        "31+ días",

    ]

    datos[
        "rango_espera"
    ] = np.select(
        condiciones,
        etiquetas,
        default="Fuera de rango"
    )

    # --------------------------------------------------------
    # Separar valores fuera de rango
    # --------------------------------------------------------

    fuera_rango = datos.loc[
        datos["rango_espera"]
        == "Fuera de rango"
    ]

    if len(fuera_rango) > 0:

        print(
            f"\nADVERTENCIA: "
            f"{len(fuera_rango):,} registros "
            f"quedaron fuera de los rangos esperados."
        )

    datos_validos = datos.loc[
        datos["rango_espera"]
        != "Fuera de rango"
    ].copy()

    # --------------------------------------------------------
    # Orden lógico
    # --------------------------------------------------------

    datos_validos[
        "rango_espera"
    ] = pd.Categorical(

        datos_validos[
            "rango_espera"
        ],

        categories=etiquetas,

        ordered=True
    )

    # --------------------------------------------------------
    # Agrupar
    # --------------------------------------------------------

    resumen = (

        datos_validos
        .groupby(
            "rango_espera",
            observed=True
        )
        .agg(

            total_citas=(
                OBJETIVO,
                "count"
            ),

            inasistencias=(
                OBJETIVO,
                "sum"
            ),

            espera_promedio=(
                VARIABLE_ESPERA,
                "mean"
            ),

        )
        .reset_index()
    )

    resumen[
        "asistencias"
    ] = (

        resumen["total_citas"]
        - resumen["inasistencias"]
    )

    resumen[
        "tasa_inasistencia_pct"
    ] = (

        100
        *
        resumen["inasistencias"]
        /
        resumen["total_citas"]

    ).round(2)

    resumen[
        "espera_promedio"
    ] = (

        resumen[
            "espera_promedio"
        ]
        .round(2)
    )

    print("\n")

    print(
        resumen.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ruta_csv = (
        salida
        / "analisis_dias_espera.csv"
    )

    resumen.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        resumen[
            "rango_espera"
        ].astype(str),

        resumen[
            "tasa_inasistencia_pct"
        ]
    )

    plt.xlabel(
        "Días de espera"
    )

    plt.ylabel(
        "Tasa de inasistencia (%)"
    )

    plt.title(
        "Tasa de inasistencia según días de espera"
    )

    for i, valor in enumerate(
        resumen[
            "tasa_inasistencia_pct"
        ]
    ):

        plt.text(
            i,
            valor + 1,
            f"{valor:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.ylim(
        0,
        min(
            100,
            max(
                resumen[
                    "tasa_inasistencia_pct"
                ].max() + 10,
                50
            )
        )
    )

    plt.tight_layout()

    ruta_png = (
        salida
        / "analisis_dias_espera.png"
    )

    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Mayor y menor tasa
    # --------------------------------------------------------

    if not resumen.empty:

        menor = resumen.loc[
            resumen[
                "tasa_inasistencia_pct"
            ].idxmin()
        ]

        mayor = resumen.loc[
            resumen[
                "tasa_inasistencia_pct"
            ].idxmax()
        ]

        print("\nMenor tasa observada:")

        print(
            f"{menor['rango_espera']} -> "
            f"{menor['tasa_inasistencia_pct']:.2f}%"
        )

        print("\nMayor tasa observada:")

        print(
            f"{mayor['rango_espera']} -> "
            f"{mayor['tasa_inasistencia_pct']:.2f}%"
        )

    print(
        f"\nCSV guardado en:\n"
        f"{ruta_csv.resolve()}"
    )

    print(
        f"\nGráfico guardado en:\n"
        f"{ruta_png.resolve()}"
    )

    return resumen


# ============================================================
# 5. ANÁLISIS POR CANAL
# ============================================================

def analizar_canales(
    df,
    salida
):

    print("\n")
    print("=" * 100)
    print("3. ANÁLISIS SEGÚN CANAL DE AGENDAMIENTO")
    print("=" * 100)

    datos = df.loc[
        df[VARIABLE_CANAL].notna()
    ].copy()

    resumen = (

        datos
        .groupby(
            VARIABLE_CANAL
        )
        .agg(

            total_citas=(
                OBJETIVO,
                "count"
            ),

            inasistencias=(
                OBJETIVO,
                "sum"
            ),

        )
        .reset_index()
    )

    resumen[
        "asistencias"
    ] = (

        resumen["total_citas"]
        - resumen["inasistencias"]
    )

    resumen[
        "tasa_inasistencia_pct"
    ] = (

        100
        *
        resumen["inasistencias"]
        /
        resumen["total_citas"]

    ).round(2)

    resumen = (

        resumen
        .sort_values(
            "tasa_inasistencia_pct",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print(
        resumen.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ruta_csv = (
        salida
        / "analisis_canal_agendamiento.csv"
    )

    resumen.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        resumen[
            VARIABLE_CANAL
        ].astype(str),

        resumen[
            "tasa_inasistencia_pct"
        ]
    )

    plt.xlabel(
        "Canal de agendamiento"
    )

    plt.ylabel(
        "Tasa de inasistencia (%)"
    )

    plt.title(
        "Tasa de inasistencia según canal de agendamiento"
    )

    plt.xticks(
        rotation=25,
        ha="right"
    )

    for i, valor in enumerate(
        resumen[
            "tasa_inasistencia_pct"
        ]
    ):

        plt.text(
            i,
            valor + 1,
            f"{valor:.1f}%",
            ha="center",
            fontsize=9
        )

    plt.ylim(
        0,
        max(
            50,
            resumen[
                "tasa_inasistencia_pct"
            ].max() + 10
        )
    )

    plt.tight_layout()

    ruta_png = (
        salida
        / "analisis_canal_agendamiento.png"
    )

    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return resumen


# ============================================================
# 6. GENERAR INFORME AUTOMÁTICO
# ============================================================

def generar_informe(
    citas_previas,
    espera,
    canales,
    total_registros,
    tasa_general,
    salida
):

    # --------------------------------------------------------
    # Citas previas
    # --------------------------------------------------------

    mayor_previas = citas_previas.loc[
        citas_previas[
            "tasa_inasistencia_pct"
        ].idxmax()
    ]

    menor_previas = citas_previas.loc[
        citas_previas[
            "tasa_inasistencia_pct"
        ].idxmin()
    ]

    # --------------------------------------------------------
    # Espera
    # --------------------------------------------------------

    mayor_espera = espera.loc[
        espera[
            "tasa_inasistencia_pct"
        ].idxmax()
    ]

    menor_espera = espera.loc[
        espera[
            "tasa_inasistencia_pct"
        ].idxmin()
    ]

    # --------------------------------------------------------
    # Canal
    # --------------------------------------------------------

    mayor_canal_tasa = canales.loc[
        canales[
            "tasa_inasistencia_pct"
        ].idxmax()
    ]

    mayor_canal_cantidad = canales.loc[
        canales[
            "inasistencias"
        ].idxmax()
    ]

    contenido = f"""
RED SALUD INTEGRAL CARAZO
ANÁLISIS DE FACTORES ASOCIADOS A LA INASISTENCIA
============================================================

REGISTROS ANALIZADOS:
{total_registros:,}

TASA GENERAL DE INASISTENCIA:
{tasa_general:.2f}%


1. CITAS PREVIAS PERDIDAS

La menor tasa observada correspondió a:
{menor_previas[VARIABLE_CITAS_PREVIAS]:.0f} citas previas perdidas
Tasa: {menor_previas["tasa_inasistencia_pct"]:.2f}%

La mayor tasa observada correspondió a:
{mayor_previas[VARIABLE_CITAS_PREVIAS]:.0f} citas previas perdidas
Tasa: {mayor_previas["tasa_inasistencia_pct"]:.2f}%


2. DÍAS DE ESPERA

Rango con menor tasa observada:
{menor_espera["rango_espera"]}
Tasa: {menor_espera["tasa_inasistencia_pct"]:.2f}%

Rango con mayor tasa observada:
{mayor_espera["rango_espera"]}
Tasa: {mayor_espera["tasa_inasistencia_pct"]:.2f}%


3. CANAL DE AGENDAMIENTO

Canal con mayor tasa de inasistencia:
{mayor_canal_tasa[VARIABLE_CANAL]}
Tasa: {mayor_canal_tasa["tasa_inasistencia_pct"]:.2f}%

Canal con mayor cantidad de inasistencias:
{mayor_canal_cantidad[VARIABLE_CANAL]}
Cantidad: {int(mayor_canal_cantidad["inasistencias"]):,}


4. INTERPRETACIÓN

Este análisis describe patrones observados en los datos.

Una mayor tasa de inasistencia asociada con un determinado
valor o categoría no demuestra que dicho factor sea la causa
de la inasistencia.

Los resultados pueden utilizarse junto con el modelo
predictivo Gradient Boosting para comprender mejor los
patrones asociados al riesgo de inasistencia.

============================================================
"""

    ruta = (
        salida
        / "informe_factores_inasistencia.txt"
    )

    with open(
        ruta,
        "w",
        encoding="utf-8"
    ) as archivo:

        archivo.write(
            contenido.strip()
        )

    print(
        f"\nInforme guardado en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 7. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 100)
    print("RED SALUD INTEGRAL CARAZO")
    print("ANÁLISIS DE FACTORES ASOCIADOS A LA INASISTENCIA")
    print("=" * 100)

    # --------------------------------------------------------
    # Carpeta resultados
    # --------------------------------------------------------

    salida = (
        Path(__file__)
        .resolve()
        .parent
        .parent
        / "resultados"
    )

    salida.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"\nCarpeta de resultados:\n"
        f"{salida.resolve()}"
    )

    # ========================================================
    # 8. CONECTAR A SQL SERVER
    # ========================================================

    print(
        "\nConectando con SQL Server..."
    )

    engine = engine_sqlserver()

    with engine.connect() as con:

        nombre_bd = (
            con
            .exec_driver_sql(
                "SELECT DB_NAME()"
            )
            .scalar()
        )

        print(
            f"Base conectada: "
            f"{nombre_bd}"
        )

        df = pd.read_sql_query(

            text("""
                SELECT
                    id_cita,
                    fecha_cita,
                    canal_agendamiento,
                    espera_dias,
                    citas_previas_perdidas,
                    no_asistio
                FROM dbo.dm_dataset_prediccion
            """),

            con
        )

    print(
        f"\nRegistros cargados: "
        f"{len(df):,}"
    )

    # ========================================================
    # 9. PREPARAR TARGET
    # ========================================================

    df = df.loc[
        df[OBJETIVO].isin(
            [0, 1]
        )
    ].copy()

    df[OBJETIVO] = (
        df[OBJETIVO]
        .astype(int)
    )

    print(
        f"Registros con resultado conocido: "
        f"{len(df):,}"
    )

    tasa_general = (
        df[OBJETIVO].mean()
        * 100
    )

    print(
        f"Tasa general de inasistencia: "
        f"{tasa_general:.2f}%"
    )

    # ========================================================
    # 10. ANÁLISIS
    # ========================================================

    citas_previas = analizar_citas_previas(
        df,
        salida
    )

    espera = analizar_dias_espera(
        df,
        salida
    )

    canales = analizar_canales(
        df,
        salida
    )

    # ========================================================
    # 11. INFORME
    # ========================================================

    generar_informe(
        citas_previas,
        espera,
        canales,
        len(df),
        tasa_general,
        salida
    )

    # ========================================================
    # 12. FINAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print("ANÁLISIS COMPLETADO")
    print("=" * 100)

    print(
        "\nArchivos generados:"
    )

    print(
        "- analisis_citas_previas_perdidas.csv"
    )

    print(
        "- analisis_citas_previas_perdidas.png"
    )

    print(
        "- analisis_dias_espera.csv"
    )

    print(
        "- analisis_dias_espera.png"
    )

    print(
        "- analisis_canal_agendamiento.csv"
    )

    print(
        "- analisis_canal_agendamiento.png"
    )

    print(
        "- informe_factores_inasistencia.txt"
    )

    print("\n")
    print(
        "IMPORTANTE:"
    )

    print(
        "Las diferencias observadas representan "
        "asociaciones en los datos y no prueban causalidad."
    )

    print("=" * 100)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()