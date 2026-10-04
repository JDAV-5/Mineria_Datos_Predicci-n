import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from database import engine_sqlserver


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

MODEL_NAME = "GradientBoosting_Final"

OBJETIVO = "no_asistio"

UMBRAL = 0.50


# ------------------------------------------------------------
# Variables numéricas
# ------------------------------------------------------------

NUM = [
    "espera_dias",
    "citas_previas_perdidas",
]


# ------------------------------------------------------------
# Variables categóricas
# ------------------------------------------------------------

CAT = [
    "centro_salud",
    "especialidad",
    "canal_agendamiento",
    "grupo_edad",
    "tipo_cobertura",
    "mes",
]


FEATURES = NUM + CAT


# ============================================================
# 2. METAS DE LOS 10 KPI
#
# IMPORTANTE:
# Son las MISMAS utilizadas anteriormente con Random Forest.
# No se modifican después de conocer los resultados.
# ============================================================

META_KPI_01 = 90.00

REFERENCIA_KPI_02 = 15.00

META_KPI_03 = 70.00

META_KPI_04 = 80.00

META_KPI_05 = 80.00

META_KPI_06 = 20.00

META_KPI_07 = 30.00

META_KPI_08 = 75.00

META_KPI_09 = 0.80

META_KPI_10 = 0.00


# ============================================================
# 3. FUNCIÓN AUXILIAR PARA DIVISIONES SEGURAS
# ============================================================

def dividir(numerador, denominador):

    if denominador == 0:
        return 0.0

    return numerador / denominador


# ============================================================
# 4. CREAR PREPROCESAMIENTO
# ============================================================

def crear_preprocesador():

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    numericas = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

    ])

    # --------------------------------------------------------
    # Variables categóricas
    # --------------------------------------------------------

    categoricas = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        ),

    ])

    # --------------------------------------------------------
    # Combinar
    # --------------------------------------------------------

    preprocesador = ColumnTransformer([

        (
            "num",
            numericas,
            NUM
        ),

        (
            "cat",
            categoricas,
            CAT
        ),

    ])

    return preprocesador


# ============================================================
# 5. CREAR MODELO FINAL
# ============================================================

def crear_modelo():

    modelo = Pipeline([

        (
            "preparar",
            crear_preprocesador()
        ),

        (
            "clasificar",
            GradientBoostingClassifier(

                n_estimators=200,

                learning_rate=0.05,

                max_depth=3,

                random_state=42,
            )
        ),

    ])

    return modelo


# ============================================================
# 6. EVALUAR MODELO
# ============================================================

def evaluar(
    y_real,
    probabilidades,
    umbral=0.50
):

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    matriz = confusion_matrix(
        y_real,
        predicciones,
        labels=[0, 1]
    )

    vn, fp, fn, vp = matriz.ravel()

    accuracy = accuracy_score(
        y_real,
        predicciones
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_real,
            predicciones
        )
    )

    recall = recall_score(
        y_real,
        predicciones,
        zero_division=0
    )

    precision = precision_score(
        y_real,
        predicciones,
        zero_division=0
    )

    f1 = f1_score(
        y_real,
        predicciones,
        zero_division=0
    )

    f2 = fbeta_score(
        y_real,
        predicciones,
        beta=2,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_real,
        probabilidades
    )

    pr_auc = average_precision_score(
        y_real,
        probabilidades
    )

    resultados = {

        "citas":
            int(len(y_real)),

        "VP":
            int(vp),

        "FP":
            int(fp),

        "FN":
            int(fn),

        "VN":
            int(vn),

        "accuracy_pct":
            round(
                accuracy * 100,
                2
            ),

        "balanced_accuracy_pct":
            round(
                balanced_accuracy * 100,
                2
            ),

        "recall_pct":
            round(
                recall * 100,
                2
            ),

        "precision_pct":
            round(
                precision * 100,
                2
            ),

        "f1_pct":
            round(
                f1 * 100,
                2
            ),

        "f2_pct":
            round(
                f2 * 100,
                2
            ),

        "roc_auc":
            round(
                roc_auc,
                4
            ),

        "pr_auc":
            round(
                pr_auc,
                4
            ),

        "predichas_inasistencias":
            int(predicciones.sum()),
    }

    return (
        resultados,
        predicciones
    )


# ============================================================
# 7. CALCULAR LOS 10 KPI
# ============================================================

def calcular_kpis(
    registros_evaluados,
    registros_validados,
    citas_resultado_conocido,
    citas_inasistencia,
    metricas,
    baseline_accuracy
):

    vp = metricas["VP"]

    fp = metricas["FP"]

    fn = metricas["FN"]

    vn = metricas["VN"]


    # ========================================================
    # KPI-01
    # PORCENTAJE DE REGISTROS VALIDADOS
    # ========================================================

    kpi_01 = (
        dividir(
            registros_validados,
            registros_evaluados
        )
        * 100
    )


    # ========================================================
    # KPI-02
    # TASA DE INASISTENCIA DE REGISTROS VALIDADOS
    # ========================================================

    kpi_02 = (
        dividir(
            citas_inasistencia,
            citas_resultado_conocido
        )
        * 100
    )


    # ========================================================
    # KPI-03
    # RECALL
    # ========================================================

    kpi_03 = (
        dividir(
            vp,
            vp + fn
        )
        * 100
    )


    # ========================================================
    # KPI-04
    # PRECISIÓN
    # ========================================================

    kpi_04 = (
        dividir(
            vp,
            vp + fp
        )
        * 100
    )


    # ========================================================
    # KPI-05
    # EXACTITUD GENERAL
    # ========================================================

    kpi_05 = (
        dividir(
            vp + vn,
            vp + vn + fp + fn
        )
        * 100
    )


    # ========================================================
    # KPI-06
    # TASA DE FALSAS ALARMAS
    #
    # FP / (FP + VN)
    # ========================================================

    kpi_06 = (
        dividir(
            fp,
            fp + vn
        )
        * 100
    )


    # ========================================================
    # KPI-07
    # INASISTENCIAS NO DETECTADAS
    #
    # FN / (VP + FN)
    # ========================================================

    kpi_07 = (
        dividir(
            fn,
            vp + fn
        )
        * 100
    )


    # ========================================================
    # KPI-08
    # F2
    # ========================================================

    kpi_08 = metricas["f2_pct"]


    # ========================================================
    # KPI-09
    # ROC-AUC
    # ========================================================

    kpi_09 = metricas["roc_auc"]


    # ========================================================
    # KPI-10
    # MEJORA FRENTE AL BASELINE
    #
    # Accuracy modelo - Accuracy baseline
    # ========================================================

    kpi_10 = (
        metricas["accuracy_pct"]
        - baseline_accuracy
    )


    # ========================================================
    # CONSTRUIR TABLA
    # ========================================================

    datos = [

        {
            "codigo":
                "KPI-01",

            "nombre":
                "Porcentaje de Registros Validados",

            "resultado":
                round(kpi_01, 2),

            "unidad":
                "%",

            "meta":
                f">= {META_KPI_01:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_01 >= META_KPI_01
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-02",

            "nombre":
                "Tasa de Inasistencia de Registros Validados",

            "resultado":
                round(kpi_02, 2),

            "unidad":
                "%",

            "meta":
                (
                    f"Referencia "
                    f"{REFERENCIA_KPI_02:.2f}%"
                ),

            "estado":
                (
                    "POR ENCIMA DE REFERENCIA"
                    if kpi_02 > REFERENCIA_KPI_02
                    else "DENTRO DE REFERENCIA"
                ),

            "tipo":
                "referencia",
        },


        {
            "codigo":
                "KPI-03",

            "nombre":
                "Tasa de Detección de Inasistencia (Recall)",

            "resultado":
                round(kpi_03, 2),

            "unidad":
                "%",

            "meta":
                f">= {META_KPI_03:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_03 >= META_KPI_03
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-04",

            "nombre":
                "Precisión de Predicción de Inasistencia",

            "resultado":
                round(kpi_04, 2),

            "unidad":
                "%",

            "meta":
                f">= {META_KPI_04:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_04 >= META_KPI_04
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-05",

            "nombre":
                "Exactitud General del Modelo",

            "resultado":
                round(kpi_05, 2),

            "unidad":
                "%",

            "meta":
                f">= {META_KPI_05:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_05 >= META_KPI_05
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-06",

            "nombre":
                "Tasa de Falsas Alarmas de Inasistencia",

            "resultado":
                round(kpi_06, 2),

            "unidad":
                "%",

            "meta":
                f"<= {META_KPI_06:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_06 <= META_KPI_06
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-07",

            "nombre":
                "Tasa de Inasistencias No Detectadas",

            "resultado":
                round(kpi_07, 2),

            "unidad":
                "%",

            "meta":
                f"<= {META_KPI_07:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_07 <= META_KPI_07
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-08",

            "nombre":
                "Rendimiento en Detección de Inasistencias (F2)",

            "resultado":
                round(kpi_08, 2),

            "unidad":
                "%",

            "meta":
                f">= {META_KPI_08:.2f}%",

            "estado":
                (
                    "CUMPLE"
                    if kpi_08 >= META_KPI_08
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-09",

            "nombre":
                "Capacidad de Diferenciación del Modelo (ROC-AUC)",

            "resultado":
                round(kpi_09, 4),

            "unidad":
                "índice",

            "meta":
                f">= {META_KPI_09:.2f}",

            "estado":
                (
                    "CUMPLE"
                    if kpi_09 >= META_KPI_09
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },


        {
            "codigo":
                "KPI-10",

            "nombre":
                "Mejora frente a la Clasificación Base",

            "resultado":
                round(kpi_10, 2),

            "unidad":
                "puntos porcentuales",

            "meta":
                f"> {META_KPI_10:.2f}",

            "estado":
                (
                    "CUMPLE"
                    if kpi_10 > META_KPI_10
                    else "NO CUMPLE"
                ),

            "tipo":
                "cumplimiento",
        },

    ]

    return pd.DataFrame(datos)


# ============================================================
# 8. MOSTRAR KPI
# ============================================================

def mostrar_kpis(df_kpis):

    print("\n")
    print("=" * 100)
    print("EVALUACIÓN DE LOS 10 KPI")
    print("=" * 100)

    for _, fila in df_kpis.iterrows():

        print(
            f"\n{fila['codigo']} - "
            f"{fila['nombre']}"
        )

        print(
            f"Resultado: "
            f"{fila['resultado']} "
            f"{fila['unidad']}"
        )

        print(
            f"Meta / referencia: "
            f"{fila['meta']}"
        )

        print(
            f"Estado: "
            f"{fila['estado']}"
        )


    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    evaluables = df_kpis.loc[
        df_kpis["tipo"]
        == "cumplimiento"
    ]

    cumplen = (
        evaluables["estado"]
        == "CUMPLE"
    ).sum()

    no_cumplen = (
        evaluables["estado"]
        == "NO CUMPLE"
    ).sum()

    print("\n")
    print("=" * 100)
    print("RESUMEN DE KPI")
    print("=" * 100)

    print(
        f"KPI totales calculados: "
        f"{len(df_kpis)}"
    )

    print(
        f"KPI con meta de cumplimiento: "
        f"{len(evaluables)}"
    )

    print(
        f"KPI que cumplen: "
        f"{cumplen}"
    )

    print(
        f"KPI que no cumplen: "
        f"{no_cumplen}"
    )

    print(
        "KPI-02: indicador comparado "
        "contra referencia operativa."
    )

    print(
        "\nRequisito indicado para "
        "la presentación: "
        "mínimo 4 KPI cumplidos."
    )

    if cumplen >= 4:

        print(
            "\nRESULTADO GENERAL: "
            "SE ALCANZA EL MÍNIMO DE 4 KPI."
        )

    else:

        print(
            "\nRESULTADO GENERAL: "
            "NO SE ALCANZA EL MÍNIMO DE 4 KPI."
        )

    print("=" * 100)


# ============================================================
# 9. MATRIZ DE CONFUSIÓN
# ============================================================

def graficar_matriz_confusion(
    y_real,
    predicciones,
    salida
):

    matriz = confusion_matrix(
        y_real,
        predicciones,
        labels=[0, 1]
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    imagen = ax.imshow(
        matriz
    )

    fig.colorbar(
        imagen,
        ax=ax
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_xticklabels([
        "Asistencia",
        "Inasistencia"
    ])

    ax.set_yticklabels([
        "Asistencia",
        "Inasistencia"
    ])

    ax.set_xlabel(
        "Predicción"
    )

    ax.set_ylabel(
        "Valor real"
    )

    ax.set_title(
        "Matriz de confusión - Gradient Boosting"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(matriz[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()

    ruta = (
        salida
        / "matriz_confusion_gradient_boosting.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nMatriz de confusión guardada en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 10. CURVA ROC
# ============================================================

def graficar_curva_roc(
    y_real,
    probabilidades,
    roc_auc,
    salida
):

    fpr, tpr, _ = roc_curve(
        y_real,
        probabilidades
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"Gradient Boosting "
            f"(AUC = {roc_auc:.4f})"
        )
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Clasificación aleatoria"
    )

    plt.xlabel(
        "Tasa de falsos positivos"
    )

    plt.ylabel(
        "Tasa de verdaderos positivos"
    )

    plt.title(
        "Curva ROC - Gradient Boosting"
    )

    plt.legend()

    plt.tight_layout()

    ruta = (
        salida
        / "curva_roc_gradient_boosting.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nCurva ROC guardada en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 11. IMPORTANCIA DE VARIABLES
# ============================================================

def graficar_importancia_variables(
    modelo,
    salida
):

    preparador = (
        modelo
        .named_steps["preparar"]
    )

    clasificador = (
        modelo
        .named_steps["clasificar"]
    )

    nombres = (
        preparador
        .get_feature_names_out()
    )

    importancias = (
        clasificador
        .feature_importances_
    )

    df_importancia = pd.DataFrame({

        "variable":
            nombres,

        "importancia":
            importancias,

    })

    df_importancia = (
        df_importancia
        .sort_values(
            "importancia",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Guardar tabla completa
    # --------------------------------------------------------

    ruta_csv = (
        salida
        / "importancia_variables_gradient_boosting.csv"
    )

    df_importancia.to_csv(
        ruta_csv,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Mostrar top 15
    # --------------------------------------------------------

    top = (
        df_importancia
        .head(15)
        .sort_values(
            "importancia",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top["variable"],
        top["importancia"]
    )

    plt.xlabel(
        "Importancia"
    )

    plt.ylabel(
        "Variable"
    )

    plt.title(
        "Variables más importantes - Gradient Boosting"
    )

    plt.tight_layout()

    ruta_png = (
        salida
        / "importancia_variables_gradient_boosting.png"
    )

    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nImportancia de variables guardada en:\n"
        f"{ruta_png.resolve()}"
    )

    print(
        f"\nTabla completa de importancia guardada en:\n"
        f"{ruta_csv.resolve()}"
    )

    return df_importancia


# ============================================================
# 12. ANÁLISIS DE INASISTENCIA POR CANAL
# ============================================================

def analizar_canales(
    df,
    salida
):

    canales = (

        df
        .groupby(
            "canal_agendamiento",
            dropna=False
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

    canales[
        "tasa_inasistencia_pct"
    ] = (

        100
        *
        canales["inasistencias"]
        /
        canales["total_citas"]

    ).round(2)

    canales = (

        canales
        .sort_values(
            "tasa_inasistencia_pct",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 100)
    print("ANÁLISIS DE INASISTENCIA POR CANAL")
    print("=" * 100)

    print(
        canales.to_string(
            index=False
        )
    )

    ruta_csv = (
        salida
        / "inasistencia_por_canal_modelo_final.csv"
    )

    canales.to_csv(
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
        canales[
            "canal_agendamiento"
        ].astype(str),

        canales[
            "tasa_inasistencia_pct"
        ]
    )

    plt.ylabel(
        "Tasa de inasistencia (%)"
    )

    plt.xlabel(
        "Canal de agendamiento"
    )

    plt.title(
        "Tasa de inasistencia por canal de agendamiento"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    ruta_png = (
        salida
        / "inasistencia_por_canal_modelo_final.png"
    )

    plt.savefig(
        ruta_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    if not canales.empty:

        mayor = canales.iloc[0]

        print(
            "\nCanal con mayor TASA "
            "de inasistencia:"
        )

        print(
            f"{mayor['canal_agendamiento']} "
            f"-> "
            f"{mayor['tasa_inasistencia_pct']:.2f}%"
        )

        # Canal con mayor cantidad absoluta
        mayor_cantidad = (

            canales
            .sort_values(
                "inasistencias",
                ascending=False
            )
            .iloc[0]
        )

        print(
            "\nCanal con mayor CANTIDAD "
            "de inasistencias:"
        )

        print(
            f"{mayor_cantidad['canal_agendamiento']} "
            f"-> "
            f"{int(mayor_cantidad['inasistencias']):,} "
            f"inasistencias"
        )

    return canales


# ============================================================
# 13. GRÁFICO DE KPI
# ============================================================

def graficar_kpis(
    df_kpis,
    salida
):

    # --------------------------------------------------------
    # Para evitar mezclar métricas con escalas diferentes,
    # mostramos los KPI porcentuales.
    #
    # NO dibujamos una línea genérica de 90%, porque cada KPI
    # tiene una meta distinta.
    # --------------------------------------------------------

    porcentuales = df_kpis.loc[
        df_kpis["unidad"] == "%"
    ].copy()

    plt.figure(
        figsize=(12, 7)
    )

    posiciones = np.arange(
        len(porcentuales)
    )

    barras = plt.bar(
        posiciones,
        porcentuales["resultado"]
    )

    plt.xticks(
        posiciones,
        porcentuales["codigo"],
        rotation=0
    )

    plt.ylabel(
        "Resultado (%)"
    )

    plt.xlabel(
        "Indicador"
    )

    plt.title(
        "Resultados de KPI porcentuales - Gradient Boosting"
    )

    # --------------------------------------------------------
    # Mostrar resultado encima de cada barra
    # --------------------------------------------------------

    for barra, valor in zip(
        barras,
        porcentuales["resultado"]
    ):

        plt.text(
            barra.get_x()
            + barra.get_width() / 2,

            barra.get_height() + 1,

            f"{valor:.2f}%",

            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylim(
        0,
        max(
            100,
            porcentuales[
                "resultado"
            ].max() + 10
        )
    )

    plt.tight_layout()

    ruta = (
        salida
        / "grafico_kpis_gradient_boosting.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nGráfico de KPI guardado en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 14. CREAR INFORME RESUMIDO
# ============================================================

def crear_informe(
    metricas_validacion,
    metricas_prueba,
    df_kpis,
    canales,
    baseline_accuracy,
    salida
):

    evaluables = df_kpis.loc[
        df_kpis["tipo"]
        == "cumplimiento"
    ]

    cumplen = (
        evaluables["estado"]
        == "CUMPLE"
    ).sum()

    no_cumplen = (
        evaluables["estado"]
        == "NO CUMPLE"
    ).sum()

    mayor_tasa = canales.iloc[0]

    mayor_cantidad = (

        canales
        .sort_values(
            "inasistencias",
            ascending=False
        )
        .iloc[0]
    )

    contenido = f"""
RED SALUD INTEGRAL CARAZO
MODELO FINAL - GRADIENT BOOSTING
============================================================

1. MODELO SELECCIONADO

Modelo:
Gradient Boosting

El modelo fue seleccionado previamente mediante la comparación
de siete algoritmos utilizando el conjunto de validación de mayo.

ROC-AUC de validación:
{metricas_validacion["roc_auc"]:.4f}


2. RESULTADOS FINALES EN JUNIO

Citas evaluadas:
{metricas_prueba["citas"]:,}

Verdaderos positivos:
{metricas_prueba["VP"]:,}

Falsos positivos:
{metricas_prueba["FP"]:,}

Falsos negativos:
{metricas_prueba["FN"]:,}

Verdaderos negativos:
{metricas_prueba["VN"]:,}

Accuracy:
{metricas_prueba["accuracy_pct"]:.2f}%

Precision:
{metricas_prueba["precision_pct"]:.2f}%

Recall:
{metricas_prueba["recall_pct"]:.2f}%

F1:
{metricas_prueba["f1_pct"]:.2f}%

F2:
{metricas_prueba["f2_pct"]:.2f}%

ROC-AUC:
{metricas_prueba["roc_auc"]:.4f}

PR-AUC:
{metricas_prueba["pr_auc"]:.4f}

Accuracy baseline:
{baseline_accuracy:.2f}%


3. ANÁLISIS POR CANAL

Canal con mayor tasa de inasistencia:
{mayor_tasa["canal_agendamiento"]}

Tasa:
{mayor_tasa["tasa_inasistencia_pct"]:.2f}%

Canal con mayor cantidad de inasistencias:
{mayor_cantidad["canal_agendamiento"]}

Cantidad:
{int(mayor_cantidad["inasistencias"]):,}


4. RESULTADO DE LOS KPI

KPI totales:
{len(df_kpis)}

KPI con meta de cumplimiento:
{len(evaluables)}

KPI que cumplen:
{cumplen}

KPI que no cumplen:
{no_cumplen}

KPI-02 se interpreta frente a una referencia operativa.


5. INTERPRETACIÓN GENERAL

El modelo Gradient Boosting permite estimar el riesgo de
inasistencia utilizando información disponible sobre las citas.

Los resultados del análisis por canal describen asociaciones
observadas en los datos. No deben interpretarse como evidencia
de que un canal de agendamiento cause la inasistencia.

La predicción puede utilizarse como apoyo para priorizar
acciones preventivas sobre las citas identificadas con mayor
riesgo, como procesos de confirmación o recordatorio.

============================================================
"""

    ruta = (
        salida
        / "informe_resumido_gradient_boosting.txt"
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
        f"\nInforme resumido guardado en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 15. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 100)
    print("RED SALUD INTEGRAL CARAZO")
    print("MODELO FINAL - GRADIENT BOOSTING")
    print("EVALUACIÓN DE 10 KPI")
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
    # 16. CONECTAR CON SQL SERVER
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

        # ----------------------------------------------------
        # Dataset predictivo
        # ----------------------------------------------------

        df = pd.read_sql_query(

            text("""
                SELECT *
                FROM dbo.dm_dataset_prediccion
            """),

            con
        )

        # ----------------------------------------------------
        # Registros originales
        #
        # Si raw_cita existe, usamos su cantidad.
        # Si no existe, utilizamos el dataset disponible.
        # ----------------------------------------------------

        try:

            registros_evaluados = (
                con
                .execute(
                    text("""
                        SELECT COUNT(*)
                        FROM dbo.raw_cita
                    """)
                )
                .scalar()
            )

        except Exception:

            registros_evaluados = len(df)


    print(
        f"\nRegistros cargados "
        f"en dataset predictivo: "
        f"{len(df):,}"
    )

    print(
        f"Registros evaluados originalmente: "
        f"{registros_evaluados:,}"
    )


    # ========================================================
    # 17. VERIFICAR COLUMNAS
    # ========================================================

    requeridas = set(

        FEATURES
        + [
            OBJETIVO,
            "id_cita",
            "fecha_cita",
        ]
    )

    faltantes = (
        requeridas
        .difference(
            df.columns
        )
    )

    if faltantes:

        raise ValueError(
            f"Faltan columnas requeridas: "
            f"{sorted(faltantes)}"
        )

    print(
        "\nColumnas requeridas verificadas."
    )


    # ========================================================
    # 18. VERIFICAR ID
    # ========================================================

    if df["id_cita"].isna().any():

        raise ValueError(
            "Existen id_cita nulos."
        )

    if df["id_cita"].duplicated().any():

        raise ValueError(
            "Existen id_cita duplicados "
            "en dm_dataset_prediccion."
        )

    print(
        "Identificadores de citas verificados."
    )


    # ========================================================
    # 19. FECHAS
    # ========================================================

    df["fecha_cita"] = pd.to_datetime(
        df["fecha_cita"],
        errors="coerce"
    )

    if df["fecha_cita"].isna().any():

        cantidad = (
            df["fecha_cita"]
            .isna()
            .sum()
        )

        print(
            f"Advertencia: "
            f"{cantidad:,} fechas inválidas."
        )

        df = df.loc[
            df["fecha_cita"].notna()
        ].copy()

    print(
        "Fechas verificadas correctamente."
    )


    # ========================================================
    # 20. TARGET
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


    # ========================================================
    # 21. VARIABLES NUMÉRICAS
    # ========================================================

    for columna in NUM:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )


    # ========================================================
    # 22. MES
    # ========================================================

    df["mes"] = (
        df["mes"]
        .astype(str)
    )


    # ========================================================
    # 23. INFORMACIÓN PARA KPI 01 Y 02
    # ========================================================

    registros_validados = len(df)

    citas_resultado_conocido = len(df)

    citas_inasistencia = int(
        df[OBJETIVO].sum()
    )

    print("\n")
    print(
        "Información para KPI-01 y KPI-02:"
    )

    print(
        f"Registros evaluados: "
        f"{registros_evaluados:,}"
    )

    print(
        f"Registros validados: "
        f"{registros_validados:,}"
    )

    print(
        f"Citas validadas con resultado conocido: "
        f"{citas_resultado_conocido:,}"
    )

    print(
        f"Citas validadas sin asistencia: "
        f"{citas_inasistencia:,}"
    )


    # ========================================================
    # 24. DISTRIBUCIÓN
    # ========================================================

    print(
        "\nDistribución de la variable objetivo:"
    )

    print(
        df[OBJETIVO]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nTasa general de inasistencia: "
        f"{100 * df[OBJETIVO].mean():.2f}%"
    )


    # ========================================================
    # 25. ANÁLISIS POR CANAL
    # ========================================================

    canales = analizar_canales(
        df,
        salida
    )


    # ========================================================
    # 26. DIVISIÓN CRONOLÓGICA
    # ========================================================

    print("\n")
    print("=" * 100)
    print("DIVISIÓN CRONOLÓGICA")
    print("=" * 100)

    train = df.loc[

        df["fecha_cita"].ge(
            "2026-01-01"
        )

        &

        df["fecha_cita"].lt(
            "2026-05-01"
        )

    ].copy()


    val = df.loc[

        df["fecha_cita"].ge(
            "2026-05-01"
        )

        &

        df["fecha_cita"].lt(
            "2026-06-01"
        )

    ].copy()


    test = df.loc[

        df["fecha_cita"].ge(
            "2026-06-01"
        )

        &

        df["fecha_cita"].lt(
            "2026-07-01"
        )

    ].copy()


    for nombre, parte in [

        (
            "Entrenamiento",
            train
        ),

        (
            "Validación",
            val
        ),

        (
            "Prueba",
            test
        ),

    ]:

        if parte.empty:

            raise ValueError(
                f"El conjunto "
                f"{nombre} está vacío."
            )

        print(
            f"{nombre}: "
            f"{len(parte):,} citas | "
            f"inasistencia "
            f"{100 * parte[OBJETIVO].mean():.2f}%"
        )


    # ========================================================
    # 27. CREAR Y ENTRENAR GRADIENT BOOSTING
    # ========================================================

    print("\n")
    print("=" * 100)
    print("ENTRENAMIENTO DEL MODELO FINAL")
    print("=" * 100)

    modelo = crear_modelo()

    print(
        "\nConfiguración:"
    )

    print(
        "Modelo: GradientBoostingClassifier"
    )

    print(
        "n_estimators: 200"
    )

    print(
        "learning_rate: 0.05"
    )

    print(
        "max_depth: 3"
    )

    print(
        "random_state: 42"
    )

    print(
        "\nEntrenando con enero-abril..."
    )

    modelo.fit(
        train[FEATURES],
        train[OBJETIVO]
    )

    print(
        "Entrenamiento completado."
    )


    # ========================================================
    # 28. VALIDACIÓN MAYO
    # ========================================================

    prob_val = (
        modelo
        .predict_proba(
            val[FEATURES]
        )[:, 1]
    )

    (
        metricas_validacion,
        pred_val
    ) = evaluar(
        val[OBJETIVO],
        prob_val,
        UMBRAL
    )

    print("\n")
    print("=" * 100)
    print("RESULTADOS DE VALIDACIÓN - MAYO")
    print("=" * 100)

    print(
        json.dumps(
            metricas_validacion,
            indent=2,
            ensure_ascii=False
        )
    )


    # ========================================================
    # 29. PRUEBA FINAL JUNIO
    # ========================================================

    prob_test = (
        modelo
        .predict_proba(
            test[FEATURES]
        )[:, 1]
    )

    (
        metricas_prueba,
        pred_test
    ) = evaluar(
        test[OBJETIVO],
        prob_test,
        UMBRAL
    )


    # ========================================================
    # 30. BASELINE
    # ========================================================

    clase_mayoritaria = (
        train[OBJETIVO]
        .value_counts()
        .idxmax()
    )

    pred_baseline = np.full(
        len(test),
        clase_mayoritaria
    )

    baseline_accuracy = (
        accuracy_score(
            test[OBJETIVO],
            pred_baseline
        )
        * 100
    )


    # --------------------------------------------------------
    # Información adicional
    # --------------------------------------------------------

    metricas_prueba.update({

        "modelo":
            MODEL_NAME,

        "umbral":
            UMBRAL,

        "entrenamiento":
            int(len(train)),

        "validacion":
            int(len(val)),

        "prueba":
            int(len(test)),

        "baseline_clase_mayoritaria_accuracy_pct":
            round(
                baseline_accuracy,
                2
            ),
    })


    print("\n")
    print("=" * 100)
    print("RESULTADOS FINALES - JUNIO")
    print("=" * 100)

    print(
        json.dumps(
            metricas_prueba,
            indent=2,
            ensure_ascii=False
        )
    )


    # ========================================================
    # 31. CALCULAR LOS 10 KPI
    # ========================================================

    df_kpis = calcular_kpis(

        registros_evaluados=
            registros_evaluados,

        registros_validados=
            registros_validados,

        citas_resultado_conocido=
            citas_resultado_conocido,

        citas_inasistencia=
            citas_inasistencia,

        metricas=
            metricas_prueba,

        baseline_accuracy=
            baseline_accuracy,
    )


    mostrar_kpis(
        df_kpis
    )


    # ========================================================
    # 32. GUARDAR KPI
    # ========================================================

    ruta_kpis = (
        salida
        / "kpis_gradient_boosting.csv"
    )

    df_kpis.to_csv(
        ruta_kpis,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"\nTabla de KPI guardada en:\n"
        f"{ruta_kpis.resolve()}"
    )


    # ========================================================
    # 33. GUARDAR PREDICCIONES
    # ========================================================

    resultados_prediccion = pd.DataFrame({

        "id_cita":
            test["id_cita"]
            .values,

        "fecha_cita":
            test["fecha_cita"]
            .values,

        "real_no_asistio":
            test[OBJETIVO]
            .values,

        "probabilidad_inasistencia":
            prob_test,

        "prediccion_no_asistio":
            pred_test,

        "modelo":
            MODEL_NAME,

    })


    ruta_predicciones = (
        salida
        / "predicciones_prueba_gradient_boosting.csv"
    )

    resultados_prediccion.to_csv(
        ruta_predicciones,
        index=False,
        encoding="utf-8-sig"
    )


    # ========================================================
    # 34. GUARDAR MÉTRICAS JSON
    # ========================================================

    metricas_completas = {

        "validacion":
            metricas_validacion,

        "prueba":
            metricas_prueba,

        "kpis":
            df_kpis.to_dict(
                orient="records"
            ),
    }

    ruta_metricas = (
        salida
        / "metricas_gradient_boosting.json"
    )

    with open(
        ruta_metricas,
        "w",
        encoding="utf-8"
    ) as archivo:

        json.dump(
            metricas_completas,
            archivo,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # 35. GUARDAR MODELO
    # ========================================================

    ruta_modelo = (
        salida
        / "modelo_gradient_boosting.joblib"
    )

    joblib.dump(
        {
            "modelo":
                modelo,

            "features":
                FEATURES,

            "numericas":
                NUM,

            "categoricas":
                CAT,

            "target":
                OBJETIVO,

            "umbral":
                UMBRAL,

            "metricas_validacion":
                metricas_validacion,

            "metricas_prueba":
                metricas_prueba,

            "kpis":
                df_kpis.to_dict(
                    orient="records"
                ),
        },

        ruta_modelo
    )


    # ========================================================
    # 36. GRÁFICOS
    # ========================================================

    graficar_matriz_confusion(
        test[OBJETIVO],
        pred_test,
        salida
    )

    graficar_curva_roc(
        test[OBJETIVO],
        prob_test,
        metricas_prueba["roc_auc"],
        salida
    )

    graficar_importancia_variables(
        modelo,
        salida
    )

    graficar_kpis(
        df_kpis,
        salida
    )


    # ========================================================
    # 37. INFORME RESUMIDO
    # ========================================================

    crear_informe(

        metricas_validacion,
        metricas_prueba,
        df_kpis,
        canales,
        baseline_accuracy,
        salida
    )


    # ========================================================
    # 38. GUARDAR RESULTADOS EN SQL SERVER
    # ========================================================

    print(
        "\nGuardando predicciones "
        "en SQL Server..."
    )

    try:

        resultados_prediccion.to_sql(

            "dm_resultados_gradient_boosting",

            engine,

            schema="dbo",

            if_exists="replace",

            index=False
        )

        print(
            "Predicciones guardadas en "
            "dbo.dm_resultados_gradient_boosting."
        )

    except Exception as e:

        print(
            "\nADVERTENCIA:"
        )

        print(
            "No se pudieron guardar "
            "las predicciones en SQL Server."
        )

        print(
            f"Detalle: {e}"
        )

        print(
            "Los archivos locales sí "
            "fueron generados."
        )


    # ========================================================
    # 39. FINAL
    # ========================================================

    print("\n")
    print("=" * 100)
    print("PROCESO COMPLETADO")
    print("=" * 100)

    print(
        f"\nModelo final:\n"
        f"Gradient Boosting"
    )

    print(
        f"\nArchivos guardados en:\n"
        f"{salida.resolve()}"
    )

    print(
        "\nArchivos generados:"
    )

    print(
        "- predicciones_prueba_gradient_boosting.csv"
    )

    print(
        "- metricas_gradient_boosting.json"
    )

    print(
        "- kpis_gradient_boosting.csv"
    )

    print(
        "- modelo_gradient_boosting.joblib"
    )

    print(
        "- matriz_confusion_gradient_boosting.png"
    )

    print(
        "- curva_roc_gradient_boosting.png"
    )

    print(
        "- importancia_variables_gradient_boosting.png"
    )

    print(
        "- importancia_variables_gradient_boosting.csv"
    )

    print(
        "- grafico_kpis_gradient_boosting.png"
    )

    print(
        "- inasistencia_por_canal_modelo_final.png"
    )

    print(
        "- inasistencia_por_canal_modelo_final.csv"
    )

    print(
        "- informe_resumido_gradient_boosting.txt"
    )

    print("=" * 100)


# ============================================================
# EJECUTAR
# ============================================================

if __name__ == "__main__":

    main()