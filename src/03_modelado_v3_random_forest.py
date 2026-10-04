import json

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from joblib import dump

from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree

from database import engine_sqlserver


# ============================================================
# 1. CONFIGURACIÓN GENERAL DEL MODELO
# ============================================================

NOMBRE_MODELO = "RandomForest_v3"

OBJETIVO = "no_asistio"

NUM = [
    "espera_dias",
    "citas_previas_perdidas",
]

CAT = [
    "centro_salud",
    "especialidad",
    "canal_agendamiento",
    "grupo_edad",
    "tipo_cobertura",
    "mes",
]

FEATURES = NUM + CAT

# Umbral utilizado para convertir probabilidades en 0 / 1
UMBRAL = 0.50


# ============================================================
# 2. METAS DE LOS KPI
# ============================================================
#
# KPI-01 y KPI-02 son los que ya habían definido.
#
# Las metas de KPI-03 a KPI-10 son propuestas para realizar
# la primera evaluación. Si el profesor establece otros
# valores, se cambian solamente aquí.
#
# IMPORTANTE:
# KPI-02 utiliza una REFERENCIA OPERATIVA, no una meta de
# aprobación del modelo.
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
# 3. FUNCIÓN SEGURA PARA DIVISIONES
# ============================================================

def dividir(numerador, denominador):

    if denominador == 0:
        return 0.0

    return numerador / denominador


# ============================================================
# 4. FUNCIÓN PARA EVALUAR EL MODELO
# ============================================================

def evaluar(y, proba, umbral=0.50):

    pred = (
        proba >= umbral
    ).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    ).ravel()

    return {

        "citas": int(len(y)),

        "VP": int(vp),
        "FP": int(fp),
        "FN": int(fn),
        "VN": int(vn),

        "accuracy_pct": round(
            100 * accuracy_score(
                y,
                pred
            ),
            2
        ),

        "balanced_accuracy_pct": round(
            100 * balanced_accuracy_score(
                y,
                pred
            ),
            2
        ),

        "recall_pct": round(
            100 * recall_score(
                y,
                pred,
                zero_division=0
            ),
            2
        ),

        "precision_pct": round(
            100 * precision_score(
                y,
                pred,
                zero_division=0
            ),
            2
        ),

        "f1_pct": round(
            100 * f1_score(
                y,
                pred,
                zero_division=0
            ),
            2
        ),

        "f2_pct": round(
            100 * fbeta_score(
                y,
                pred,
                beta=2,
                zero_division=0
            ),
            2
        ),

        "roc_auc": round(
            float(
                roc_auc_score(
                    y,
                    proba
                )
            ),
            4
        ),

        "pr_auc": round(
            float(
                average_precision_score(
                    y,
                    proba
                )
            ),
            4
        ),

        "predichas_inasistencias": int(
            pred.sum()
        ),
    }


# ============================================================
# 5. FUNCIÓN PARA CALCULAR LOS 10 KPI
# ============================================================

def calcular_kpis(
    registros_evaluados,
    registros_validados,
    citas_validadas_sin_asistencia,
    citas_validadas_resultado_conocido,
    metricas,
    baseline_accuracy
):

    """
    Calcula los 10 KPI definidos para el proyecto.

    KPI-01:
        Porcentaje de Registros Validados

    KPI-02:
        Tasa de Inasistencia de Registros Validados

    KPI-03:
        Tasa de Detección de Inasistencia (Recall)

    KPI-04:
        Precisión de Predicción de Inasistencia

    KPI-05:
        Exactitud General del Modelo

    KPI-06:
        Tasa de Falsas Alarmas

    KPI-07:
        Tasa de Inasistencias No Detectadas

    KPI-08:
        Rendimiento en Detección de Inasistencias (F2)

    KPI-09:
        Capacidad de Diferenciación (ROC-AUC)

    KPI-10:
        Mejora frente a Clasificación Base
    """

    vp = metricas["VP"]
    fp = metricas["FP"]
    fn = metricas["FN"]
    vn = metricas["VN"]

    # --------------------------------------------------------
    # KPI-01
    # Porcentaje de Registros Validados
    # --------------------------------------------------------

    prv = round(
        100 * dividir(
            registros_validados,
            registros_evaluados
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-02
    # Tasa de Inasistencia de Registros Validados
    # --------------------------------------------------------

    tirv = round(
        100 * dividir(
            citas_validadas_sin_asistencia,
            citas_validadas_resultado_conocido
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-03
    # Recall
    # --------------------------------------------------------

    recall = round(
        100 * dividir(
            vp,
            vp + fn
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-04
    # Precision
    # --------------------------------------------------------

    precision = round(
        100 * dividir(
            vp,
            vp + fp
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-05
    # Accuracy
    # --------------------------------------------------------

    accuracy = round(
        100 * dividir(
            vp + vn,
            vp + vn + fp + fn
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-06
    # Tasa de Falsas Alarmas
    #
    # De quienes realmente asistirían,
    # cuántos fueron clasificados como inasistencia.
    # --------------------------------------------------------

    falsas_alarmas = round(
        100 * dividir(
            fp,
            fp + vn
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-07
    # Tasa de Inasistencias No Detectadas
    # --------------------------------------------------------

    no_detectadas = round(
        100 * dividir(
            fn,
            vp + fn
        ),
        2
    )

    # --------------------------------------------------------
    # KPI-08
    # F2
    # --------------------------------------------------------

    f2 = metricas["f2_pct"]

    # --------------------------------------------------------
    # KPI-09
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = metricas["roc_auc"]

    # --------------------------------------------------------
    # KPI-10
    # Mejora frente al baseline
    # --------------------------------------------------------

    mejora_baseline = round(
        accuracy - baseline_accuracy,
        2
    )

    # ========================================================
    # CONSTRUIR TABLA DE KPI
    # ========================================================

    kpis = [

        {
            "codigo": "KPI-01",
            "nombre": "Porcentaje de Registros Validados",
            "resultado": prv,
            "unidad": "%",
            "meta": f">= {META_KPI_01:.2f}%",
            "estado": (
                "CUMPLE"
                if prv >= META_KPI_01
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-02",
            "nombre":
                "Tasa de Inasistencia de Registros Validados",
            "resultado": tirv,
            "unidad": "%",
            "meta":
                f"Referencia {REFERENCIA_KPI_02:.2f}%",
            "estado": (
                "POR ENCIMA DE REFERENCIA"
                if tirv > REFERENCIA_KPI_02
                else "DENTRO DE REFERENCIA"
            ),
            "tipo": "referencia",
        },

        {
            "codigo": "KPI-03",
            "nombre":
                "Tasa de Detección de Inasistencia (Recall)",
            "resultado": recall,
            "unidad": "%",
            "meta": f">= {META_KPI_03:.2f}%",
            "estado": (
                "CUMPLE"
                if recall >= META_KPI_03
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-04",
            "nombre":
                "Precisión de Predicción de Inasistencia",
            "resultado": precision,
            "unidad": "%",
            "meta": f">= {META_KPI_04:.2f}%",
            "estado": (
                "CUMPLE"
                if precision >= META_KPI_04
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-05",
            "nombre":
                "Exactitud General del Modelo",
            "resultado": accuracy,
            "unidad": "%",
            "meta": f">= {META_KPI_05:.2f}%",
            "estado": (
                "CUMPLE"
                if accuracy >= META_KPI_05
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-06",
            "nombre":
                "Tasa de Falsas Alarmas de Inasistencia",
            "resultado": falsas_alarmas,
            "unidad": "%",
            "meta": f"<= {META_KPI_06:.2f}%",
            "estado": (
                "CUMPLE"
                if falsas_alarmas <= META_KPI_06
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-07",
            "nombre":
                "Tasa de Inasistencias No Detectadas",
            "resultado": no_detectadas,
            "unidad": "%",
            "meta": f"<= {META_KPI_07:.2f}%",
            "estado": (
                "CUMPLE"
                if no_detectadas <= META_KPI_07
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-08",
            "nombre":
                "Rendimiento en Detección de Inasistencias (F2)",
            "resultado": f2,
            "unidad": "%",
            "meta": f">= {META_KPI_08:.2f}%",
            "estado": (
                "CUMPLE"
                if f2 >= META_KPI_08
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-09",
            "nombre":
                "Capacidad de Diferenciación del Modelo (ROC-AUC)",
            "resultado": roc_auc,
            "unidad": "índice",
            "meta": f">= {META_KPI_09:.2f}",
            "estado": (
                "CUMPLE"
                if roc_auc >= META_KPI_09
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },

        {
            "codigo": "KPI-10",
            "nombre":
                "Mejora frente a la Clasificación Base",
            "resultado": mejora_baseline,
            "unidad": "puntos porcentuales",
            "meta": f"> {META_KPI_10:.2f}",
            "estado": (
                "CUMPLE"
                if mejora_baseline > META_KPI_10
                else "NO CUMPLE"
            ),
            "tipo": "cumplimiento",
        },
    ]

    return pd.DataFrame(kpis)


# ============================================================
# 6. MOSTRAR RESULTADOS DE LOS KPI
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

    # Solo KPI que realmente tienen criterio
    # de cumplimiento.
    evaluables = df_kpis[
        df_kpis["tipo"] == "cumplimiento"
    ]

    cumplen = int(
        (
            evaluables["estado"] == "CUMPLE"
        ).sum()
    )

    no_cumplen = int(
        (
            evaluables["estado"] == "NO CUMPLE"
        ).sum()
    )

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
        "\nRequisito indicado para la presentación: "
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
            "NO SE ALCANZA TODAVÍA "
            "EL MÍNIMO DE 4 KPI."
        )

    print("=" * 100)

    return cumplen


# ============================================================
# 7. GRÁFICO DE LOS KPI
# ============================================================

def generar_grafico_kpis(df_kpis, salida):

    # Para un gráfico comparable utilizamos solamente
    # los KPI expresados como porcentaje.
    datos = df_kpis[
        df_kpis["unidad"] == "%"
    ].copy()

    if datos.empty:
        return

    plt.figure(
        figsize=(13, 7)
    )

    plt.bar(
        datos["codigo"],
        datos["resultado"]
    )

    plt.axhline(
        y=90,
        linestyle="--",
        linewidth=1,
        label="Referencia visual 90%"
    )

    plt.xlabel("KPI")

    plt.ylabel("Resultado (%)")

    plt.title(
        "Resultados de KPI expresados en porcentaje"
    )

    plt.ylim(
        0,
        max(
            100,
            datos["resultado"].max() + 10
        )
    )

    plt.legend()

    plt.tight_layout()

    ruta = (
        salida
        / "grafico_kpis_random_forest_v3.png"
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
# 8. MATRIZ DE CONFUSIÓN
# ============================================================

def generar_matriz_confusion(
    y_real,
    pred,
    salida
):

    matriz = confusion_matrix(
        y_real,
        pred,
        labels=[0, 1]
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        matriz,
        interpolation="nearest"
    )

    plt.title(
        "Matriz de confusión - Random Forest"
    )

    plt.colorbar()

    clases = [
        "Asiste",
        "No asiste"
    ]

    posiciones = range(
        len(clases)
    )

    plt.xticks(
        posiciones,
        clases
    )

    plt.yticks(
        posiciones,
        clases
    )

    plt.xlabel(
        "Predicción del modelo"
    )

    plt.ylabel(
        "Resultado real"
    )

    for i in range(
        matriz.shape[0]
    ):

        for j in range(
            matriz.shape[1]
        ):

            plt.text(
                j,
                i,
                str(matriz[i, j]),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=14
            )

    plt.tight_layout()

    ruta = (
        salida
        / "matriz_confusion_random_forest_v3.png"
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
# 9. CURVA ROC
# ============================================================

def generar_curva_roc(
    y_real,
    proba,
    salida
):

    fpr, tpr, _ = roc_curve(
        y_real,
        proba
    )

    auc = roc_auc_score(
        y_real,
        proba
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Random Forest (AUC = {auc:.4f})"
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
        "Curva ROC - Random Forest"
    )

    plt.legend()

    plt.tight_layout()

    ruta = (
        salida
        / "curva_roc_random_forest_v3.png"
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
# 10. IMPORTANCIA DE VARIABLES
# ============================================================

def generar_importancia_variables(
    modelo,
    salida,
    top_n=15
):

    preprocesador = modelo.named_steps[
        "preparar"
    ]

    random_forest = modelo.named_steps[
        "clasificar"
    ]

    nombres = (
        preprocesador
        .get_feature_names_out()
    )

    importancias = (
        random_forest
        .feature_importances_
    )

    importancia = pd.DataFrame({

        "variable": nombres,

        "importancia": importancias,

    })

    importancia = (
        importancia
        .sort_values(
            "importancia",
            ascending=False
        )
        .head(top_n)
        .sort_values(
            "importancia",
            ascending=True
        )
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.barh(
        importancia["variable"],
        importancia["importancia"]
    )

    plt.xlabel(
        "Importancia"
    )

    plt.ylabel(
        "Variable"
    )

    plt.title(
        "Variables con mayor importancia "
        "en Random Forest"
    )

    plt.tight_layout()

    ruta = (
        salida
        / "importancia_variables_random_forest_v3.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nImportancia de variables guardada en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 11. ÁRBOL REPRESENTATIVO DEL RANDOM FOREST
# ============================================================

def generar_arbol_visual(
    modelo,
    salida
):

    print(
        "\nGenerando visualización "
        "de un árbol del Random Forest..."
    )

    preprocesador = modelo.named_steps[
        "preparar"
    ]

    nombres_variables = (
        preprocesador
        .get_feature_names_out()
    )

    random_forest = modelo.named_steps[
        "clasificar"
    ]

    arbol = (
        random_forest
        .estimators_[0]
    )

    print(
        f"Total de árboles en el Random Forest: "
        f"{len(random_forest.estimators_)}"
    )

    print(
        "Árbol seleccionado para visualizar: 1"
    )

    plt.figure(
        figsize=(30, 16)
    )

    plot_tree(
        arbol,
        feature_names=nombres_variables,
        class_names=[
            "Asiste",
            "No asiste"
        ],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,
        impurity=True,
        proportion=False,
        precision=2,
    )

    plt.title(
        "Árbol de decisión No. 1 "
        "del modelo Random Forest",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()

    ruta_arbol = (
        salida
        / "arbol_random_forest_v3.png"
    )

    plt.savefig(
        ruta_arbol,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Árbol guardado correctamente en:\n"
        f"{ruta_arbol.resolve()}"
    )


# ============================================================
# 12. GENERAR INFORME RESUMIDO
# ============================================================

def generar_informe(
    salida,
    df_kpis,
    met_test,
    registros_evaluados,
    registros_validados,
    cumplen
):

    ruta = (
        salida
        / "informe_resumido_random_forest_v3.txt"
    )

    lineas = []

    lineas.append(
        "=" * 72
    )

    lineas.append(
        "RED SALUD INTEGRAL CARAZO"
    )

    lineas.append(
        "INFORME RESUMIDO - RANDOM FOREST V3"
    )

    lineas.append(
        "=" * 72
    )

    lineas.append("")

    lineas.append(
        "OBJETIVO:"
    )

    lineas.append(
        "Estimar la probabilidad de inasistencia "
        "a las citas médicas."
    )

    lineas.append("")

    lineas.append(
        "DATOS:"
    )

    lineas.append(
        f"Registros evaluados: "
        f"{registros_evaluados:,}"
    )

    lineas.append(
        f"Registros validados: "
        f"{registros_validados:,}"
    )

    lineas.append("")

    lineas.append(
        "CONFIGURACIÓN DEL MODELO:"
    )

    lineas.append(
        "Modelo: Random Forest"
    )

    lineas.append(
        "Número de árboles: 300"
    )

    lineas.append(
        "Profundidad máxima: 4"
    )

    lineas.append(
        "Mínimo de registros por hoja: 10"
    )

    lineas.append(
        'Variables por división: sqrt'
    )

    lineas.append(
        f"Umbral de clasificación: {UMBRAL}"
    )

    lineas.append("")

    lineas.append(
        "MATRIZ DE CONFUSIÓN:"
    )

    lineas.append(
        f"VP: {met_test['VP']}"
    )

    lineas.append(
        f"FP: {met_test['FP']}"
    )

    lineas.append(
        f"FN: {met_test['FN']}"
    )

    lineas.append(
        f"VN: {met_test['VN']}"
    )

    lineas.append("")

    lineas.append(
        "RESULTADOS DE LOS KPI:"
    )

    for _, fila in df_kpis.iterrows():

        lineas.append(
            f"{fila['codigo']} - "
            f"{fila['nombre']}: "
            f"{fila['resultado']} "
            f"{fila['unidad']} | "
            f"{fila['meta']} | "
            f"{fila['estado']}"
        )

    lineas.append("")

    lineas.append(
        f"KPI con criterio de cumplimiento "
        f"que alcanzan su meta: {cumplen}"
    )

    if cumplen >= 4:

        lineas.append(
            "Se alcanza el requisito mínimo "
            "de cuatro KPI cumplidos."
        )

    else:

        lineas.append(
            "Todavía no se alcanza el requisito "
            "mínimo de cuatro KPI cumplidos."
        )

    lineas.append("")

    lineas.append(
        "NOTA:"
    )

    lineas.append(
        "KPI-02 se interpreta frente a una "
        "referencia operativa del 15%; "
        "no se contabiliza automáticamente "
        "como KPI aprobado o reprobado."
    )

    lineas.append("")

    lineas.append(
        "=" * 72
    )

    ruta.write_text(
        "\n".join(lineas),
        encoding="utf-8"
    )

    print(
        f"\nInforme resumido guardado en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 13. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 68)
    print("RED SALUD INTEGRAL CARAZO")
    print("MODELO PREDICTIVO V3 - RANDOM FOREST")
    print("EVALUACIÓN DE 10 KPI")
    print("=" * 68)

    # --------------------------------------------------------
    # PASO 1: Crear carpeta de resultados
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
        "\nCarpeta de resultados:"
    )

    print(
        salida.resolve()
    )

    # --------------------------------------------------------
    # PASO 2: Conectar con SQL Server
    # --------------------------------------------------------

    print(
        "\nConectando con SQL Server..."
    )

    engine = engine_sqlserver()

    with engine.connect() as con:

        base = con.exec_driver_sql(
            "SELECT DB_NAME()"
        ).scalar()

        print(
            f"Base conectada: {base}"
        )

        # Dataset analítico
        df = pd.read_sql_query(

            text("""
                SELECT *
                FROM dbo.dm_dataset_prediccion
            """),

            con
        )

        # ----------------------------------------------------
        # Obtener cantidad original de registros evaluados
        # para KPI-01.
        # ----------------------------------------------------

        try:

            registros_evaluados = int(
                con.execute(
                    text("""
                        SELECT COUNT(*)
                        FROM dbo.raw_cita
                    """)
                ).scalar_one()
            )

        except Exception:

            print(
                "\nAVISO: No fue posible consultar "
                "dbo.raw_cita."
            )

            print(
                "KPI-01 utilizará temporalmente "
                "el total del dataset predictivo."
            )

            registros_evaluados = int(
                len(df)
            )

    print(
        f"\nRegistros cargados en dataset predictivo: "
        f"{len(df):,}"
    )

    print(
        f"Registros evaluados originalmente: "
        f"{registros_evaluados:,}"
    )

    # --------------------------------------------------------
    # PASO 3: Verificar columnas necesarias
    # --------------------------------------------------------

    requeridas = set(
        FEATURES
        + [
            OBJETIVO,
            "id_cita",
            "fecha_cita"
        ]
    )

    faltantes = requeridas.difference(
        df.columns
    )

    if faltantes:

        raise ValueError(
            f"Faltan columnas: "
            f"{sorted(faltantes)}"
        )

    print(
        "\nColumnas requeridas verificadas."
    )

    # --------------------------------------------------------
    # PASO 4: Verificar identificadores
    # --------------------------------------------------------

    if (
        df["id_cita"].isna().any()
        or
        df["id_cita"].duplicated().any()
    ):

        raise ValueError(
            "El dataset predictivo contiene "
            "id_cita nulos o duplicados."
        )

    print(
        "Identificadores de citas verificados."
    )

    # --------------------------------------------------------
    # PASO 5: Preparar fechas
    # --------------------------------------------------------

    df["fecha_cita"] = pd.to_datetime(
        df["fecha_cita"],
        errors="coerce"
    )

    if df["fecha_cita"].isna().any():

        raise ValueError(
            "Existen fechas de cita "
            "no interpretables."
        )

    print(
        "Fechas verificadas correctamente."
    )

    # --------------------------------------------------------
    # PASO 6:
    # Guardar información para KPI-01 y KPI-02
    # ANTES de dividir entrenamiento / prueba.
    # --------------------------------------------------------

    mascara_resultado = (
        df[OBJETIVO].isin(
            [0, 1]
        )
    )

    df_resultado_conocido = (
        df.loc[
            mascara_resultado
        ]
        .copy()
    )

    # Para este código, consideramos validados los registros
    # que llegaron al dataset analítico y poseen resultado
    # conocido.
    #
    # Si 02_calidad.sql posee una bandera específica como
    # es_valido = 1, posteriormente podemos sustituir este
    # cálculo por esa bandera.
    registros_validados = int(
        len(
            df_resultado_conocido
        )
    )

    citas_validadas_resultado_conocido = int(
        len(
            df_resultado_conocido
        )
    )

    citas_validadas_sin_asistencia = int(
        (
            df_resultado_conocido[
                OBJETIVO
            ] == 1
        ).sum()
    )

    print(
        "\nInformación para KPI-01 y KPI-02:"
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
        f"{citas_validadas_resultado_conocido:,}"
    )

    print(
        f"Citas validadas sin asistencia: "
        f"{citas_validadas_sin_asistencia:,}"
    )

    # --------------------------------------------------------
    # PASO 7: Preparar variable objetivo
    # --------------------------------------------------------

    df = (
        df.loc[
            mascara_resultado
        ]
        .copy()
    )

    df[OBJETIVO] = (
        df[OBJETIVO]
        .astype(int)
    )

    print(
        "\nDistribución de la variable objetivo:"
    )

    print(
        df[OBJETIVO]
        .value_counts()
    )

    print(
        "\nTasa general de inasistencia: "
        f"{100 * df[OBJETIVO].mean():.2f}%"
    )

    # --------------------------------------------------------
    # PASO 8: Preparar variables numéricas
    # --------------------------------------------------------

    for columna in NUM:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="raise"
        )

    # --------------------------------------------------------
    # PASO 9: Preparar mes como categórica
    # --------------------------------------------------------

    df["mes"] = (
        df["mes"]
        .astype(str)
    )

    # --------------------------------------------------------
    # PASO 10: División cronológica
    # --------------------------------------------------------

    print(
        "\nDividiendo dataset cronológicamente..."
    )

    # Entrenamiento: enero - abril
    train = df.loc[
        df["fecha_cita"].ge(
            "2026-01-01"
        )
        &
        df["fecha_cita"].lt(
            "2026-05-01"
        )
    ].copy()

    # Validación: mayo
    val = df.loc[
        df["fecha_cita"].ge(
            "2026-05-01"
        )
        &
        df["fecha_cita"].lt(
            "2026-06-01"
        )
    ].copy()

    # Prueba final: junio
    test = df.loc[
        df["fecha_cita"].ge(
            "2026-06-01"
        )
        &
        df["fecha_cita"].lt(
            "2026-07-01"
        )
    ].copy()

    # --------------------------------------------------------
    # PASO 11: Verificar conjuntos
    # --------------------------------------------------------

    for nombre, parte in (

        (
            "entrenamiento",
            train
        ),

        (
            "validación",
            val
        ),

        (
            "prueba",
            test
        ),

    ):

        if (
            parte.empty
            or
            parte[OBJETIVO].nunique() != 2
        ):

            raise ValueError(
                f"{nombre}: está vacío "
                "o no contiene ambas clases."
            )

        print(
            f"{nombre}: "
            f"{len(parte):,} citas; "
            f"inasistencia "
            f"{100 * parte[OBJETIVO].mean():.2f}%"
        )

    # --------------------------------------------------------
    # PASO 12: Pipeline numérico
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
    # PASO 13: Pipeline categórico
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
                handle_unknown="ignore"
            )
        ),

    ])

    # --------------------------------------------------------
    # PASO 14: Integrar preparación
    # --------------------------------------------------------

    preparar = ColumnTransformer([

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

    # --------------------------------------------------------
    # PASO 15: Crear Random Forest
    # --------------------------------------------------------

    modelo = Pipeline([

        (
            "preparar",
            preparar
        ),

        (
            "clasificar",

            RandomForestClassifier(

                n_estimators=300,

                max_depth=4,

                min_samples_leaf=10,

                max_features="sqrt",

                random_state=42,

                n_jobs=-1,
            )
        ),

    ])

    # --------------------------------------------------------
    # PASO 16: Entrenar
    # --------------------------------------------------------

    print(
        "\nEntrenando Random Forest..."
    )

    modelo.fit(
        train[FEATURES],
        train[OBJETIVO]
    )

    print(
        "Entrenamiento completado."
    )

    # --------------------------------------------------------
    # PASO 17: Generar árbol
    # --------------------------------------------------------

    generar_arbol_visual(
        modelo,
        salida
    )

    # --------------------------------------------------------
    # PASO 18: Validación
    # --------------------------------------------------------

    prob_val = (
        modelo
        .predict_proba(
            val[FEATURES]
        )[:, 1]
    )

    met_val = evaluar(
        val[OBJETIVO],
        prob_val,
        UMBRAL
    )

    print(
        "\nRESULTADOS DE VALIDACIÓN (MAYO):"
    )

    print(
        json.dumps(
            met_val,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # PASO 19: Prueba final
    # --------------------------------------------------------

    prob_test = (
        modelo
        .predict_proba(
            test[FEATURES]
        )[:, 1]
    )

    pred_test = (
        prob_test >= UMBRAL
    ).astype(int)

    met_test = evaluar(
        test[OBJETIVO],
        prob_test,
        UMBRAL
    )

    # --------------------------------------------------------
    # PASO 20: Baseline
    # --------------------------------------------------------

    base_accuracy = round(

        100
        *
        max(
            test[OBJETIVO].mean(),
            1 - test[OBJETIVO].mean()
        ),

        2
    )

    # --------------------------------------------------------
    # PASO 21: Información adicional
    # --------------------------------------------------------

    met_test.update({

        "modelo":
            NOMBRE_MODELO,

        "umbral":
            UMBRAL,

        "entrenamiento":
            int(len(train)),

        "validacion":
            int(len(val)),

        "prueba":
            int(len(test)),

        "baseline_clase_mayoritaria_accuracy_pct":
            base_accuracy,

        "intervalo_solicitado_80_85_pct":
            (
                80
                <= met_test["accuracy_pct"]
                <= 85
            ),
    })

    # --------------------------------------------------------
    # PASO 22: Mostrar resultado final
    # --------------------------------------------------------

    print(
        "\nRESULTADOS FINALES EN JUNIO (PRUEBA):"
    )

    print(
        json.dumps(
            met_test,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # PASO 23: CALCULAR LOS 10 KPI
    # --------------------------------------------------------

    df_kpis = calcular_kpis(

        registros_evaluados=
            registros_evaluados,

        registros_validados=
            registros_validados,

        citas_validadas_sin_asistencia=
            citas_validadas_sin_asistencia,

        citas_validadas_resultado_conocido=
            citas_validadas_resultado_conocido,

        metricas=
            met_test,

        baseline_accuracy=
            base_accuracy
    )

    # --------------------------------------------------------
    # PASO 24: Mostrar KPI
    # --------------------------------------------------------

    cumplen = mostrar_kpis(
        df_kpis
    )

    # --------------------------------------------------------
    # PASO 25: Guardar tabla de KPI
    # --------------------------------------------------------

    ruta_kpis = (
        salida
        / "kpis_random_forest_v3.csv"
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

    # --------------------------------------------------------
    # PASO 26: Generar gráficos
    # --------------------------------------------------------

    generar_grafico_kpis(
        df_kpis,
        salida
    )

    generar_matriz_confusion(
        test[OBJETIVO],
        pred_test,
        salida
    )

    generar_curva_roc(
        test[OBJETIVO],
        prob_test,
        salida
    )

    generar_importancia_variables(
        modelo,
        salida
    )

    # --------------------------------------------------------
    # PASO 27: Preparar predicciones
    # --------------------------------------------------------

    res = test[
        [
            "id_cita",
            "fecha_cita",
            OBJETIVO
        ]
    ].copy()

    res["fecha_cita"] = (
        res["fecha_cita"]
        .dt
        .date
    )

    res[
        "prob_inasistencia"
    ] = prob_test

    res[
        "pred_inasistencia"
    ] = pred_test

    res[
        "conjunto"
    ] = "prueba"

    res[
        "nombre_modelo"
    ] = NOMBRE_MODELO

    # --------------------------------------------------------
    # PASO 28: Guardar predicciones SQL Server
    # --------------------------------------------------------

    print(
        "\nGuardando predicciones "
        "en SQL Server..."
    )

    with engine.begin() as con:

        con.execute(

            text("""
                DELETE
                FROM dbo.dm_resultados_prediccion
                WHERE nombre_modelo = :nombre
                  AND conjunto = 'prueba'
            """),

            {
                "nombre":
                    NOMBRE_MODELO
            },
        )

        res.to_sql(

            "dm_resultados_prediccion",

            con=con,

            schema="dbo",

            if_exists="append",

            index=False,

            chunksize=500,
        )

    # --------------------------------------------------------
    # PASO 29: Guardar predicciones CSV
    # --------------------------------------------------------

    ruta_predicciones = (
        salida
        / "predicciones_prueba_random_forest_v3.csv"
    )

    res.to_csv(
        ruta_predicciones,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # PASO 30: Guardar métricas JSON
    # --------------------------------------------------------

    ruta_metricas = (
        salida
        / "metricas_random_forest_v3.json"
    )

    ruta_metricas.write_text(

        json.dumps(

            {
                "validacion":
                    met_val,

                "prueba":
                    met_test,

                "kpi":
                    df_kpis.to_dict(
                        orient="records"
                    ),
            },

            indent=2,

            ensure_ascii=False,
        ),

        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PASO 31: Guardar modelo
    # --------------------------------------------------------

    ruta_modelo = (
        salida
        / "modelo_random_forest_v3.joblib"
    )

    dump(

        {
            "modelo":
                modelo,

            "umbral":
                UMBRAL,

            "features":
                FEATURES,

            "kpis":
                df_kpis.to_dict(
                    orient="records"
                ),
        },

        ruta_modelo,
    )

    # --------------------------------------------------------
    # PASO 32: Generar informe resumido
    # --------------------------------------------------------

    generar_informe(

        salida,

        df_kpis,

        met_test,

        registros_evaluados,

        registros_validados,

        cumplen
    )

    # --------------------------------------------------------
    # PASO 33: Resultado final
    # --------------------------------------------------------

    print("\n")
    print("=" * 68)
    print("PROCESO COMPLETADO")
    print("=" * 68)

    print(
        "\nPredicciones guardadas en "
        "dbo.dm_resultados_prediccion."
    )

    print(
        "\nArchivos locales guardados en:"
    )

    print(
        salida.resolve()
    )

    print(
        "\nArchivos generados:"
    )

    print(
        "- predicciones_prueba_random_forest_v3.csv"
    )

    print(
        "- metricas_random_forest_v3.json"
    )

    print(
        "- kpis_random_forest_v3.csv"
    )

    print(
        "- modelo_random_forest_v3.joblib"
    )

    print(
        "- arbol_random_forest_v3.png"
    )

    print(
        "- matriz_confusion_random_forest_v3.png"
    )

    print(
        "- curva_roc_random_forest_v3.png"
    )

    print(
        "- importancia_variables_random_forest_v3.png"
    )

    print(
        "- grafico_kpis_random_forest_v3.png"
    )

    print(
        "- informe_resumido_random_forest_v3.txt"
    )

    print("=" * 68)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()