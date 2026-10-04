import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from database import engine_sqlserver


warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================

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

UMBRAL = 0.50


# ============================================================
# 2. FUNCIÓN PARA CALCULAR MÉTRICAS
# ============================================================

def evaluar_modelo(
    y_real,
    probabilidades,
    umbral=0.50
):

    predicciones = (
        probabilidades >= umbral
    ).astype(int)

    vn, fp, fn, vp = confusion_matrix(
        y_real,
        predicciones,
        labels=[0, 1]
    ).ravel()

    return {

        "Accuracy":
            accuracy_score(
                y_real,
                predicciones
            ),

        "Precision":
            precision_score(
                y_real,
                predicciones,
                zero_division=0
            ),

        "Recall":
            recall_score(
                y_real,
                predicciones,
                zero_division=0
            ),

        "F1":
            f1_score(
                y_real,
                predicciones,
                zero_division=0
            ),

        "F2":
            fbeta_score(
                y_real,
                predicciones,
                beta=2,
                zero_division=0
            ),

        "ROC_AUC":
            roc_auc_score(
                y_real,
                probabilidades
            ),

        "PR_AUC":
            average_precision_score(
                y_real,
                probabilidades
            ),

        "VP": int(vp),

        "FP": int(fp),

        "FN": int(fn),

        "VN": int(vn),
    }


# ============================================================
# 3. CREAR PREPROCESAMIENTO
# ============================================================

def crear_preprocesador(escalar=False):

    # --------------------------------------------------------
    # Variables numéricas
    # --------------------------------------------------------

    pasos_numericos = [

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ]

    # Algunos modelos se benefician del escalado
    if escalar:

        pasos_numericos.append(

            (
                "scaler",
                StandardScaler()
            )

        )

    numericas = Pipeline(
        pasos_numericos
    )

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
    # Integrar
    # --------------------------------------------------------

    return ColumnTransformer([

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


# ============================================================
# 4. CREAR MODELOS
# ============================================================

def crear_modelos():

    modelos = {

        # ----------------------------------------------------
        # 1. REGRESIÓN LOGÍSTICA
        # ----------------------------------------------------

        "1. Regresión Logística":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=True
                    )
                ),

                (
                    "clasificar",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42
                    )
                ),

            ]),

        # ----------------------------------------------------
        # 2. ÁRBOL DE DECISIÓN
        # ----------------------------------------------------

        "2. Árbol de Decisión":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=False
                    )
                ),

                (
                    "clasificar",
                    DecisionTreeClassifier(
                        max_depth=8,
                        min_samples_split=20,
                        min_samples_leaf=10,
                        class_weight="balanced",
                        random_state=42
                    )
                ),

            ]),

        # ----------------------------------------------------
        # 3. RANDOM FOREST
        #
        # Usamos exactamente la configuración del
        # Random Forest V3 que ya evaluamos.
        # ----------------------------------------------------

        "3. Random Forest":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=False
                    )
                ),

                (
                    "clasificar",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=4,
                        min_samples_leaf=10,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1
                    )
                ),

            ]),

        # ----------------------------------------------------
        # 4. GRADIENT BOOSTING
        # ----------------------------------------------------

        "4. Gradient Boosting":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=False
                    )
                ),

                (
                    "clasificar",
                    GradientBoostingClassifier(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    )
                ),

            ]),

        # ----------------------------------------------------
        # 5. KNN
        # ----------------------------------------------------

        "5. KNN (k=10)":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=True
                    )
                ),

                (
                    "clasificar",
                    KNeighborsClassifier(
                        n_neighbors=10,
                        weights="distance",
                        n_jobs=-1
                    )
                ),

            ]),

        # ----------------------------------------------------
        # 6. NAIVE BAYES
        # ----------------------------------------------------

        "6. Naive Bayes":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=False
                    )
                ),

                (
                    "clasificar",
                    GaussianNB()
                ),

            ]),

        # ----------------------------------------------------
        # 7. SVM
        # ----------------------------------------------------

        "7. SVM (RBF)":

            Pipeline([

                (
                    "preparar",
                    crear_preprocesador(
                        escalar=True
                    )
                ),

                (
                    "clasificar",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42
                    )
                ),

            ]),
    }

    return modelos


# ============================================================
# 5. ANALIZAR INASISTENCIA POR CANAL
# ============================================================

def analizar_canales(
    df,
    salida
):

    print("\n")
    print("=" * 90)
    print("ANÁLISIS DE INASISTENCIA POR CANAL DE AGENDAMIENTO")
    print("=" * 90)

    # --------------------------------------------------------
    # Agrupar por canal
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Calcular tasa
    # --------------------------------------------------------

    canales[
        "tasa_inasistencia_pct"
    ] = (

        100
        *
        canales["inasistencias"]
        /
        canales["total_citas"]

    ).round(2)

    # --------------------------------------------------------
    # Ordenar
    # --------------------------------------------------------

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

    print(
        canales.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Guardar CSV
    # --------------------------------------------------------

    ruta_csv = (
        salida
        / "inasistencia_por_canal.csv"
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
        figsize=(11, 6)
    )

    plt.bar(
        canales[
            "canal_agendamiento"
        ].astype(str),

        canales[
            "tasa_inasistencia_pct"
        ]
    )

    plt.title(
        "Tasa de inasistencia "
        "por canal de agendamiento"
    )

    plt.xlabel(
        "Canal de agendamiento"
    )

    plt.ylabel(
        "Tasa de inasistencia (%)"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    ruta_grafico = (
        salida
        / "inasistencia_por_canal.png"
    )

    plt.savefig(
        ruta_grafico,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Mostrar canal con mayor tasa
    # --------------------------------------------------------

    if not canales.empty:

        mayor = canales.iloc[0]

        print("\nCanal con mayor tasa de inasistencia:")

        print(
            f"{mayor['canal_agendamiento']}"
        )

        print(
            f"Tasa: "
            f"{mayor['tasa_inasistencia_pct']:.2f}%"
        )

        print(
            f"Inasistencias: "
            f"{int(mayor['inasistencias']):,}"
        )

        print(
            f"Total de citas: "
            f"{int(mayor['total_citas']):,}"
        )

    print(
        f"\nResultados guardados en:\n"
        f"{ruta_csv.resolve()}"
    )

    print(
        f"\nGráfico guardado en:\n"
        f"{ruta_grafico.resolve()}"
    )

    return canales


# ============================================================
# 6. GENERAR GRÁFICO COMPARATIVO DE MODELOS
# ============================================================

def grafico_modelos(
    resultados,
    salida
):

    datos = (
        resultados
        .sort_values(
            "ROC_AUC",
            ascending=True
        )
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.barh(
        datos["Modelo"],
        datos["ROC_AUC"]
    )

    plt.xlabel(
        "ROC-AUC"
    )

    plt.ylabel(
        "Modelo"
    )

    plt.title(
        "Comparación de modelos predictivos"
    )

    plt.xlim(
        0,
        1
    )

    plt.tight_layout()

    ruta = (
        salida
        / "comparacion_modelos_roc_auc.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nGráfico comparativo guardado en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 7. GENERAR CURVAS ROC
# ============================================================

def grafico_curvas_roc(
    curvas,
    salida
):

    plt.figure(
        figsize=(9, 7)
    )

    for nombre, datos in curvas.items():

        plt.plot(
            datos["fpr"],
            datos["tpr"],
            linewidth=2,
            label=(
                f"{nombre} "
                f"(AUC={datos['auc']:.3f})"
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
        "Curvas ROC de los modelos"
    )

    plt.legend(
        fontsize=8
    )

    plt.tight_layout()

    ruta = (
        salida
        / "curvas_roc_modelos.png"
    )

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nCurvas ROC guardadas en:\n"
        f"{ruta.resolve()}"
    )


# ============================================================
# 8. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 90)
    print("RED SALUD INTEGRAL CARAZO")
    print("COMPARACIÓN DE MODELOS PREDICTIVOS")
    print("=" * 90)

    # --------------------------------------------------------
    # PASO 1: Carpeta de resultados
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

    # --------------------------------------------------------
    # PASO 2: Cargar dataset
    # --------------------------------------------------------

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
                SELECT *
                FROM dbo.dm_dataset_prediccion
            """),

            con
        )

    print(
        f"Registros cargados: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # PASO 3: Verificar columnas
    # --------------------------------------------------------

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
            f"Faltan columnas: "
            f"{sorted(faltantes)}"
        )

    # --------------------------------------------------------
    # PASO 4: Preparar fecha
    # --------------------------------------------------------

    df["fecha_cita"] = pd.to_datetime(
        df["fecha_cita"],
        errors="coerce"
    )

    df = df.loc[
        df["fecha_cita"].notna()
    ].copy()

    # --------------------------------------------------------
    # PASO 5: Target válido
    # --------------------------------------------------------

    df = df.loc[
        df[OBJETIVO].isin(
            [0, 1]
        )
    ].copy()

    df[OBJETIVO] = (
        df[OBJETIVO]
        .astype(int)
    )

    # --------------------------------------------------------
    # PASO 6: Variables numéricas
    # --------------------------------------------------------

    for columna in NUM:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="coerce"
        )

    # --------------------------------------------------------
    # PASO 7: Mes categórico
    # --------------------------------------------------------

    df["mes"] = (
        df["mes"]
        .astype(str)
    )

    print(
        f"\nDataset listo: "
        f"{len(df):,} registros"
    )

    print(
        f"Tasa general de inasistencia: "
        f"{100 * df[OBJETIVO].mean():.2f}%"
    )

    # ========================================================
    # PASO 8:
    # ANÁLISIS DESCRIPTIVO POR CANAL
    # ========================================================

    analizar_canales(
        df,
        salida
    )

    # ========================================================
    # PASO 9:
    # DIVISIÓN CRONOLÓGICA
    # ========================================================

    print("\n")
    print("=" * 90)
    print("DIVISIÓN CRONOLÓGICA")
    print("=" * 90)

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
                f"El conjunto {nombre} "
                f"está vacío."
            )

        print(
            f"{nombre}: "
            f"{len(parte):,} registros | "
            f"Inasistencia: "
            f"{100 * parte[OBJETIVO].mean():.2f}%"
        )

    # --------------------------------------------------------
    # PASO 10: Crear modelos
    # --------------------------------------------------------

    modelos = crear_modelos()

    print("\n")
    print("=" * 90)
    print(
        f"MODELOS CONFIGURADOS: "
        f"{len(modelos)}"
    )
    print("=" * 90)

    for nombre in modelos:

        print(
            f"- {nombre}"
        )

    # ========================================================
    # PASO 11:
    # ENTRENAR EN ENERO-ABRIL
    # EVALUAR EN MAYO
    # ========================================================

    resultados_validacion = []

    print("\n")
    print("=" * 90)
    print("COMPARACIÓN INICIAL EN VALIDACIÓN - MAYO")
    print("=" * 90)

    for nombre, modelo in modelos.items():

        print(
            f"\nEntrenando: {nombre}"
        )

        inicio = time.time()

        try:

            modelo.fit(
                train[FEATURES],
                train[OBJETIVO]
            )

            probabilidades = (
                modelo
                .predict_proba(
                    val[FEATURES]
                )[:, 1]
            )

            metricas = evaluar_modelo(
                val[OBJETIVO],
                probabilidades,
                UMBRAL
            )

            tiempo = (
                time.time()
                - inicio
            )

            resultados_validacion.append({

                "Modelo":
                    nombre,

                "Accuracy":
                    metricas["Accuracy"],

                "Precision":
                    metricas["Precision"],

                "Recall":
                    metricas["Recall"],

                "F1":
                    metricas["F1"],

                "F2":
                    metricas["F2"],

                "ROC_AUC":
                    metricas["ROC_AUC"],

                "PR_AUC":
                    metricas["PR_AUC"],

                "Tiempo_s":
                    tiempo,
            })

            print(
                f"Accuracy: "
                f"{metricas['Accuracy']:.4f}"
            )

            print(
                f"Precision: "
                f"{metricas['Precision']:.4f}"
            )

            print(
                f"Recall: "
                f"{metricas['Recall']:.4f}"
            )

            print(
                f"ROC-AUC: "
                f"{metricas['ROC_AUC']:.4f}"
            )

            print(
                f"Tiempo: "
                f"{tiempo:.2f}s"
            )

        except Exception as e:

            print(
                f"ERROR: {e}"
            )

    # --------------------------------------------------------
    # PASO 12: Tabla de validación
    # --------------------------------------------------------

    df_val = pd.DataFrame(
        resultados_validacion
    )

    if df_val.empty:

        raise ValueError(
            "Ningún modelo pudo ser evaluado."
        )

    df_val = (
        df_val
        .sort_values(
            "ROC_AUC",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 110)
    print("TABLA COMPARATIVA - VALIDACIÓN MAYO")
    print("=" * 110)

    print(
        df_val.to_string(
            index=False,
            formatters={

                "Accuracy":
                    lambda x: f"{x:.4f}",

                "Precision":
                    lambda x: f"{x:.4f}",

                "Recall":
                    lambda x: f"{x:.4f}",

                "F1":
                    lambda x: f"{x:.4f}",

                "F2":
                    lambda x: f"{x:.4f}",

                "ROC_AUC":
                    lambda x: f"{x:.4f}",

                "PR_AUC":
                    lambda x: f"{x:.4f}",

                "Tiempo_s":
                    lambda x: f"{x:.2f}",
            }
        )
    )

    # --------------------------------------------------------
    # PASO 13:
    # Seleccionar mejor modelo por ROC-AUC en MAYO
    # --------------------------------------------------------

    mejor_nombre = (
        df_val
        .iloc[0]["Modelo"]
    )

    mejor_auc_val = (
        df_val
        .iloc[0]["ROC_AUC"]
    )

    print("\n")
    print("=" * 90)
    print("MODELO SELECCIONADO EN VALIDACIÓN")
    print("=" * 90)

    print(
        f"Modelo: "
        f"{mejor_nombre}"
    )

    print(
        f"ROC-AUC validación: "
        f"{mejor_auc_val:.4f}"
    )

    # ========================================================
    # PASO 14:
    # EVALUACIÓN FINAL EN JUNIO
    #
    # IMPORTANTE:
    # junio no se usa para seleccionar el modelo.
    # ========================================================

    print("\n")
    print("=" * 90)
    print("EVALUACIÓN FINAL EN JUNIO")
    print("=" * 90)

    resultados_prueba = []
    curvas = {}

    for nombre, modelo in modelos.items():

        inicio = time.time()

        try:

            # El modelo ya fue entrenado
            # con enero-abril.
            #
            # Para mantener exactamente la misma lógica
            # que nuestro Random Forest V3, NO reentrenamos
            # con mayo antes de junio.

            probabilidades = (
                modelo
                .predict_proba(
                    test[FEATURES]
                )[:, 1]
            )

            metricas = evaluar_modelo(
                test[OBJETIVO],
                probabilidades,
                UMBRAL
            )

            tiempo = (
                time.time()
                - inicio
            )

            resultados_prueba.append({

                "Modelo":
                    nombre,

                "Accuracy":
                    metricas["Accuracy"],

                "Precision":
                    metricas["Precision"],

                "Recall":
                    metricas["Recall"],

                "F1":
                    metricas["F1"],

                "F2":
                    metricas["F2"],

                "ROC_AUC":
                    metricas["ROC_AUC"],

                "PR_AUC":
                    metricas["PR_AUC"],

                "VP":
                    metricas["VP"],

                "FP":
                    metricas["FP"],

                "FN":
                    metricas["FN"],

                "VN":
                    metricas["VN"],

                "Tiempo_s":
                    tiempo,
            })

            fpr, tpr, _ = roc_curve(
                test[OBJETIVO],
                probabilidades
            )

            curvas[nombre] = {

                "fpr":
                    fpr,

                "tpr":
                    tpr,

                "auc":
                    metricas["ROC_AUC"],
            }

        except Exception as e:

            print(
                f"\nError evaluando "
                f"{nombre} en junio: {e}"
            )

    # --------------------------------------------------------
    # PASO 15: Tabla de prueba
    # --------------------------------------------------------

    df_test = pd.DataFrame(
        resultados_prueba
    )

    df_test = (
        df_test
        .sort_values(
            "ROC_AUC",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    print("\n")
    print("=" * 125)
    print("TABLA COMPARATIVA FINAL - JUNIO")
    print("=" * 125)

    print(
        df_test.to_string(
            index=False,
            formatters={

                "Accuracy":
                    lambda x: f"{x:.4f}",

                "Precision":
                    lambda x: f"{x:.4f}",

                "Recall":
                    lambda x: f"{x:.4f}",

                "F1":
                    lambda x: f"{x:.4f}",

                "F2":
                    lambda x: f"{x:.4f}",

                "ROC_AUC":
                    lambda x: f"{x:.4f}",

                "PR_AUC":
                    lambda x: f"{x:.4f}",

                "Tiempo_s":
                    lambda x: f"{x:.2f}",
            }
        )
    )

    # --------------------------------------------------------
    # PASO 16:
    # Resultado del modelo seleccionado
    # --------------------------------------------------------

    fila_seleccionada = df_test.loc[
        df_test["Modelo"]
        == mejor_nombre
    ]

    print("\n")
    print("=" * 90)
    print("RESULTADO FINAL DEL MODELO SELECCIONADO")
    print("=" * 90)

    if not fila_seleccionada.empty:

        fila = (
            fila_seleccionada
            .iloc[0]
        )

        print(
            f"Modelo seleccionado: "
            f"{mejor_nombre}"
        )

        print(
            f"Accuracy: "
            f"{fila['Accuracy']:.4f}"
        )

        print(
            f"Precision: "
            f"{fila['Precision']:.4f}"
        )

        print(
            f"Recall: "
            f"{fila['Recall']:.4f}"
        )

        print(
            f"F1: "
            f"{fila['F1']:.4f}"
        )

        print(
            f"F2: "
            f"{fila['F2']:.4f}"
        )

        print(
            f"ROC-AUC: "
            f"{fila['ROC_AUC']:.4f}"
        )

        print(
            f"PR-AUC: "
            f"{fila['PR_AUC']:.4f}"
        )

    # --------------------------------------------------------
    # PASO 17: Guardar tablas
    # --------------------------------------------------------

    ruta_val = (
        salida
        / "comparacion_modelos_validacion.csv"
    )

    ruta_test = (
        salida
        / "comparacion_modelos_prueba.csv"
    )

    df_val.to_csv(
        ruta_val,
        index=False,
        encoding="utf-8-sig"
    )

    df_test.to_csv(
        ruta_test,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # PASO 18: Gráficos
    # --------------------------------------------------------

    grafico_modelos(
        df_test,
        salida
    )

    grafico_curvas_roc(
        curvas,
        salida
    )

    # --------------------------------------------------------
    # PASO 19: Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("PROCESO COMPLETADO")
    print("=" * 90)

    print(
        f"\nModelo seleccionado utilizando "
        f"validación de mayo:\n"
        f"{mejor_nombre}"
    )

    print(
        "\nIMPORTANTE:"
    )

    print(
        "La selección se realizó con mayo. "
        "Junio se conservó como prueba final."
    )

    print(
        "\nArchivos generados:"
    )

    print(
        "- inasistencia_por_canal.csv"
    )

    print(
        "- inasistencia_por_canal.png"
    )

    print(
        "- comparacion_modelos_validacion.csv"
    )

    print(
        "- comparacion_modelos_prueba.csv"
    )

    print(
        "- comparacion_modelos_roc_auc.png"
    )

    print(
        "- curvas_roc_modelos.png"
    )

    print("=" * 90)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":

    main()