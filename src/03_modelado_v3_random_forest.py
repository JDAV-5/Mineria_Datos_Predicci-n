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
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree

from database import engine_sqlserver


# ============================================================
# 1. CONFIGURACIÓN DEL MODELO
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

UMBRAL = 0.50


# ============================================================
# 2. FUNCIÓN PARA EVALUAR EL MODELO
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
            100 * accuracy_score(y, pred),
            2
        ),

        "balanced_accuracy_pct": round(
            100 * balanced_accuracy_score(y, pred),
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
# 3. FUNCIÓN PARA GENERAR UN ÁRBOL DEL RANDOM FOREST
# ============================================================

def generar_arbol_visual(modelo, salida):

    print(
        "\nGenerando visualización de un árbol "
        "del Random Forest..."
    )

    # --------------------------------------------------------
    # Obtener el preprocesador ya entrenado
    # --------------------------------------------------------

    preprocesador = modelo.named_steps[
        "preparar"
    ]

    # --------------------------------------------------------
    # Obtener nombres de todas las variables
    # después del OneHotEncoder
    # --------------------------------------------------------

    nombres_variables = (
        preprocesador.get_feature_names_out()
    )

    # --------------------------------------------------------
    # Obtener el Random Forest entrenado
    # --------------------------------------------------------

    random_forest = modelo.named_steps[
        "clasificar"
    ]

    # --------------------------------------------------------
    # Extraer el primer árbol
    #
    # El Random Forest tiene 300 árboles.
    # estimators_[0] corresponde al árbol número 1.
    # --------------------------------------------------------

    arbol = random_forest.estimators_[0]

    print(
        f"Total de árboles en el Random Forest: "
        f"{len(random_forest.estimators_)}"
    )

    print(
        "Árbol seleccionado para visualizar: 1"
    )

    # --------------------------------------------------------
    # Crear figura
    # --------------------------------------------------------

    plt.figure(
        figsize=(30, 16)
    )

    # --------------------------------------------------------
    # Dibujar árbol
    #
    # max_depth=3 limita solamente la visualización.
    # NO modifica el árbol entrenado.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Ruta de salida
    # --------------------------------------------------------

    ruta_arbol = (
        salida
        / "arbol_random_forest_v3.png"
    )

    # --------------------------------------------------------
    # Guardar imagen
    # --------------------------------------------------------

    plt.savefig(
        ruta_arbol,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Árbol guardado correctamente en:\n"
        f"{ruta_arbol.resolve()}"
    )


# ============================================================
# 4. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 68)
    print("RED SALUD INTEGRAL CARAZO")
    print("MODELO PREDICTIVO V3 - RANDOM FOREST")
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

        df = pd.read_sql_query(

            text("""
                SELECT *
                FROM dbo.dm_dataset_prediccion
            """),

            con
        )

    print(
        f"Registros cargados: {len(df):,}"
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
    # PASO 4: Verificar ID de citas
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
    # PASO 6: Preparar variable objetivo
    # --------------------------------------------------------

    df = df.loc[
        df[OBJETIVO].isin(
            [0, 1]
        )
    ].copy()

    df[OBJETIVO] = df[
        OBJETIVO
    ].astype(int)

    print(
        "\nDistribución de la variable objetivo:"
    )

    print(
        df[OBJETIVO].value_counts()
    )

    print(
        "\nTasa general de inasistencia: "
        f"{100 * df[OBJETIVO].mean():.2f}%"
    )

    # --------------------------------------------------------
    # PASO 7: Preparar variables numéricas
    # --------------------------------------------------------

    for columna in NUM:

        df[columna] = pd.to_numeric(
            df[columna],
            errors="raise"
        )

    # --------------------------------------------------------
    # PASO 8: Preparar mes como variable categórica
    # --------------------------------------------------------

    # Aunque mes esté almacenado como entero
    # en SQL Server, para el modelo se considera
    # una variable categórica.

    df["mes"] = df[
        "mes"
    ].astype(str)

    # --------------------------------------------------------
    # PASO 9: División cronológica
    # --------------------------------------------------------

    print(
        "\nDividiendo dataset cronológicamente..."
    )

    # Entrenamiento:
    # Enero - Abril 2026

    train = df.loc[
        df["fecha_cita"].ge(
            "2026-01-01"
        )
        &
        df["fecha_cita"].lt(
            "2026-05-01"
        )
    ].copy()

    # Validación:
    # Mayo 2026

    val = df.loc[
        df["fecha_cita"].ge(
            "2026-05-01"
        )
        &
        df["fecha_cita"].lt(
            "2026-06-01"
        )
    ].copy()

    # Prueba:
    # Junio 2026

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
    # PASO 10: Verificar conjuntos
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
    # PASO 11: Pipeline para variables numéricas
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
    # PASO 12: Pipeline para variables categóricas
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
    # PASO 13: Integrar preparación de variables
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
    # PASO 14: Crear Random Forest
    # --------------------------------------------------------

    # Los hiperparámetros fueron seleccionados
    # utilizando el conjunto de validación de mayo.
    #
    # Junio se mantiene como conjunto de prueba final.

    modelo = Pipeline([

        (
            "preparar",
            preparar
        ),

        (
            "clasificar",

            RandomForestClassifier(

                # Cantidad de árboles del bosque
                n_estimators=300,

                # Profundidad máxima de cada árbol
                max_depth=4,

                # Número mínimo de registros
                # permitido en una hoja
                min_samples_leaf=10,

                # Cantidad de variables evaluadas
                # en cada división
                max_features="sqrt",

                # Permite reproducibilidad
                random_state=42,

                # Utilizar todos los núcleos disponibles
                n_jobs=-1,
            )
        ),

    ])

    # --------------------------------------------------------
    # PASO 15: Entrenar modelo
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

    # ========================================================
    # PASO 16: GENERAR UN ÁRBOL DEL RANDOM FOREST
    # ========================================================

    generar_arbol_visual(
        modelo,
        salida
    )

    # --------------------------------------------------------
    # PASO 17: Evaluación en validación
    # --------------------------------------------------------

    prob_val = modelo.predict_proba(
        val[FEATURES]
    )[:, 1]

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
    # PASO 18: Predicción sobre conjunto de prueba
    # --------------------------------------------------------

    prob_test = modelo.predict_proba(
        test[FEATURES]
    )[:, 1]

    pred_test = (
        prob_test >= UMBRAL
    ).astype(int)

    met_test = evaluar(
        test[OBJETIVO],
        prob_test,
        UMBRAL
    )

    # --------------------------------------------------------
    # PASO 19: Baseline
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
    # PASO 20: Agregar información del modelo
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
    # PASO 21: Mostrar resultados finales
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

    if met_test[
        "intervalo_solicitado_80_85_pct"
    ]:

        print(
            "\nOK: el Accuracy final "
            "está dentro del intervalo 80-85%."
        )

    else:

        print(
            "\nAVISO: el Accuracy final "
            "quedó fuera de 80-85%. "
            "No se alterarán etiquetas "
            "ni predicciones para forzar "
            "la cifra."
        )

    # --------------------------------------------------------
    # PASO 22: Preparar predicciones
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
    # PASO 23: Guardar resultados en SQL Server
    # --------------------------------------------------------

    print(
        "\nGuardando predicciones "
        "en SQL Server..."
    )

    with engine.begin() as con:

        # Eliminar resultados anteriores
        # del mismo modelo y conjunto.

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

        # Insertar las predicciones nuevas

        res.to_sql(

            "dm_resultados_prediccion",

            con=con,

            schema="dbo",

            if_exists="append",

            index=False,

            chunksize=500,
        )

    # --------------------------------------------------------
    # PASO 24: Guardar predicciones como CSV
    # --------------------------------------------------------

    ruta_predicciones = (
        salida
        / "predicciones_prueba_random_forest_v3.csv"
    )

    res.to_csv(

        ruta_predicciones,

        index=False,

        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # PASO 25: Guardar métricas como JSON
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
            },

            indent=2,

            ensure_ascii=False,
        ),

        encoding="utf-8",
    )

    # --------------------------------------------------------
    # PASO 26: Guardar modelo entrenado
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
        },

        ruta_modelo,
    )

    # --------------------------------------------------------
    # PASO 27: Resultado final
    # --------------------------------------------------------

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
        "- modelo_random_forest_v3.joblib"
    )

    print(
        "- arbol_random_forest_v3.png"
    )

    print("=" * 68)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    main()