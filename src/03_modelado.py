import json

from pathlib import Path

import numpy as np
import pandas as pd

from sqlalchemy import text

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from joblib import dump

from database import engine_sqlserver


# ============================================================
# 1. CONFIGURACION DEL MODELO
# ============================================================

MODEL_NAME = "RegresionLogistica_v1"

TARGET = "no_asistio"

NUM = [
    "espera_dias",
    "citas_previas_perdidas"
]

CAT = [
    "centro_salud",
    "especialidad",
    "canal_agendamiento",
    "grupo_edad",
    "tipo_cobertura",
    "mes"
]

FEATURES = NUM + CAT


# ============================================================
# 2. FUNCION PARA CALCULAR METRICAS
# ============================================================

def metricas(y, pred, proba):

    matriz = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    )

    vn, fp, fn, vp = matriz.ravel()

    resultados = {

        "total_citas_prueba": int(len(y)),

        "verdaderos_positivos": int(vp),

        "falsos_positivos": int(fp),

        "falsos_negativos": int(fn),

        "verdaderos_negativos": int(vn),

        "recall": float(
            recall_score(
                y,
                pred,
                zero_division=0
            )
        ),

        "precision": float(
            precision_score(
                y,
                pred,
                zero_division=0
            )
        ),

        "f2": float(
            fbeta_score(
                y,
                pred,
                beta=2,
                zero_division=0
            )
        )
    }

    if y.nunique() == 2:

        resultados["roc_auc"] = float(
            roc_auc_score(y, proba)
        )

        resultados["pr_auc"] = float(
            average_precision_score(y, proba)
        )

    else:

        resultados["roc_auc"] = None

        resultados["pr_auc"] = None

    return resultados


# ============================================================
# 3. FUNCION PRINCIPAL
# ============================================================

def main():

    print("=" * 60)

    print("RED SALUD INTEGRAL CARAZO")

    print("MODELO PREDICTIVO DE INASISTENCIA")

    print("=" * 60)


    # --------------------------------------------------------
    # PASO 1: CONECTAR A SQL SERVER
    # --------------------------------------------------------

    print("\nConectando con SQL Server...")

    engine = engine_sqlserver()

    with engine.connect() as con:

        df = pd.read_sql_query(

            text("""
                SELECT *
                FROM dbo.dm_dataset_prediccion
            """),

            con

        )

    print("\nConexion y carga completadas.")

    print(
        f"Registros cargados: {len(df):,}"
    )


    # --------------------------------------------------------
    # PASO 2: VERIFICAR COLUMNAS
    # --------------------------------------------------------

    columnas_necesarias = (
        FEATURES
        + [TARGET, "id_cita", "fecha_cita"]
    )

    faltantes = set(
        columnas_necesarias
    ).difference(df.columns)

    if faltantes:

        raise ValueError(
            f"Columnas faltantes: {sorted(faltantes)}"
        )


    # --------------------------------------------------------
    # PASO 3: PREPARAR FECHAS
    # --------------------------------------------------------

    df["fecha_cita"] = pd.to_datetime(
        df["fecha_cita"],
        errors="coerce"
    )

    if df["fecha_cita"].isna().any():

        raise ValueError(
            "Existen fechas de cita invalidas. "
            "Revisar el dataset predictivo."
        )


    # --------------------------------------------------------
    # PASO 4: PREPARAR VARIABLE OBJETIVO
    # --------------------------------------------------------

    df = df[
        df[TARGET].isin([0, 1])
    ].copy()

    df[TARGET] = df[TARGET].astype(int)

    print("\nDistribucion de la variable objetivo:")

    print(
        df[TARGET].value_counts()
    )

    print(
        "\nTasa general de inasistencia: "
        f"{df[TARGET].mean() * 100:.2f}%"
    )


    # --------------------------------------------------------
    # PASO 5: DIVISION CRONOLOGICA
    # --------------------------------------------------------

    print("\nDividiendo datos cronologicamente...")

    # Entrenamiento: Enero - Abril

    train = df[

        (df["fecha_cita"] >= "2026-01-01")

        &

        (df["fecha_cita"] < "2026-05-01")

    ].copy()


    # Validacion: Mayo

    val = df[

        (df["fecha_cita"] >= "2026-05-01")

        &

        (df["fecha_cita"] < "2026-06-01")

    ].copy()


    # Prueba: Junio

    test = df[

        (df["fecha_cita"] >= "2026-06-01")

        &

        (df["fecha_cita"] < "2026-07-01")

    ].copy()


    print(
        f"\nEntrenamiento: {len(train):,} registros"
    )

    print(
        f"Validacion: {len(val):,} registros"
    )

    print(
        f"Prueba: {len(test):,} registros"
    )


    # --------------------------------------------------------
    # PASO 6: VERIFICAR CONJUNTOS
    # --------------------------------------------------------

    for nombre, conjunto in [

        ("Entrenamiento", train),

        ("Validacion", val),

        ("Prueba", test)

    ]:

        if len(conjunto) == 0:

            raise ValueError(
                f"El conjunto {nombre} esta vacio."
            )

        if conjunto[TARGET].nunique() < 2:

            raise ValueError(
                f"El conjunto {nombre} no contiene "
                "ambas clases de asistencia."
            )


    # --------------------------------------------------------
    # PASO 7: PREPARAR VARIABLES
    # --------------------------------------------------------

    print("\nPreparando variables predictoras...")


    # Variables numericas

    num_pipe = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scale",
            StandardScaler()
        )

    ])


    # Variables categoricas

    cat_pipe = Pipeline([

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
        )

    ])


    # Integracion de transformaciones

    prep = ColumnTransformer([

        (
            "numericas",
            num_pipe,
            NUM
        ),

        (
            "categoricas",
            cat_pipe,
            CAT
        )

    ])


    # --------------------------------------------------------
    # PASO 8: CREAR MODELO
    # --------------------------------------------------------

    print("\nCreando modelo de Regresion Logistica...")

    modelo = Pipeline([

        (
            "preprocesar",
            prep
        ),

        (
            "clasificador",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            )
        )

    ])


    # --------------------------------------------------------
    # PASO 9: ENTRENAMIENTO
    # --------------------------------------------------------

    print("\nEntrenando modelo predictivo...")

    X_train = train[FEATURES]

    y_train = train[TARGET]

    modelo.fit(
        X_train,
        y_train
    )

    print("\nEntrenamiento completado.")


    # --------------------------------------------------------
    # PASO 10: VALIDACION
    # --------------------------------------------------------

    print("\nEvaluando conjunto de validacion...")

    X_val = val[FEATURES]

    y_val = val[TARGET]

    pval = modelo.predict_proba(
        X_val
    )[:, 1]


    # Seleccionar umbral usando F2

    candidatos = np.round(

        np.arange(
            0.20,
            0.801,
            0.02
        ),

        2

    )

    umbral = float(

        max(

            candidatos,

            key=lambda t: fbeta_score(

                y_val,

                (pval >= t).astype(int),

                beta=2,

                zero_division=0

            )

        )

    )

    print(
        f"\nUmbral seleccionado: {umbral:.2f}"
    )


    # --------------------------------------------------------
    # PASO 11: PREDICCION FINAL
    # --------------------------------------------------------

    print("\nRealizando predicciones sobre junio...")

    X_test = test[FEATURES]

    y_test = test[TARGET]

    ptest = modelo.predict_proba(
        X_test
    )[:, 1]

    pred = (
        ptest >= umbral
    ).astype(int)


    # --------------------------------------------------------
    # PASO 12: CALCULAR METRICAS
    # --------------------------------------------------------

    met = metricas(
        y_test,
        pred,
        ptest
    )

    met.update({

        "modelo": MODEL_NAME,

        "umbral": umbral,

        "entrenamiento": int(len(train)),

        "validacion": int(len(val)),

        "prueba": int(len(test))

    })


    print("\nRESULTADOS DEL MODELO")

    print("=" * 60)

    print(
        json.dumps(
            met,
            indent=2,
            ensure_ascii=False
        )
    )


    # --------------------------------------------------------
    # PASO 13: CREAR CARPETA DE RESULTADOS
    # --------------------------------------------------------

    salida = (
        Path(__file__).resolve().parent.parent
        / "resultados"
    )

    salida.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # PASO 14: PREPARAR RESULTADOS
    # --------------------------------------------------------

    res = test[[

        "id_cita",
        "fecha_cita",
        "no_asistio"

    ]].copy()

    res["fecha_cita"] = (
        res["fecha_cita"].dt.date
    )

    res["prob_inasistencia"] = ptest

    res["pred_inasistencia"] = pred

    res["conjunto"] = "prueba"

    res["nombre_modelo"] = MODEL_NAME


    # --------------------------------------------------------
    # PASO 15: EXPORTAR PREDICCIONES
    # --------------------------------------------------------

    res.to_csv(

        salida / "predicciones_prueba.csv",

        index=False,

        encoding="utf-8-sig"

    )


    # --------------------------------------------------------
    # PASO 16: GUARDAR PREDICCIONES EN SQL SERVER
    # --------------------------------------------------------

    print("\nGuardando predicciones en SQL Server...")

    with engine.begin() as con:

        # Eliminar solamente evaluaciones anteriores
        # de este mismo modelo y conjunto de prueba.

        con.execute(

            text("""
                DELETE FROM dbo.dm_resultados_prediccion

                WHERE nombre_modelo = :modelo
                  AND conjunto = :conjunto
            """),

            {

                "modelo": MODEL_NAME,

                "conjunto": "prueba"

            }

        )


        # Insertar las nuevas predicciones

        res.to_sql(

            "dm_resultados_prediccion",

            con=con,

            schema="dbo",

            if_exists="append",

            index=False,

            chunksize=500

        )


    # --------------------------------------------------------
    # PASO 17: GUARDAR METRICAS
    # --------------------------------------------------------

    (
        salida / "metricas_modelo.json"
    ).write_text(

        json.dumps(
            met,
            indent=2,
            ensure_ascii=False
        ),

        encoding="utf-8"

    )


    # --------------------------------------------------------
    # PASO 18: GUARDAR MODELO ENTRENADO
    # --------------------------------------------------------

    dump(

        {

            "modelo": modelo,

            "umbral": umbral,

            "features": FEATURES

        },

        salida / "modelo_inasistencia.joblib"

    )


    print("\n" + "=" * 60)

    print("MODELO PREDICTIVO FINALIZADO")

    print("=" * 60)

    print("\nResultados guardados en:")

    print(salida.resolve())

    print("\nArchivos generados:")

    print("- predicciones_prueba.csv")

    print("- metricas_modelo.json")

    print("- modelo_inasistencia.joblib")

    print(
        "\nPredicciones almacenadas en "
        "dbo.dm_resultados_prediccion"
    )

    print("=" * 60)


# ============================================================
# EJECUCION
# ============================================================

if __name__ == "__main__":

    main()
