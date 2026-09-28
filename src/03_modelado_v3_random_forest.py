

import json
from pathlib import Path

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

from database import engine_sqlserver

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


def evaluar(y, proba, umbral=0.50):
    pred = (proba >= umbral).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "citas": int(len(y)),
        "VP": int(vp),
        "FP": int(fp),
        "FN": int(fn),
        "VN": int(vn),
        "accuracy_pct": round(100 * accuracy_score(y, pred), 2),
        "balanced_accuracy_pct": round(100 * balanced_accuracy_score(y, pred), 2),
        "recall_pct": round(100 * recall_score(y, pred, zero_division=0), 2),
        "precision_pct": round(100 * precision_score(y, pred, zero_division=0), 2),
        "f1_pct": round(100 * f1_score(y, pred, zero_division=0), 2),
        "f2_pct": round(100 * fbeta_score(y, pred, beta=2, zero_division=0), 2),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "pr_auc": round(float(average_precision_score(y, proba)), 4),
        "predichas_inasistencias": int(pred.sum()),
    }


def main():
    print("=" * 68)
    print("RED SALUD INTEGRAL CARAZO")
    print("MODELO PREDICTIVO V3 - RANDOM FOREST")
    print("=" * 68)

    salida = Path(__file__).resolve().parent.parent / "resultados"
    salida.mkdir(parents=True, exist_ok=True)

    engine = engine_sqlserver()

    with engine.connect() as con:
        base = con.exec_driver_sql("SELECT DB_NAME()").scalar()
        print(f"Base conectada: {base}")
        df = pd.read_sql_query(
            text("SELECT * FROM dbo.dm_dataset_prediccion"), con
        )

    print(f"Registros cargados: {len(df):,}")

    requeridas = set(FEATURES + [OBJETIVO, "id_cita", "fecha_cita"])
    faltantes = requeridas.difference(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    if df["id_cita"].isna().any() or df["id_cita"].duplicated().any():
        raise ValueError("El dataset predictivo contiene id_cita nulos o duplicados.")

    df["fecha_cita"] = pd.to_datetime(df["fecha_cita"], errors="coerce")
    if df["fecha_cita"].isna().any():
        raise ValueError("Existen fechas de cita no interpretables.")

    df = df.loc[df[OBJETIVO].isin([0, 1])].copy()
    df[OBJETIVO] = df[OBJETIVO].astype(int)

    for columna in NUM:
        df[columna] = pd.to_numeric(df[columna], errors="raise")

    # Aunque mes esté almacenado como entero en SQL, aquí se trata como categoría.
    df["mes"] = df["mes"].astype(str)

    train = df.loc[
        df["fecha_cita"].ge("2026-01-01")
        & df["fecha_cita"].lt("2026-05-01")
    ].copy()

    val = df.loc[
        df["fecha_cita"].ge("2026-05-01")
        & df["fecha_cita"].lt("2026-06-01")
    ].copy()

    test = df.loc[
        df["fecha_cita"].ge("2026-06-01")
        & df["fecha_cita"].lt("2026-07-01")
    ].copy()

    for nombre, parte in (
        ("entrenamiento", train),
        ("validacion", val),
        ("prueba", test),
    ):
        if parte.empty or parte[OBJETIVO].nunique() != 2:
            raise ValueError(
                f"{nombre}: está vacío o no contiene ambas clases."
            )
        print(
            f"{nombre}: {len(parte):,} citas; "
            f"inasistencia {100 * parte[OBJETIVO].mean():.2f}%"
        )

    numericas = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categoricas = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preparar = ColumnTransformer([
        ("num", numericas, NUM),
        ("cat", categoricas, CAT),
    ])

    # Hiperparámetros elegidos usando el conjunto de validación (mayo),
    # no modificando las etiquetas ni mirando el resultado individual de junio.
    modelo = Pipeline([
        ("preparar", preparar),
        (
            "clasificar",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    print("\nEntrenando Random Forest...")
    modelo.fit(train[FEATURES], train[OBJETIVO])

    prob_val = modelo.predict_proba(val[FEATURES])[:, 1]
    met_val = evaluar(val[OBJETIVO], prob_val, UMBRAL)

    print("\nRESULTADOS DE VALIDACION (MAYO):")
    print(json.dumps(met_val, indent=2, ensure_ascii=False))

    prob_test = modelo.predict_proba(test[FEATURES])[:, 1]
    pred_test = (prob_test >= UMBRAL).astype(int)
    met_test = evaluar(test[OBJETIVO], prob_test, UMBRAL)

    base_accuracy = round(
        100 * max(test[OBJETIVO].mean(), 1 - test[OBJETIVO].mean()), 2
    )

    met_test.update({
        "modelo": NOMBRE_MODELO,
        "umbral": UMBRAL,
        "entrenamiento": int(len(train)),
        "validacion": int(len(val)),
        "prueba": int(len(test)),
        "baseline_clase_mayoritaria_accuracy_pct": base_accuracy,
        "intervalo_solicitado_80_85_pct": (
            80 <= met_test["accuracy_pct"] <= 85
        ),
    })

    print("\nRESULTADOS FINALES EN JUNIO (PRUEBA):")
    print(json.dumps(met_test, indent=2, ensure_ascii=False))

    if met_test["intervalo_solicitado_80_85_pct"]:
        print("\nOK: el Accuracy final está dentro del intervalo 80-85%.")
    else:
        print(
            "\nAVISO: el Accuracy final quedó fuera de 80-85%. "
            "No se alterarán etiquetas ni predicciones para forzar la cifra."
        )

    res = test[["id_cita", "fecha_cita", OBJETIVO]].copy()
    res["fecha_cita"] = res["fecha_cita"].dt.date
    res["prob_inasistencia"] = prob_test
    res["pred_inasistencia"] = pred_test
    res["conjunto"] = "prueba"
    res["nombre_modelo"] = NOMBRE_MODELO

    with engine.begin() as con:
        con.execute(
            text("""
                DELETE FROM dbo.dm_resultados_prediccion
                WHERE nombre_modelo = :nombre
                  AND conjunto = 'prueba'
            """),
            {"nombre": NOMBRE_MODELO},
        )

        res.to_sql(
            "dm_resultados_prediccion",
            con=con,
            schema="dbo",
            if_exists="append",
            index=False,
            chunksize=500,
        )

    res.to_csv(
        salida / "predicciones_prueba_random_forest_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    (salida / "metricas_random_forest_v3.json").write_text(
        json.dumps(
            {
                "validacion": met_val,
                "prueba": met_test,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    dump(
        {
            "modelo": modelo,
            "umbral": UMBRAL,
            "features": FEATURES,
        },
        salida / "modelo_random_forest_v3.joblib",
    )

    print("\nPredicciones guardadas en dbo.dm_resultados_prediccion.")
    print("Archivos locales guardados en:", salida.resolve())
    print("=" * 68)


if __name__ == "__main__":
    main()
