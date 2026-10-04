
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(
    page_title="Red Salud Integral Carazo",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. RUTAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RESULTADOS_DIR = BASE_DIR / "resultados"


# ============================================================
# 3. FUNCIONES PARA CARGAR ARCHIVOS
# ============================================================

def cargar_json(nombre_archivo):
    ruta = RESULTADOS_DIR / nombre_archivo

    if not ruta.exists():
        return None

    try:
        with open(ruta, "r", encoding="utf-8") as archivo:
            return json.load(archivo)
    except Exception as error:
        st.warning(
            f"No se pudo cargar {nombre_archivo}: {error}"
        )
        return None


def cargar_csv(nombre_archivo):
    ruta = RESULTADOS_DIR / nombre_archivo

    if not ruta.exists():
        return None

    try:
        return pd.read_csv(ruta)
    except Exception as error:
        st.warning(
            f"No se pudo cargar {nombre_archivo}: {error}"
        )
        return None


def obtener_valor(diccionario, clave, defecto):
    if diccionario is None:
        return defecto

    valor = diccionario.get(clave)

    if valor is None:
        return defecto

    return valor


# ============================================================
# 4. FUNCIÓN PARA GRÁFICOS
# ============================================================

def configurar_grafico(titulo, ylabel="Inasistencia (%)"):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.set_title(
        titulo,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=10,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    ax.set_axisbelow(True)

    for borde in [
        "top",
        "right",
    ]:
        ax.spines[borde].set_visible(False)

    return fig, ax


# ============================================================
# 5. VERIFICAR CARPETA DE RESULTADOS
# ============================================================

if not RESULTADOS_DIR.exists():
    st.error(
        "No se encontró la carpeta 'resultados'. "
        "Verifica que dashboard.py esté ubicado en "
        "la carpeta principal del proyecto."
    )
    st.stop()


# ============================================================
# 6. CARGAR RESULTADOS
# ============================================================

metricas = cargar_json(
    "metricas_gradient_boosting.json"
)

citas_previas = cargar_csv(
    "analisis_citas_previas_perdidas.csv"
)

dias_espera = cargar_csv(
    "analisis_dias_espera.csv"
)

canales = cargar_csv(
    "analisis_canal_agendamiento.csv"
)

importancias = cargar_csv(
    "importancia_variables_gradient_boosting.csv"
)

kpis_archivo = cargar_csv(
    "kpis_gradient_boosting.csv"
)


# ============================================================
# 7. MÉTRICAS DEL MODELO
# ============================================================

accuracy = float(
    obtener_valor(
        metricas,
        "accuracy_pct",
        83.15
    )
)

precision = float(
    obtener_valor(
        metricas,
        "precision_pct",
        80.16
    )
)

recall = float(
    obtener_valor(
        metricas,
        "recall_pct",
        71.76
    )
)

f1 = float(
    obtener_valor(
        metricas,
        "f1_pct",
        75.73
    )
)

f2 = float(
    obtener_valor(
        metricas,
        "f2_pct",
        73.29
    )
)

roc_auc = float(
    obtener_valor(
        metricas,
        "roc_auc",
        0.8886
    )
)

pr_auc = float(
    obtener_valor(
        metricas,
        "pr_auc",
        0.8280
    )
)

vp = int(
    obtener_valor(
        metricas,
        "VP",
        404
    )
)

fp = int(
    obtener_valor(
        metricas,
        "FP",
        100
    )
)

fn = int(
    obtener_valor(
        metricas,
        "FN",
        159
    )
)

vn = int(
    obtener_valor(
        metricas,
        "VN",
        874
    )
)

citas_prueba = int(
    obtener_valor(
        metricas,
        "citas",
        1537
    )
)

baseline = float(
    obtener_valor(
        metricas,
        "baseline_clase_mayoritaria_accuracy_pct",
        63.37
    )
)


# ============================================================
# 8. DATOS GENERALES
# ============================================================

TOTAL_CITAS = 8996
TOTAL_INASISTENCIAS = 3381
TOTAL_ASISTENCIAS = 5615

TASA_INASISTENCIA = 37.58
REFERENCIA_INASISTENCIA = 15.00

REGISTROS_ORIGINALES = 10080
REGISTROS_VALIDADOS = 8996

PRV = (
    REGISTROS_VALIDADOS
    / REGISTROS_ORIGINALES
    * 100
)

FALSAS_ALARMAS = (
    fp / (fp + vn) * 100
)

NO_DETECTADAS = (
    fn / (vp + fn) * 100
)

MEJORA_BASELINE = (
    accuracy - baseline
)


# ============================================================
# 9. ENCABEZADO
# ============================================================

st.title("Red Salud Integral Carazo")

st.subheader(
    "Análisis y predicción de inasistencia a citas médicas"
)

st.caption(
    "Dashboard de apoyo a la gestión de citas y toma de decisiones"
)

st.divider()


# ============================================================
# 10. SITUACIÓN GENERAL
# ============================================================

st.header("Situación general")

st.write(
    "Panorama de las citas utilizadas en el análisis."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Citas analizadas",
        value=f"{TOTAL_CITAS:,}",
    )

with col2:
    st.metric(
        label="Inasistencias",
        value=f"{TOTAL_INASISTENCIAS:,}",
    )

with col3:
    st.metric(
        label="Asistencias",
        value=f"{TOTAL_ASISTENCIAS:,}",
    )

with col4:
    st.metric(
        label="Tasa de inasistencia",
        value=f"{TASA_INASISTENCIA:.2f}%",
        delta=f"{TASA_INASISTENCIA - REFERENCIA_INASISTENCIA:.2f} pp sobre referencia",
        delta_color="inverse",
    )


# ============================================================
# 11. INDICADOR PRINCIPAL
# ============================================================

st.subheader("Inasistencia observada")

col_actual, col_referencia = st.columns([2, 1])

with col_actual:

    st.metric(
        "Tasa actual de inasistencia",
        f"{TASA_INASISTENCIA:.2f}%",
    )

    st.progress(
        min(
            TASA_INASISTENCIA / 100,
            1.0
        )
    )

with col_referencia:

    st.metric(
        "Referencia operativa",
        f"{REFERENCIA_INASISTENCIA:.0f}%",
    )

    diferencia = (
        TASA_INASISTENCIA
        - REFERENCIA_INASISTENCIA
    )

    st.write(
        f"La tasa observada se encuentra "
        f"**{diferencia:.2f} puntos porcentuales** "
        f"por encima de la referencia."
    )

st.divider()


# ============================================================
# 12. FACTORES ASOCIADOS
# ============================================================

st.header("¿Qué factores están asociados con la inasistencia?")

st.write(
    "Los datos muestran diferencias importantes según "
    "el historial del paciente y las características de la cita."
)


# ============================================================
# 13. CITAS PREVIAS PERDIDAS
# ============================================================

st.subheader("Historial de citas perdidas")

if citas_previas is not None:

    grafico_col, hallazgo_col = st.columns(
        [2, 1]
    )

    with grafico_col:

        fig, ax = configurar_grafico(
            "Inasistencia según historial de citas perdidas"
        )

        x = (
            citas_previas[
                "citas_previas_perdidas"
            ]
            .astype(int)
            .astype(str)
        )

        y = citas_previas[
            "tasa_inasistencia_pct"
        ]

        barras = ax.bar(
            x,
            y,
        )

        ax.set_xlabel(
            "Número de citas previas perdidas"
        )

        ax.set_ylim(
            0,
            100
        )

        for barra, dato in zip(
            barras,
            y
        ):
            ax.text(
                barra.get_x()
                + barra.get_width() / 2,
                dato + 2,
                f"{dato:.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with hallazgo_col:

        st.info(
            "### Hallazgo principal\n\n"
            "El historial de citas perdidas presenta "
            "una diferencia marcada en la inasistencia."
        )

        st.metric(
            "Sin citas previas perdidas",
            "8.26%",
        )

        st.metric(
            "Con 4 citas previas perdidas",
            "91.70%",
        )

        st.write(
            "La tasa observada aumenta considerablemente "
            "en pacientes que ya presentan antecedentes "
            "de inasistencia."
        )

else:

    st.warning(
        "No se encontró "
        "'analisis_citas_previas_perdidas.csv'."
    )


# ============================================================
# 14. DÍAS DE ESPERA
# ============================================================

st.subheader("Tiempo de espera")

if dias_espera is not None:

    grafico_col, hallazgo_col = st.columns(
        [2, 1]
    )

    with grafico_col:

        fig, ax = configurar_grafico(
            "Inasistencia según días de espera"
        )

        x = dias_espera[
            "rango_espera"
        ].astype(str)

        y = dias_espera[
            "tasa_inasistencia_pct"
        ]

        barras = ax.bar(
            x,
            y,
        )

        ax.set_xlabel(
            "Días entre agendamiento y cita"
        )

        ax.set_ylim(
            0,
            max(
                60,
                float(y.max()) + 10
            )
        )

        for barra, dato in zip(
            barras,
            y
        ):
            ax.text(
                barra.get_x()
                + barra.get_width() / 2,
                dato + 1.5,
                f"{dato:.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.xticks(
            rotation=10
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with hallazgo_col:

        st.info(
            "### Hallazgo principal\n\n"
            "Las citas con períodos de espera más largos "
            "presentan tasas de inasistencia más elevadas "
            "en los datos analizados."
        )

        st.metric(
            "Espera de 8 a 14 días",
            "25.54%",
        )

        st.metric(
            "Espera de 15 a 30 días",
            "48.99%",
        )

        st.write(
            "El grupo de 15 a 30 días registró la mayor "
            "tasa observada: **48.99%**."
        )

else:

    st.warning(
        "No se encontró 'analisis_dias_espera.csv'."
    )


# ============================================================
# 15. CANAL DE AGENDAMIENTO
# ============================================================

st.subheader("Canal de agendamiento")

if canales is not None:

    datos_canal = canales.sort_values(
        "tasa_inasistencia_pct",
        ascending=False
    )

    grafico_col, hallazgo_col = st.columns(
        [2, 1]
    )

    with grafico_col:

        fig, ax = configurar_grafico(
            "Inasistencia según canal de agendamiento"
        )

        x = datos_canal[
            "canal_agendamiento"
        ]

        y = datos_canal[
            "tasa_inasistencia_pct"
        ]

        barras = ax.bar(
            x,
            y,
        )

        ax.set_xlabel(
            "Canal de agendamiento"
        )

        ax.set_ylim(
            0,
            max(
                50,
                float(y.max()) + 8
            )
        )

        for barra, dato in zip(
            barras,
            y
        ):
            ax.text(
                barra.get_x()
                + barra.get_width() / 2,
                dato + 1,
                f"{dato:.1f}%",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

        plt.xticks(
            rotation=10
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    with hallazgo_col:

        st.info(
            "### Hallazgo principal\n\n"
            "Las tasas entre los principales canales son "
            "similares, aunque existen diferencias observables."
        )

        st.metric(
            "Mayor tasa",
            "39.00%",
            "Ventanilla",
        )

        st.metric(
            "Mayor cantidad",
            "1,195",
            "Call Center",
        )

        st.write(
            "**Ventanilla** registra la mayor tasa de "
            "inasistencia, mientras que **Call Center** "
            "concentra la mayor cantidad de casos."
        )

else:

    st.warning(
        "No se encontró "
        "'analisis_canal_agendamiento.csv'."
    )


st.caption(
    "Las diferencias mostradas representan asociaciones "
    "observadas en los datos y no demuestran causalidad."
)

st.divider()


# ============================================================
# 16. MODELO PREDICTIVO
# ============================================================

st.header("¿Qué tan bien podemos anticipar una inasistencia?")

st.write(
    "Se utilizó un modelo **Gradient Boosting** para "
    "estimar anticipadamente el riesgo de que un paciente "
    "no asista a su cita."
)


# ============================================================
# 17. MÉTRICAS PRINCIPALES
# ============================================================

m1, m2, m3, m4 = st.columns(4)

with m1:

    st.metric(
        "Predicciones correctas",
        f"{accuracy:.2f}%",
    )

    st.caption(
        "Accuracy"
    )

with m2:

    st.metric(
        "Inasistencias detectadas",
        f"{recall:.2f}%",
    )

    st.caption(
        "Recall"
    )

with m3:

    st.metric(
        "Precisión de las alertas",
        f"{precision:.2f}%",
    )

    st.caption(
        "Precision"
    )

with m4:

    st.metric(
        "Capacidad de diferenciación",
        f"{roc_auc:.4f}",
    )

    st.caption(
        "ROC-AUC"
    )


# ============================================================
# 18. INTERPRETACIÓN PARA EL PÚBLICO
# ============================================================

st.subheader("¿Qué significan estos resultados?")

col_a, col_b, col_c = st.columns(3)

with col_a:

    st.success(
        f"**{accuracy:.2f}% de las citas fueron "
        f"clasificadas correctamente.**"
    )

with col_b:

    st.success(
        f"El modelo logró identificar "
        f"**{recall:.2f}% de las inasistencias reales**."
    )

with col_c:

    st.success(
        f"De las citas señaladas como posibles "
        f"inasistencias, **{precision:.2f}% realmente "
        f"terminaron en inasistencia**."
    )


# ============================================================
# 19. RESULTADOS DE JUNIO
# ============================================================

st.subheader("Resultado de la evaluación final")

st.write(
    f"El modelo fue evaluado con **{citas_prueba:,} citas "
    f"correspondientes al período de prueba de junio**."
)

r1, r2, r3, r4 = st.columns(4)

with r1:
    st.metric(
        "Inasistencias detectadas",
        f"{vp:,}",
    )

with r2:
    st.metric(
        "Asistencias identificadas",
        f"{vn:,}",
    )

with r3:
    st.metric(
        "Alertas incorrectas",
        f"{fp:,}",
    )

with r4:
    st.metric(
        "Inasistencias no detectadas",
        f"{fn:,}",
    )


# ============================================================
# 20. MATRIZ DE CONFUSIÓN VISUAL
# ============================================================

st.subheader("Resultado de las predicciones")

matriz = pd.DataFrame(
    {
        "Resultado": [
            "Inasistencia detectada",
            "Asistencia identificada",
            "Alerta incorrecta",
            "Inasistencia no detectada",
        ],
        "Cantidad": [
            vp,
            vn,
            fp,
            fn,
        ],
    }
)

fig, ax = configurar_grafico(
    "Distribución de resultados",
    "Cantidad de citas"
)

barras = ax.bar(
    matriz["Resultado"],
    matriz["Cantidad"],
)

for barra, dato in zip(
    barras,
    matriz["Cantidad"]
):
    ax.text(
        barra.get_x()
        + barra.get_width() / 2,
        dato + 15,
        f"{dato:,}",
        ha="center",
        fontsize=9,
        fontweight="bold",
    )

plt.xticks(
    rotation=12,
    ha="right"
)

plt.tight_layout()

st.pyplot(
    fig,
    use_container_width=True
)

plt.close(fig)


# ============================================================
# 21. VARIABLES MÁS IMPORTANTES
# ============================================================

st.subheader("¿Qué información utiliza más el modelo?")

if importancias is not None:

    top_importancias = (
        importancias
        .sort_values(
            "importancia",
            ascending=False
        )
        .head(8)
        .copy()
    )

    nombres_amigables = {
        "num__citas_previas_perdidas":
            "Citas previas perdidas",

        "num__espera_dias":
            "Días de espera",

        "cat__canal_agendamiento_REFERENCIA MÉDICA":
            "Canal: Referencia médica",

        "cat__tipo_cobertura_PÚBLICA":
            "Cobertura pública",

        "cat__especialidad_MEDICINA GENERAL":
            "Medicina general",

        "cat__grupo_edad_45-59":
            "Edad: 45-59",

        "cat__grupo_edad_30-44":
            "Edad: 30-44",

        "cat__mes_4":
            "Mes: abril",
    }

    top_importancias["nombre"] = (
        top_importancias["variable"]
        .map(nombres_amigables)
        .fillna(
            top_importancias["variable"]
        )
    )

    top_importancias = (
        top_importancias
        .sort_values(
            "importancia",
            ascending=True
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 4.8)
    )

    barras = ax.barh(
        top_importancias["nombre"],
        top_importancias["importancia"],
    )

    ax.set_title(
        "Variables con mayor importancia para el modelo",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel(
        "Importancia relativa"
    )

    ax.grid(
        axis="x",
        alpha=0.20
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for barra, dato in zip(
        barras,
        top_importancias["importancia"]
    ):
        ax.text(
            dato + 0.005,
            barra.get_y()
            + barra.get_height() / 2,
            f"{dato * 100:.1f}%",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.info(
        "El **historial de citas previas perdidas** es la "
        "variable con mayor peso dentro de las decisiones "
        "del modelo, seguida por los **días de espera**."
    )

    st.caption(
        "La importancia de una variable dentro del modelo "
        "no implica que esa variable sea la causa de la inasistencia."
    )

else:

    st.warning(
        "No se encontró el archivo de importancia de variables."
    )


st.divider()


# ============================================================
# 22. KPI
# ============================================================

st.header("Indicadores principales")

st.write(
    "Cuatro indicadores permiten resumir el desempeño "
    "alcanzado por el modelo."
)

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.metric(
        "Detección de inasistencias",
        f"{recall:.2f}%",
        f"{recall - 70:.2f} pp sobre meta"
    )

    st.caption(
        "Meta ≥ 70%"
    )

with k2:

    st.metric(
        "Precisión de predicción",
        f"{precision:.2f}%",
        f"{precision - 80:.2f} pp sobre meta"
    )

    st.caption(
        "Meta ≥ 80%"
    )

with k3:

    st.metric(
        "Exactitud general",
        f"{accuracy:.2f}%",
        f"{accuracy - 80:.2f} pp sobre meta"
    )

    st.caption(
        "Meta ≥ 80%"
    )

with k4:

    st.metric(
        "Capacidad de diferenciación",
        f"{roc_auc:.4f}",
        f"{roc_auc - 0.80:.4f} sobre meta"
    )

    st.caption(
        "Meta ≥ 0.80"
    )

st.success(
    "Los cuatro indicadores principales mostrados "
    "alcanzaron sus metas establecidas."
)


# ============================================================
# 23. LOS 10 KPI
# ============================================================

with st.expander(
    "Ver evaluación completa de los 10 KPI"
):

    tabla_kpis = pd.DataFrame(
        {
            "KPI": [
                "Registros validados",
                "Tasa de inasistencia",
                "Detección de inasistencias",
                "Precisión de predicción",
                "Exactitud general",
                "Falsas alarmas",
                "Inasistencias no detectadas",
                "Rendimiento F2",
                "Capacidad de diferenciación",
                "Mejora frente a clasificación base",
            ],

            "Resultado": [
                f"{PRV:.2f}%",
                f"{TASA_INASISTENCIA:.2f}%",
                f"{recall:.2f}%",
                f"{precision:.2f}%",
                f"{accuracy:.2f}%",
                f"{FALSAS_ALARMAS:.2f}%",
                f"{NO_DETECTADAS:.2f}%",
                f"{f2:.2f}%",
                f"{roc_auc:.4f}",
                f"{MEJORA_BASELINE:.2f} pp",
            ],

            "Meta / referencia": [
                "≥ 90%",
                "Referencia: 15%",
                "≥ 70%",
                "≥ 80%",
                "≥ 80%",
                "≤ 20%",
                "≤ 30%",
                "≥ 75%",
                "≥ 0.80",
                "> 0 pp",
            ],

            "Estado": [
                "No cumple",
                "Sobre referencia",
                "Cumple",
                "Cumple",
                "Cumple",
                "Cumple",
                "Cumple",
                "No cumple",
                "Cumple",
                "Cumple",
            ],
        }
    )

    st.dataframe(
        tabla_kpis,
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "**Resultado:** 7 KPI cumplen, 2 no cumplen "
        "y 1 indicador se compara contra una referencia operativa."
    )


# ============================================================
# 24. ACCIONES RECOMENDADAS
# ============================================================

st.divider()

st.header("Acciones recomendadas")

st.write(
    "A partir de los resultados observados, se proponen "
    "las siguientes acciones de apoyo a la gestión:"
)

col_accion1, col_accion2 = st.columns(2)

with col_accion1:

    st.info(
        "### 🎯 Priorizar citas de mayor riesgo\n\n"
        "Utilizar la probabilidad estimada por el modelo "
        "para priorizar las citas que requieren seguimiento."
    )

    st.info(
        "### 📅 Considerar el historial del paciente\n\n"
        "Dar mayor atención a pacientes que presentan "
        "antecedentes de citas perdidas."
    )

with col_accion2:

    st.info(
        "### ⏱️ Vigilar períodos de espera prolongados\n\n"
        "Dar seguimiento a citas con tiempos de espera "
        "en los rangos donde se observaron mayores tasas."
    )

    st.info(
        "### 📞 Priorizar confirmaciones y recordatorios\n\n"
        "Orientar los esfuerzos de confirmación hacia "
        "las citas identificadas con mayor riesgo."
    )


# ============================================================
# 25. RESUMEN FINAL
# ============================================================

st.divider()

st.header("Resumen")

res1, res2, res3, res4 = st.columns(4)

with res1:
    st.metric(
        "Citas analizadas",
        "8,996"
    )

with res2:
    st.metric(
        "Inasistencia",
        "37.58%"
    )

with res3:
    st.metric(
        "Exactitud del modelo",
        f"{accuracy:.2f}%"
    )

with res4:
    st.metric(
        "Inasistencias detectadas",
        f"{recall:.2f}%"
    )

st.success(
    "El análisis permite identificar patrones asociados "
    "a la inasistencia y utilizar un modelo predictivo "
    "como apoyo para priorizar acciones preventivas."
)

st.caption(
    "Red Salud Integral Carazo · "
    "Análisis de inasistencia a citas médicas"
)