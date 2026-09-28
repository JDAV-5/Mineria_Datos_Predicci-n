

import sys

from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import pandas as pd

from sqlalchemy import text

from database import engine_sqlserver


# ============================================================
# 1. COLUMNAS DEL DATASET
# ============================================================

COLS = [
    'id_cita',
    'fecha_cita',
    'centro_salud',
    'canal_agendamiento',
    'grupo_edad',
    'especialidad',
    'costo_estimado_usd',
    'espera_dias',
    'recordatorios_enviados',
    'citas_previas_perdidas',
    'tipo_cobertura',
    'sistema_origen',
    'no_asistio',
    'consentimiento_contacto',
    'fecha_carga'
]


# ============================================================
# 2. FUNCIÓN PARA CONVERTIR ENTEROS
# ============================================================

def convertir_entero(valor, columna, fila_excel):

    """
    Convierte valores como 33.0 a 33.

    Conserva:
    - Números negativos.
    - Valores nulos.

    No modifica el Excel original.

    Si encuentra un valor no entero o no numérico,
    detiene la importación para revisar el problema.
    """

    # Conservar valores vacíos
    if pd.isna(valor):
        return None

    # Tratar cadenas vacías como valores nulos
    if isinstance(valor, str) and not valor.strip():
        return None

    try:

        numero = Decimal(str(valor).strip())

    except InvalidOperation:

        raise ValueError(
            f"Valor no numérico en la columna '{columna}', "
            f"fila {fila_excel} del Excel: {valor!r}"
        )

    # Verificar que sea un número entero
    if (
        not numero.is_finite()
        or numero != numero.to_integral_value()
    ):

        raise ValueError(
            f"Valor no entero en la columna '{columna}', "
            f"fila {fila_excel} del Excel: {valor!r}. "
            "Revisar sin modificar el Excel original."
        )

    return int(numero)


# ============================================================
# 3. FUNCIÓN PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("RED SALUD INTEGRAL CARAZO")
    print("IMPORTACIÓN DE DATOS A SQL SERVER")
    print("=" * 60)

    # --------------------------------------------------------
    # PASO 1: Validar argumentos
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        raise SystemExit(
            "Uso: python src/01_cargar_excel.py "
            "<archivo.xlsx> <v0|ampliado>"
        )

    ruta = Path(sys.argv[1])

    origen = sys.argv[2]

    if origen not in ('v0', 'ampliado'):

        raise SystemExit(
            "El origen debe ser 'v0' o 'ampliado'."
        )

    # --------------------------------------------------------
    # PASO 2: Verificar archivo Excel
    # --------------------------------------------------------

    if not ruta.exists():

        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta}"
        )

    print(f"\nArchivo encontrado: {ruta}")

    # --------------------------------------------------------
    # PASO 3: Leer Excel
    # --------------------------------------------------------

    print("\nLeyendo archivo Excel...")

    df = pd.read_excel(
        ruta,
        sheet_name='C02_Citas',
        dtype={
            'id_cita': str
        }
    )

    print(
        f"Registros encontrados: {len(df):,}"
    )

    print(
        f"Columnas encontradas: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # PASO 4: Validar existencia de columnas
    # --------------------------------------------------------

    faltan = set(COLS).difference(df.columns)

    if faltan:

        raise ValueError(
            f"Columnas faltantes: {sorted(faltan)}"
        )

    print("\nColumnas verificadas correctamente.")

    # Conservar las columnas esperadas
    df = df[COLS].copy()

    # --------------------------------------------------------
    # PASO 5: Convertir fechas
    # --------------------------------------------------------

    print("\nPreparando campos de fecha...")

    for columna in (
        'fecha_cita',
        'fecha_carga'
    ):

        df[columna] = pd.to_datetime(
            df[columna],
            errors='coerce'
        ).dt.date

    # --------------------------------------------------------
    # PASO 6: Preparar variable objetivo
    # --------------------------------------------------------

    print("\nPreparando variable no_asistio...")

    def convertir_resultado(valor):

        if pd.isna(valor):
            return None

        if (
            isinstance(valor, float)
            and valor.is_integer()
        ):
            return str(int(valor))

        return str(valor)

    df['no_asistio'] = df['no_asistio'].map(
        convertir_resultado
    )

    # --------------------------------------------------------
    # PASO 7: Corregir representación de números enteros
    # --------------------------------------------------------

    print("\nPreparando columnas numéricas...")

    columnas_enteras = [
        'espera_dias',
        'recordatorios_enviados',
        'citas_previas_perdidas'
    ]

    for columna in columnas_enteras:

        valores_convertidos = []

        for indice, valor in df[columna].items():

            numero = convertir_entero(
                valor,
                columna,
                indice + 2
            )

            valores_convertidos.append(numero)

        # Mantener números enteros de Python y valores None
        df[columna] = pd.Series(
            valores_convertidos,
            index=df.index,
            dtype=object
        )

        print(
            f"Columna preparada: {columna}"
        )

    # --------------------------------------------------------
    # PASO 8: Añadir origen del dataset
    # --------------------------------------------------------

    df['origen_dataset'] = origen

    # --------------------------------------------------------
    # PASO 9: Generar identificador del lote
    # --------------------------------------------------------

    lote = datetime.now(
        timezone.utc
    ).strftime(
        '%Y%m%dT%H%M%S%fZ'
    )

    df['lote_carga'] = lote

    # --------------------------------------------------------
    # PASO 10: Preparar valores nulos para SQL Server
    # --------------------------------------------------------

    df = df.astype(object).where(
        pd.notnull(df),
        None
    )

    print("\nDatos preparados para importación.")

    # --------------------------------------------------------
    # PASO 11: Conectar con SQL Server
    # --------------------------------------------------------

    print("\nConectando con SQL Server...")

    engine = engine_sqlserver()

    # --------------------------------------------------------
    # PASO 12: Verificar que el dataset no esté importado
    # --------------------------------------------------------

    with engine.begin() as con:

        total_existente = con.execute(
            text("""
                SELECT COUNT(*)
                FROM dbo.raw_cita
                WHERE origen_dataset = :origen
            """),
            {
                'origen': origen
            }
        ).scalar_one()

        if total_existente > 0:

            raise SystemExit(
                f"\nATENCIÓN: Ya existen "
                f"{total_existente:,} registros "
                f"del dataset '{origen}' "
                "en dbo.raw_cita.\n"
                "Se canceló la importación para "
                "evitar duplicar una carga anterior."
            )

        # ----------------------------------------------------
        # PASO 13: Importar registros
        # ----------------------------------------------------

        print("\nImportando registros a dbo.raw_cita...")

        df.to_sql(
            'raw_cita',
            con=con,
            schema='dbo',
            if_exists='append',
            index=False,
            chunksize=500
        )

    # --------------------------------------------------------
    # PASO 14: Mostrar resultados
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("IMPORTACIÓN COMPLETADA CORRECTAMENTE")

    print("=" * 60)

    print(
        f"\nRegistros importados: {len(df):,}"
    )

    print(
        f"Origen del dataset: {origen}"
    )

    print(
        f"Lote de carga: {lote}"
    )

    print(
        "\nTabla destino: dbo.raw_cita"
    )

    print(
        "\nLos registros originales fueron conservados."
    )

    print(
        "No se realizó depuración analítica."
    )

    print(
        "\nSiguiente paso: ejecutar 02_calidad.sql"
    )

    print("=" * 60)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == '__main__':

    main()
