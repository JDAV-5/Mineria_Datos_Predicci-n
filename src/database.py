"""Conexión local a SQL Server. No almacenar contraseñas en Git."""
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()


def engine_sqlserver():
    server = os.getenv('DB_SERVER', 'localhost')
    database = os.getenv('DB_NAME', 'RedSaludCarazoDW')
    driver = os.getenv('DB_DRIVER', 'ODBC Driver 17 for SQL Server')
    if os.getenv('DB_TRUSTED', 'yes').lower() == 'yes':
        authentication = 'Trusted_Connection=yes;'
    else:
        username = os.environ['DB_USER']
        password = os.environ['DB_PASSWORD']
        authentication = f'UID={username};PWD={password};'
    conn_str = (
        f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};'
        + authentication + 'TrustServerCertificate=yes;'
    )
    return create_engine('mssql+pyodbc:///?odbc_connect=' + quote_plus(conn_str),
                         future=True)
