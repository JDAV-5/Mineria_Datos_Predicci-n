USE RedSaludCarazoDW_V2;
GO

IF OBJECT_ID(N'dbo.raw_cita', N'U') IS NULL
CREATE TABLE dbo.raw_cita (
    id_raw BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    lote_carga NVARCHAR(45) NOT NULL,
    origen_dataset NVARCHAR(20) NOT NULL, -- 'v0' o 'ampliado'
    id_cita NVARCHAR(40) NULL,
    fecha_cita DATE NULL,
    centro_salud NVARCHAR(80) NULL,
    canal_agendamiento NVARCHAR(80) NULL,
    grupo_edad NVARCHAR(30) NULL,
    especialidad NVARCHAR(100) NULL,
    costo_estimado_usd DECIMAL(12,2) NULL,
    espera_dias INT NULL,
    recordatorios_enviados INT NULL,
    citas_previas_perdidas INT NULL,
    tipo_cobertura NVARCHAR(50) NULL,
    sistema_origen NVARCHAR(50) NULL,
    no_asistio NVARCHAR(20) NULL,
    consentimiento_contacto NVARCHAR(10) NULL,
    fecha_carga DATE NULL,
    fecha_ingesta DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
IF OBJECT_ID(N'dbo.dim_tiempo', N'U') IS NULL
CREATE TABLE dbo.dim_tiempo (
    fecha DATE NOT NULL PRIMARY KEY,
    anio INT NOT NULL,
    mes INT NOT NULL,
    trimestre INT NOT NULL,
    dia INT NOT NULL,
    dia_semana_lunes_1 INT NOT NULL,
    es_fin_semana BIT NOT NULL
);
GO
IF OBJECT_ID(N'dbo.dim_centro_salud', N'U') IS NULL
CREATE TABLE dbo.dim_centro_salud (
    id_centro INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    centro_salud NVARCHAR(80) NOT NULL UNIQUE
);
GO
IF OBJECT_ID(N'dbo.dim_especialidad', N'U') IS NULL
CREATE TABLE dbo.dim_especialidad (
    id_especialidad INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    especialidad NVARCHAR(100) NOT NULL UNIQUE
);
GO
IF OBJECT_ID(N'dbo.dim_canal', N'U') IS NULL
CREATE TABLE dbo.dim_canal (
    id_canal INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    canal_agendamiento NVARCHAR(80) NOT NULL UNIQUE
);
GO
IF OBJECT_ID(N'dbo.hecho_cita', N'U') IS NULL
CREATE TABLE dbo.hecho_cita (
    id_hecho BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    id_raw BIGINT NOT NULL UNIQUE REFERENCES dbo.raw_cita(id_raw),
    id_cita NVARCHAR(40) NOT NULL UNIQUE,
    fecha_cita DATE NOT NULL REFERENCES dbo.dim_tiempo(fecha),
    id_centro INT NOT NULL REFERENCES dbo.dim_centro_salud(id_centro),
    id_especialidad INT NOT NULL REFERENCES dbo.dim_especialidad(id_especialidad),
    id_canal INT NOT NULL REFERENCES dbo.dim_canal(id_canal),
    grupo_edad NVARCHAR(30) NOT NULL,
    tipo_cobertura NVARCHAR(50) NULL,
    espera_dias INT NOT NULL,
    recordatorios_enviados INT NOT NULL,
    citas_previas_perdidas INT NOT NULL,
    costo_estimado_usd DECIMAL(12,2) NULL,
    sistema_origen NVARCHAR(50) NULL,
    consentimiento_contacto NVARCHAR(10) NULL,
    no_asistio TINYINT NOT NULL CHECK (no_asistio IN (0,1)),
    fecha_carga DATE NULL
);
GO
IF OBJECT_ID(N'dbo.dm_resultados_prediccion', N'U') IS NULL
CREATE TABLE dbo.dm_resultados_prediccion (
    id_resultado BIGINT IDENTITY(1,1) NOT NULL PRIMARY KEY,
    id_cita NVARCHAR(40) NOT NULL,
    fecha_cita DATE NOT NULL,
    no_asistio TINYINT NOT NULL,
    prob_inasistencia FLOAT NOT NULL,
    pred_inasistencia TINYINT NOT NULL,
    conjunto NVARCHAR(20) NOT NULL,
    nombre_modelo NVARCHAR(60) NOT NULL,
    fecha_evaluacion DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE object_id=OBJECT_ID(N'dbo.raw_cita') AND name=N'IX_raw_lote_id')
    CREATE INDEX IX_raw_lote_id ON dbo.raw_cita(origen_dataset,lote_carga,id_cita);
GO
PRINT N'Esquema RedSaludCarazoDW listo.';
GO
