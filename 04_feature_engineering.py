# 04_feature_engineering.py

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# ── Conexión directa ───────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Valeria2004@localhost/fleetdb")
engine  = create_engine(DATABASE_URL)

# ── QUERY 1: Telemetría con ventanas de tiempo ─────────────
QUERY_TELEMETRY = """
WITH telem_7d AS (
    SELECT
        truck_id,
        AVG(engine_temp_c)      AS avg_temp_7d,
        MAX(engine_temp_c)      AS max_temp_7d,
        STDDEV(engine_temp_c)   AS std_temp_7d,
        AVG(oil_pressure_bar)   AS avg_oil_7d,
        MIN(oil_pressure_bar)   AS min_oil_7d,
        AVG(rpm)                AS avg_rpm_7d,
        AVG(coolant_temp_c)     AS avg_coolant_7d,
        AVG(battery_v)          AS avg_battery_7d,
        COUNT(fault_code)       AS fault_codes_7d,
        AVG(speed_kmh)          AS avg_speed_7d
    FROM telemetry
    WHERE recorded_at > NOW() - INTERVAL '7 days'
    GROUP BY truck_id
),
telem_30d AS (
    SELECT
        truck_id,
        AVG(engine_temp_c)                      AS avg_temp_30d,
        MAX(engine_temp_c)                      AS max_temp_30d,
        AVG(oil_pressure_bar)                   AS avg_oil_30d,
        AVG(rpm)                                AS avg_rpm_30d,
        COUNT(fault_code)                       AS fault_codes_30d,
        MAX(odometer_km) - MIN(odometer_km)     AS km_recorridos_30d
    FROM telemetry
    WHERE recorded_at > NOW() - INTERVAL '30 days'
    GROUP BY truck_id
)
SELECT
    t.id                                            AS truck_id,
    t.odometer_km,
    t.engine_hours,
    t.year,
    (EXTRACT(YEAR FROM NOW()) - t.year)             AS truck_age_years,
    t7.avg_temp_7d,   t7.max_temp_7d,   t7.std_temp_7d,
    t7.avg_oil_7d,    t7.min_oil_7d,
    t7.avg_rpm_7d,    t7.avg_coolant_7d, t7.avg_battery_7d,
    t7.fault_codes_7d, t7.avg_speed_7d,
    t30.avg_temp_30d,  t30.max_temp_30d,
    t30.avg_oil_30d,   t30.avg_rpm_30d,
    t30.fault_codes_30d, t30.km_recorridos_30d
FROM trucks t
LEFT JOIN telem_7d  t7  ON t7.truck_id  = t.id
LEFT JOIN telem_30d t30 ON t30.truck_id = t.id
"""

# ── QUERY 2: Combustible ───────────────────────────────────
QUERY_FUEL = """
SELECT
    truck_id,
    AVG(km_per_liter)       AS avg_kpl_30d,
    MIN(km_per_liter)       AS min_kpl_30d,
    STDDEV(km_per_liter)    AS std_kpl_30d,
    COUNT(*)                AS n_recargas_30d,
    SUM(total_cost_mxn)     AS costo_total_30d,
    AVG(liters)             AS avg_litros,
    AVG(CASE WHEN rn <= 4 THEN km_per_liter END) -
    AVG(CASE WHEN rn > n_total - 4 THEN km_per_liter END) AS kpl_trend
FROM (
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY truck_id ORDER BY fueled_at DESC) AS rn,
           COUNT(*) OVER (PARTITION BY truck_id) AS n_total
    FROM fuel_logs
) sub
GROUP BY truck_id
"""

# ── QUERY 3: Historial de mantenimiento ────────────────────
QUERY_MAINT = """
SELECT
    truck_id,
    COUNT(*)                                        AS total_eventos,
    SUM(CASE WHEN is_failure THEN 1 ELSE 0 END)     AS total_fallos,
    SUM(CASE WHEN is_failure THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(*), 0)                       AS ratio_fallos,
    SUM(downtime_days)                              AS total_downtime_days,
    SUM(cost_mxn)                                   AS total_cost_mxn,
    EXTRACT(DAY FROM NOW() - MAX(event_date))       AS days_since_last_maint
FROM maintenance_events
GROUP BY truck_id
"""

# ── QUERY 4: Target — camiones que tuvieron fallos ─────────
QUERY_TARGET = """
SELECT DISTINCT truck_id, TRUE AS will_fail
FROM maintenance_events
WHERE is_failure = TRUE
"""


def build_ml_dataset():
    print("⚙️  Construyendo dataset de ML desde SQL...")

    with engine.connect() as conn:
        df_telem  = pd.read_sql(text(QUERY_TELEMETRY), conn)
        df_fuel   = pd.read_sql(text(QUERY_FUEL),      conn)
        df_maint  = pd.read_sql(text(QUERY_MAINT),     conn)
        df_target = pd.read_sql(text(QUERY_TARGET),    conn)

    print(f"  Telemetría : {df_telem.shape}")
    print(f"  Combustible: {df_fuel.shape}")
    print(f"  Mant.      : {df_maint.shape}")
    print(f"  Target     : {df_target.shape}")

    # Eliminar columnas duplicadas
    df_telem  = df_telem.loc[:,  ~df_telem.columns.duplicated()]
    df_fuel   = df_fuel.loc[:,   ~df_fuel.columns.duplicated()]
    df_maint  = df_maint.loc[:,  ~df_maint.columns.duplicated()]
    df_target = df_target.loc[:, ~df_target.columns.duplicated()]

    # Merge de todas las fuentes
    df = (df_telem
          .merge(df_fuel,   on="truck_id", how="left")
          .merge(df_maint,  on="truck_id", how="left")
          .merge(df_target, on="truck_id", how="left"))

    # Target: 1 si tuvo fallo, 0 si no
    df["will_fail"] = df["will_fail"].fillna(False).astype(int)

    # Features derivados
    df["temp_trend"]    = df["avg_temp_7d"].fillna(0)  - df["avg_temp_30d"].fillna(0)
    df["oil_trend"]     = df["avg_oil_7d"].fillna(0)   - df["avg_oil_30d"].fillna(0)
    df["fault_accel"]   = df["fault_codes_7d"].fillna(0) / (df["fault_codes_30d"].fillna(0) + 1)
    df["overdue_maint"] = (df["days_since_last_maint"].fillna(0) > 90).astype(int)

    # Lista final de features
    FEATURES = [
        "truck_age_years", "odometer_km",    "engine_hours",
        "avg_temp_7d",     "max_temp_7d",    "std_temp_7d",
        "avg_oil_7d",      "min_oil_7d",
        "avg_rpm_7d",      "avg_coolant_7d", "avg_battery_7d",
        "fault_codes_7d",  "fault_codes_30d",
        "avg_kpl_30d",     "min_kpl_30d",    "kpl_trend",
        "total_fallos",    "ratio_fallos",   "days_since_last_maint",
        "total_downtime_days", "km_recorridos_30d",
        "temp_trend",      "oil_trend",      "fault_accel",
        "overdue_maint",
    ]
    TARGET = "will_fail"

    # Usar solo columnas que existen
    features_ok = [f for f in FEATURES if f in df.columns]
    missing     = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"\n  ⚠️  Columnas faltantes (se ignoran): {missing}")

    df_out = df[features_ok + [TARGET, "truck_id"]].fillna(0)

    print(f"\n📊 Dataset final: {df_out.shape}")
    print(f"   Tasa de fallos: {df_out[TARGET].mean():.1%}")
    print(f"   Camiones con fallo: {df_out[TARGET].sum()} de {len(df_out)}")

    # Guardar
    os.makedirs("data", exist_ok=True)
    df_out.to_csv("data/fleet_features.csv", index=False)
    df_out.to_sql("ml_fleet_features", engine, if_exists="replace", index=False)
    print("✅ Guardado: data/fleet_features.csv + tabla ml_fleet_features")

    return df_out, features_ok, TARGET


if __name__ == "__main__":
    df, features, target = build_ml_dataset()

    print("\n📈 Correlación con target (top 10):")
    corr = df[features].corrwith(df[target]).abs().sort_values(ascending=False)
    print(corr.head(10).round(3).to_string())

    print("\n📋 Muestra del dataset:")
    print(df[["truck_id", "odometer_km", "avg_temp_7d",
              "total_fallos", "will_fail"]].head(10).to_string(index=False))