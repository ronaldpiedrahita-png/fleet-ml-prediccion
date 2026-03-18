# 03_fuel_api.py
# API de precios CRE: https://api.datos.gob.mx/v1/precio-gasolina-diaria-por-estacion
# Sin API key. Datos abiertos del gobierno mexicano.

import os, requests, time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Valeria2004@localhost/fleetdb")
engine  = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)

# API pública datos.gob.mx  (datos abiertos, no requiere key)
CRE_API = "https://api.datos.gob.mx/v1/precio-gasolina-diaria-por-estacion"

# Nota: si el endpoint CRE no está disponible, se usa fallback de precios simulados
FALLBACK_DIESEL_PRICES = {
    "Monterrey":    23.80,
    "CDMX":         24.10,
    "Guadalajara":  23.65,
    "Tijuana":      25.30,
    "Querétaro":    23.95,
}


def fetch_diesel_prices() -> dict:
    """
    Intenta obtener precios reales de la CRE.
    Si falla, usa precios de referencia (simulados pero realistas).
    Retorna dict: {ciudad: precio_diesel_mxn}
    """
    print("⛽ Consultando precios de diésel (CRE México)...")
    try:
        params = {"tipo_combustible": "diesel", "pageSize": 200}
        r = requests.get(CRE_API, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if "results" not in data:
            raise ValueError("Formato inesperado")

        df = pd.DataFrame(data["results"])
        # Campos típicos: estado, municipio, precio_diesel, fecha
        precio_por_estado = (
            df.groupby("estado")["precio_diesel"]
            .mean()
            .round(2)
            .to_dict()
        )
        print(f"  ✅ {len(precio_por_estado)} estados con datos de precios")
        return precio_por_estado

    except Exception as e:
        print(f"  ⚠️  API CRE no disponible ({e}). Usando precios de referencia.")
        return FALLBACK_DIESEL_PRICES


def create_fuel_prices_table():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fuel_market_prices (
                id          SERIAL PRIMARY KEY,
                city        VARCHAR(50),
                diesel_mxn  FLOAT,
                fetched_at  TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
        print("✅ Tabla fuel_market_prices creada")


def save_prices(prices: dict):
    """Guarda los precios de hoy en SQL."""
    session = Session()
    for city, price in prices.items():
        session.execute(text(
            "INSERT INTO fuel_market_prices (city, diesel_mxn) VALUES (:city, :price)"
        ), {"city": city, "price": price})
    session.commit()
    session.close()
    print(f"  ✅ {len(prices)} precios guardados en base de datos")


def analyze_fleet_fuel_efficiency():
    """
    JOIN entre fuel_logs y fuel_market_prices para:
    1. Detectar camiones con bajo rendimiento (señal de fallo motor)
    2. Calcular costo real vs precio de mercado
    """
    QUERY = """
        SELECT
            fl.truck_id,
            t.plate,
            t.model,
            t.odometer_km,
            COUNT(fl.id)                 AS total_cargas,
            AVG(fl.km_per_liter)         AS avg_rendimiento,
            MIN(fl.km_per_liter)         AS min_rendimiento,
            AVG(fl.liters)               AS avg_litros_por_carga,
            SUM(fl.total_cost_mxn)       AS costo_total_30d,
            AVG(fmp.diesel_mxn)          AS precio_mercado_avg,
            AVG(fl.price_per_liter)      AS precio_pagado_avg,
            AVG(fl.price_per_liter) - AVG(fmp.diesel_mxn) AS sobrecosto_litro
        FROM fuel_logs fl
        JOIN trucks t ON t.id = fl.truck_id
        LEFT JOIN fuel_market_prices fmp ON fmp.city = fl.station_city
        WHERE fl.fueled_at > NOW() - INTERVAL '30 days'
        GROUP BY fl.truck_id, t.plate, t.model, t.odometer_km
        ORDER BY avg_rendimiento ASC
    """
    df = pd.read_sql(text(QUERY), engine)
    print(f"\n📊 Análisis de rendimiento de combustible ({len(df)} camiones):")

    # Camiones con bajo rendimiento = posible fallo de motor
    bajos = df[df["avg_rendimiento"] < 5.0]
    if not bajos.empty:
        print(f"\n🔴 {len(bajos)} camiones con rendimiento < 5.0 km/l (RIESGO FALLO):")
        for _, r in bajos.iterrows():
            print(f"   🚛 #{r.truck_id} {r.plate} ({r.model}): {r.avg_rendimiento:.2f} km/l")

    df.to_csv("data/fuel_analysis.csv", index=False)
    print("\n✅ Análisis guardado en data/fuel_analysis.csv")
    return df


if __name__ == "__main__":
    import os; os.makedirs("data", exist_ok=True)
    create_fuel_prices_table()
    prices = fetch_diesel_prices()
    save_prices(prices)
    df = analyze_fleet_fuel_efficiency()
    print(df[["plate", "avg_rendimiento", "costo_total_30d", "sobrecosto_litro"]].head(10).to_string())