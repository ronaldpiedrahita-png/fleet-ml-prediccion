# 02b_telemetry_ingestor.py
import os, time, requests
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# â”€â”€ ConexiÃ³n directa sin .env â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATABASE_URL = "postgresql://postgres:Valeria2004@localhost/fleetdb"
engine       = create_engine(DATABASE_URL)
Session      = sessionmaker(bind=engine)

# â”€â”€ Config API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
API_BASE  = "http://localhost:8001"
API_TOKEN = "fleet-secret-token-2024"
HEADERS   = {"Authorization": f"Bearer {API_TOKEN}"}
N_TRUCKS  = 200
INTERVAL  = 30

# â”€â”€ Reglas de anomalÃ­as â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ANOMALY_RULES = {
    "engine_temp_c":    (None, 105),
    "oil_pressure_bar": (2.0,  None),
    "battery_v":        (13.0, None),
    "coolant_temp_c":   (None, 108),
}

def check_anomalies(reading: dict) -> list:
    alerts = []
    for field, (mn, mx) in ANOMALY_RULES.items():
        val = reading.get(field)
        if val is None:
            continue
        if mn and val < mn:
            alerts.append(f"{field}={val} BAJO_MIN({mn})")
        if mx and val > mx:
            alerts.append(f"{field}={val} SOBRE_MAX({mx})")
    if reading.get("fault_code"):
        alerts.append(f"OBD_FAULT={reading['fault_code']}")
    return alerts

def ingest_cycle():
    session  = Session()
    ok, errs = 0, 0

    for truck_id in range(1, N_TRUCKS + 1):
        try:
            r = requests.get(
                f"{API_BASE}/telemetry/{truck_id}/live",
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=3
            )

            if r.status_code != 200:
                print(f"  âŒ CamiÃ³n {truck_id}: {r.status_code} â†’ {r.text}")
                errs += 1
                continue

            d = r.json()

            # Insertar en base de datos
            session.execute(text("""
                INSERT INTO telemetry
                  (truck_id, recorded_at, latitude, longitude, speed_kmh,
                   rpm, engine_temp_c, oil_pressure_bar, coolant_temp_c,
                   battery_v, odometer_km, fault_code)
                VALUES
                  (:tid, :rat, :lat, :lon, :spd,
                   :rpm, :etmp, :oilp, :ctmp,
                   :bat, :odo, :flt)
            """), {
                "tid":  d["truck_id"],
                "rat":  d["recorded_at"],
                "lat":  d["latitude"],
                "lon":  d["longitude"],
                "spd":  d["speed_kmh"],
                "rpm":  d["rpm"],
                "etmp": d["engine_temp_c"],
                "oilp": d["oil_pressure_bar"],
                "ctmp": d["coolant_temp_c"],
                "bat":  d["battery_v"],
                "odo":  d["odometer_km"],
                "flt":  d.get("fault_code")
            })

            # Detectar anomalÃ­as
            alerts = check_anomalies(d)
            if alerts:
                print(f"  âš ï¸  CamiÃ³n {truck_id}: {', '.join(alerts)}")

            ok += 1
            time.sleep(0.1)

        except Exception as e:
            print(f"  âŒ ExcepciÃ³n camiÃ³n {truck_id}: {e}")
            errs += 1

    session.commit()
    session.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ciclo: {ok} OK Â· {errs} errores")


if __name__ == "__main__":
    print(f"ðŸš› Ingestor activo â€” ciclo cada {INTERVAL}s. Ctrl+C para detener.")
    print(f"   Conectado a: {DATABASE_URL}")
    print(f"   API: {API_BASE}")
    print(f"   Camiones monitoreados: {N_TRUCKS}\n")
    while True:
        ingest_cycle()
        time.sleep(INTERVAL)
