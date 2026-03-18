# 02a_telemetry_server.py  ← correr con: uvicorn 02a_telemetry_server:app --port 8001
# pip install fastapi uvicorn numpy

import random, time
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import Optional

app       = FastAPI(title="Fleet Telemetry Mock API", version="1.0")
API_TOKEN = "fleet-secret-token-2024"  # en producción: JWT / OAuth2

# Estado "vivo" de cada camión (simula degradación progresiva)
truck_state: dict[int, dict] = {}

def get_truck_state(truck_id: int) -> dict:
    if truck_id not in truck_state:
        # Inicializar estado base del camión
        truck_state[truck_id] = {
            "odometer":       random.uniform(50_000, 500_000),
            "base_temp":      random.uniform(86, 92),
            "degradation":    random.random() > 0.75,  # 25% de camiones degradados
            "lat":            random.uniform(19, 26),
            "lon":            random.uniform(-105, -97),
        }
    return truck_state[truck_id]


class TelemetryReading(BaseModel):
    truck_id:         int
    recorded_at:      str
    latitude:         float
    longitude:        float
    speed_kmh:        float
    rpm:              float
    engine_temp_c:    float
    oil_pressure_bar: float
    coolant_temp_c:   float
    battery_v:        float
    odometer_km:      float
    fault_code:       Optional[str] = None


def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {API_TOKEN}":
        raise HTTPException(401, detail="Token inválido")


@app.get("/telemetry/{truck_id}/live", response_model=TelemetryReading)
def get_live_telemetry(truck_id: int, authorization: str = Header(...)):
    """Retorna lectura en vivo de sensores para un camión."""
    verify_token(authorization)
    state = get_truck_state(truck_id)
    deg   = state["degradation"]

    moving  = random.random() > 0.3
    speed   = random.uniform(60, 105) if moving else 0.0
    rpm     = random.uniform(1200, 1900) if moving else 700

    # Simular degradación gradual del motor
    temp_base  = state["base_temp"] + (random.uniform(5, 20) if deg else 0)
    oil_press  = random.uniform(1.5, 3.0) if deg else random.uniform(3.5, 5.0)
    fault      = None
    if deg and random.random() > 0.95:
        fault = random.choice(["P0217", "P0524", "P0118", "P0300", "P0562"])

    state["odometer"] += speed / 3600
    state["lat"]       += np.random.normal(0, 0.001)
    state["lon"]       += np.random.normal(0, 0.001)

    return TelemetryReading(
        truck_id         = truck_id,
        recorded_at      = datetime.utcnow().isoformat(),
        latitude         = round(state["lat"], 6),
        longitude        = round(state["lon"], 6),
        speed_kmh        = round(speed, 1),
        rpm              = round(rpm, 0),
        engine_temp_c    = round(temp_base + np.random.normal(0, 2), 1),
        oil_pressure_bar = round(oil_press, 2),
        coolant_temp_c   = round(random.uniform(85, 108 if deg else 95), 1),
        battery_v        = round(random.uniform(13.4, 14.8), 2),
        odometer_km      = round(state["odometer"], 1),
        fault_code       = fault,
    )


@app.get("/fleet/status")
def fleet_status(authorization: str = Header(...)):
    verify_token(authorization)
    return {"active_trucks": len(truck_state), "server_time": datetime.utcnow().isoformat()}