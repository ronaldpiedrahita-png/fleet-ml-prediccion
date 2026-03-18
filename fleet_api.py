# 06_fleet_api.py
# Ejecutar: uvicorn 06_fleet_api:app --reload --port 8000
# Docs: http://localhost:8000/docs

import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from fastapi.middleware.cors import CORSMiddleware

# ── Conexión directa ───────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:Valeria2004@localhost/fleetdb")
engine        = create_engine(DATABASE_URL)
SessionLocal  = sessionmaker(bind=engine)
MODEL_VER     = "v1.0"
THRESHOLD     = 0.40
store         = {}

# ── Cargar modelo al iniciar ───────────────────────────────
@asynccontextmanager
async def lifespan(app):
    try:
        store["model"]    = joblib.load("models/fleet_model.pkl")
        store["features"] = joblib.load("models/feature_names.pkl")
        store["shap_df"]  = joblib.load("models/shap_importance.pkl")
        print(f"✅ Modelo {MODEL_VER} cargado. Threshold={THRESHOLD}")
        print(f"   Features: {len(store['features'])}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
    yield
    store.clear()

app = FastAPI(
    title="Fleet Maintenance Prediction API",
    description="Predice probabilidad de fallo de motor en flota de tracto-camiones",
    version=MODEL_VER,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# ── Dependency base de datos ───────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ── Schemas Pydantic ───────────────────────────────────────
class TruckFeatures(BaseModel):
    truck_id:              int
    truck_age_years:       float = 0.0
    odometer_km:           float = 0.0
    engine_hours:          float = 0.0
    avg_temp_7d:           float = 0.0
    max_temp_7d:           float = 0.0
    std_temp_7d:           float = 0.0
    avg_oil_7d:            float = 0.0
    min_oil_7d:            float = 0.0
    avg_rpm_7d:            float = 0.0
    avg_coolant_7d:        float = 0.0
    avg_battery_7d:        float = 0.0
    fault_codes_7d:        float = 0.0
    fault_codes_30d:       float = 0.0
    avg_kpl_30d:           float = 0.0
    min_kpl_30d:           float = 0.0
    kpl_trend:             float = 0.0
    total_fallos:          float = 0.0
    ratio_fallos:          float = 0.0
    days_since_last_maint: float = 0.0
    total_downtime_days:   float = 0.0
    km_recorridos_30d:     float = 0.0
    temp_trend:            float = 0.0
    oil_trend:             float = 0.0
    fault_accel:           float = 0.0
    overdue_maint:         float = 0.0

class PredictionOut(BaseModel):
    truck_id:        int
    failure_prob:    float
    alert_level:     str
    top_risk_factor: str
    predicted_at:    str
    model_version:   str
    recommendation:  str

# ── Helpers ────────────────────────────────────────────────
def prob_to_alert(p: float):
    if   p > 0.80: return "CRITICAL", "⛔ Sacar de servicio inmediatamente"
    elif p > 0.60: return "HIGH",     "⚠️  Programar mantenimiento esta semana"
    elif p > 0.40: return "WATCH",    "👀 Monitorear — revisión en 15 días"
    else:          return "OK",       "✅ Sin alertas activas"

def get_top_risk() -> str:
    try:
        shap_df = store["shap_df"]
        top     = shap_df.iloc[0]["feature"]
        return top.replace("_", " ").title()
    except:
        return "Sin datos SHAP"

# ── Endpoints ──────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status":  "ok",
        "model":   MODEL_VER,
        "docs":    "http://localhost:8000/docs",
        "health":  "http://localhost:8000/health"
    }

@app.get("/fleet/positions")
def fleet_positions(db: Session = Depends(get_db)):
    """Retorna la última posición conocida de cada camión."""
    rows = db.execute(text("""
        SELECT DISTINCT ON (t.truck_id)
            t.truck_id,
            tr.plate,
            tr.model,
            tr.base_city,
            t.latitude,
            t.longitude,
            t.speed_kmh,
            t.engine_temp_c,
            t.oil_pressure_bar,
            t.recorded_at,
            COALESCE(mp.alert_level, 'OK') AS alert_level,
            COALESCE(mp.failure_prob, 0)   AS failure_prob
        FROM telemetry t
        JOIN trucks tr ON tr.id = t.truck_id
        LEFT JOIN (
            SELECT DISTINCT ON (truck_id)
                truck_id, alert_level, failure_prob
            FROM ml_predictions
            ORDER BY truck_id, predicted_at DESC
        ) mp ON mp.truck_id = t.truck_id
        WHERE t.latitude IS NOT NULL
        ORDER BY t.truck_id, t.recorded_at DESC
    """)).fetchall()

    return [{
        "truck_id":      r[0],
        "plate":         r[1],
        "model":         r[2],
        "city":          r[3],
        "lat":           r[4],
        "lon":           r[5],
        "speed_kmh":     r[6],
        "engine_temp_c": r[7],
        "oil_pressure":  r[8],
        "last_seen":     str(r[9]),
        "alert_level":   r[10],
        "failure_prob":  r[11]
    } for r in rows]

@app.get("/health")
def health():
    return {
        "status":       "healthy",
        "model_loaded": "model" in store,
        "model_version": MODEL_VER,
        "threshold":    THRESHOLD,
        "db":           DATABASE_URL.split("@")[1]  # solo host/db, sin contraseña
    }

@app.post("/predict/truck", response_model=PredictionOut)
def predict_failure(data: TruckFeatures, db: Session = Depends(get_db)):
    """
    Predice la probabilidad de fallo de motor en los próximos 30 días.
    Guarda la predicción en SQL para auditoría y monitoreo.
    """
    if "model" not in store:
        raise HTTPException(500, detail="Modelo no cargado")

    model    = store["model"]
    features = store["features"]

    # Construir vector de features en el orden correcto
    X = np.array([[getattr(data, f, 0.0) for f in features]])

    prob       = float(model.predict_proba(X)[0, 1])
    level, rec = prob_to_alert(prob)
    top        = get_top_risk()
    now        = datetime.utcnow()

    # Guardar predicción en SQL
    try:
        db.execute(text("""
            INSERT INTO ml_predictions
              (truck_id, failure_prob, alert_level, predicted_at, model_version, top_risk_factor)
            VALUES
              (:tid, :prob, :lvl, :ts, :ver, :top)
        """), {
            "tid":  data.truck_id,
            "prob": prob,
            "lvl":  level,
            "ts":   now,
            "ver":  MODEL_VER,
            "top":  top
        })
        db.commit()
    except Exception as e:
        print(f"  ⚠️  Error guardando predicción: {e}")

    return PredictionOut(
        truck_id        = data.truck_id,
        failure_prob    = round(prob, 4),
        alert_level     = level,
        top_risk_factor = top,
        predicted_at    = now.isoformat(),
        model_version   = MODEL_VER,
        recommendation  = rec
    )


@app.get("/predict/truck/{truck_id}/auto")
def predict_from_db(truck_id: int, db: Session = Depends(get_db)):
    """
    Predice automáticamente usando los datos más recientes del camión
    que ya están en la base de datos. No necesitas enviar JSON manual.
    """
    # Obtener features del camión desde SQL
    row = db.execute(text("""
        SELECT * FROM ml_fleet_features WHERE truck_id = :id LIMIT 1
    """), {"id": truck_id}).fetchone()

    if not row:
        raise HTTPException(404, detail=f"Camión {truck_id} no encontrado en ml_fleet_features")

    features = store["features"]
    cols     = list(row._mapping.keys())

    # Construir vector
    X = np.array([[
        float(row._mapping.get(f, 0.0) or 0.0)
        for f in features
    ]])

    prob       = float(store["model"].predict_proba(X)[0, 1])
    level, rec = prob_to_alert(prob)
    top        = get_top_risk()
    now        = datetime.utcnow()

    # Guardar en SQL
    try:
        db.execute(text("""
            INSERT INTO ml_predictions
              (truck_id, failure_prob, alert_level, predicted_at, model_version, top_risk_factor)
            VALUES (:tid, :prob, :lvl, :ts, :ver, :top)
        """), {
            "tid": truck_id, "prob": prob, "lvl": level,
            "ts": now, "ver": MODEL_VER, "top": top
        })
        db.commit()
    except Exception as e:
        print(f"  ⚠️  Error guardando: {e}")

    return {
        "truck_id":        truck_id,
        "failure_prob":    round(prob, 4),
        "alert_level":     level,
        "top_risk_factor": top,
        "recommendation":  rec,
        "predicted_at":    now.isoformat()
    }


@app.get("/fleet/alerts")
def get_active_alerts(db: Session = Depends(get_db)):
    """Lista camiones con alertas HIGH o CRITICAL en las últimas 24 horas."""
    rows = db.execute(text("""
        SELECT DISTINCT ON (mp.truck_id)
            mp.truck_id, t.plate, t.model, t.base_city,
            mp.failure_prob, mp.alert_level,
            mp.top_risk_factor, mp.predicted_at
        FROM ml_predictions mp
        JOIN trucks t ON t.id = mp.truck_id
        WHERE mp.alert_level IN ('HIGH', 'CRITICAL')
          AND mp.predicted_at > NOW() - INTERVAL '24 hours'
        ORDER BY mp.truck_id, mp.predicted_at DESC
    """)).fetchall()

    return {
        "total_alerts": len(rows),
        "alerts": [{
            "truck_id":    r[0],
            "plate":       r[1],
            "model":       r[2],
            "city":        r[3],
            "failure_prob": r[4],
            "alert_level": r[5],
            "top_risk":    r[6],
            "last_check":  str(r[7])
        } for r in rows]
    }


@app.get("/fleet/summary")
def fleet_summary(db: Session = Depends(get_db)):
    """Resumen del estado de toda la flota."""
    rows = db.execute(text("""
        SELECT
            alert_level,
            COUNT(*) as cantidad
        FROM (
            SELECT DISTINCT ON (truck_id)
                truck_id, alert_level
            FROM ml_predictions
            ORDER BY truck_id, predicted_at DESC
        ) latest
        GROUP BY alert_level
        ORDER BY alert_level
    """)).fetchall()

    return {
        "summary":    {r[0]: r[1] for r in rows},
        "total_predictions": db.execute(
            text("SELECT COUNT(*) FROM ml_predictions")
        ).scalar()
    }


@app.get("/trucks/{truck_id}/history")
def truck_history(truck_id: int, db: Session = Depends(get_db)):
    """Historial de predicciones de un camión específico."""
    # Verificar que el camión existe
    truck = db.execute(text(
        "SELECT plate, model FROM trucks WHERE id = :id"
    ), {"id": truck_id}).fetchone()

    if not truck:
        raise HTTPException(404, detail=f"Camión {truck_id} no existe")

    rows = db.execute(text("""
        SELECT failure_prob, alert_level, top_risk_factor, predicted_at
        FROM ml_predictions
        WHERE truck_id = :id
        ORDER BY predicted_at DESC
        LIMIT 30
    """), {"id": truck_id}).fetchall()

    return {
        "truck_id": truck_id,
        "plate":    truck[0],
        "model":    truck[1],
        "predictions": [{
            "prob":  r[0],
            "level": r[1],
            "risk":  r[2],
            "at":    str(r[3])
        } for r in rows]
    }