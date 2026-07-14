"""Tests de la API de prediccion (FastAPI TestClient).

No requieren PostgreSQL: la escritura de la prediccion en BD esta protegida con
try/except, asi que el endpoint responde igual sin base de datos.
"""
import os

# Usar SQLite en memoria: el test no necesita PostgreSQL ni el driver psycopg2.
# Debe fijarse ANTES de importar fleetml.api (que crea el engine al importarse).
os.environ["DATABASE_URL"] = "sqlite://"

from fastapi.testclient import TestClient  # noqa: E402

from fleetml import api as fleet_api  # noqa: E402


def test_prob_to_alert_thresholds():
    assert fleet_api.prob_to_alert(0.90)[0] == "CRITICAL"
    assert fleet_api.prob_to_alert(0.70)[0] == "HIGH"
    assert fleet_api.prob_to_alert(0.50)[0] == "WATCH"
    assert fleet_api.prob_to_alert(0.10)[0] == "OK"


def test_root_endpoint_ok():
    # El context manager dispara el lifespan, que carga el modelo.
    with TestClient(fleet_api.app) as client:
        resp = client.get("/")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_health_reports_model_loaded():
    with TestClient(fleet_api.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["model_loaded"] is True


def test_predict_returns_valid_prediction():
    # Un camion muy desgastado -> el endpoint debe devolver una prediccion valida.
    payload = {
        "truck_id": 1, "odometer_km": 780_000, "truck_age_years": 9,
        "avg_temp_7d": 104, "max_temp_7d": 112, "avg_oil_7d": 1.9,
        "min_oil_7d": 1.4, "avg_kpl_30d": 5.2, "fault_codes_7d": 4,
    }
    with TestClient(fleet_api.app) as client:
        resp = client.post("/predict/truck", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["failure_prob"] <= 1.0
        assert body["alert_level"] in {"OK", "WATCH", "HIGH", "CRITICAL"}
        assert body["truck_id"] == 1
