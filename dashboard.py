# dashboard.py
# Ejecutar: uvicorn dashboard:app --port 8080
# Ver en:   http://localhost:8080

import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

os.makedirs("templates", exist_ok=True)

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
app      = FastAPI(title="Fleet Dashboard")
templates = Jinja2Templates(directory="templates")


def get_fleet_data():
    """Obtiene todos los datos necesarios de la API."""
    try:
        summary = requests.get(f"{API_BASE}/fleet/summary", timeout=3).json()
        alerts  = requests.get(f"{API_BASE}/fleet/alerts",  timeout=3).json()
        health  = requests.get(f"{API_BASE}/health",        timeout=3).json()
        return summary, alerts, health
    except Exception as e:
        return {}, {"alerts": []}, {}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    summary, alerts, health = get_fleet_data()
    return templates.TemplateResponse("dashboard.html", {
        "request":  request,
        "summary":  summary,
        "alerts":   alerts,
        "health":   health,
        "api_base": API_BASE
    })


@app.get("/api/data")
def api_data():
    """Endpoint para actualización en tiempo real desde el frontend."""
    summary, alerts, health = get_fleet_data()
    return {
        "summary": summary,
        "alerts":  alerts,
        "health":  health
    }