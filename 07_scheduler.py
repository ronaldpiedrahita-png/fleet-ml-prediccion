# 07_scheduler.py
# Ejecutar: python 07_scheduler.py
# Predice automáticamente todos los camiones cada hora

import requests
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# ── Configuración ──────────────────────────────────────────
API_BASE  = "http://localhost:8000"
N_TRUCKS  = 200
LOG_FILE  = "logs/scheduler.log"

# ── Logging ────────────────────────────────────────────────
import os
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Jobs ───────────────────────────────────────────────────
def predict_all_trucks():
    """
    Job principal: predice todos los camiones y loggea alertas.
    Se ejecuta automáticamente cada hora.
    """
    log.info(f"Iniciando ciclo de predicción — {N_TRUCKS} camiones")
    start    = datetime.now()
    ok       = 0
    errs     = 0
    critical = []
    high     = []

    for truck_id in range(1, N_TRUCKS + 1):
        try:
            r = requests.get(
                f"{API_BASE}/predict/truck/{truck_id}/auto",
                timeout=5
            )
            if r.status_code == 200:
                data = r.json()
                level = data["alert_level"]
                prob  = data["failure_prob"]

                if level == "CRITICAL":
                    critical.append((truck_id, prob))
                elif level == "HIGH":
                    high.append((truck_id, prob))
                ok += 1
            else:
                log.warning(f"  Camión {truck_id}: status {r.status_code}")
                errs += 1

        except Exception as e:
            log.error(f"  Camión {truck_id}: {e}")
            errs += 1

    elapsed = (datetime.now() - start).seconds

    # Resumen del ciclo
    log.info(f"Ciclo completado en {elapsed}s — {ok} OK · {errs} errores")
    log.info(f"Alertas CRITICAL: {len(critical)} | HIGH: {len(high)}")

    # Loggear camiones críticos
    if critical:
        log.warning(f"CRITICAL ({len(critical)} camiones):")
        for tid, prob in critical:
            log.warning(f"  → Camión {tid}: prob={prob}")

    if high:
        log.info(f"HIGH ({len(high)} camiones):")
        for tid, prob in high:
            log.info(f"  → Camión {tid}: prob={prob}")


def check_api_health():
    """
    Job secundario: verifica que la API esté funcionando.
    Se ejecuta cada 5 minutos.
    """
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            log.info(f"Health check OK — modelo: {data['model_version']}")
        else:
            log.error(f"Health check FAILED — status: {r.status_code}")
    except Exception as e:
        log.error(f"API no disponible: {e}")


def fleet_summary_report():
    """
    Job de reporte: imprime resumen de la flota.
    Se ejecuta cada 30 minutos.
    """
    try:
        r = requests.get(f"{API_BASE}/fleet/summary", timeout=3)
        if r.status_code == 200:
            data = r.json()
            log.info(f"Resumen flota: {data['summary']} | Total predicciones: {data['total_predictions']}")
    except Exception as e:
        log.error(f"Error obteniendo resumen: {e}")


# ── Scheduler ──────────────────────────────────────────────
if __name__ == "__main__":
    scheduler = BlockingScheduler(timezone="America/Bogota")

    # Job 1: predecir todos los camiones cada 1 hora
    scheduler.add_job(
        predict_all_trucks,
        trigger=IntervalTrigger(hours=1),
        id="predict_fleet",
        name="Predicción completa de flota",
        replace_existing=True
    )

    # Job 2: health check cada 5 minutos
    scheduler.add_job(
        check_api_health,
        trigger=IntervalTrigger(minutes=5),
        id="health_check",
        name="Health check API",
        replace_existing=True
    )

    # Job 3: reporte cada 30 minutos
    scheduler.add_job(
        fleet_summary_report,
        trigger=IntervalTrigger(minutes=30),
        id="fleet_report",
        name="Reporte de flota",
        replace_existing=True
    )

    # Ejecutar inmediatamente al iniciar
    log.info("Scheduler iniciado. Ejecutando primera predicción...")
    predict_all_trucks()
    fleet_summary_report()

    log.info("Jobs programados:")
    log.info("  Predicción completa → cada 1 hora")
    log.info("  Health check        → cada 5 minutos")
    log.info("  Reporte de flota    → cada 30 minutos")
    log.info("Presiona Ctrl+C para detener\n")

    try:
        scheduler.start()
    except KeyboardInterrupt:
        log.info("Scheduler detenido.")