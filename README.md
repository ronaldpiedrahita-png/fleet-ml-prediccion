# 🚛 FleetML — Sistema de Mantenimiento Predictivo

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple?logo=railway)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)

Sistema end-to-end de **mantenimiento predictivo** para flotas de tracto-camiones. Predice la probabilidad de fallo de motor en los próximos 30 días usando telemetría en tiempo real, datos de combustible y Machine Learning.

**Demo en producción:** https://fleet-ml-prediccion-production.up.railway.app/docs

---

## Arquitectura del Sistema

```
Telemetría GPS (API Mock)
        ↓
Ingesta cada 30s → PostgreSQL
        ↓
Feature Engineering (SQL con ventanas de tiempo)
        ↓
Modelo ML (Random Forest · AUC=1.0)
        ↓
API REST en producción (FastAPI)
        ↓
Dashboard Web con mapa en tiempo real
```

---

## Stack Tecnológico

| Capa | Tecnología |
|---|---|
| Base de datos | PostgreSQL + SQLAlchemy ORM |
| API de sensores | FastAPI (servidor mock de telemetría) |
| API externa | CRE México (precios de combustible) |
| ML Pipeline | scikit-learn · XGBoost · Random Forest |
| Explainability | SHAP values |
| Experiment tracking | MLflow |
| API de predicción | FastAPI + Pydantic v2 |
| Automatización | APScheduler |
| Frontend | HTML + CSS + Leaflet.js (mapa GPS) |
| Containerización | Docker + docker-compose |
| Deploy | Railway (nube) |

---

## Características Principales

- **200 tracto-camiones** monitoreados en tiempo real
- **Telemetría por sensores**: temperatura motor, RPM, presión de aceite, refrigerante, batería, GPS
- **Ingesta paralela** con 20 hilos simultáneos (ThreadPoolExecutor)
- **Feature engineering** con ventanas de tiempo SQL (7d, 30d)
- **Modelo explicable** — SHAP values indican cuál sensor causó la alerta
- **4 niveles de alerta**: OK / WATCH / HIGH / CRITICAL
- **Dashboard en tiempo real** con mapa de México, auto-refresh cada 30s
- **Schedule automático** — predicciones cada hora sin intervención
- **API documentada** con Swagger UI
- **Deploy completo** en Docker y Railway

---

## Pipeline de Datos — 6 Etapas

### Etapa 1 — Base de Datos SQL
Esquema relacional completo con SQLAlchemy ORM. Tablas: `trucks`, `telemetry`, `fuel_logs`, `maintenance_events`, `ml_predictions`. Seed de 200 camiones con 90 días de historial sintético realista.

### Etapa 2 — API de Telemetría
Servidor FastAPI que simula una API de telemática real (similar a Samsara/Geotab). Cliente de ingesta paralela que consulta los 200 camiones con autenticación Bearer Token y detecta anomalías en tiempo de ingesta.

### Etapa 3 — API de Combustible
Consume la API pública de la CRE México (datos abiertos, sin API key). Calcula rendimiento km/l por camión y detecta camiones con rendimiento bajo — señal de fallo de motor inminente.

### Etapa 4 — Feature Engineering
SQL avanzado con `GROUP BY`, `CASE WHEN`, `JOIN`, funciones de ventana `ROW_NUMBER()` para extraer 25 features de series de tiempo por camión.

### Etapa 5 — Entrenamiento ML
Comparación Random Forest vs XGBoost en MLflow. SMOTE para desbalance de clases. SHAP values para explainability. AUC-ROC: 1.0.

### Etapa 6 — API de Predicción
FastAPI con Pydantic v2, endpoint de predicción automática desde SQL, historial por camión y resumen de flota.

---

## Endpoints de la API

```
GET  /health                         → Estado del sistema
GET  /fleet/summary                  → Resumen de alertas de toda la flota
GET  /fleet/alerts                   → Camiones HIGH y CRITICAL activos
GET  /fleet/positions                → Posición GPS de todos los camiones
GET  /predict/truck/{id}/auto        → Predicción automática desde BD
POST /predict/truck                  → Predicción con datos manuales
GET  /trucks/{id}/history            → Historial de predicciones
GET  /docs                           → Documentación Swagger
```

---

## Estructura del Proyecto

```
fleet-ml-prediccion/
├── 01_fleet_db_setup.py        # Esquema SQL + seed data
├── 02a_telemetry_server.py     # Servidor mock de sensores
├── 02b_telemetry_ingestor.py   # Cliente de ingesta paralela
├── 03_fuel_api.py              # API de combustible CRE
├── 04_feature_engineering.py   # Features desde SQL
├── 05_train_model.py           # Entrenamiento + MLflow + SHAP
├── fleet_api.py                # API de predicción (FastAPI)
├── dashboard.py                # Servidor del dashboard web
├── 07_scheduler.py             # Jobs automáticos (APScheduler)
├── templates/
│   └── dashboard.html          # Dashboard con mapa Leaflet
├── models/                     # Modelo entrenado (.pkl)
├── data/                       # Datasets generados (.csv)
├── Dockerfile                  # Imagen Docker
├── docker-compose.yml          # Orquestación de servicios
└── requirements.txt            # Dependencias Python
```

---

## Ejecución Local

### Requisitos
- Python 3.11+
- PostgreSQL 15
- Docker Desktop (para deploy en contenedores)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/TuUsuario/fleet-ml-prediccion.git
cd fleet-ml-prediccion

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos
# Crear archivo .env con:
# DATABASE_URL=postgresql://postgres:tu_password@localhost/fleetdb
```

### Ejecutar el pipeline completo

```bash
# 1. Crear BD y datos
python 01_fleet_db_setup.py

# 2. Servidor de telemetría (Terminal 1)
uvicorn 02a_telemetry_server:app --port 8001

# 3. Ingestor de datos (Terminal 2)
python 02b_telemetry_ingestor.py

# 4. API de combustible
python 03_fuel_api.py

# 5. Feature engineering
python 04_feature_engineering.py

# 6. Entrenar modelo
python 05_train_model.py

# 7. API de predicción (Terminal 3)
uvicorn fleet_api:app --port 8000

# 8. Dashboard web (Terminal 4)
uvicorn dashboard:app --port 8080

# 9. Scheduler automático (Terminal 5)
python 07_scheduler.py
```

### Con Docker (recomendado)

```bash
# Construir y levantar todo el sistema
docker compose up --build -d

# Ver logs
docker compose logs -f

# Detener
docker compose down
```

**URLs disponibles:**
- `http://localhost:8000/docs` → API de predicción
- `http://localhost:8080` → Dashboard web
- `http://localhost:8001/docs` → Mock de telemetría
- `http://localhost:5050` → PgAdmin

---

## Resultados del Modelo

| Métrica | Random Forest | XGBoost |
|---|---|---|
| AUC-ROC | 1.0000 | 1.0000 |
| F1 Score | 1.0000 | 1.0000 |
| Recall (fallo) | 1.0000 | 1.0000 |

**Top features (SHAP):**

| Feature | Importancia |
|---|---|
| total_downtime_days | 0.097 |
| total_fallos | 0.082 |
| ratio_fallos | 0.068 |
| max_temp_7d | 0.033 |
| avg_temp_7d | 0.031 |
| avg_oil_7d | 0.027 |

---

## Lo que aprendí en este proyecto

- Diseño de esquemas SQL para datos industriales de series de tiempo
- Consumo y creación de APIs REST con autenticación Bearer Token
- Feature engineering avanzado con ventanas de tiempo en SQL
- Mantenimiento predictivo con modelos de clasificación binaria
- Explainability con SHAP values para contextos industriales
- Deploy de modelos ML como APIs REST con FastAPI
- Containerización con Docker y docker-compose
- Deploy en producción con Railway

---

## Autor

**Ronald** — ingeniero industrial - especialista Analitica bigdata

[![GitHub](https://img.shields.io/badge/GitHub-TuUsuario-black?logo=github)](https://github.com/TuUsuario)

---

*Proyecto desarrollado como parte del portafolio de Data Science y Machine Learning en producción.*
