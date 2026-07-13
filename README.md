# 🚛 FleetML — Sistema de Mantenimiento Predictivo

[![CI](https://github.com/ronaldpiedrahita-png/fleet-ml-prediccion/actions/workflows/ci.yml/badge.svg)](https://github.com/ronaldpiedrahita-png/fleet-ml-prediccion/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-blue?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)

Sistema end-to-end de **mantenimiento predictivo** para flotas de tracto-camiones. Predice la probabilidad de fallo de motor en los próximos 30 días usando telemetría en tiempo real, datos de combustible y Machine Learning.

> Se ejecuta localmente con Docker (ver "Ejecución Local"). Documentación
> interactiva de la API disponible en `/docs` (Swagger UI).

---

## Problema de negocio e impacto

En el transporte de carga, un **fallo de motor no planificado** no es solo una
reparación: es un tracto-camión parado en carretera, entregas incumplidas, una
grúa, y una reparación de emergencia que cuesta varias veces más que una
programada. En la industria, el costo de una avería mayor y sus días fuera de
servicio se mide en **decenas de miles de dólares por evento**.

Hay tres formas de mantener una flota:

| Estrategia | Cómo funciona | Problema |
|---|---|---|
| **Reactiva** | Reparar cuando se rompe | Averías en ruta, costo máximo |
| **Preventiva** | Calendario fijo (cada X km) | Sobre-mantiene equipos sanos |
| **Predictiva** (este proyecto) | Reparar *justo antes* del fallo | Requiere datos + modelo |

**Qué resuelve FleetML:** anticipa con ~30 días la probabilidad de fallo de motor
por camión a partir de su telemetría, para pasar de apagar incendios a **planificar
el taller**. El modelo prioriza *recall* (detectar el máximo de fallos reales)
porque, en este dominio, **no detectar una avería cuesta mucho más que una falsa
alarma** — y esa decisión de negocio es la que fija el umbral de clasificación.

> Las cifras de costo son contexto del sector (ilustrativas), no resultados medidos
> de este proyecto, que usa datos sintéticos (ver más abajo).

---

## Arquitectura del Sistema

```
Telemetría GPS (API Mock)
        ↓
Ingesta cada 30s → PostgreSQL
        ↓
Feature Engineering (SQL con ventanas de tiempo)
        ↓
Modelo ML (Random Forest · AUC≈0.78, validado 5-fold)
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
| Deploy | Docker (self-host) |

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
- **Deploy completo** con Docker y docker-compose

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
Comparación Random Forest vs XGBoost en MLflow. Ponderación de clases (`class_weight="balanced"` / `scale_pos_weight`) para el desbalance. Evaluación con **validación cruzada 5-fold** y **predicciones out-of-fold** (más robusto que un único split con 200 filas). SHAP values para explainability. **AUC-ROC ≈ 0.78.**

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
├── 01_fleet_db_setup.py        # Esquema SQL + seed data (desgaste + fallo probabilístico)
├── 02a_telemetry_server.py     # Servidor mock de sensores
├── 02b_telemetry_ingestor.py   # Cliente de ingesta paralela
├── 03_fuel_api.py              # API de combustible CRE
├── 04_feature_engineering.py   # Features desde SQL (sin columnas con fuga)
├── 05_train_model.py           # Entrenamiento + CV + SHAP
├── generate_dataset.py         # Genera el dataset sintético SIN base de datos
├── ml_features.py              # Selección de features (fuente única, anti-fuga)
├── fleet_api.py                # API de predicción (FastAPI)
├── dashboard.py                # Servidor del dashboard web
├── 07_scheduler.py             # Jobs automáticos (APScheduler)
├── tests/                      # Suite de pytest (fuga, modelo, API, datos)
├── .github/workflows/ci.yml    # Integración continua (GitHub Actions)
├── templates/dashboard.html    # Dashboard con mapa Leaflet
├── models/                     # Modelo entrenado (.pkl)
├── data/                       # Datasets generados (.csv)
├── .env.example                # Plantilla de variables de entorno
├── Dockerfile · docker-compose.yml
├── requirements.txt · pytest.ini · LICENSE
```

---

## Ejecución Local

### Requisitos
- Python 3.12
- PostgreSQL 15 *(opcional — ver "Opción rápida" más abajo)*
- Docker Desktop (para deploy en contenedores)

### Instalación

```bash
# Clonar repositorio
git clone https://github.com/ronaldpiedrahita-png/fleet-ml-prediccion.git
cd fleet-ml-prediccion

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Instalar dependencias
pip install -r requirements.txt

# Configurar base de datos: copiar la plantilla y poner tu contraseña
cp .env.example .env      # luego edita DATABASE_URL en .env
```

### Opción rápida — reproducir el modelo sin base de datos

El dataset de features se puede **regenerar sin PostgreSQL** (datos sintéticos
realistas). Ideal para revisar el modelo end-to-end en segundos:

```bash
python generate_dataset.py      # genera data/fleet_features.csv (sin BD)
python 05_train_model.py        # entrena y reporta métricas honestas (AUC ~0.78)
pytest                          # corre los 11 tests (incluye guarda anti-fuga)
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

## Datos y transparencia

Los datos son **100% sintéticos** — no provienen de una flota real. Se generan con
un modelo físico simplificado: cada camión tiene un *desgaste* latente que crece
con el kilometraje y la edad, se refleja en sus sensores **con ruido de medición**
(las lecturas de camiones sanos y averiados se solapan, como en la realidad) y
eleva la probabilidad de fallo de forma **probabilística**.

Esto es deliberado. Una versión anterior generaba los datos con un umbral duro que
hacía el problema trivial (AUC = 1.0 por fuga de datos). Los datos actuales
producen un problema **aprendible pero no trivial**, con métricas creíbles, y un
test automático impide que la fuga reaparezca.

---

## Resultados del Modelo

Métricas por **validación cruzada 5-fold** sobre 200 camiones (24.5% con fallo).
El umbral se fija priorizando **recall** (en mantenimiento predictivo, no detectar
un fallo cuesta más que una falsa alarma).

| Métrica | Random Forest | XGBoost |
|---|---|---|
| AUC-ROC (out-of-fold) | **0.78** | 0.74 |
| Recall (fallo) | 0.76 | 0.71 |
| Precision (fallo) | 0.39 | 0.44 |
| F1 (fallo) | 0.52 | 0.54 |

> **Nota de honestidad:** una versión anterior reportaba AUC = 1.0. Eso era
> **fuga de datos** (features derivadas de la propia etiqueta de fallo) sobre un
> generador de datos degenerado. Se corrigió: datos sintéticos realistas (fallo
> probabilístico, sensores con ruido que se solapan) y un test automático
> (`tests/test_no_target_leakage.py`) que impide que la fuga reaparezca. Un AUC
> de 0.78 honesto vale más que un 1.0 falso.

**Top features (SHAP)** — ahora señales físicas legítimas, no derivadas del target:

| Feature | Importancia | Interpretación |
|---|---|---|
| temp_trend | 0.054 | Tendencia al alza de temperatura |
| avg_kpl_30d | 0.050 | Caída de rendimiento de combustible |
| avg_oil_7d | 0.036 | Presión de aceite baja |
| min_oil_7d | 0.032 | Mínimo de presión de aceite |
| std_temp_7d | 0.028 | Inestabilidad térmica |
| avg_coolant_7d | 0.025 | Temperatura de refrigerante |

---

## Lo que aprendí en este proyecto

- **Detectar y corregir fuga de datos (*data leakage*)**: un AUC de 1.0 casi nunca
  es un logro — aquí era una señal de que features derivadas del target y un
  generador de datos degenerado hacían el problema trivial. Aprendí a diagnosticarlo
  (AUC individual por feature), corregirlo y **blindarlo con un test automático**.
- **Elegir el umbral desde el negocio**, no desde la métrica: priorizar *recall*
  porque un falso negativo (avería no detectada) cuesta más que un falso positivo.
- **Validación honesta en datasets pequeños**: validación cruzada 5-fold y
  predicciones out-of-fold en lugar de un único split.
- Diseño de esquemas SQL para datos industriales de series de tiempo
- Feature engineering con ventanas de tiempo en SQL
- Explainability con SHAP para contextos industriales
- Deploy de modelos ML como APIs REST con FastAPI + tests + CI
- Containerización con Docker para un despliegue reproducible

---

## Autor

**Ronald Piedrahita** — Ingeniero Industrial · Especialista en Analítica y Big Data

[![GitHub](https://img.shields.io/badge/GitHub-ronaldpiedrahita--png-black?logo=github)](https://github.com/ronaldpiedrahita-png)

---

*Proyecto desarrollado como parte del portafolio de Data Science y Machine Learning en producción.*
