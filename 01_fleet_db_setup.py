# 01_fleet_db_setup.py
# pip install sqlalchemy psycopg2-binary python-dotenv faker numpy

import os, random
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, Float,
    String, Boolean, DateTime, ForeignKey, Text, Enum
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import enum

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/fleetdb")
engine       = create_engine(DATABASE_URL, echo=False)
Session      = sessionmaker(bind=engine)
Base         = declarative_base()

class Truck(Base):
    __tablename__ = "trucks"
    id             = Column(Integer, primary_key=True)
    plate          = Column(String(12), unique=True)
    model          = Column(String(60))
    brand          = Column(String(30))
    year           = Column(Integer)
    odometer_km    = Column(Float)     # km acumulados
    engine_hours   = Column(Float)     # horas de motor
    status         = Column(String(20), default="active")
    base_city      = Column(String(40))
    created_at     = Column(DateTime, default=datetime.utcnow)

    telemetry   = relationship("Telemetry",    back_populates="truck")
    fuel_logs   = relationship("FuelLog",      back_populates="truck")
    maintenance = relationship("MaintenanceEvent", back_populates="truck")


class Telemetry(Base):
    """Registro de sensores cada 15 minutos por camiÃ³n."""
    __tablename__      = "telemetry"
    id                 = Column(Integer, primary_key=True)
    truck_id           = Column(Integer, ForeignKey("trucks.id"), index=True)
    recorded_at        = Column(DateTime, index=True)
    latitude           = Column(Float)
    longitude          = Column(Float)
    speed_kmh          = Column(Float)
    rpm                = Column(Float)        # revoluciones por minuto
    engine_temp_c      = Column(Float)        # temperatura motor Â°C
    oil_pressure_bar   = Column(Float)        # presiÃ³n aceite bar
    coolant_temp_c     = Column(Float)        # temperatura refrigerante
    battery_v          = Column(Float)        # voltaje baterÃ­a
    odometer_km        = Column(Float)        # snapshot del odÃ³metro
    fault_code         = Column(String(20), nullable=True)  # cÃ³digo OBD-II

    truck = relationship("Truck", back_populates="telemetry")


class FuelLog(Base):
    """Registro de carga de combustible en cada parada."""
    __tablename__  = "fuel_logs"
    id             = Column(Integer, primary_key=True)
    truck_id       = Column(Integer, ForeignKey("trucks.id"), index=True)
    fueled_at      = Column(DateTime)
    liters         = Column(Float)      # litros cargados
    price_per_liter = Column(Float)    # precio del diÃ©sel ese dÃ­a
    total_cost_mxn = Column(Float)     # costo total en MXN
    km_since_last  = Column(Float)     # km desde Ãºltima carga
    km_per_liter   = Column(Float)     # rendimiento calculado
    station_city   = Column(String(40))

    truck = relationship("Truck", back_populates="fuel_logs")


class MaintenanceEvent(Base):
    """Historial de mantenimientos correctivos y preventivos."""
    __tablename__  = "maintenance_events"
    id             = Column(Integer, primary_key=True)
    truck_id       = Column(Integer, ForeignKey("trucks.id"), index=True)
    event_date     = Column(DateTime)
    event_type     = Column(String(30))  # engine_failure / brake / tire / oil / transmission
    is_failure     = Column(Boolean)    # TRUE = fallo no planificado (target ML)
    cost_mxn       = Column(Float)
    downtime_days  = Column(Integer)   # dÃ­as fuera de servicio
    odometer_at_event = Column(Float)
    notes          = Column(Text, nullable=True)

    truck = relationship("Truck", back_populates="maintenance")


class MLPrediction(Base):
    """Predicciones del modelo guardadas para monitoreo."""
    __tablename__  = "ml_predictions"
    id             = Column(Integer, primary_key=True)
    truck_id       = Column(Integer, ForeignKey("trucks.id"))
    failure_prob   = Column(Float)
    alert_level    = Column(String(10))  # LOW / MEDIUM / HIGH / CRITICAL
    predicted_at   = Column(DateTime, default=datetime.utcnow)
    model_version  = Column(String(20))
    top_risk_factor = Column(String(50), nullable=True)


# â”€â”€ SEED DATA â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TRUCK_MODELS = [
    ("Kenworth",     "T680"),
    ("Peterbilt",    "579"),
    ("Freightliner", "Cascadia"),
    ("Volvo",        "FH16"),
    ("International","LT Series"),
]

CITIES = [
    ("Monterrey", 25.67, -100.31),
    ("CDMX",       19.43,  -99.13),
    ("Guadalajara",20.66, -103.35),
    ("Tijuana",    32.52, -117.03),
    ("QuerÃ©taro",  20.59,  -100.39),
]

FAILURE_TYPES = ["engine_failure", "brake_system", "transmission",
                  "cooling_system", "electrical"]
MAINT_TYPES   = ["oil_change", "tire_rotation", "brake_inspection",
                  "filter_change", "full_service"]


def simulate_engine_temp(base_temp=88, wear=0.0):
    """Temperatura crece de forma continua con el desgaste, CON ruido.

    Sano (~88C) y desgastado (~104C) se solapan: el ruido de medicion impide
    que la temperatura por si sola separe perfectamente los camiones que fallan.
    """
    return round(base_temp + 16 * wear + np.random.normal(0, 3), 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def seed_fleet(n_trucks=200, days_history=90):
    session = Session()
    print(f"ðŸš› Generando flota de {n_trucks} camiones con {days_history} dÃ­as de historial...")

    for i in range(n_trucks):
        brand, model = random.choice(TRUCK_MODELS)
        city_name, lat, lon = random.choice(CITIES)
        year        = random.randint(2016, 2023)
        odometer    = random.uniform(50_000, 600_000)
        eng_hours   = odometer / random.uniform(55, 75)

        # Desgaste latente CONTINUO (0 = nuevo, ~1 = muy desgastado): km + edad
        # + variacion individual. Reemplaza al viejo umbral duro odometer>400k,
        # que hacia el problema trivial (AUC=1.0 por fuga de datos).
        wear = min(1.2, max(0.0,
            0.60 * (odometer - 50_000) / 550_000
            + 0.25 * (2023 - year) / 7
            + np.random.normal(0, 0.15)))

        # Propension a fallo: sigmoide del desgaste + un shock NO observado.
        # Al ser probabilistica (no un umbral), un camion desgastado puede no
        # fallar y uno nuevo si -> el modelo aprende, pero nunca perfecto.
        fail_propensity = sigmoid(4.5 * (wear - 0.62) + np.random.normal(0, 0.8))
        truck_will_fail = random.random() < fail_propensity

        truck = Truck(
            plate       = f"{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}{random.choice('ABCDEFGHJKLMNPRSTUVWXYZ')}-{random.randint(1000,9999)}",
            model       = model,
            brand       = brand,
            year        = year,
            odometer_km = round(odometer, 1),
            engine_hours = round(eng_hours, 1),
            status      = "critical" if truck_will_fail and random.random() > 0.7 else "active",
            base_city   = city_name,
        )
        session.add(truck)
        session.flush()

        # â”€â”€ TelemetrÃ­a: 1 registro cada hora durante dÃ­as_history â”€â”€
        odo_cursor = odometer - (days_history * random.uniform(200, 500))
        for h in range(days_history * 24):
            ts       = datetime.utcnow() - timedelta(hours=(days_history*24 - h))
            moving   = random.random() > 0.35
            speed    = random.uniform(60, 110) if moving else 0
            rpm      = random.uniform(1200, 2000) if moving else random.uniform(600, 800)
            odo_cursor += speed / 3600  # km por hora
            fault    = None
            if random.random() < 0.005 + 0.05 * wear:   # mas probable con desgaste
                fault = random.choice(["P0217", "P0524", "P0118", "P0300"])

            tl = Telemetry(
                truck_id       = truck.id,
                recorded_at    = ts,
                latitude       = lat + np.random.normal(0, 0.5),
                longitude      = lon + np.random.normal(0, 0.5),
                speed_kmh      = round(speed, 1),
                rpm            = round(rpm, 0),
                engine_temp_c  = simulate_engine_temp(wear=wear),
                oil_pressure_bar = round(max(0.8, 4.6 - 2.3 * wear + np.random.normal(0, 0.4)), 2),
                coolant_temp_c = round(88 + 12 * wear + np.random.normal(0, 3), 1),
                battery_v      = round(random.uniform(13.5, 14.8), 2),
                odometer_km    = round(odo_cursor, 1),
                fault_code     = fault,
            )
            session.add(tl)

        # â”€â”€ Combustible: carga cada ~2 dÃ­as â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        diesel_price = random.uniform(22.5, 25.8)
        last_km = odometer - (days_history * 350)
        for d in range(0, days_history, random.randint(2, 3)):
            km_driven = random.uniform(400, 800)
            kpl       = max(3.5, 7.6 - 2.2 * wear + random.gauss(0, 0.5))  # rendimiento cae con desgaste
            liters    = km_driven / kpl
            session.add(FuelLog(
                truck_id       = truck.id,
                fueled_at      = datetime.utcnow() - timedelta(days=days_history-d),
                liters         = round(liters, 1),
                price_per_liter = diesel_price,
                total_cost_mxn = round(liters * diesel_price, 2),
                km_since_last  = round(km_driven, 1),
                km_per_liter   = round(km_driven / liters, 2),
                station_city   = city_name,
            ))

        # â”€â”€ Mantenimientos: 2-8 eventos por camiÃ³n â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Si el camion falla (segun su propension probabilistica), 1..n de sus
        # eventos son fallos no planificados; si no, ninguno lo es.
        n_events   = random.randint(2, 8)
        n_fail     = random.randint(1, n_events) if truck_will_fail else 0
        fail_flags = [True] * n_fail + [False] * (n_events - n_fail)
        random.shuffle(fail_flags)
        for is_failure in fail_flags:
            etype      = random.choice(FAILURE_TYPES if is_failure else MAINT_TYPES)
            session.add(MaintenanceEvent(
                truck_id          = truck.id,
                event_date        = datetime.utcnow() - timedelta(days=random.randint(1, 85)),
                event_type        = etype,
                is_failure        = is_failure,
                cost_mxn          = random.uniform(3_000 if not is_failure else 15_000, 80_000),
                downtime_days     = 0 if not is_failure else random.randint(1, 14),
                odometer_at_event = truck.odometer_km - random.uniform(0, 50_000),
                notes             = f"Auto-generado Â· tipo: {etype}",
            ))

    session.commit()
    session.close()
    print(f"âœ… {n_trucks} camiones Â· telemetrÃ­a Â· combustible Â· mantenimientos insertados.")


if __name__ == "__main__":
    Base.metadata.create_all(engine)
    print("âœ… Tablas creadas: trucks, telemetry, fuel_logs, maintenance_events, ml_predictions")
    seed_fleet(n_trucks=200, days_history=90)
