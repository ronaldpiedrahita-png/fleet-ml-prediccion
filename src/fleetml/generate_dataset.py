"""Genera un dataset sintetico *realista* de la flota — sin necesidad de PostgreSQL.

Reemplaza al generador degenerado original, donde un umbral duro
(`odometer > 400_000`) controlaba a la vez TODOS los sensores y el fallo, con
distribuciones que no se solapaban. Eso producia un AUC=1.0 falso (fuga de datos).

Aqui el fallo es *probabilistico*: cada camion tiene un desgaste latente continuo
(`wear`) que:
  - se refleja en los sensores CON RUIDO (las distribuciones sano/degradado se
    solapan, como en la vida real), y
  - eleva la probabilidad de fallo via una sigmoide, con un componente
    NO observado (`shock`) que impide que ningun feature prediga perfecto.

Resultado: un problema de ML aprendible pero no trivial (AUC realista ~0.80-0.88).

Uso:
    python -m fleetml.generate_dataset          # escribe data/fleet_features.csv
    python -m fleetml.generate_dataset --check  # solo imprime diagnostico, no escribe
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Features del modelo (SIN las columnas derivadas del fallo: total_fallos,
# ratio_fallos, total_downtime_days — esas eran fuga directa del target).
FEATURES = [
    "truck_age_years", "odometer_km", "engine_hours",
    "avg_temp_7d", "max_temp_7d", "std_temp_7d",
    "avg_oil_7d", "min_oil_7d",
    "avg_rpm_7d", "avg_coolant_7d", "avg_battery_7d",
    "fault_codes_7d", "fault_codes_30d",
    "avg_kpl_30d", "min_kpl_30d", "kpl_trend",
    "days_since_last_maint", "km_recorridos_30d",
    "temp_trend", "oil_trend", "fault_accel", "overdue_maint",
]

TARGET = "will_fail"


def simulate_fleet(n_trucks: int = 200, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con `FEATURES` + target `will_fail` + `truck_id`.

    Determinista dado `seed` (reproducible y testeable).
    """
    rng = np.random.default_rng(seed)
    n = n_trucks

    # ── Caracteristicas del vehiculo ───────────────────────────────
    odometer = rng.uniform(50_000, 800_000, n)
    age = rng.integers(2, 11, n).astype(float)
    engine_hours = odometer / rng.uniform(55, 75, n)

    # ── Desgaste latente (continuo, NO es un feature directo) ──────
    # Combinacion normalizada de km, edad y horas-motor + variacion individual.
    drivers = (
        0.55 * (odometer - 50_000) / 750_000
        + 0.25 * (age - 2) / 8
        + 0.20 * (engine_hours - engine_hours.min())
        / (engine_hours.max() - engine_hours.min())
    )
    wear = np.clip(drivers + rng.normal(0, 0.14, n), 0, 1.3)

    # ── Sensores: reflejan el desgaste CON RUIDO DE MEDICION ───────
    # El ruido hace que las distribuciones sano/degradado se solapen.
    avg_temp_7d = 86 + 16 * wear + rng.normal(0, 2.4, n)
    max_temp_7d = avg_temp_7d + np.abs(rng.normal(6, 2.0, n)) + 4 * wear
    std_temp_7d = np.abs(2 + 3 * wear + rng.normal(0, 0.96, n))
    avg_oil_7d = 4.6 - 2.3 * wear + rng.normal(0, 0.28, n)
    min_oil_7d = avg_oil_7d - np.abs(rng.normal(0.4, 0.3, n))
    avg_rpm_7d = rng.normal(1500, 190, n)                       # casi sin senal
    avg_coolant_7d = 88 + 12 * wear + rng.normal(0, 2.08, n)
    avg_battery_7d = 14.2 - 0.5 * wear + rng.normal(0, 0.176, n)
    fault_codes_7d = rng.poisson(np.clip(3.8 * wear, 0, None), n).astype(float)
    fault_codes_30d = fault_codes_7d + rng.poisson(np.clip(5 * wear, 0, None), n)
    avg_kpl_30d = 7.6 - 2.2 * wear + rng.normal(0, 0.36, n)
    min_kpl_30d = avg_kpl_30d - np.abs(rng.normal(0.5, 0.3, n))
    kpl_trend = -0.4 * wear + rng.normal(0, 0.45, n)
    km_recorridos_30d = rng.uniform(6_000, 18_000, n)           # operativo, sin senal
    days_since_last_maint = rng.uniform(1, 180, n)              # operativo, sin senal
    temp_trend = 2 * wear + rng.normal(0, 0.88, n)
    oil_trend = -0.3 * wear + rng.normal(0, 0.28, n)
    fault_accel = fault_codes_7d / (fault_codes_30d + 1)
    overdue_maint = (days_since_last_maint > 90).astype(int)

    # ── Fallo en los proximos 30 dias: PROBABILISTICO ─────────────
    # El riesgo depende del desgaste + un shock NO observado (no es feature),
    # que es lo que impide un AUC perfecto y hace el problema realista.
    shock = rng.normal(0, 1.0, n)
    risk = 7.5 * (wear - 0.72) + 0.45 * shock
    p_fail = 1.0 / (1.0 + np.exp(-risk))
    will_fail = (rng.uniform(0, 1, n) < p_fail).astype(int)

    df = pd.DataFrame({
        "truck_age_years": age,
        "odometer_km": np.round(odometer, 1),
        "engine_hours": np.round(engine_hours, 1),
        "avg_temp_7d": np.round(avg_temp_7d, 1),
        "max_temp_7d": np.round(max_temp_7d, 1),
        "std_temp_7d": np.round(std_temp_7d, 2),
        "avg_oil_7d": np.round(avg_oil_7d, 2),
        "min_oil_7d": np.round(min_oil_7d, 2),
        "avg_rpm_7d": np.round(avg_rpm_7d, 0),
        "avg_coolant_7d": np.round(avg_coolant_7d, 1),
        "avg_battery_7d": np.round(avg_battery_7d, 2),
        "fault_codes_7d": fault_codes_7d,
        "fault_codes_30d": fault_codes_30d.astype(float),
        "avg_kpl_30d": np.round(avg_kpl_30d, 2),
        "min_kpl_30d": np.round(min_kpl_30d, 2),
        "kpl_trend": np.round(kpl_trend, 3),
        "days_since_last_maint": np.round(days_since_last_maint, 0),
        "km_recorridos_30d": np.round(km_recorridos_30d, 0),
        "temp_trend": np.round(temp_trend, 2),
        "oil_trend": np.round(oil_trend, 3),
        "fault_accel": np.round(fault_accel, 3),
        "overdue_maint": overdue_maint,
        TARGET: will_fail,
        "truck_id": np.arange(1, n + 1),
    })
    return df[FEATURES + [TARGET, "truck_id"]]


def _diagnostics(df: pd.DataFrame) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    y = df[TARGET].astype(int)
    print(f"Filas: {len(df)} | Tasa de fallos: {y.mean():.1%} "
          f"({y.sum()} fallos / {(y == 0).sum()} sanos)")

    aucs = []
    for f in FEATURES:
        a = roc_auc_score(y, df[f].fillna(0))
        aucs.append((f, max(a, 1 - a)))
    aucs.sort(key=lambda r: -r[1])
    print("\nAUC individual (top 5):")
    for f, a in aucs[:5]:
        print(f"  {a:.3f}  {f}")
    print(f"Max AUC individual: {aucs[0][1]:.3f}  (umbral de fuga: 0.97)")

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, df[FEATURES].fillna(0), y, cv=cv, scoring="roc_auc")
    print(f"\nModelo (RF) 5-fold CV AUC: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"  folds: {np.round(scores, 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Solo imprime diagnostico, no escribe el CSV.")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = simulate_fleet(n_trucks=args.n, seed=args.seed)
    _diagnostics(df)

    if not args.check:
        out = Path("data") / "fleet_features.csv"
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nGuardado: {out}")
