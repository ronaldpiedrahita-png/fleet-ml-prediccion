"""Guarda contra fuga de datos (target leakage) en el dataset de ML.

Un feature con fuga es aquel que, por sí solo, separa casi perfectamente el
target. En un problema real ningún sensor logra eso: si ocurre, es que el
feature deriva de la etiqueta (o de un proceso de datos degenerado).
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_features import LEAKING_FEATURES, select_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "fleet_features.csv"
NON_FEATURES = {"will_fail", "truck_id"}

# Umbral: ningún feature debería, por sí solo, predecir el fallo con AUC >= 0.97.
# Un separador casi perfecto individual es la firma clásica de la fuga de datos.
MAX_SINGLE_FEATURE_AUC = 0.97


def _model_features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES]


def test_no_single_feature_almost_perfectly_predicts_target():
    df = pd.read_csv(DATA)
    y = df["will_fail"].astype(int)

    offenders = {}
    for feature in _model_features(df):
        x = df[feature].fillna(0)
        auc = roc_auc_score(y, x)
        auc = max(auc, 1 - auc)  # el feature podría estar invertido
        if auc >= MAX_SINGLE_FEATURE_AUC:
            offenders[feature] = round(float(auc), 4)

    assert not offenders, (
        "Features con fuga (predicen el target casi perfectos por sí solos): "
        f"{offenders}"
    )


def test_select_features_excludes_known_leaks_and_metadata():
    columns = ["odometer_km", "total_fallos", "ratio_fallos",
               "total_downtime_days", "will_fail", "truck_id", "avg_temp_7d"]
    result = select_features(columns)

    assert not (set(result) & LEAKING_FEATURES), "select_features dejo pasar una fuga"
    assert "will_fail" not in result and "truck_id" not in result
    assert "odometer_km" in result and "avg_temp_7d" in result

