"""Guarda de regresion: el modelo debe rendir en un rango realista.

Un AUC ~1.0 significaria que reaparecio la fuga de datos; un AUC ~0.5 que el
modelo no aprende nada. El rango (0.65, 0.95) es lo esperable en mantenimiento
predictivo con senales de sensores ruidosas.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_features import select_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "fleet_features.csv"


def test_model_auc_is_realistic_not_perfect():
    df = pd.read_csv(DATA)
    features = select_features(df.columns)
    X = df[features].fillna(0)
    y = df["will_fail"].astype(int)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc").mean()

    assert 0.65 < auc < 0.95, f"AUC fuera del rango realista esperado: {auc:.3f}"
