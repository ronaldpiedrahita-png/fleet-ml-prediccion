# 05_train_model.py
#
# Entrena y compara Random Forest vs XGBoost para predecir fallo de motor.
# La evaluacion usa validacion cruzada 5-fold (AUC robusto) + un conjunto de
# prueba retenido para la matriz de confusion. La seleccion de features excluye
# columnas con fuga de datos (ver ml_features.py).

import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from xgboost import XGBClassifier

from ml_features import select_features


def choose_threshold(y_true, prob, target_recall=0.70):
    """Umbral mas exigente (mayor precision) que aun logra el recall objetivo.

    En mantenimiento predictivo un fallo no detectado (FN) cuesta mucho mas que
    una falsa alarma (FP), asi que priorizamos recall sobre precision.
    """
    best_t, best_prec = 0.5, -1.0
    for t in np.linspace(0.05, 0.90, 86):
        pred = (prob >= t).astype(int)
        if recall_score(y_true, pred, zero_division=0) >= target_recall:
            p = precision_score(y_true, pred, zero_division=0)
            if p > best_prec:
                best_prec, best_t = p, float(t)
    return best_t

# MLflow es opcional: si no esta disponible, el entrenamiento no debe fallar.
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except Exception:                                    # pragma: no cover
    MLFLOW_AVAILABLE = False

Path("models").mkdir(exist_ok=True)

# ── Cargar datos ───────────────────────────────────────────
df = pd.read_csv("data/fleet_features.csv")
FEATURES = select_features(df.columns)               # excluye target, id y fugas
TARGET = "will_fail"

X = df[FEATURES].fillna(0)
y = df[TARGET].astype(int)
print(f"Dataset: {X.shape}  |  Features: {len(FEATURES)}")
print(f"Tasa de fallos: {y.mean():.1%}  ({y.sum()} fallos / {(y == 0).sum()} sanos)")

pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

# ── Definir modelos ────────────────────────────────────────
MODELS = {
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ]),
    "xgboost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight,
            eval_metric="auc", random_state=42, verbosity=0,
        )),
    ]),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("fleet-maintenance-prediction")

best_auc, best_name, best_model = -1.0, None, None

for name, pipeline in MODELS.items():
    print(f"\n=== {name} ===")

    # Predicciones out-of-fold: cada fila se predice con un modelo que NO la vio.
    # Mejor que un unico split retenido cuando el dataset es pequeno (200 filas).
    oof_prob = cross_val_predict(pipeline, X, y, cv=cv,
                                 method="predict_proba")[:, 1]
    oof_auc = roc_auc_score(y, oof_prob)

    thr = choose_threshold(y, oof_prob, target_recall=0.70)
    y_pred = (oof_prob >= thr).astype(int)
    f1 = f1_score(y, y_pred, zero_division=0)
    recall_fail = recall_score(y, y_pred, zero_division=0)
    precision_fail = precision_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)

    print(f"  AUC (out-of-fold):   {oof_auc:.3f}")
    print(f"  Umbral (recall>=0.70): {thr:.2f}")
    print(f"  Recall (fallo):      {recall_fail:.3f}")
    print(f"  Precision (fallo):   {precision_fail:.3f}")
    print(f"  F1 (fallo):          {f1:.3f}")
    print(f"  Matriz de confusion [[TN FP] [FN TP]]:\n{cm}")

    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model": name, "n_features": len(FEATURES),
                                   "n_rows": len(X), "threshold": round(thr, 2)})
                mlflow.log_metrics({
                    "oof_auc": float(oof_auc), "f1": float(f1),
                    "recall_failure": float(recall_fail),
                    "precision_failure": float(precision_fail),
                })
                try:
                    mlflow.sklearn.log_model(pipeline, name=name)          # mlflow >= 3
                except TypeError:
                    mlflow.sklearn.log_model(pipeline, artifact_path=name)  # mlflow < 3
        except Exception as e:                           # pragma: no cover
            print(f"  (MLflow: no se registro el run: {e})")

    if oof_auc > best_auc:
        best_auc, best_name, best_model = oof_auc, name, pipeline

# ── Ajuste final sobre TODOS los datos (para desplegar) ────
print(f"\nMejor modelo: {best_name}  (AUC out-of-fold = {best_auc:.3f})")
best_model.fit(X, y)

# ── SHAP sobre el mejor modelo ─────────────────────────────
print("Calculando SHAP values...")

try:
    X_scaled = best_model.named_steps["scaler"].transform(X)
    clf = best_model.named_steps["clf"]
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(np.asarray(shap_values).shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_shap": np.abs(sv).mean(axis=0),
    }).sort_values("mean_shap", ascending=False)
    print("\nTop 8 features (SHAP):")
    print(shap_df.head(8).to_string(index=False))
except Exception as e:                               # pragma: no cover
    print(f"  SHAP no disponible ({e}); usando feature_importances_")
    clf = best_model.named_steps["clf"]
    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "mean_shap": clf.feature_importances_,
    }).sort_values("mean_shap", ascending=False)
    print(shap_df.head(8).to_string(index=False))

# ── Guardar artefactos ─────────────────────────────────────
joblib.dump(best_model, "models/fleet_model.pkl")
joblib.dump(FEATURES, "models/feature_names.pkl")
joblib.dump(shap_df, "models/shap_importance.pkl")
print(f"\nModelo '{best_name}' guardado en models/fleet_model.pkl")
