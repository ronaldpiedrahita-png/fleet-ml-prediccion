# 05_train_model.py

import pandas as pd
import numpy as np
import mlflow, mlflow.sklearn
import joblib, shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble        import RandomForestClassifier
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (roc_auc_score, f1_score,
                                     classification_report)
from xgboost import XGBClassifier

Path("models").mkdir(exist_ok=True)

# ── Cargar datos ───────────────────────────────────────────
df = pd.read_csv("data/fleet_features.csv")
FEATURES = [c for c in df.columns if c not in ["will_fail", "truck_id"]]
TARGET   = "will_fail"

X = df[FEATURES]
y = df[TARGET].astype(int)
print(f"Dataset: {X.shape}  |  Tasa fallos: {y.mean():.1%}")
print(f"Camiones con fallo: {y.sum()} | Sin fallo: {(y==0).sum()}")

# ── Split ──────────────────────────────────────────────────
# Con 50 registros usamos 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Fallos en train: {y_train.sum()} | Fallos en test: {y_test.sum()}")

# ── Calcular peso para clases desbalanceadas ───────────────
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight: {pos_weight:.2f}")

# ── Definir modelos ────────────────────────────────────────
MODELS = {
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "xgboost": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            random_state=42,
            verbosity=0
        ))
    ]),
}

# ── Entrenar con MLflow ────────────────────────────────────
mlflow.set_experiment("fleet-maintenance-prediction")
best_auc, best_name, best_model = 0, None, None

for name, pipeline in MODELS.items():
    with mlflow.start_run(run_name=name):
        print(f"\n🚀 Entrenando {name}...")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        THRESHOLD  = 0.4
        y_pred_adj = (y_prob >= THRESHOLD).astype(int)

        auc = roc_auc_score(y_test, y_prob)
        f1  = f1_score(y_test, y_pred_adj, zero_division=0)

        # Manejo seguro del classification report
        report = classification_report(
            y_test, y_pred_adj,
            output_dict=True,
            zero_division=0
        )
        recall_fail    = report.get("1", {}).get("recall",    0)
        precision_fail = report.get("1", {}).get("precision", 0)

        print(f"  AUC-ROC:        {auc:.4f}")
        print(f"  F1:             {f1:.4f}")
        print(f"  Recall fallo:   {recall_fail:.4f}")
        print(f"  Precision fallo:{precision_fail:.4f}")

        mlflow.log_params({
            "model":     name,
            "threshold": THRESHOLD,
            "n_train":   len(X_train),
            "n_test":    len(X_test)
        })
        mlflow.log_metrics({
            "auc_roc":           auc,
            "f1":                f1,
            "recall_failure":    recall_fail,
            "precision_failure": precision_fail
        })
        mlflow.sklearn.log_model(pipeline, name)

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, pipeline

# ── SHAP ───────────────────────────────────────────────────
print(f"\n🏆 Mejor modelo: {best_name} (AUC={best_auc:.4f})")
print("🔍 Calculando SHAP values...")

try:
    X_test_scaled = best_model.named_steps["scaler"].transform(X_test)
    clf           = best_model.named_steps["clf"]
    explainer     = shap.TreeExplainer(clf)
    shap_values   = explainer.shap_values(X_test_scaled)

    # Compatibilidad con diferentes versiones de SHAP
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    shap_df = pd.DataFrame({
        "feature":   FEATURES,
        "mean_shap": np.abs(sv).mean(axis=0)
    }).sort_values("mean_shap", ascending=False)

    print("\n🔎 Top 8 features más importantes (SHAP):")
    print(shap_df.head(8).to_string(index=False))

except Exception as e:
    print(f"  ⚠️  SHAP no disponible: {e}")
    # Usar feature importance del modelo como fallback
    clf      = best_model.named_steps["clf"]
    shap_df  = pd.DataFrame({
        "feature":   FEATURES,
        "mean_shap": clf.feature_importances_
    }).sort_values("mean_shap", ascending=False)
    print("\n🔎 Top 8 features por importancia del modelo:")
    print(shap_df.head(8).to_string(index=False))

# ── Guardar modelo ─────────────────────────────────────────
joblib.dump(best_model, "models/fleet_model.pkl")
joblib.dump(FEATURES,   "models/feature_names.pkl")
joblib.dump(shap_df,    "models/shap_importance.pkl")

print(f"\n✅ Modelo '{best_name}' guardado en models/fleet_model.pkl")
print("💡 Ejecuta: mlflow ui → ver experimentos en http://localhost:5000")