"""
Trains and compares two churn-propensity models:
  (a) Logistic Regression (interpretable baseline)
  (b) XGBoost (higher-capacity comparison)

Reports AUC, precision/recall, and feature importance -- then serializes the
winning model to models/model.pkl for deployment (see azure_deploy/).
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_churn.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEPLOY_MODEL_PATH = PROJECT_ROOT / "azure_deploy" / "model.pkl"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customer_id", "churned"])
y = df["churned"]

numeric_cols = ["tenure_months", "monthly_charge", "num_support_calls", "num_products", "satisfaction_score", "has_autopay"]
categorical_cols = ["contract_type"]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first"), categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# --- Logistic Regression ---
logreg_pipe = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=1000))])
logreg_pipe.fit(X_train, y_train)
logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_proba)

# --- XGBoost ---
xgb_pipe = Pipeline([("prep", preprocess), ("clf", XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.08, eval_metric="logloss", random_state=42
))])
xgb_pipe.fit(X_train, y_train)
xgb_proba = xgb_pipe.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_proba)

print(f"Logistic Regression AUC: {logreg_auc:.4f}")
print(f"XGBoost AUC:             {xgb_auc:.4f}")
print("\n=== XGBoost classification report (threshold=0.5) ===")
print(classification_report(y_test, (xgb_proba >= 0.5).astype(int)))

# Pick the better model
best_name, best_pipe, best_proba, best_auc = (
    ("xgboost", xgb_pipe, xgb_proba, xgb_auc)
    if xgb_auc >= logreg_auc
    else ("logistic_regression", logreg_pipe, logreg_proba, logreg_auc)
)
print(f"\nSelected model for deployment: {best_name} (AUC={best_auc:.4f})")

joblib.dump(best_pipe, OUTPUT_DIR / "model.pkl")
joblib.dump(best_pipe, DEPLOY_MODEL_PATH)

with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump({
        "logistic_regression_auc": round(logreg_auc, 4),
        "xgboost_auc": round(xgb_auc, 4),
        "selected_model": best_name,
    }, f, indent=2)

# --- ROC curve plot ---
plt.figure(figsize=(6, 6))
for name, proba, auc in [("Logistic Regression", logreg_proba, logreg_auc), ("XGBoost", xgb_proba, xgb_auc)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Churn Propensity Model — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
print("Saved outputs/model.pkl, metrics.json, roc_curve.png")

# --- Feature importance (XGBoost) ---
feature_names = (
    numeric_cols
    + list(xgb_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_cols))
)
importances = xgb_pipe.named_steps["clf"].feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
imp_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
print("\nTop churn drivers:")
print(imp_df.head(5).to_string(index=False))
