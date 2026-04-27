import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

FEATURE_COLS = [
    "sm_clock", "sm_clock_std", "sm_clock_drop", "sm_clock_ratio",
    "gpu_temp", "temp_slope", "temp_slope_2m", "temp_max",
    "power_usage", "power_slope", "power_std",
    "mem_util_pct", "mem_slope",
    "tensor_mean", "util_mean",
    "ttft_delta", "ttft_slope", "queue_mean", "queue_slope",
]

df = pd.read_csv("features_combined.csv")
available = [f for f in FEATURE_COLS if f in df.columns]
df_clean = df[available + ["future_throttle"]].dropna()
X = df_clean[available].values
y = df_clean["future_throttle"].astype(int).values
print(f"Shape: {X.shape}, Throttle rate: {y.mean()*100:.1f}% ({y.sum()}/{len(y)})\n")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ("Random Forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)),
    ("Logistic Regression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
]

results = []
for name, model in models:
    y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
    p = precision_score(y, y_pred, zero_division=0)
    r = recall_score(y, y_pred, zero_division=0)
    f = f1_score(y, y_pred, zero_division=0)
    print(f"{name}: P={p:.3f} R={r:.3f} F1={f:.3f}")
    results.append((name, p, r, f, model))

best = max(results, key=lambda x: x[3])
print(f"\nBEST: {best[0]} — P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f}")

# Save best model
best[4].fit(X_scaled, y)
joblib.dump({"model": best[4], "scaler": scaler, "features": available, "model_type": "supervised"}, "models/gpuwatch_model.pkl")
print("Model saved.")
