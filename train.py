"""
train.py
========
Trains an Isolation Forest anomaly detector on GPU telemetry data.
Evaluates its ability to predict thermal throttling 45 seconds ahead.

Inputs:
    features.csv       — from features.py

Outputs:
    models/gpuwatch_model.pkl     — trained scaler + model
    results/classification_report.txt
    results/lag_correlation.png   — SM clock drop vs TTFT delta scatter

Usage:
    python train.py
    python train.py --input features.csv --contamination 0.05
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
import sys
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

FEATURE_COLS = [
    "sm_clock", "sm_clock_std", "sm_clock_drop", "sm_clock_ratio",
    "gpu_temp", "temp_slope", "temp_slope_2m", "temp_max",
    "power_usage", "power_slope", "power_std",
    "mem_util_pct", "mem_slope",
    "tensor_mean", "util_mean",
    "ttft_delta", "ttft_slope", "queue_mean", "queue_slope",
]


def load_features(path):
    print(f"[train] Loading {path}...")
    df = pd.read_csv(path)
    print(f"[train] Shape: {df.shape}")

    # Use only features that exist in this dataset
    available = [f for f in FEATURE_COLS if f in df.columns]
    missing = [f for f in FEATURE_COLS if f not in df.columns]
    if missing:
        print(f"[train] Skipping missing features: {missing}")

    df_clean = df[available + ["throttled", "future_throttle"]].dropna()
    print(f"[train] Clean rows: {len(df_clean)}")
    print(f"[train] Throttle rate: {df_clean['throttled'].mean()*100:.1f}%")

    if len(df_clean) < 200:
        print("[train] WARNING: Very few rows. Collect more data for better results.")
        print("         Target: 2000+ rows (2.5+ hours of collection)")

    if df_clean["throttled"].sum() < 20:
        print("[train] WARNING: Very few throttle events detected.")
        print("         Run Locust with more users to stress the GPU more.")

    return df_clean, available


def train_isolation_forest(df, features, contamination=0.05):
    print(f"\n[train] Training Isolation Forest on {len(features)} features...")

    # Train ONLY on healthy samples — key to unsupervised anomaly detection
    X_healthy = df[df["throttled"] == 0][features]
    print(f"[train] Healthy samples for training: {len(X_healthy)}")

    # Scale features
    scaler = StandardScaler()
    X_healthy_scaled = scaler.fit_transform(X_healthy)

    # Train Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,  # expected anomaly fraction
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_healthy_scaled)
    print("[train] Training complete.")

    return iso, scaler


def evaluate(df, iso, scaler, features):
    print("\n[train] Evaluating on full dataset...")

    X_all = scaler.transform(df[features])

    # Anomaly scores (lower = more anomalous)
    df = df.copy()
    df["anomaly_score"]     = iso.score_samples(X_all)
    df["predicted_anomaly"] = (iso.predict(X_all) == -1).astype(int)

    # ── Current throttle detection ────────────────────────────────
    print("\n── Current Throttle Detection ──")
    print(classification_report(
        df["throttled"],
        df["predicted_anomaly"],
        target_names=["Normal", "Throttling"],
        digits=3,
    ))

    # ── Future throttle prediction (45s ahead) ────────────────────
    valid = df.dropna(subset=["future_throttle"]).copy()
    valid["future_throttle"] = valid["future_throttle"].astype(int)

    print("── Predictive Performance (45s ahead) ──")
    report = classification_report(
        valid["future_throttle"],
        valid["predicted_anomaly"],
        target_names=["Normal", "Pre-Throttle"],
        digits=3,
        output_dict=True,
    )
    print(classification_report(
        valid["future_throttle"],
        valid["predicted_anomaly"],
        target_names=["Normal", "Pre-Throttle"],
        digits=3,
    ))

    # Save report
    with open("results/classification_report.txt", "w") as f:
        f.write("GPUWatch Anomaly Detector — Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("── Current Throttle Detection ──\n")
        f.write(classification_report(
            df["throttled"], df["predicted_anomaly"],
            target_names=["Normal", "Throttling"], digits=3
        ))
        f.write("\n── Predictive Performance (45s ahead) ──\n")
        f.write(classification_report(
            valid["future_throttle"], valid["predicted_anomaly"],
            target_names=["Normal", "Pre-Throttle"], digits=3
        ))
        f.write(f"\nTotal rows evaluated: {len(df)}\n")
        f.write(f"Throttle events: {df['throttled'].sum()}\n")
        f.write(f"Prediction horizon: 45 seconds\n")

    print("[train] Report saved to results/classification_report.txt")

    return report, df


def plot_results(df):
    """Generate lag correlation plot — SM clock drop vs TTFT delta."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GPUWatch — GPU Health vs Inference SLO Correlation", fontsize=14)

        # Panel 1: SM Clock over time colored by anomaly
        ax = axes[0, 0]
        normal = df[df["predicted_anomaly"] == 0]
        anomaly = df[df["predicted_anomaly"] == 1]
        ax.scatter(range(len(normal)), normal["sm_clock"], s=1, alpha=0.3, c="blue", label="Normal")
        ax.scatter(anomaly.index, anomaly["sm_clock"], s=5, alpha=0.8, c="red", label="Anomaly")
        ax.set_title("SM Clock — Anomaly Detection")
        ax.set_ylabel("SM Clock (MHz)")
        ax.set_xlabel("Time (samples)")
        ax.legend()

        # Panel 2: SM Clock Ratio distribution
        ax = axes[0, 1]
        if "sm_clock_ratio" in df.columns:
            ax.hist(df[df["throttled"]==0]["sm_clock_ratio"].dropna(), bins=50, alpha=0.7, label="Normal", color="blue")
            ax.hist(df[df["throttled"]==1]["sm_clock_ratio"].dropna(), bins=50, alpha=0.7, label="Throttled", color="red")
            ax.axvline(0.92, color="orange", linestyle="--", label="Threshold (0.92)")
            ax.set_title("SM Clock Ratio Distribution")
            ax.set_xlabel("SM Clock Ratio")
            ax.legend()

        # Panel 3: Anomaly score distribution
        ax = axes[1, 0]
        ax.hist(df[df["throttled"]==0]["anomaly_score"].dropna(), bins=50, alpha=0.7, label="Normal", color="blue")
        ax.hist(df[df["throttled"]==1]["anomaly_score"].dropna(), bins=50, alpha=0.7, label="Throttled", color="red")
        ax.set_title("Isolation Forest Anomaly Scores")
        ax.set_xlabel("Score (lower = more anomalous)")
        ax.legend()

        # Panel 4: SM Clock Drop vs TTFT Delta scatter
        ax = axes[1, 1]
        if "ttft_delta" in df.columns and "sm_clock_drop" in df.columns:
            scatter = ax.scatter(
                df["sm_clock_drop"].clip(0, 200),
                df["ttft_delta"].clip(-0.5, 2),
                c=df["throttled"], cmap="RdBu_r", alpha=0.3, s=2
            )
            plt.colorbar(scatter, ax=ax, label="Throttled")
            ax.set_title("SM Clock Drop vs TTFT Delta\n(Core Lag Correlation)")
            ax.set_xlabel("SM Clock Drop from Max (MHz)")
            ax.set_ylabel("TTFT P95 Delta (s)")

        plt.tight_layout()
        plt.savefig("results/lag_correlation.png", dpi=150, bbox_inches="tight")
        print("[train] Plot saved to results/lag_correlation.png")
        plt.close()

    except ImportError:
        print("[train] matplotlib not installed — skipping plots")
    except Exception as e:
        print(f"[train] Plot error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="features.csv")
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly fraction (default: 0.05 = 5%%)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[train] ERROR: {args.input} not found.")
        print("         Run: python features.py  first")
        sys.exit(1)

    # Load
    df, features = load_features(args.input)

    # Train
    iso, scaler = train_isolation_forest(df, features, args.contamination)

    # Evaluate
    report, df_eval = evaluate(df, iso, scaler, features)

    # Plot
    plot_results(df_eval)

    # Save model
    model_path = "models/gpuwatch_model.pkl"
    joblib.dump({"model": iso, "scaler": scaler, "features": features}, model_path)
    print(f"\n[train] Model saved to {model_path}")

    # Print resume-ready numbers
    pre_throttle = report.get("Pre-Throttle", {})
    print("\n" + "=" * 50)
    print("RESUME BULLET NUMBERS:")
    print(f"  Precision: {pre_throttle.get('precision', 0):.2f}")
    print(f"  Recall:    {pre_throttle.get('recall', 0):.2f}")
    print(f"  F1:        {pre_throttle.get('f1-score', 0):.2f}")
    print("  Lead time: 45 seconds")
    print("=" * 50)


if __name__ == "__main__":
    main()
