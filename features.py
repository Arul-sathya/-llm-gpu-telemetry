"""
features.py
===========
Transforms raw GPU telemetry CSV into ML-ready features
for the Isolation Forest anomaly detector.

Inputs:
    gpu_telemetry.csv  — from collect_metrics.py

Outputs:
    features.csv       — engineered features + throttle labels
    feature_stats.txt  — summary statistics

Usage:
    python features.py
    python features.py --input gpu_telemetry.csv --output features.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys

# Rolling window = 5 minutes at 5s intervals
WINDOW = 60

# Throttle threshold — SM clock below 92% of recent max = throttled
THROTTLE_RATIO = 0.92

# Prediction horizon — label throttle events N steps ahead
# 9 steps * 5s = 45 seconds prediction window
PREDICTION_HORIZON = 9


def load_and_validate(path):
    print(f"[features] Loading {path}...")
    df = pd.read_csv(path)
    print(f"[features] Shape: {df.shape}")
    print(f"[features] Columns: {list(df.columns)}")
    print(f"[features] Null counts:\n{df.isnull().sum()}\n")

    required = ["timestamp", "sm_clock", "gpu_temp", "power_usage"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[features] ERROR: Missing required columns: {missing}")
        sys.exit(1)

    return df


def engineer_features(df):
    print("[features] Engineering features...")
    df = df.copy().sort_values("timestamp").reset_index(drop=True)

    W = WINDOW

    # ── SM Clock features ─────────────────────────────────────────
    df["sm_clock_mean"]    = df["sm_clock"].rolling(W).mean()
    df["sm_clock_std"]     = df["sm_clock"].rolling(W).std()
    df["sm_clock_min"]     = df["sm_clock"].rolling(W).min()
    df["sm_clock_max"]     = df["sm_clock"].rolling(W).max()
    # Drop from recent max — key throttle onset signal
    df["sm_clock_drop"]    = df["sm_clock"].rolling(W).max() - df["sm_clock"]
    # SM clock as fraction of rolling max (1.0 = healthy, <0.92 = throttled)
    df["sm_clock_ratio"]   = df["sm_clock"] / df["sm_clock"].rolling(W).max().clip(lower=1)

    # ── Temperature features ──────────────────────────────────────
    df["temp_mean"]        = df["gpu_temp"].rolling(W).mean()
    df["temp_slope"]       = df["gpu_temp"].diff(12)       # 1-min trend
    df["temp_slope_2m"]    = df["gpu_temp"].diff(24)       # 2-min trend
    df["temp_max"]         = df["gpu_temp"].rolling(W).max()

    # ── Power features ────────────────────────────────────────────
    df["power_mean"]       = df["power_usage"].rolling(W).mean()
    df["power_slope"]      = df["power_usage"].diff(12)
    df["power_std"]        = df["power_usage"].rolling(W).std()

    # ── Memory features ───────────────────────────────────────────
    if "fb_used" in df.columns and "fb_total" in df.columns:
        df["mem_util_pct"] = 100 * df["fb_used"] / df["fb_total"].clip(lower=1)
        df["mem_slope"]    = df["fb_used"].diff(12)
    elif "mem_util_pct" not in df.columns:
        df["mem_util_pct"] = 0.0
        df["mem_slope"]    = 0.0

    # ── Inference SLO features ────────────────────────────────────
    if "ttft_p95" in df.columns:
        df["ttft_mean"]    = df["ttft_p95"].rolling(W).mean()
        df["ttft_slope"]   = df["ttft_p95"].diff(12)
        df["ttft_lag1"]    = df["ttft_p95"].shift(12)
        df["ttft_delta"]   = df["ttft_p95"] - df["ttft_lag1"]
    else:
        df["ttft_mean"]    = 0.0
        df["ttft_slope"]   = 0.0
        df["ttft_delta"]   = 0.0

    if "queue_depth" in df.columns:
        df["queue_mean"]   = df["queue_depth"].rolling(W).mean()
        df["queue_slope"]  = df["queue_depth"].diff(12)
    else:
        df["queue_mean"]   = 0.0
        df["queue_slope"]  = 0.0

    if "tensor_active" in df.columns:
        df["tensor_mean"]  = df["tensor_active"].rolling(W).mean()
    else:
        df["tensor_mean"]  = 0.0

    if "gpu_util" in df.columns:
        df["util_mean"]    = df["gpu_util"].rolling(W).mean()
    else:
        df["util_mean"]    = 0.0

    # ── Throttle label ────────────────────────────────────────────
    # Current throttle: SM clock ratio below threshold
    base_clock = df["sm_clock"].quantile(0.95)
    df["throttled"] = (df["sm_clock"] < base_clock * THROTTLE_RATIO).astype(int)

    # Future throttle: does throttling happen in next 45 seconds?
    # This is what we're predicting — forward-shifted label
    df["future_throttle"] = df["throttled"].shift(-PREDICTION_HORIZON)

    print(f"[features] Base SM clock (95th pct): {base_clock:.0f} MHz")
    print(f"[features] Throttle threshold: {base_clock * THROTTLE_RATIO:.0f} MHz")
    print(f"[features] Current throttle events: {df['throttled'].sum()} / {len(df)}")
    print(f"[features] Future throttle labels: {df['future_throttle'].sum()} / {len(df)}")

    return df


def select_features(df):
    """Return the feature columns used for model training."""
    FEATURES = [
        # SM clock signals (most important)
        "sm_clock",
        "sm_clock_std",
        "sm_clock_drop",
        "sm_clock_ratio",

        # Temperature
        "gpu_temp",
        "temp_slope",
        "temp_slope_2m",
        "temp_max",

        # Power
        "power_usage",
        "power_slope",
        "power_std",

        # Memory
        "mem_util_pct",
        "mem_slope",

        # Compute
        "tensor_mean",
        "util_mean",

        # Inference SLOs
        "ttft_delta",
        "ttft_slope",
        "queue_mean",
        "queue_slope",
    ]

    # Only include features that actually exist in the dataframe
    available = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[features] Warning: Missing features (will skip): {missing}")

    return available


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="gpu_telemetry.csv")
    parser.add_argument("--output", default="features.csv")
    args = parser.parse_args()

    df = load_and_validate(args.input)
    df = engineer_features(df)

    FEATURES = select_features(df)
    print(f"\n[features] Selected {len(FEATURES)} features: {FEATURES}")

    # Save full feature set including labels
    save_cols = ["timestamp"] + FEATURES + ["throttled", "future_throttle"]
    save_cols = [c for c in save_cols if c in df.columns]
    df_save = df[save_cols].dropna(subset=FEATURES)

    print(f"\n[features] Rows before dropna: {len(df)}")
    print(f"[features] Rows after dropna:  {len(df_save)}")
    print(f"[features] Throttle rate: {df_save['throttled'].mean()*100:.1f}%")

    df_save.to_csv(args.output, index=False)
    print(f"\n[features] Saved to {args.output}")

    # Save stats
    with open("feature_stats.txt", "w") as f:
        f.write("GPUWatch Feature Statistics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total rows: {len(df_save)}\n")
        f.write(f"Throttle events: {df_save['throttled'].sum()} ({df_save['throttled'].mean()*100:.1f}%)\n")
        if "future_throttle" in df_save:
            valid = df_save.dropna(subset=["future_throttle"])
            f.write(f"Future throttle labels: {valid['future_throttle'].sum()} ({valid['future_throttle'].mean()*100:.1f}%)\n\n")
        f.write(df_save[FEATURES].describe().to_string())
    print("[features] Stats saved to feature_stats.txt")

    return FEATURES


if __name__ == "__main__":
    main()
