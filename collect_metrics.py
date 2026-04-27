"""
collect_metrics.py
==================
Scrapes Prometheus every 5 seconds and saves GPU + inference
metrics to a CSV file for anomaly detector training.

Run AFTER docker compose up -d and Locust load generator is running.
Let it collect for at least 2-3 hours to capture enough throttle events.

Usage:
    python collect_metrics.py                    # runs until Ctrl+C
    python collect_metrics.py --duration 7200    # run for 2 hours
    python collect_metrics.py --output my_run.csv

Output:
    gpu_telemetry.csv  — raw time-series, one row per 5s interval
"""

import requests
import pandas as pd
import time
import argparse
import signal
import sys
from datetime import datetime

PROMETHEUS = "http://localhost:9090"
INTERVAL   = 5    # seconds between scrapes
OUTPUT     = "gpu_telemetry.csv"

# ── Prometheus queries ────────────────────────────────────────────
# Each key maps to a PromQL expression.
# Results are stored as columns in the CSV.
QUERIES = {
    # GPU hardware (DCGM or nvidia-smi exporter)
    "sm_clock":         "DCGM_FI_DEV_SM_CLOCK",
    "mem_clock":        "DCGM_FI_DEV_MEM_CLOCK",
    "gpu_temp":         "DCGM_FI_DEV_GPU_TEMP",
    "power_usage":      "DCGM_FI_DEV_POWER_USAGE",
    "fb_used":          "DCGM_FI_DEV_FB_USED",
    "fb_free":          "DCGM_FI_DEV_FB_FREE",
    "fb_total":         "DCGM_FI_DEV_FB_TOTAL",
    "gpu_util":         "DCGM_FI_DEV_GPU_UTIL",
    "tensor_active":    "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",

    # Inference SLOs (vLLM)
    "ttft_p95":         "histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[1m]))",
    "ttft_p50":         "histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket[1m]))",
    "queue_depth":      "vllm:num_requests_waiting",
    "running_requests": "vllm:num_requests_running",
    "token_throughput": "rate(vllm:generation_tokens_total[1m])",
    "request_rate":     "rate(vllm:request_success_total[1m])",
    "kv_cache_usage":   "vllm:gpu_cache_usage_perc",

    # Derived (recording rules from rules.yml)
    "sm_clock_ratio":   "gpuwatch:sm_clock_ratio",
    "mem_util_pct":     "gpuwatch:mem_util_pct",
    "temp_slope":       "gpuwatch:temp_slope",
    "power_slope":      "gpuwatch:power_slope",
    "ttft_p95_ms":      "gpuwatch:ttft_p95_ms",
}


def query_prometheus(expr):
    """Query Prometheus instant value. Returns float or None."""
    try:
        r = requests.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": expr},
            timeout=4,
        )
        result = r.json().get("data", {}).get("result", [])
        if result:
            return float(result[0]["value"][1])
        return None
    except Exception:
        return None


def scrape_all():
    """Scrape all metrics in one round. Returns dict."""
    row = {"timestamp": time.time(), "datetime": datetime.utcnow().isoformat()}
    for name, expr in QUERIES.items():
        row[name] = query_prometheus(expr)
    return row


def main():
    parser = argparse.ArgumentParser(description="GPUWatch metric collector")
    parser.add_argument("--duration", type=int, default=0,
                        help="Collection duration in seconds (0 = run until Ctrl+C)")
    parser.add_argument("--output", default=OUTPUT,
                        help="Output CSV filename")
    parser.add_argument("--interval", type=int, default=INTERVAL,
                        help="Scrape interval in seconds")
    args = parser.parse_args()

    rows = []
    start = time.time()
    count = 0

    def save_and_exit(sig=None, frame=None):
        print(f"\n[collector] Saving {len(rows)} rows to {args.output}...")
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(args.output, index=False)
            print(f"[collector] Saved. Shape: {df.shape}")
            print(f"[collector] Null counts:\n{df.isnull().sum()}")

            # Quick throttle event summary
            if "sm_clock_ratio" in df.columns:
                df_clean = df.dropna(subset=["sm_clock_ratio"])
                if len(df_clean) > 0:
                    base = df_clean["sm_clock_ratio"].quantile(0.95)
                    throttled = (df_clean["sm_clock_ratio"] < 0.92).sum()
                    print(f"\n[collector] Throttle events detected: {throttled} / {len(df_clean)} rows")
                    print(f"[collector] SM clock ratio range: {df_clean['sm_clock_ratio'].min():.3f} - {df_clean['sm_clock_ratio'].max():.3f}")
        sys.exit(0)

    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    print(f"[collector] Starting collection → {args.output}")
    print(f"[collector] Scrape interval: {args.interval}s")
    print(f"[collector] Duration: {'until Ctrl+C' if args.duration == 0 else f'{args.duration}s'}")
    print(f"[collector] Prometheus: {PROMETHEUS}")
    print("[collector] Press Ctrl+C to stop and save\n")

    # Verify Prometheus is reachable
    try:
        r = requests.get(f"{PROMETHEUS}/-/healthy", timeout=3)
        print(f"[collector] Prometheus: OK")
    except Exception:
        print(f"[collector] ERROR: Cannot reach Prometheus at {PROMETHEUS}")
        print("           Make sure docker compose is running.")
        sys.exit(1)

    while True:
        row = scrape_all()
        rows.append(row)
        count += 1

        # Progress log every 60 rows (~5 minutes)
        if count % 60 == 0:
            elapsed = (time.time() - start) / 60
            sm = row.get("sm_clock", "N/A")
            temp = row.get("gpu_temp", "N/A")
            ttft = row.get("ttft_p95_ms", "N/A")
            ratio = row.get("sm_clock_ratio", "N/A")
            throttled = "⚠️ THROTTLING" if isinstance(ratio, float) and ratio < 0.92 else "OK"
            print(f"[{elapsed:.0f}m] rows={count} | SM={sm}MHz | Temp={temp}C | TTFT_P95={ttft}ms | {throttled}")

        # Auto-save every 10 minutes
        if count % 120 == 0:
            df = pd.DataFrame(rows)
            df.to_csv(args.output, index=False)
            print(f"[collector] Auto-saved {len(rows)} rows")

        # Check duration
        if args.duration > 0 and (time.time() - start) >= args.duration:
            save_and_exit()

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
