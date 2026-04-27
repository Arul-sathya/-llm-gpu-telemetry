"""
nvidia_smi_exporter.py
======================
Fallback GPU metrics exporter using nvidia-smi instead of DCGM.

Use this when:
  - Running on Modal (DCGM needs privileged container access)
  - Running on Vast.ai / RunPod and DCGM image won't start
  - Quick local testing without DCGM

Exposes the same metric names as the DCGM Exporter so Prometheus
config and Grafana dashboards work without changes.

Usage:
  pip install prometheus-client
  python nvidia_smi_exporter.py
  # Metrics at http://localhost:9400/metrics

On Modal:
  Deploy this as a background task in the same container as vLLM.
"""

import subprocess
import time
import re
import sys
from prometheus_client import (
    Gauge, Counter, start_http_server, REGISTRY, PROCESS_COLLECTOR, PLATFORM_COLLECTOR
)

# Remove default metrics to keep output clean
REGISTRY.unregister(PROCESS_COLLECTOR)
REGISTRY.unregister(PLATFORM_COLLECTOR)

PORT = 9400
SCRAPE_INTERVAL = 5  # seconds

# ── Metric Definitions (matching DCGM field names) ────────────────
LABELS = ["gpu", "modelName", "UUID"]

sm_clock    = Gauge("DCGM_FI_DEV_SM_CLOCK",    "SM clock frequency (MHz)",       LABELS)
mem_clock   = Gauge("DCGM_FI_DEV_MEM_CLOCK",   "Memory clock frequency (MHz)",   LABELS)
gpu_temp    = Gauge("DCGM_FI_DEV_GPU_TEMP",    "GPU temperature (C)",             LABELS)
power_usage = Gauge("DCGM_FI_DEV_POWER_USAGE", "GPU power draw (W)",             LABELS)
fb_used     = Gauge("DCGM_FI_DEV_FB_USED",     "GPU framebuffer used (MiB)",     LABELS)
fb_free     = Gauge("DCGM_FI_DEV_FB_FREE",     "GPU framebuffer free (MiB)",     LABELS)
fb_total    = Gauge("DCGM_FI_DEV_FB_TOTAL",    "GPU framebuffer total (MiB)",    LABELS)
gpu_util    = Gauge("DCGM_FI_DEV_GPU_UTIL",    "GPU utilization (%)",            LABELS)

# Approximated metrics (nvidia-smi doesn't expose these directly)
# tensor_active is approximated from GPU utilization when SM clock is healthy
tensor_active = Gauge("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "Tensor core active ratio (approx)", LABELS)


def get_gpu_info():
    """
    Query nvidia-smi for all GPU metrics in one call.
    Returns list of dicts, one per GPU.
    """
    query_fields = [
        "index",
        "name",
        "uuid",
        "clocks.sm",           # SM clock MHz
        "clocks.mem",          # Memory clock MHz
        "temperature.gpu",     # Temperature C
        "power.draw",          # Power W
        "memory.used",         # VRAM used MiB
        "memory.free",         # VRAM free MiB
        "memory.total",        # VRAM total MiB
        "utilization.gpu",     # GPU util %
    ]

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={','.join(query_fields)}",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
        ).decode("utf-8").strip()
    except subprocess.CalledProcessError as e:
        print(f"ERROR: nvidia-smi failed: {e}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("ERROR: nvidia-smi not found. Is the NVIDIA driver installed?", file=sys.stderr)
        sys.exit(1)

    gpus = []
    for line in output.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 11:
            continue

        def safe_float(val, default=0.0):
            try:
                # nvidia-smi returns "[N/A]" for some fields
                return float(val) if val not in ["[N/A]", "N/A", ""] else default
            except ValueError:
                return default

        gpus.append({
            "index":    parts[0],
            "name":     parts[1],
            "uuid":     parts[2],
            "sm_clock": safe_float(parts[3]),
            "mem_clock":safe_float(parts[4]),
            "temp":     safe_float(parts[5]),
            "power":    safe_float(parts[6]),
            "mem_used": safe_float(parts[7]),
            "mem_free": safe_float(parts[8]),
            "mem_total":safe_float(parts[9]),
            "util":     safe_float(parts[10]),
        })

    return gpus


def update_metrics():
    """Poll nvidia-smi and update Prometheus gauges."""
    gpus = get_gpu_info()
    for g in gpus:
        labels = [g["index"], g["name"], g["uuid"]]

        sm_clock.labels(*labels).set(g["sm_clock"])
        mem_clock.labels(*labels).set(g["mem_clock"])
        gpu_temp.labels(*labels).set(g["temp"])
        power_usage.labels(*labels).set(g["power"])
        fb_used.labels(*labels).set(g["mem_used"])
        fb_free.labels(*labels).set(g["mem_free"])
        fb_total.labels(*labels).set(g["mem_total"])
        gpu_util.labels(*labels).set(g["util"])

        # Approximate tensor active from GPU util (0-1 scale)
        # Not precise but good enough for anomaly detection features
        tensor_active.labels(*labels).set(g["util"] / 100.0)

    if gpus:
        names = [f"GPU{g['index']} ({g['name']})" for g in gpus]
        print(f"[{time.strftime('%H:%M:%S')}] Scraped: {', '.join(names)}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] WARNING: No GPUs found")


def main():
    print(f"Starting nvidia-smi exporter on port {PORT}")
    print(f"Metrics: http://localhost:{PORT}/metrics")
    print(f"Scrape interval: {SCRAPE_INTERVAL}s")
    print("---")

    start_http_server(PORT)

    while True:
        try:
            update_metrics()
        except Exception as e:
            print(f"ERROR in update_metrics: {e}", file=sys.stderr)
        time.sleep(SCRAPE_INTERVAL)


if __name__ == "__main__":
    main()
