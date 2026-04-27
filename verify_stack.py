#!/usr/bin/env python3
"""
verify_stack.py
===============
Run after `docker compose up -d` to confirm all services
are healthy and metrics are flowing correctly.

Usage:
  python verify_stack.py

Expected output (all green):
  [OK] DCGM Exporter — SM clock metric found: 1410.0 MHz
  [OK] vLLM — health endpoint responding
  [OK] vLLM — Prometheus metrics endpoint responding
  [OK] Prometheus — DCGM target is UP
  [OK] Prometheus — vLLM target is UP
  [OK] Grafana — API responding
  [OK] Recording rules — gpuwatch:sm_clock_ratio found
  All checks passed. Stack is healthy.
"""

import requests
import sys
import time

DCGM    = "http://localhost:9400"
VLLM    = "http://localhost:8000"
PROM    = "http://localhost:9090"
GRAFANA = "http://localhost:3000"

OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

errors = 0

def check(label, condition, detail=""):
    global errors
    if condition:
        print(f"{OK} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"{FAIL} {label}" + (f" — {detail}" if detail else ""))
        errors += 1

def query_prometheus(expr):
    try:
        r = requests.get(f"{PROM}/api/v1/query", params={"query": expr}, timeout=5)
        result = r.json().get("data", {}).get("result", [])
        return result
    except Exception:
        return []


print("\nGPUWatch Stack Verification")
print("=" * 50)
print("Waiting 5s for services to stabilize...")
time.sleep(5)
print()

# ── DCGM Exporter ─────────────────────────────────────────────────
try:
    r = requests.get(f"{DCGM}/metrics", timeout=5)
    sm_line = [l for l in r.text.split("\n") if "DCGM_FI_DEV_SM_CLOCK" in l and not l.startswith("#")]
    if sm_line:
        val = sm_line[0].split()[-1]
        check("DCGM Exporter", True, f"SM clock: {val} MHz")
    else:
        check("DCGM Exporter — metrics endpoint up but DCGM_FI_DEV_SM_CLOCK not found", False,
              "Check dcgm-metrics.csv and that GPU is visible to container")
except Exception as e:
    check("DCGM Exporter", False, f"Cannot reach {DCGM}/metrics: {e}")
    print(f"  Tip: Try `nvidia_smi_exporter.py` as fallback")

# ── vLLM ──────────────────────────────────────────────────────────
try:
    r = requests.get(f"{VLLM}/health", timeout=10)
    check("vLLM health", r.status_code == 200, "endpoint responding")
except Exception as e:
    check("vLLM health", False, f"{e} — model may still be loading, wait 60s and retry")

try:
    r = requests.get(f"{VLLM}/metrics", timeout=5)
    has_metrics = "vllm" in r.text
    check("vLLM Prometheus metrics", has_metrics, "vllm: metrics found")
except Exception as e:
    check("vLLM Prometheus metrics", False, str(e))

# ── Quick inference test ───────────────────────────────────────────
try:
    r = requests.post(f"{VLLM}/v1/completions", json={
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello",
        "max_tokens": 10,
    }, timeout=30)
    ok = r.status_code == 200 and "choices" in r.json()
    check("vLLM inference", ok, "test completion successful")
except Exception as e:
    check("vLLM inference", False, str(e))

# ── Prometheus ────────────────────────────────────────────────────
try:
    r = requests.get(f"{PROM}/api/v1/targets", timeout=5)
    targets = r.json()["data"]["activeTargets"]
    dcgm_up = any(t["labels"].get("job") == "dcgm" and t["health"] == "up" for t in targets)
    vllm_up = any(t["labels"].get("job") == "vllm" and t["health"] == "up" for t in targets)
    check("Prometheus — DCGM target", dcgm_up, "health=up")
    check("Prometheus — vLLM target", vllm_up, "health=up")
except Exception as e:
    check("Prometheus targets", False, str(e))

# ── Recording rules ───────────────────────────────────────────────
result = query_prometheus("gpuwatch:sm_clock_ratio")
check("Recording rule — gpuwatch:sm_clock_ratio", len(result) > 0,
      f"value={result[0]['value'][1][:6] if result else 'missing'}")

result = query_prometheus("gpuwatch:ttft_p95_ms")
check("Recording rule — gpuwatch:ttft_p95_ms", len(result) > 0,
      f"value={result[0]['value'][1][:8] if result else 'missing — need traffic first'}")

# ── Grafana ───────────────────────────────────────────────────────
try:
    r = requests.get(f"{GRAFANA}/api/health", timeout=5)
    check("Grafana", r.status_code == 200, "API responding")
except Exception as e:
    check("Grafana", False, str(e))

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 50)
if errors == 0:
    print("\033[92mAll checks passed. Stack is healthy.\033[0m")
    print("\nNext steps:")
    print("  1. Open Grafana: http://localhost:3000")
    print("  2. Start load:   locust -f locustfile.py --host http://localhost:8000")
    print("  3. Watch SM Clock drop as GPU heats up under load")
else:
    print(f"\033[91m{errors} check(s) failed. Fix above issues before collecting data.\033[0m")
    sys.exit(1)
