#!/usr/bin/env python3
"""
verify_modal.py
===============
Health check for the Mac + Modal setup.
Run after setup_modal.py completes.

Usage:
  python verify_modal.py
"""

import requests
import sys
import re

OK   = "\033[92m[OK]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"

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
        r = requests.get(
            "http://localhost:9090/api/v1/query",
            params={"query": expr}, timeout=5
        )
        return r.json().get("data", {}).get("result", [])
    except Exception:
        return []

# ── Read Modal URLs from prometheus-modal.yml ─────────────────────
vllm_url = None
metrics_url = None
try:
    with open("prometheus-modal.yml") as f:
        content = f.read()
    urls = re.findall(r"targets: \['([^']+)'\]", content)
    for u in urls:
        if "YOUR_MODAL" in u:
            print(f"{FAIL} prometheus-modal.yml not configured — run setup_modal.py first")
            sys.exit(1)
        if vllm_url is None:
            vllm_url = u
        else:
            metrics_url = u
except FileNotFoundError:
    print(f"{FAIL} prometheus-modal.yml not found — are you in the gpuwatch directory?")
    sys.exit(1)

print("\nGPUWatch Modal Stack Verification")
print("=" * 50)
print(f"vLLM endpoint:    https://{vllm_url}")
print(f"Metrics endpoint: https://{metrics_url}")
print()

# ── Check Modal vLLM ──────────────────────────────────────────────
try:
    r = requests.get(f"https://{vllm_url}/health", timeout=15)
    check("Modal vLLM health", r.status_code == 200, "endpoint responding")
except Exception as e:
    check("Modal vLLM health", False, f"{e} — is Modal app deployed and running?")

try:
    r = requests.post(f"https://{vllm_url}/v1/completions", json={
        "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt": "Hello",
        "max_tokens": 5,
    }, timeout=30)
    ok = r.status_code == 200 and "choices" in r.json()
    check("Modal vLLM inference", ok, "test completion successful")
except Exception as e:
    check("Modal vLLM inference", False, str(e))

# ── Check Modal metrics ───────────────────────────────────────────
try:
    r = requests.get(f"https://{metrics_url}/metrics", timeout=10)
    has_sm = "DCGM_FI_DEV_SM_CLOCK" in r.text
    check("Modal GPU metrics", has_sm, "DCGM_FI_DEV_SM_CLOCK found")
    if has_sm:
        line = [l for l in r.text.split("\n") if "DCGM_FI_DEV_SM_CLOCK" in l and not l.startswith("#")]
        if line:
            val = line[0].split()[-1]
            print(f"     SM Clock: {val} MHz")
except Exception as e:
    check("Modal GPU metrics", False, str(e))

# ── Check local Prometheus ────────────────────────────────────────
try:
    r = requests.get("http://localhost:9090/-/healthy", timeout=5)
    check("Prometheus (local)", r.status_code == 200, "healthy")
except Exception as e:
    check("Prometheus (local)", False, f"{e} — run: docker compose -f docker-compose-mac.yml up -d")

try:
    r = requests.get("http://localhost:9090/api/v1/targets", timeout=5)
    targets = r.json()["data"]["activeTargets"]
    vllm_up = any(t["labels"].get("job") == "vllm" and t["health"] == "up" for t in targets)
    gpu_up  = any(t["labels"].get("job") == "gpu"  and t["health"] == "up" for t in targets)
    check("Prometheus — vLLM target", vllm_up, "state=up")
    check("Prometheus — GPU target",  gpu_up,  "state=up")
except Exception as e:
    check("Prometheus targets", False, str(e))

# ── Check recording rules ─────────────────────────────────────────
result = query_prometheus("gpuwatch:sm_clock_ratio")
check("Recording rule — sm_clock_ratio", len(result) > 0,
      result[0]['value'][1][:6] if result else "not yet — need 5min of data")

# ── Check Grafana ─────────────────────────────────────────────────
try:
    r = requests.get("http://localhost:3000/api/health", timeout=5)
    check("Grafana (local)", r.status_code == 200)
except Exception as e:
    check("Grafana (local)", False, str(e))

# ── Summary ───────────────────────────────────────────────────────
print()
print("=" * 50)
if errors == 0:
    print("\033[92mAll checks passed.\033[0m")
    print("\nOpen Grafana: http://localhost:3000")
    print(f"Start load:   locust -f locustfile.py --host https://{vllm_url}")
else:
    print(f"\033[91m{errors} check(s) failed.\033[0m")
    sys.exit(1)
