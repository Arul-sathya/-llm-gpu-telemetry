#!/usr/bin/env python3
"""
setup_modal.py
==============
After `modal deploy modal_app.py`, run this script.
It fetches your Modal endpoint URLs and patches prometheus-modal.yml
automatically so you don't have to edit it manually.

Usage:
  python setup_modal.py

Requirements:
  pip install modal requests
  modal setup  (authenticate first)
"""

import subprocess
import sys
import re

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.stderr.strip()

print("GPUWatch Modal Setup")
print("=" * 50)

# ── Step 1: Check modal is installed ──────────────────────────────
stdout, _ = run("modal --version")
if not stdout:
    print("ERROR: modal not installed. Run: pip install modal")
    sys.exit(1)
print(f"[OK] Modal CLI: {stdout}")

# ── Step 2: List deployed endpoints ───────────────────────────────
print("\nFetching Modal app endpoints...")
stdout, stderr = run("modal app list")
if "gpuwatch" not in stdout:
    print("ERROR: gpuwatch app not deployed yet.")
    print("Run: modal deploy modal_app.py")
    sys.exit(1)

# ── Step 3: Get endpoint URLs ─────────────────────────────────────
stdout, _ = run("modal app logs gpuwatch --json 2>/dev/null || echo ''")

# Try to get URLs from modal serve output
print("\nTo get your endpoint URLs, run:")
print("  modal deploy modal_app.py")
print("\nThen look for lines like:")
print("  vllm_server => https://YOUR-NAME--gpuwatch-vllm-server.modal.run")
print("  metrics_server => https://YOUR-NAME--gpuwatch-metrics-server.modal.run")

print("\n" + "=" * 50)
vllm_url = input("Paste your vLLM URL (without https://): ").strip()
metrics_url = input("Paste your metrics URL (without https://): ").strip()

if not vllm_url or not metrics_url:
    print("ERROR: Both URLs are required.")
    sys.exit(1)

# ── Step 4: Patch prometheus-modal.yml ────────────────────────────
with open("prometheus-modal.yml", "r") as f:
    content = f.read()

content = content.replace("YOUR_MODAL_VLLM_URL", vllm_url)
content = content.replace("YOUR_MODAL_METRICS_URL", metrics_url)

with open("prometheus-modal.yml", "w") as f:
    f.write(content)

print(f"\n[OK] prometheus-modal.yml updated:")
print(f"     vLLM    → https://{vllm_url}/metrics")
print(f"     GPU     → https://{metrics_url}/metrics")

# ── Step 5: Start local stack ─────────────────────────────────────
print("\nStarting Prometheus + Grafana locally...")
stdout, stderr = subprocess.run(
    "docker compose -f docker-compose-mac.yml up -d",
    shell=True, capture_output=True, text=True
).stdout, ""

print("[OK] Local stack started")
print("\nNext steps:")
print("  1. Open Grafana:   http://localhost:3000")
print("  2. Open Prometheus: http://localhost:9090")
print("  3. Check targets:  http://localhost:9090/targets")
print("     Both 'vllm' and 'gpu' jobs should show state=UP")
print("  4. Start load:")
print("     locust -f locustfile.py --host https://" + vllm_url)
