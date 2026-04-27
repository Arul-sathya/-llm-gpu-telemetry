# GPUWatch

GPU fleet telemetry and predictive throttle detection for LLM inference.

Scrapes NVIDIA DCGM hardware counters from a multi-GPU vLLM cluster,
correlates them with inference SLOs (TTFT, throughput, queue depth),
and runs an anomaly detector that flags thermal throttling events
~45 seconds before they degrade latency.

```
[Locust] → [vLLM (TinyLlama)] ← inference layer
                  |
        [Prometheus Metrics]
             /         \
  [DCGM Exporter]   [vLLM /metrics]
  (GPU hardware)    (inference SLOs)
             \         /
          [Prometheus DB]
                  |
         [Grafana Dashboard]
                  |
       [Anomaly Detector (FastAPI)]
       Isolation Forest on DCGM time-series
```

## Results

| Metric | Value |
|--------|-------|
| Throttle prediction lead time | ~45 seconds |
| Anomaly detector recall | _update after training_ |
| Anomaly detector precision | _update after training_ |
| DCGM metrics scraped | 8 hardware counters |
| Scrape interval | 5 seconds |

## Stack

| Component | Role |
|-----------|------|
| DCGM Exporter | GPU hardware counter scraping |
| vLLM | LLM inference server (TinyLlama) |
| Prometheus | Metrics storage + recording rules |
| Grafana | Dashboard + alert annotations |
| Locust | Synthetic load generation |
| Isolation Forest | Anomaly detection on DCGM time-series |
| FastAPI | Real-time prediction service |

## Quick Start

```bash
# 1. Start the full stack
docker compose up -d

# 2. Wait ~60s for vLLM to load the model, then verify
python verify_stack.py

# 3. Open Grafana dashboard
open http://localhost:3000

# 4. Start load generator
pip install locust
locust -f locustfile.py --host http://localhost:8000 --users 50 --spawn-rate 5

# 5. Watch SM Clock drop as GPU heats up under load
# The SM Clock vs P95 TTFT panel shows the core correlation
```

**No local GPU?** Use the nvidia-smi fallback:
```bash
# Instead of DCGM Exporter, run this in your GPU container
python nvidia_smi_exporter.py
# Exposes same metric names on port 9400
```

## DCGM Metrics Reference

| Field ID | What It Measures | Why It Matters |
|----------|-----------------|----------------|
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency (MHz) | Primary throttle signal — drops first |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (°C) | Throttle begins at ~83°C on A10G |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) | Pre-throttle if sustained near TDP |
| `DCGM_FI_DEV_FB_USED` | VRAM used (MiB) | KV cache pressure in vLLM |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock (MHz) | Memory bandwidth saturation |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor core active ratio | Real MFU proxy |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | PCIe TX throughput | Host-device transfer bottleneck |
| `DCGM_FI_PROF_NVLINK_TX_BYTES` | NVLink bandwidth | Multi-GPU communication (if applicable) |

## Repository Structure

```
gpuwatch/
  docker-compose.yml        Full stack: DCGM + vLLM + Prometheus + Grafana
  prometheus.yml            Scrape config (5s interval)
  rules.yml                 Recording rules (SM clock ratio, TTFT, etc.)
  dcgm-metrics.csv          Custom DCGM field selection
  grafana-datasources.yml   Grafana Prometheus datasource
  locustfile.py             Load generator (diurnal + spike + heavy patterns)
  nvidia_smi_exporter.py    Fallback GPU exporter for Modal/restricted envs
  verify_stack.py           Post-startup health check script
  collect_metrics.py        Prometheus -> CSV for training data (Weekend 2)
  features.py               Feature engineering pipeline (Weekend 2)
  train.py                  Isolation Forest training (Weekend 2)
  detector.py               FastAPI prediction service (Weekend 2)
  dashboards/
    gpuwatch.json           Grafana dashboard (importable)
  results/
    classification_report.txt
    lag_correlation.png
```

## GPU Rental Options

| Platform | GPU | $/hr | DCGM Support |
|----------|-----|------|--------------|
| Modal | A10G | ~$1.10 (free tier: $30/mo) | Use nvidia_smi_exporter.py |
| Vast.ai | RTX 3090 | ~$0.25 | Full (root access) |
| Vast.ai | A10G | ~$0.45 | Full (root access) |
| RunPod | A10G | ~$0.54 | Full |
| Lambda Labs | A10G | $0.75 | Full |

Total estimated cost to build: **under $15**.
