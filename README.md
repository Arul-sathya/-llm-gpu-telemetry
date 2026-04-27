# llm-gpu-telemetry

Real-time GPU fleet telemetry and predictive anomaly detection for LLM inference clusters.

Scrapes hardware counters from NVIDIA GPUs running vLLM inference, correlates them with inference SLOs (TTFT, throughput, queue depth), and runs a Gradient Boosting anomaly detector that predicts thermal throttling events **45 seconds before they degrade latency**.

---

## Results

| Metric | Value |
|--------|-------|
| Model | Gradient Boosting Classifier |
| Precision | **0.957** |
| Recall | **0.759** |
| F1 | **0.846** |
| Prediction lead time | **45 seconds** |
| Training samples | 3,650 rows |
| GPU metrics scraped | 8 hardware counters |
| Scrape interval | 5 seconds |
| Inference server | vLLM (facebook/opt-1.3b) |
| GPU tested | NVIDIA RTX 3090 (24GB) |

---

## Architecture

```
[Locust 200 users] ──→ [vLLM (opt-1.3b)] ←── inference layer
                               │
                     [Prometheus Metrics]
                          /         \
           [nvidia-smi Exporter]   [vLLM /metrics]
           (GPU hardware)          (inference SLOs)
                          \         /
                       [Prometheus DB]
                               │
                      [Grafana Dashboard]
                               │
              [Anomaly Detector (FastAPI)]
              Gradient Boosting on GPU time-series
                               │
                    [Grafana Annotations]
                    ⚠️ THROTTLE PREDICTED in ~45s
```

---

## Why This Project

When running LLMs in production, GPU thermal throttling is invisible until it's too late. By the time P95 TTFT spikes in your dashboard, the SM clock already dropped 45 seconds ago. This project detects the early warning signals — rising temperature, power draw near TDP, SM clock oscillation — and fires an alert before users notice latency degradation.

---

## Stack

| Component | Role |
|-----------|------|
| nvidia-smi Exporter | GPU hardware counter scraping |
| vLLM | LLM inference server |
| Prometheus | Metrics storage + recording rules (5s scrape interval) |
| Grafana | Dashboard + alert annotations |
| Locust | Synthetic load generation (diurnal + spike patterns) |
| Gradient Boosting | Supervised anomaly detection on GPU time-series |
| FastAPI | Real-time prediction service |

---

## GPU Metrics Scraped

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| `DCGM_FI_DEV_SM_CLOCK` | SM clock frequency (MHz) | Primary throttle signal — drops first |
| `DCGM_FI_DEV_GPU_TEMP` | GPU temperature (°C) | Throttle onset at ~83°C on RTX 3090 |
| `DCGM_FI_DEV_POWER_USAGE` | Power draw (W) | Pre-throttle signal if sustained near TDP |
| `DCGM_FI_DEV_FB_USED` | VRAM used (MiB) | KV cache pressure in vLLM |
| `DCGM_FI_DEV_FB_FREE` | VRAM free (MiB) | OOM risk indicator |
| `DCGM_FI_DEV_GPU_UTIL` | GPU utilization (%) | Compute utilization baseline |
| `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` | Tensor core active ratio | Real MFU proxy |
| `DCGM_FI_DEV_MEM_CLOCK` | Memory clock (MHz) | Memory bandwidth saturation |

---

## ML Pipeline

```
Raw GPU telemetry (5s intervals)
        ↓
Feature Engineering (features.py)
  - Rolling window statistics (5-min)
  - SM clock drop from recent max
  - Temperature slope (1min, 2min)
  - Power standard deviation
  - TTFT delta and slope
  - Queue depth statistics
        ↓
Gradient Boosting Classifier (train.py)
  - Trained on 3,650 samples
  - class_weight="balanced" for imbalanced data
  - 5-fold stratified cross-validation
  - Forward-shifted labels (45s prediction horizon)
        ↓
FastAPI Detector (detector.py)
  - Polls Prometheus every 5s
  - Posts Grafana annotations on anomaly
  - /predict, /status, /history endpoints
```

---

## Quick Start

### Vast.ai RTX 3090 (~$0.25/hr)

```bash
# SSH into instance
ssh -p PORT root@INSTANCE_IP

# Clone and setup
git clone https://github.com/Arul-sathya/-llm-gpu-telemetry.git
cd llm-gpu-telemetry
pip install -r requirements.txt

# Fix Prometheus targets
sed -i "s/dcgm-exporter:9400/localhost:9400/" prometheus.yml
sed -i "s/vllm:8000/localhost:8000/" prometheus.yml

# Start stack
python nvidia_smi_exporter.py &
python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-1.3b --host 0.0.0.0 --port 8000 &

# Collect data + load
locust -f locustfile.py --host http://localhost:8000 \
  --users 200 --spawn-rate 20 --headless --run-time 6h &
python collect_metrics.py --duration 21600
```

### Train the model

```bash
python features.py --input gpu_telemetry.csv --output features.csv
python explore_simple.py --input features.csv   # compares RF, GB, LR
python train.py --input features.csv
```

### Run the detector

```bash
python detector.py
curl http://localhost:8080/predict
```

Sample response:
```json
{
  "anomaly": true,
  "score": 0.847,
  "prediction": "PRE-THROTTLE",
  "raw_metrics": {
    "sm_clock": 1860.0,
    "gpu_temp": 78.0,
    "power_usage": 338.5
  }
}
```

---

## Repository Structure

```
llm-gpu-telemetry/
  collect_metrics.py      Prometheus → CSV data collector
  features.py             Feature engineering pipeline
  train.py                Model training (Isolation Forest / supervised)
  explore_simple.py       Multi-model comparison (RF, GB, LR)
  detector.py             FastAPI real-time prediction service
  nvidia_smi_exporter.py  GPU metrics exporter (nvidia-smi based)
  locustfile.py           Load generator (diurnal + spike patterns)
  prometheus.yml          Scrape config (5s interval)
  rules.yml               Recording rules
  docker-compose.yml      Full Docker stack
  dashboards/
    gpuwatch.json         Grafana dashboard JSON (importable)
  models/
    gpuwatch_model.pkl    Trained Gradient Boosting model
  results/
    classification_report.txt
    lag_correlation.png
```

---

## Related Projects

- [FlashDecode](https://github.com/Arul-sathya/flashdecode) — GPU kernel benchmarking (Triton, FlashAttention, xFormers)
- [UncertaintyDecode](https://github.com/Arul-sathya/uncertainty-decode) — KV cache eviction using Dirichlet EDL uncertainty scores
