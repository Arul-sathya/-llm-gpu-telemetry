"""
detector.py
===========
FastAPI service that runs the trained Isolation Forest in real time.
Queries Prometheus every 5 seconds, builds feature vectors,
runs inference, and posts alert annotations to Grafana when
a throttle event is predicted.

Usage:
    pip install fastapi uvicorn joblib scikit-learn requests
    uvicorn detector:app --host 0.0.0.0 --port 8080

    # Or run directly:
    python detector.py

Endpoints:
    GET /predict    — run one prediction, returns score + anomaly flag
    GET /status     — model info + prediction stats
    GET /health     — liveness check
"""

import time
import threading
import logging
from datetime import datetime, timezone
from collections import deque

import joblib
import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("detector")

# ── Config ────────────────────────────────────────────────────────
PROMETHEUS  = "http://localhost:9090"
GRAFANA     = "http://localhost:3000"
MODEL_PATH  = "models/gpuwatch_model.pkl"
POLL_INTERVAL = 5   # seconds between auto-predictions
PORT        = 8080

# ── Prometheus queries (must match features.py) ───────────────────
QUERIES = {
    "sm_clock":         "DCGM_FI_DEV_SM_CLOCK",
    "sm_clock_std":     "stddev_over_time(DCGM_FI_DEV_SM_CLOCK[5m])",
    "sm_clock_drop":    "max_over_time(DCGM_FI_DEV_SM_CLOCK[5m]) - DCGM_FI_DEV_SM_CLOCK",
    "sm_clock_ratio":   "gpuwatch:sm_clock_ratio",
    "gpu_temp":         "DCGM_FI_DEV_GPU_TEMP",
    "temp_slope":       "deriv(DCGM_FI_DEV_GPU_TEMP[1m]) * 60",
    "temp_slope_2m":    "deriv(DCGM_FI_DEV_GPU_TEMP[2m]) * 60",
    "temp_max":         "max_over_time(DCGM_FI_DEV_GPU_TEMP[5m])",
    "power_usage":      "DCGM_FI_DEV_POWER_USAGE",
    "power_slope":      "deriv(DCGM_FI_DEV_POWER_USAGE[1m]) * 60",
    "power_std":        "stddev_over_time(DCGM_FI_DEV_POWER_USAGE[5m])",
    "mem_util_pct":     "gpuwatch:mem_util_pct",
    "mem_slope":        "deriv(DCGM_FI_DEV_FB_USED[1m]) * 60",
    "tensor_mean":      "avg_over_time(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE[5m])",
    "util_mean":        "avg_over_time(DCGM_FI_DEV_GPU_UTIL[5m])",
    "ttft_delta":       "deriv(gpuwatch:ttft_p95_ms[2m])",
    "ttft_slope":       "deriv(gpuwatch:ttft_p95_ms[1m])",
    "queue_mean":       "avg_over_time(vllm:num_requests_waiting[5m])",
    "queue_slope":      "deriv(vllm:num_requests_waiting[1m])",
}


# ── State ─────────────────────────────────────────────────────────
class DetectorState:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = None
        self.loaded = False
        self.predictions = deque(maxlen=200)   # rolling history
        self.total_predictions = 0
        self.total_anomalies = 0
        self.last_prediction = None
        self.last_anomaly_time = None
        self.consecutive_anomalies = 0

state = DetectorState()
app = FastAPI(title="GPUWatch Anomaly Detector")


# ── Model loading ─────────────────────────────────────────────────
def load_model():
    try:
        artifacts = joblib.load(MODEL_PATH)
        state.model   = artifacts["model"]
        state.scaler  = artifacts["scaler"]
        state.features = artifacts["features"]
        state.loaded  = True
        log.info(f"Model loaded: {len(state.features)} features")
        log.info(f"Features: {state.features}")
    except FileNotFoundError:
        log.warning(f"Model not found at {MODEL_PATH}")
        log.warning("Run: python train.py  to train the model first")
    except Exception as e:
        log.error(f"Model load error: {e}")


# ── Prometheus scraping ───────────────────────────────────────────
def query_prometheus(expr):
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


def get_feature_vector():
    """Scrape current metrics and build feature vector."""
    if not state.features:
        return None, {}

    raw = {}
    for fname in state.features:
        if fname in QUERIES:
            raw[fname] = query_prometheus(QUERIES[fname])
        else:
            raw[fname] = None

    # Check for missing values
    missing = [k for k, v in raw.items() if v is None]
    if len(missing) > len(state.features) // 2:
        log.warning(f"Too many missing features: {missing}")
        return None, raw

    # Fill remaining None with 0
    vector = np.array([raw.get(f, 0.0) or 0.0 for f in state.features]).reshape(1, -1)
    return vector, raw


# ── Grafana annotation ────────────────────────────────────────────
def post_grafana_annotation(text, tags=None):
    try:
        payload = {
            "time": int(datetime.now(timezone.utc).timestamp() * 1000),
            "text": text,
            "tags": tags or ["gpuwatch", "anomaly"],
        }
        r = requests.post(
            f"{GRAFANA}/api/annotations",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=3,
        )
        if r.status_code in (200, 201):
            log.info(f"Grafana annotation posted: {text}")
        else:
            log.warning(f"Grafana annotation failed: {r.status_code}")
    except Exception as e:
        log.warning(f"Grafana annotation error: {e}")


# ── Core prediction ───────────────────────────────────────────────
def run_prediction():
    """Run one prediction cycle. Returns result dict."""
    if not state.loaded:
        return {"error": "Model not loaded — run train.py first"}

    X, raw = get_feature_vector()
    if X is None:
        return {"error": "Could not build feature vector — Prometheus unreachable?"}

    X_scaled = state.scaler.transform(X)
    score    = float(state.model.score_samples(X_scaled)[0])
    is_anomaly = state.model.predict(X_scaled)[0] == -1

    result = {
        "timestamp":   datetime.utcnow().isoformat(),
        "anomaly":     bool(is_anomaly),
        "score":       round(score, 4),
        "raw_metrics": {k: round(v, 2) if v else None for k, v in raw.items()},
        "prediction":  "PRE-THROTTLE" if is_anomaly else "NORMAL",
    }

    # Update state
    state.total_predictions += 1
    state.predictions.append(result)
    state.last_prediction = result

    if is_anomaly:
        state.total_anomalies += 1
        state.consecutive_anomalies += 1
        state.last_anomaly_time = datetime.utcnow().isoformat()

        # Post Grafana annotation on first detection (not every 5s)
        if state.consecutive_anomalies == 1:
            sm = raw.get("sm_clock", "?")
            temp = raw.get("gpu_temp", "?")
            post_grafana_annotation(
                f"⚠️ THROTTLE PREDICTED in ~45s | SM={sm}MHz Temp={temp}C",
                tags=["gpuwatch", "throttle-prediction"],
            )
            log.warning(f"ANOMALY DETECTED — score={score:.4f} SM={sm}MHz Temp={temp}C")
    else:
        state.consecutive_anomalies = 0

    return result


# ── Background poller ─────────────────────────────────────────────
def background_poller():
    """Poll and predict every POLL_INTERVAL seconds."""
    log.info(f"Background poller started (every {POLL_INTERVAL}s)")
    while True:
        try:
            result = run_prediction()
            if "error" not in result:
                status = "⚠️  ANOMALY" if result["anomaly"] else "   normal"
                score = result.get("score", "?")
                log.info(f"{status} | score={score:.4f}")
        except Exception as e:
            log.error(f"Poller error: {e}")
        time.sleep(POLL_INTERVAL)


# ── API endpoints ─────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    load_model()
    t = threading.Thread(target=background_poller, daemon=True)
    t.start()
    log.info(f"GPUWatch detector running on :{PORT}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.loaded}


@app.get("/predict")
def predict():
    """Run a single prediction and return result."""
    result = run_prediction()
    status_code = 200
    if "error" in result:
        status_code = 503
    return JSONResponse(content=result, status_code=status_code)


@app.get("/status")
def status():
    """Return model info and prediction history stats."""
    anomaly_rate = (
        state.total_anomalies / state.total_predictions
        if state.total_predictions > 0 else 0
    )
    recent = list(state.predictions)[-10:]
    recent_anomaly_rate = (
        sum(1 for r in recent if r.get("anomaly")) / len(recent)
        if recent else 0
    )

    return {
        "model_loaded":           state.loaded,
        "features":               state.features,
        "total_predictions":      state.total_predictions,
        "total_anomalies":        state.total_anomalies,
        "anomaly_rate":           round(anomaly_rate, 3),
        "recent_anomaly_rate":    round(recent_anomaly_rate, 3),
        "consecutive_anomalies":  state.consecutive_anomalies,
        "last_anomaly_time":      state.last_anomaly_time,
        "last_prediction":        state.last_prediction,
    }


@app.get("/history")
def history(n: int = 50):
    """Return last N predictions."""
    return {"predictions": list(state.predictions)[-n:]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
