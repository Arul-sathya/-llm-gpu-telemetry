"""
modal_app.py - GPUWatch inference server
Uses transformers directly — no vLLM dependency hell.
"""
import modal
import threading
import subprocess
import time

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.0",
        "accelerate==0.33.0",
        "fastapi==0.110.0",
        "uvicorn==0.27.1",
        "prometheus-client==0.20.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
)

app = modal.App("gpuwatch", image=image)
hf_cache = modal.Volume.from_name("gpuwatch-hf-cache", create_if_missing=True)
MODEL = "facebook/opt-125m"


def run_metrics_exporter(port=9400):
    from prometheus_client import (
        Gauge, start_http_server, REGISTRY,
        PROCESS_COLLECTOR, PLATFORM_COLLECTOR
    )
    try:
        REGISTRY.unregister(PROCESS_COLLECTOR)
        REGISTRY.unregister(PLATFORM_COLLECTOR)
    except Exception:
        pass

    LABELS = ["gpu", "modelName", "UUID"]
    sm_clock    = Gauge("DCGM_FI_DEV_SM_CLOCK",    "SM clock MHz",   LABELS)
    gpu_temp    = Gauge("DCGM_FI_DEV_GPU_TEMP",    "Temperature C",  LABELS)
    power_usage = Gauge("DCGM_FI_DEV_POWER_USAGE", "Power W",        LABELS)
    fb_used     = Gauge("DCGM_FI_DEV_FB_USED",     "VRAM used MiB",  LABELS)
    fb_free     = Gauge("DCGM_FI_DEV_FB_FREE",     "VRAM free MiB",  LABELS)
    fb_total    = Gauge("DCGM_FI_DEV_FB_TOTAL",    "VRAM total MiB", LABELS)
    gpu_util    = Gauge("DCGM_FI_DEV_GPU_UTIL",    "GPU util pct",   LABELS)
    tensor_est  = Gauge("DCGM_FI_PROF_PIPE_TENSOR_ACTIVE", "Tensor", LABELS)

    start_http_server(port)
    print(f"[metrics] running on :{port}")

    while True:
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,name,uuid,clocks.sm,temperature.gpu,"
                "power.draw,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits"
            ], timeout=10).decode().strip()
            for line in out.split("\n"):
                p = [x.strip() for x in line.split(",")]
                if len(p) < 10:
                    continue
                def s(v):
                    try: return float(v)
                    except: return 0.0
                lb = [p[0], p[1], p[2]]
                sm_clock.labels(*lb).set(s(p[3]))
                gpu_temp.labels(*lb).set(s(p[4]))
                power_usage.labels(*lb).set(s(p[5]))
                fb_used.labels(*lb).set(s(p[6]))
                fb_free.labels(*lb).set(s(p[7]))
                fb_total.labels(*lb).set(s(p[8]))
                gpu_util.labels(*lb).set(s(p[9]))
                tensor_est.labels(*lb).set(s(p[9]) / 100.0)
        except Exception as e:
            print(f"[metrics] error: {e}")
        time.sleep(5)


@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=300,
    volumes={"/root/.cache/huggingface": hf_cache},
)
@modal.web_server(8000, startup_timeout=300)
def vllm_server():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
    import uuid
    import time as t

    # Start GPU metrics in background
    thread = threading.Thread(target=run_metrics_exporter, args=(9400,), daemon=True)
    thread.start()

    print(f"[server] loading {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print("[server] model loaded, starting API")

    api = FastAPI()

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.get("/metrics")
    def metrics():
        # Proxy nvidia-smi metrics as text for Prometheus
        try:
            out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=index,name,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]).decode().strip()
            lines = []
            for line in out.split("\n"):
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 6:
                    lines.append(f'# GPU {p[0]} {p[1]}: SM={p[2]}MHz Temp={p[3]}C Power={p[4]}W Util={p[5]}%')
            return "\n".join(lines)
        except Exception as e:
            return str(e)

    @api.post("/v1/completions")
    def completions(body: dict):
        prompt = body.get("prompt", "Hello")
        max_tokens = min(body.get("max_tokens", 50), 100)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
            "object": "text_completion",
            "model": MODEL,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(output[0]) - inputs["input_ids"].shape[1]}
        }

    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")


@app.function(
    gpu="A10G",
    timeout=3600,
    scaledown_window=300,
)
@modal.web_server(9400, startup_timeout=60)
def metrics_server():
    thread = threading.Thread(target=run_metrics_exporter, args=(9400,), daemon=True)
    thread.start()
    print("[metrics] standalone server started")
    while True:
        time.sleep(30)
