"""
GPUWatch Load Generator
=======================
Simulates realistic LLM inference traffic patterns to stress the GPU
and trigger observable thermal/throttling events.

Traffic shapes:
  - Diurnal: slow ramp up, hold at peak, ramp down
  - Spike: sudden burst then back to baseline (flash crowd)
  - Sustained heavy: constant max load to force thermal throttle
  - Mixed: realistic workload with short + long prompts

Usage:
  pip install locust

  # Basic run (50 users, ramp over 30s)
  locust -f locustfile.py --host http://localhost:8000 \
    --users 50 --spawn-rate 5 --run-time 2h

  # Headless with spike shape
  locust -f locustfile.py --host http://localhost:8000 \
    --users 80 --spawn-rate 20 --run-time 30m --headless

  # Open browser UI at http://localhost:8089 (default)
  locust -f locustfile.py --host http://localhost:8000

Data collection note:
  Run for at least 4 hours across all shapes to get enough
  throttled samples for the anomaly detector training data.
  Target: >100 throttle events across the dataset.
"""

import random
import time
import json
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner

# ── Prompt Bank ───────────────────────────────────────────────────
# Mix of short and long prompts to vary token generation pressure

SHORT_PROMPTS = [
    "What is attention in transformers?",
    "Explain gradient descent in 2 sentences.",
    "What is a KV cache?",
    "Define RLHF.",
    "What does beam search do?",
    "What is quantization in ML?",
    "What is the difference between FP16 and BF16?",
    "Explain softmax.",
    "What is perplexity?",
    "What is a LoRA adapter?",
]

MEDIUM_PROMPTS = [
    "Explain how transformer self-attention works, including the role of Q, K, and V matrices.",
    "Compare Adam and AdamW optimizers. When would you use each?",
    "What is the difference between encoder-only, decoder-only, and encoder-decoder architectures?",
    "Explain the PagedAttention algorithm used in vLLM and why it improves GPU memory efficiency.",
    "How does speculative decoding reduce latency in LLM inference?",
    "Describe the training process for RLHF. What are the reward model and PPO stages?",
    "What is the difference between fine-tuning and RAG? When would you use each approach?",
    "Explain how Flash Attention reduces memory complexity from O(n^2) to O(n).",
]

LONG_PROMPTS = [
    "Write a detailed technical explanation of how the KV cache works in transformer inference, "
    "why it grows with sequence length, and what strategies like sliding window attention "
    "and StreamingLLM use to manage it for long contexts.",

    "Explain the full training pipeline for a large language model from scratch: "
    "dataset curation, tokenization, pretraining objectives, compute infrastructure, "
    "gradient checkpointing, mixed precision training, and evaluation.",

    "Compare the following inference optimization techniques in detail: "
    "quantization (GPTQ, AWQ, SqueezeLLM), pruning, speculative decoding, "
    "continuous batching, and tensor parallelism. When would you apply each? "
    "What are the tradeoffs in terms of throughput, latency, and quality?",

    "Describe how you would design a production LLM inference system to serve "
    "100K requests per day with P95 TTFT under 1 second and P95 ITL under 50ms, "
    "using a fleet of 8xA100 GPUs. Cover autoscaling, load balancing, "
    "model sharding, observability, and failure handling.",
]

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ── User Classes ──────────────────────────────────────────────────

class ShortRequestUser(HttpUser):
    """
    Fast interactive users — short prompts, low token output.
    Simulates chat-style usage. High concurrency, low per-request load.
    """
    wait_time = between(0.2, 1.0)
    weight = 3  # 3x more common than heavy users

    @task(4)
    def short_completion(self):
        prompt = random.choice(SHORT_PROMPTS)
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.7,
            },
            name="/v1/completions [short]",
        )

    @task(1)
    def medium_completion(self):
        prompt = random.choice(MEDIUM_PROMPTS)
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.7,
            },
            name="/v1/completions [medium]",
        )

    @task(1)
    def health_check(self):
        """Occasional health check — doesn't count toward GPU load."""
        self.client.get("/health", name="/health")


class HeavyRequestUser(HttpUser):
    """
    Heavy batch users — long prompts, max token output.
    Simulates batch inference jobs that saturate the GPU.
    These are what trigger thermal throttling events.
    """
    wait_time = between(1.0, 3.0)
    weight = 1

    @task(2)
    def long_completion(self):
        prompt = random.choice(LONG_PROMPTS)
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.8,
            },
            name="/v1/completions [long]",
            timeout=60,
        )

    @task(1)
    def medium_completion(self):
        # Long prompt + medium output — high KV cache pressure
        prompt = random.choice(MEDIUM_PROMPTS) * 3
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.5,
            },
            name="/v1/completions [medium-long]",
            timeout=30,
        )


class SpikeUser(HttpUser):
    """
    Flash crowd simulation — burst of requests with no wait.
    Spawn these suddenly to simulate a traffic spike event.
    Use: increase user count rapidly in Locust UI from 10 -> 100.
    """
    wait_time = between(0.05, 0.2)
    weight = 1

    @task
    def rapid_fire(self):
        prompt = random.choice(SHORT_PROMPTS + MEDIUM_PROMPTS)
        self.client.post(
            "/v1/completions",
            json={
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 1.0,
            },
            name="/v1/completions [spike]",
        )


# ── Load Shape ────────────────────────────────────────────────────
# Custom load shape for automated diurnal + spike pattern
# Use this instead of --users flag for scripted data collection runs

from locust import LoadTestShape

class DiurnalWithSpikes(LoadTestShape):
    """
    Simulates a realistic daily traffic pattern with random spikes.
    
    Timeline (compressed to ~2 hours for data collection):
      0:00 -  0:15  Ramp up (0 -> 20 users)
      0:15 -  0:45  Morning peak (20 users)
      0:45 -  1:00  Midday dip (10 users)
      1:00 -  1:30  Afternoon peak (40 users) - triggers throttling
      1:30 -  1:45  Spike event (80 users for 5 min)
      1:45 -  2:00  Ramp down (80 -> 5 users)

    To use this shape, run WITHOUT --users and --spawn-rate:
      locust -f locustfile.py --host http://localhost:8000 --run-time 2h --headless
    """

    stages = [
        # (duration_seconds, target_users, spawn_rate)
        (900,  20,  2),   # 0:00-0:15  ramp up
        (1800, 20,  5),   # 0:15-0:45  morning peak
        (900,  10,  5),   # 0:45-1:00  midday dip
        (1800, 40,  3),   # 1:00-1:30  afternoon peak (thermal stress)
        (300,  80, 20),   # 1:30-1:35  spike! (flash crowd)
        (600,  40,  5),   # 1:35-1:45  post-spike settle
        (900,   5,  5),   # 1:45-2:00  ramp down
    ]

    def tick(self):
        run_time = self.get_run_time()
        cumulative = 0
        for duration, users, spawn_rate in self.stages:
            cumulative += duration
            if run_time < cumulative:
                return (users, spawn_rate)
        return None  # stop after all stages


# ── Event Hooks ───────────────────────────────────────────────────
# Log important events to a file for later correlation with DCGM data

import csv
from datetime import datetime

_log_file = open("load_events.csv", "w", newline="")
_log_writer = csv.writer(_log_file)
_log_writer.writerow(["timestamp", "event", "user_count", "notes"])

@events.spawning_complete.add_listener
def on_spawning_complete(user_count, **kwargs):
    _log_writer.writerow([
        datetime.utcnow().isoformat(),
        "spawn_complete",
        user_count,
        f"All {user_count} users spawned"
    ])
    _log_file.flush()

@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    _log_file.close()
