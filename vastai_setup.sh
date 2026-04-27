#!/usr/bin/env bash
# vastai_setup.sh
# ===============
# Run this on a fresh Vast.ai instance to set up GPUWatch in one shot.
#
# Usage:
#   ssh root@INSTANCE_IP -p PORT
#   curl -sSL https://raw.githubusercontent.com/Arul-sathya/gpuwatch/main/vastai_setup.sh | bash
#
# Or manually:
#   bash vastai_setup.sh

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     GPUWatch — Vast.ai Setup         ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── 1. Verify GPU is visible ──────────────────────────────────────
echo "[1/6] Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── 2. Install Docker if not present ─────────────────────────────
echo "[2/6] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    systemctl start docker
    systemctl enable docker
fi
docker --version

# ── 3. Install nvidia-container-toolkit ──────────────────────────
echo "[3/6] Checking nvidia-container-toolkit..."
if ! nvidia-ctk --version &> /dev/null; then
    echo "Installing nvidia-container-toolkit..."
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
fi
nvidia-ctk --version

# ── 4. Clone repo ─────────────────────────────────────────────────
echo "[4/6] Cloning GPUWatch..."
if [ -d "gpuwatch" ]; then
    echo "Directory exists, pulling latest..."
    cd gpuwatch && git pull
else
    git clone https://github.com/Arul-sathya/gpuwatch.git
    cd gpuwatch
fi

# ── 5. Install Python deps ────────────────────────────────────────
echo "[5/6] Installing Python dependencies..."
pip install -q requests pandas numpy scikit-learn joblib \
    fastapi uvicorn prometheus-client locust \
    matplotlib seaborn

# ── 6. Start stack ────────────────────────────────────────────────
echo "[6/6] Starting GPUWatch stack..."
docker compose pull
docker compose up -d

echo ""
echo "Waiting 30s for services to initialize..."
sleep 30

# ── Verify ────────────────────────────────────────────────────────
echo ""
echo "Running health check..."
python3 verify_stack.py

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  GPUWatch is running!                                ║"
echo "║                                                      ║"
echo "║  Grafana:    http://$(curl -s ifconfig.me):3000      ║"
echo "║  Prometheus: http://$(curl -s ifconfig.me):9090      ║"
echo "║                                                      ║"
echo "║  Next: start load generator                          ║"
echo "║  locust -f locustfile.py --host http://localhost:8000║"
echo "╚══════════════════════════════════════════════════════╝"
