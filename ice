#!/usr/bin/env bash
set -e

# ── Load environment variables from .env ─────────────────
if [ -f "$(dirname "$0")/.env" ]; then
    set -a
    source "$(dirname "$0")/.env"
    set +a
fi

# ── Configuration ──────────────────────────────────────
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

# ── Kill any previous instances ─────────────────────────
echo "Stopping old services..."
docker compose -f "$PROJECT_ROOT/docker/docker-compose.yml" down 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true

# ── Start PostgreSQL ─────────────────────────────────────
echo "Starting PostgreSQL..."
docker compose -f "$PROJECT_ROOT/docker/docker-compose.yml" up -d

# ── Background model (C7 shared-first) ───────────────────
# Shared mode (default): background work reuses the main Ollama model — no
# separate server, the proxy's in-process maintenance runtime idle-gates it.
# Dedicated mode is a power-user config; start its server MANUALLY, e.g.:
#   vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8002 --max-model-len 8192 \
#       --gpu-memory-utilization 0.25 --enforce-eager --kv-cache-dtype fp8 \
#       > logs/vllm_bg.log 2>&1 &
BG_MODE="${BACKGROUND_MODEL_MODE:-shared}"
if [ "$BG_MODE" = "dedicated" ]; then
    echo "Dedicated bg mode: expecting a manually-started bg server on :8002 (see comment in this script)."
fi

# ── Start FastAPI proxy (owns background maintenance in-process since C7) ──
echo "Starting ICE proxy..."
nohup uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/proxy.log" 2>&1 &

# ── Tail logs ────────────────────────────────────────────
echo "All services started. Tailing proxy log (Ctrl+C to stop viewing, services keep running)."
tail -f "$LOG_DIR/proxy.log"
