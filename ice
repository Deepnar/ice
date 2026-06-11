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
pkill -f "celery" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
pkill -f "vllm serve" 2>/dev/null || true

# ── Start PostgreSQL + Redis ─────────────────────────────
echo "Starting PostgreSQL and Redis..."
docker compose -f "$PROJECT_ROOT/docker/docker-compose.yml" up -d

# ── Start vLLM background model ──────────────────────────
# Start vLLM background model (only in dedicated mode)
BG_MODE="${BACKGROUND_MODEL_MODE:-dedicated}"
if [ "$BG_MODE" = "dedicated" ]; then
    echo "Starting background model (vllm-bg)..."
    nohup vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ \
        --port 8002 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.25 \
        --enforce-eager \
        --kv-cache-dtype fp8 \
        > "$LOG_DIR/vllm_bg.log" 2>&1 &
else
    echo "Shared mode – skipping background model (will use main LLM)."
fi
# ── Start Celery worker + beat ───────────────────────────
echo "Starting Celery worker and beat..."
nohup uv run celery -A src.workers.celery_app worker -B --loglevel=info \
    > "$LOG_DIR/celery.log" 2>&1 &

# ── Start FastAPI proxy ──────────────────────────────────
echo "Starting ICE proxy..."
nohup uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/proxy.log" 2>&1 &

# ── Tail logs ────────────────────────────────────────────
echo "All services started. Tailing unified log (Ctrl+C to stop viewing, services keep running)."
tail -f "$LOG_DIR/vllm_bg.log" "$LOG_DIR/celery.log" "$LOG_DIR/proxy.log"
