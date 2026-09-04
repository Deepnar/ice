#!/usr/bin/env bash
# ⛔ DIAGNOSTIC ONLY — NOT USED BY experiments/lme/run_lme_cloud.sh.
#
# Serve the rejected vLLM background candidate on :8002 for reproducing the
# parity failure documented in experiments/lme/results/cloud_stack_calibration.md.
# The final matched run uses the exact Ollama qwen3:4b-instruct-bg instead.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${LME_BG_MODEL:-Qwen/Qwen3-4B-AWQ}"
REVISION="${LME_BG_REVISION:-74d4bd2bd4bff9cafc9345221320bffb08b406a3}"
SERVED_NAME="${LME_BG_SERVED_NAME:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
PORT="${LME_BG_PORT:-8002}"

cd "$REPO"
exec uv run --no-sync vllm serve "$MODEL" \
  --revision "$REVISION" \
  --served-model-name "$SERVED_NAME" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --chat-template "$REPO/scripts/oneoff/qwen3_bg_ollama_chat_template.jinja" \
  --chat-template-content-format string \
  --generation-config vllm \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.45 \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching
