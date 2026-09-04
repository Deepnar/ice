#!/usr/bin/env bash
# Start/resume the matched cloud-answerer + Ollama-background ICE-v2 LME study.
# Generation only; judging remains a separate command after a phase completes.
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHASE="${1:-}"
ANSWER_PROFILE="${LME_CLOUD_ANSWER_PROFILE:-opencode-luna}"
RUN_ROOT="${LME_CLOUD_RUN_ROOT:-$HARNESS/runs/v2-paper-eval-cloud-v1}"

if [[ "$PHASE" != "oracle" && "$PHASE" != "full" ]]; then
  echo "usage: $0 {oracle|full} [runner options]" >&2
  exit 2
fi
shift

echo "answerer : $ANSWER_PROFILE"
echo "background: qwen3:4b-instruct-bg (Ollama, kept resident)"
echo "artifacts : $RUN_ROOT"
echo

export LME_DIRECT_JOURNAL=1

exec "$HARNESS/run_lme.sh" "$PHASE" "$@" \
  --answer-profile "$ANSWER_PROFILE" \
  --background-provider ollama \
  --answer-max-tokens 4096 \
  --out "$RUN_ROOT"
