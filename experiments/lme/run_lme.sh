#!/usr/bin/env bash
# Start (or resume) a LongMemEval run against ICE v2 -- the frozen system at tag
# `v2-paper-eval`. Safe to kill at any time; re-run the same command to continue.
#
#   experiments/lme/run_lme.sh oracle          # the control -- do this first
#   experiments/lme/run_lme.sh abstention      # all 30 _abs instances
#   experiments/lme/run_lme.sh stratified 60   # 60, spread across question types
#   experiments/lme/run_lme.sh full             # all 500 LongMemEval-S questions
#   experiments/lme/run_lme.sh oracle --plan   # show the plan, touch nothing
#
# Callable by ABSOLUTE PATH from anywhere -- it resolves its own location and cds
# to the worktree itself. You do NOT need to be in the ice directory, and you do
# NOT need a venv activated: `uv run` resolves the worktree's own lockfile.
#
# Scoring is deliberately NOT run here. When a phase finishes:
#   cd /home/deepnar/Programs/ice-worktrees/v2-paper-eval
#   uv run python /home/deepnar/Programs/ice/experiments/lme/score.py --phase <phase>
set -euo pipefail

WORKTREE="${ICE_V2_WORKTREE:-/home/deepnar/Programs/ice-worktrees/v2-paper-eval}"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HARNESS/lme_run.py"
PG_CONTAINER="${ICE_PG_CONTAINER:-ice_postgres}"
OLLAMA="${OLLAMA_HOST_URL:-http://localhost:11434}"
export OLLAMA_HOST_URL="$OLLAMA"

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 {oracle|abstention|stratified|full} [limit] [--plan]" >&2
  exit 2
fi
shift || true

EXTRA=()
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then EXTRA+=(--limit "$1"); shift || true; fi
PLAN_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --plan|--plan-only) PLAN_ONLY=1 ;;
    *) EXTRA+=("$arg") ;;
  esac
done

ANSWER_PROFILE="local-gemma26"
BACKGROUND_PROVIDER="ollama"
for ((arg_i = 0; arg_i < ${#EXTRA[@]}; arg_i++)); do
  if [[ "${EXTRA[$arg_i]}" == "--answer-profile" ]]; then
    ANSWER_PROFILE="${EXTRA[$((arg_i + 1))]:-}"
  elif [[ "${EXTRA[$arg_i]}" == --answer-profile=* ]]; then
    ANSWER_PROFILE="${EXTRA[$arg_i]#--answer-profile=}"
  elif [[ "${EXTRA[$arg_i]}" == "--background-provider" ]]; then
    BACKGROUND_PROVIDER="${EXTRA[$((arg_i + 1))]:-}"
  elif [[ "${EXTRA[$arg_i]}" == --background-provider=* ]]; then
    BACKGROUND_PROVIDER="${EXTRA[$arg_i]#--background-provider=}"
  fi
done

die() { echo "⛔ $*" >&2; exit 1; }

# --- preconditions -----------------------------------------------------------
[[ -d "$WORKTREE" ]] || die "v2 worktree not found at: $WORKTREE
   create it with:
   git worktree add -b lme/v2-paper-eval $WORKTREE v2-paper-eval"

[[ -f "$HARNESS/data/longmemeval_s" ]] || die "corpus not fetched. run:
   cd $WORKTREE && uv run python $HARNESS/fetch_dataset.py"

# --plan needs nothing else -- no database, no GPU, no models. Skip the rest so it
# stays free to run just to see what a phase would cost.
if [[ "$PLAN_ONLY" -eq 0 ]]; then

  # Postgres: start it if it exists but is stopped, rather than making the user do it.
  if ! docker ps --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
    if docker ps -a --format '{{.Names}}' | grep -qx "$PG_CONTAINER"; then
      echo "starting $PG_CONTAINER ..."
      docker start "$PG_CONTAINER" >/dev/null
      for _ in $(seq 1 30); do
        docker exec "$PG_CONTAINER" pg_isready -U ice >/dev/null 2>&1 && break
        sleep 1
      done
    else
      die "postgres container '$PG_CONTAINER' does not exist. bring it up with:
   docker compose -f docker/docker-compose.yml up -d"
    fi
  fi
  docker exec "$PG_CONTAINER" pg_isready -U ice >/dev/null 2>&1 \
    || die "$PG_CONTAINER is running but not accepting connections."

  # The isolated database must exist. Never fall back to ice_db: the v2 schema is
  # 24 migrations behind main's, and the runner TRUNCATES every mapped table.
  docker exec "$PG_CONTAINER" psql -U ice -d ice_lme_v2 -c 'SELECT 1' >/dev/null 2>&1 \
    || die "database ice_lme_v2 is missing or unreachable. create it with:
   docker exec $PG_CONTAINER psql -U ice -d postgres -c 'CREATE DATABASE ice_lme_v2 OWNER ice;'
   cd $WORKTREE && uv run python $HARNESS/setup_v2_db.py"

  REQUIRED_MODELS=()
  [[ "$ANSWER_PROFILE" == "local-gemma26" ]] \
    && REQUIRED_MODELS+=(gemma4:26b-a4b-it-q4_K_M)
  [[ "$BACKGROUND_PROVIDER" == "ollama" ]] \
    && REQUIRED_MODELS+=(qwen3:4b-instruct-bg)

  if (( ${#REQUIRED_MODELS[@]} )); then
    curl -sf --max-time 5 "$OLLAMA/api/tags" >/dev/null \
      || die "Ollama is not responding at $OLLAMA -- start it with: ollama serve"
    INSTALLED="$(curl -sf --max-time 10 "$OLLAMA/api/tags")"
    MISSING=()
    for m in "${REQUIRED_MODELS[@]}"; do
      grep -q "\"$m\"" <<<"$INSTALLED" || MISSING+=("$m")
    done
    if (( ${#MISSING[@]} )); then
      printf '⛔ missing Ollama model(s):\n' >&2
      printf '   %s\n' "${MISSING[@]}" >&2
      printf '   pull with: ollama pull %s\n' "${MISSING[@]}" >&2
      exit 1
    fi
  fi

  if [[ "$BACKGROUND_PROVIDER" == "vllm" ]]; then
    curl -sf --max-time 5 http://127.0.0.1:8002/v1/models >/dev/null \
      || die "vLLM background diagnostic is not responding on :8002. The final
   cloud run uses Ollama. Rejected-path reproduction only:
   $HARNESS/../../scripts/oneoff/lme_vllm_background_rejected.sh"
  fi
fi

# Respect the runner's --out override. Without this, a smoke run writing answers
# under /tmp still appended its launcher output to the real phase's run.log.
RUN_ROOT="$HARNESS/runs/v2-paper-eval"
for ((arg_i = 0; arg_i < ${#EXTRA[@]}; arg_i++)); do
  if [[ "${EXTRA[$arg_i]}" == "--out" && $((arg_i + 1)) -lt ${#EXTRA[@]} ]]; then
    RUN_ROOT="${EXTRA[$((arg_i + 1))]}"
  elif [[ "${EXTRA[$arg_i]}" == --out=* ]]; then
    RUN_ROOT="${EXTRA[$arg_i]#--out=}"
  fi
done

LOG_DIR="$RUN_ROOT/$PHASE"
if [[ "$PLAN_ONLY" -eq 1 ]]; then
  EXTRA+=(--plan-only)
fi

if [[ "$PLAN_ONLY" -eq 0 ]]; then
  mkdir -p "$LOG_DIR"
fi

echo "worktree : $WORKTREE"
echo "phase    : $PHASE"
echo "log      : $LOG_DIR/run.log"
echo

# ⚑ The run MUST execute from the worktree. v2 stores Vector(384) and embeds with
# truncate_dim=384; `main` uses the same model name at 1024. Running under main's
# environment yields silently wrong vectors -- no crash, no warning. `uv run` from
# the worktree resolves v2's own uv.lock. The runner also asserts the embedding
# dimension at startup and refuses to proceed if it is not 384.
cd "$WORKTREE"

# Interactive runs keep a durable clean stdout log. A supervisor already owns a
# journal and must execute Python directly: stopping a `python | tee` control
# group killed tee first and turned the runner's signal handler into a
# BrokenPipeError. Set LME_DIRECT_JOURNAL=1 under systemd.
if [[ "${LME_DIRECT_JOURNAL:-0}" == "1" ]]; then
  exec uv run --no-sync python "$RUNNER" --phase "$PHASE" "${EXTRA[@]}"
fi
if [[ "$PLAN_ONLY" -eq 1 ]]; then
  exec uv run --no-sync python "$RUNNER" --phase "$PHASE" "${EXTRA[@]}"
fi
uv run --no-sync python "$RUNNER" --phase "$PHASE" "${EXTRA[@]}" | tee -a "$LOG_DIR/run.log"
