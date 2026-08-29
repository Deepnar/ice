#!/usr/bin/env bash
# Start (or resume) a LongMemEval run against ICE v2 -- the frozen system at tag
# `v2-paper-eval`. Safe to kill at any time; re-run the same command to continue.
#
#   ./experiments/lme/run_lme.sh oracle          # the control -- do this first
#   ./experiments/lme/run_lme.sh abstention      # all 30 _abs instances
#   ./experiments/lme/run_lme.sh stratified 60   # 60, spread across question types
#   ./experiments/lme/run_lme.sh oracle --plan   # show the plan, touch nothing
#
# Everything is logged to experiments/lme/runs/v2-paper-eval/<phase>/run.log, so
# you can close the laptop, come back, and read what happened.
set -euo pipefail

WORKTREE="${ICE_V2_WORKTREE:-/home/deepnar/Programs/ice-worktrees/v2-paper-eval}"
HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HARNESS/lme_run.py"

PHASE="${1:-}"
if [[ -z "$PHASE" ]]; then
  echo "usage: $0 {oracle|abstention|stratified} [limit] [--plan]" >&2
  exit 2
fi
shift || true

EXTRA=()
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then EXTRA+=(--limit "$1"); shift || true; fi
if [[ "${1:-}" == "--plan" ]]; then EXTRA+=(--plan-only); shift || true; fi
EXTRA+=("$@")

# ⚑ The run MUST execute from the worktree. v2 stores Vector(384) and embeds with
# truncate_dim=384; `main` uses the same model name at 1024. Running under main's
# environment yields silently wrong vectors -- no crash, no warning. `uv run` from
# the worktree resolves v2's own uv.lock (sentence-transformers 5.5.1, torch
# 2.11.0). The runner also asserts the embedding dim at startup and refuses to
# proceed if it is not 384.
if [[ ! -d "$WORKTREE" ]]; then
  echo "⛔ v2 worktree not found at: $WORKTREE" >&2
  echo "   create it with:" >&2
  echo "   git worktree add -b lme/v2-paper-eval $WORKTREE v2-paper-eval" >&2
  exit 1
fi

if [[ ! -f "$HARNESS/data/longmemeval_s" ]]; then
  echo "⛔ corpus not fetched. run:" >&2
  echo "   uv run python $HARNESS/fetch_dataset.py" >&2
  exit 1
fi

LOG_DIR="$HARNESS/runs/v2-paper-eval/$PHASE"
mkdir -p "$LOG_DIR"

echo "worktree : $WORKTREE"
echo "phase    : $PHASE"
echo "log      : $LOG_DIR/run.log"
echo

cd "$WORKTREE"
# tee so progress is visible live AND survives the terminal being closed.
uv run python "$RUNNER" --phase "$PHASE" "${EXTRA[@]}" 2>&1 | tee -a "$LOG_DIR/run.log"
