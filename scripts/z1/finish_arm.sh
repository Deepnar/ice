#!/bin/sh
# Close out a seeded arm: drain batch summaries (asserted), then snapshot.
#
# ⚑ WHY THESE TWO STEPS AND THIS ORDER.
#  1. `seed_store` calls `batch_summarize()` inside a try/except. A single
#     transient failure there left an arm with ZERO summaries on 2026-08-17,
#     which made `summary_synthesis` (40 probes) unscoreable and was misread for
#     a session as "the summariser is broken". The drain asserts the row count.
#  2. The snapshot happens AFTER the drain, so the snapshot contains the
#     summaries. Snapshot first and the restored store is missing them.
#  3. It ABORTS rather than continuing if either step fails. A snapshot of a
#     store whose drain failed looks identical to a good one months later.
#
# ⚑ AND NOTHING SCORES HERE. Scoring a store while it is still changing
# invalidated a run earlier the same day. Scoring is a separate, later step
# against the snapshot.
#
# Usage:
#   sh scripts/z1/finish_arm.sh ner-a-micro qwen3:4b-instruct
set -u

cd "$(dirname "$0")/../.." || exit 1

ARM="${1:?usage: finish_arm.sh <arm-name> [bg-model]}"
MODEL="${2:-qwen3:4b-instruct}"
LOG="${FINISH_ARM_LOG:-/tmp/ice_finish_${ARM}.log}"
: > "$LOG"

echo "=== FINISH ARM $ARM (bg model $MODEL) ===" | tee -a "$LOG"
date -u +"start %Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"

echo "--- 1/2 batch summaries (asserted, summaries >= 1) ---" | tee -a "$LOG"
# ⚠ --bg-model is not optional: .env pins BACKGROUND_MODEL_NAME to a model no
# arm uses, and this runs in a different process than seed_store's own override.
uv run python scripts/z1/drain_batch_summaries.py \
    --expect-min 1 --bg-model "$MODEL" 2>&1 | tee -a "$LOG"
drain=$?
if [ "$drain" -ne 0 ]; then
  echo "⛔ DRAIN FAILED (exit $drain) — NOT snapshotting, NOT proceeding." \
    | tee -a "$LOG"
  echo "   The store is intact; summary_synthesis is unscoreable until fixed." \
    | tee -a "$LOG"
  exit 1
fi

echo "--- 2/2 snapshot ---" | tee -a "$LOG"
uv run python scripts/z1/snapshot.py save --arm "$ARM" 2>&1 | tee -a "$LOG"
snap=$?
if [ "$snap" -ne 0 ]; then
  echo "⛔ SNAPSHOT FAILED (exit $snap) — the arm is NOT preserved." | tee -a "$LOG"
  exit 1
fi

echo "--- final counts ---" | tee -a "$LOG"
uv run python scripts/z1/snapshot.py counts 2>&1 | tee -a "$LOG"

date -u +"end   %Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
echo "ARM_CLOSED=$ARM" | tee -a "$LOG"
echo
echo "NOT scored — deliberately. Score against the snapshot, in a session that"
echo "is not also writing to the store."
