#!/bin/sh
# Z1: seed the FIXED pipeline once per candidate background model.
#
# ⚑ WHY THIS LIVES IN THE REPO. The first version of this driver was written
# into a session scratchpad and referenced from HANDOFF.md. The scratchpad was
# wiped between sessions, so the next session found an instruction pointing at a
# file that no longer existed and had to stop. An experiment driver is part of
# the experiment: it belongs next to the scripts it calls, under version
# control, or the run is not reproducible.
#
# ── THE MODELS, AND WHY THESE TWO ────────────────────────────────────────────
# Top 2 of the 2026-08-12 eight-arm read (PROVENANCE, A12), big model excluded
# by user decision:
#   qwen3:4b-instruct  (2.5 GB) — the record's stated "practical pick": tied
#                                 with e4b on summary quality at a quarter the size
#   gemma4:e4b         (9.6 GB) — top of the raw ranking
# ⚠ `.env` pins BACKGROUND_MODEL_NAME=gemma4:26b-a4b-it-q4_K_M. That pin is
# STALE, not a decision — A12 ranked the 26B third. Each arm passes --bg-model,
# which overrides the setting in-process (seed_store.py sets it before reading
# it into seed_model, and post_flight forwards model_used to both extractors),
# so the pin does not affect this run. It affects everything else.
#
# ── CODEX_NODE_PROMOTION: THE ANSWER TO THE OBVIOUS QUESTION ─────────────────
# `codex_node_promotion` defaults to **False** in config.py, and that default is
# deliberate: promotion merges two entity identities, and "plan" is not always
# "master plan", so it must never happen unattended in production.
#
# This run overrides it to true, and the two facts do not conflict — they are
# different scopes. G44's second half (repairing generic nodes already stored)
# is only exercised if something turns it on, and a seeding run is exactly where
# to do that: the store is disposable and snapshotted, and only ZERO-EDGE stubs
# are eligible, so the merge cannot re-attribute a fact that exists.
#
# **The override is per-run and does not change the default.** If you want the
# arms seeded without it, comment the export out — do not edit config.py.
#
# Run:
#   sh scripts/z1/two_arm_seed.sh
# Roughly 50 minutes per arm. Snapshots let either arm be re-examined later
# without re-seeding.
set -u

cd "$(dirname "$0")/../.." || exit 1
LOG="${TWO_ARM_LOG:-/tmp/ice_two_arm_seed.log}"
: > "$LOG"
echo "log: $LOG"

# See the block above before changing this.
export CODEX_NODE_PROMOTION=true

run_arm() {
  model="$1"; arm="$2"
  echo "=== ARM $arm ($model) — clean ===" | tee -a "$LOG"
  uv run python scripts/z1/seed_store.py --clean >> "$LOG" 2>&1 || {
    echo "ARM_FAILED_CLEAN=$arm" | tee -a "$LOG"; return 1; }

  echo "=== ARM $arm — seeding 293 turns ===" | tee -a "$LOG"
  date -u +"start %Y-%m-%dT%H:%M:%SZ" >> "$LOG"
  uv run python scripts/z1/seed_store.py --bg-model "$model" >> "$LOG" 2>&1
  echo "SEED_EXIT_$arm=$?" >> "$LOG"
  date -u +"end   %Y-%m-%dT%H:%M:%SZ" >> "$LOG"

  echo "=== ARM $arm — snapshot ===" | tee -a "$LOG"
  uv run python scripts/z1/snapshot.py save --arm "$arm" >> "$LOG" 2>&1
  echo "ARM_DONE=$arm" | tee -a "$LOG"
}

run_arm "qwen3:4b-instruct" "fixed-qwen3-4b-instruct"
run_arm "gemma4:e4b"        "fixed-gemma4-e4b"

echo "BOTH_ARMS_DONE" | tee -a "$LOG"
echo
echo "Next: validate the fixes on the new store (HANDOFF step 2), then"
echo "  uv run python scripts/z1/score_typed.py --tag post-reseed"
echo "  uv run python scripts/z1/harvest_probe_context.py --sample 25"
