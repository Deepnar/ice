#!/bin/sh
# ⚑ ONE experiment answering TWO questions (2026-08-22).
#
#   Q1 NOISE FLOOR — how much do two IDENTICAL runs differ? Never measured.
#      The 293-turn re-seed produced 24% fewer triplets than arm B on identical
#      inputs and I could not attribute it. If run-to-run spread is that large,
#      no single-run model comparison is readable — and A12 ran every model
#      exactly ONCE, which would make its ranking partly noise.
#
#   Q2 DIRECTION RULE — does telling the extractor which argument is the
#      subject reduce the ~25-28% reversal rate? Blocking 1,611 canonicalisation
#      merges did not move it, so the reversals are generated, not merged in.
#
# Design: each prompt run TWICE. Within-prompt spread = noise. Between-prompt
# difference = effect. Reading the effect without the floor is what this
# project has been doing and is the reason A12 is unreadable.
#
# Everything else pinned: same 60 turns, same model, same NER tier, promotion
# on. `codex_extraction_direction_rule` is the ONLY variable.
set -u
cd "$(dirname "$0")/../.." || exit 1

MODEL="${MODEL:-qwen3:4b-instruct}"
TIER="${TIER:-background}"
TURNS="${TURNS:-60}"
LOG="logs/prompt_ab_noisefloor.log"
: > "$LOG"
export CODEX_NODE_PROMOTION=true

run() {
  flag="$1"; rep="$2"; arm="dir-${flag}-run${rep}"
  echo "=== ARM $arm (direction_rule=$flag, ${TURNS} turns) ===" | tee -a "$LOG"
  uv run python scripts/z1/seed_store.py --clean >> "$LOG" 2>&1 || {
    echo "ARM_FAILED_CLEAN=$arm" | tee -a "$LOG"; return 1; }
  CODEX_EXTRACTION_DIRECTION_RULE="$flag" CODEX_EXTRACTION_NER_TIER="$TIER" \
    uv run python scripts/z1/seed_store.py --bg-model "$MODEL" --limit "$TURNS" >> "$LOG" 2>&1
  echo "SEED_EXIT_$arm=$?" | tee -a "$LOG"
  uv run python scripts/z1/snapshot.py save --arm "$arm" >> "$LOG" 2>&1
  echo "ARM_DONE=$arm" | tee -a "$LOG"
}

run false 1
run true  1
run false 2
run true  2
echo "ALL ARMS DONE" | tee -a "$LOG"
