#!/bin/sh
# ⚑ CAN THIS PROJECT REPRODUCE ITS OWN RESULT? Nobody has ever checked.
#
# Seeds the SAME turns TWICE with identical configuration and compares what
# came out. If the two disagree, every comparison this project has ever made —
# every arm, every ablation, every model ranking — carries an unknown amount of
# run-to-run difference that was never separated from the effect being claimed.
#
# WHY THIS EXISTS (2026-08-22). A 293-turn re-seed produced 24% fewer triplets
# than the arm it was compared against, on identical inputs. Two 180-turn arms
# with identical config differed by 1.1%; two others were byte-identical. A
# direct three-call test then showed extraction deterministic at temperature 0
# — and an hour earlier the same test had shown it was not. The behaviour is
# intermittent and THE CAUSE IS NOT KNOWN.
#
# ⚠ SO THIS IS A DETECTOR, NOT A FIX. No warm-up call, no seed pinning, no
# workaround — those would be a remedy for a cause nobody has established, and
# writing one before measuring is the exact failure this file exists to catch.
# Measure first. Fix second. In that order.
#
# ⚠ DESTRUCTIVE: --cleans the store twice. Snapshot anything you care about,
# and never run it while a judge or scorer is reading the database.
#
#   sh scripts/z1/check_reproducible.sh          # 20 turns/conversation
#   TURNS=40 sh scripts/z1/check_reproducible.sh
set -u
cd "$(dirname "$0")/../.." || exit 1

TURNS="${TURNS:-20}"
MODEL="${MODEL:-qwen3:4b-instruct}"
TIER="${TIER:-background}"
LOG="logs/reproducibility_check.log"
: > "$LOG"
export CODEX_NODE_PROMOTION=true

echo "reproducibility check: 2 identical seeds, ${TURNS} turns/conversation" | tee -a "$LOG"
echo "model=$MODEL ner_tier=$TIER" | tee -a "$LOG"

for rep in A B; do
  echo "=== PASS $rep ===" | tee -a "$LOG"
  uv run python scripts/z1/seed_store.py --clean >> "$LOG" 2>&1 || {
    echo "FAILED_CLEAN pass=$rep" | tee -a "$LOG"; exit 1; }
  CODEX_EXTRACTION_NER_TIER="$TIER" \
    uv run python scripts/z1/seed_store.py --bg-model "$MODEL" --limit "$TURNS" >> "$LOG" 2>&1
  echo "PASS_${rep}_EXIT=$?" | tee -a "$LOG"
  uv run python scripts/z1/dump_graph_fingerprint.py --label "$rep" >> "$LOG" 2>&1
done

uv run python scripts/z1/dump_graph_fingerprint.py --compare A B | tee -a "$LOG"
echo "REPRODUCIBILITY CHECK DONE" | tee -a "$LOG"
