#!/usr/bin/env bash
# Z1/A12 stage 1: seed the SAME turn subset with each candidate background model,
# then measure the store each one built.
#
# Why a partial SEED per arm rather than direct extractor calls: a seed runs the
# whole post-flight chain (density, grounded summary, chunking, codex,
# procedural) plus clustering and batch summaries, so store_report.py can then
# measure EVERY background job. Calling the two functions that happen to take
# raw text would have covered 2 jobs out of ~12.
#
# --limit 20 is 20 turns PER CONVERSATION, so each arm sees the same 60 turns
# spread across all three — paired, and no single topic decides the ranking.
#
# ⚠ The store is wiped between arms. Arm 1's FULL 293-turn seed is preserved in
# experiments/curation_files/snapshots/gemma4-26b.sql; verify that restores
# before running this, because rebuilding it costs an hour.
set -uo pipefail
cd "$(dirname "$0")/../.."

LIMIT="${LIMIT:-20}"
ARMS=(
  "gemma4:26b-a4b-it-q4_K_M"
  "granite4:micro"
  "granite4:tiny-h"
  "ministral-3:8b"
  "gemma4:e4b"
  "nemotron-mini:4b"
  "qwen3.5:4b"
  "qwen3:4b-instruct"
)

mkdir -p experiments/curation_files/arm_stores
for MODEL in "${ARMS[@]}"; do
  SAFE="${MODEL//[:\/]/_}"
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "ARM: $MODEL   ($(date +%H:%M:%S))"
  echo "════════════════════════════════════════════════════════════"

  uv run python -u scripts/z1/seed_store.py --clean 2>&1 | grep -E "^removed|nothing to clean"

  # Ollama keeps the previous model resident; on a 24 GB card the next one may
  # not fit beside it. Evict explicitly rather than hoping.
  ollama stop "$MODEL" >/dev/null 2>&1 || true

  if ! uv run python -u scripts/z1/seed_store.py --limit "$LIMIT" --bg-model "$MODEL" \
        > "experiments/curation_files/arm_stores/${SAFE}.seed.log" 2>&1; then
    echo "  !! SEED FAILED — see arm_stores/${SAFE}.seed.log"
    tail -3 "experiments/curation_files/arm_stores/${SAFE}.seed.log"
    continue
  fi
  grep -E "seeded .* new turns|store now" "experiments/curation_files/arm_stores/${SAFE}.seed.log" | tail -2

  uv run python -u scripts/z1/store_report.py --arm "$SAFE" --save 2>&1 \
    | sed -n '/derived/,/standalone triplets/p'

  uv run python -u scripts/z1/sample_bg_output.py --n 12 --md --seed 5 >/dev/null 2>&1
  cp experiments/curation_files/BG_QUALITY_SAMPLE.md \
     "experiments/curation_files/arm_stores/${SAFE}.sample.md" 2>/dev/null || true
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "ALL ARMS DONE — reports in experiments/curation_files/store_reports/"
echo "                samples in experiments/curation_files/arm_stores/"
echo "════════════════════════════════════════════════════════════"
