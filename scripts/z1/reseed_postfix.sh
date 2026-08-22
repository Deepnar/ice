#!/bin/sh
# One arm, arm-B configuration, seeded with TODAY'S extractor fixes.
#
# ⚑ WHY ONE ARM AND NOT TWO. `ner_arm_seed.sh` runs preflight + background to
# answer "which NER". That question is settled (A9b). This run answers a
# different one: **what do the 2026-08-22 extractor fixes change?** So it holds
# arm B's configuration exactly — background NER tier, qwen3:4b-instruct, node
# promotion on — and varies only the code. The comparison is against the
# `ner-b-nuner` snapshot, which differs from this run in nothing else.
#
# The fixes under test:
#   · G51  supersession no longer retires an open-vocabulary predecessor
#          (667 of 817 expiries on the old arm were true facts)
#   · G45  polarity + argument role settled before similarity
#          (blocks 1,611 of 4,012 real merges, led by `is` -> `in` x942)
#   · A8   negation-shaped relations normalised and flagged (646 edges)
#   · G50  merge_key no longer rendered as a fact (was 6,271 of 6,271)
#   · G57  summariser ceiling 300 -> 900 tokens (40% were truncated)
#   · G57  summary coverage scored on prose, not on the model's own index
#
# ⚠ LOG GOES UNDER logs/, NEVER /tmp — TRAPS #40 destroyed an arm's grounding
# record exactly that way, and `ner_arm_seed.sh` still defaults to /tmp.
set -u
cd "$(dirname "$0")/../.." || exit 1

ARM="${ARM:-ner-b-postfix}"
MODEL="${MODEL:-qwen3:4b-instruct}"
TIER="${TIER:-background}"
LOG="logs/reseed_${ARM}.log"
: > "$LOG"

export CODEX_NODE_PROMOTION=true

echo "ARM=$ARM MODEL=$MODEL TIER=$TIER" | tee -a "$LOG"
date -u +"start %Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"

echo "=== clean ===" | tee -a "$LOG"
uv run python scripts/z1/seed_store.py --clean >> "$LOG" 2>&1 || {
  echo "RESEED_FAILED_CLEAN" | tee -a "$LOG"; exit 1; }

echo "=== seeding 293 turns ===" | tee -a "$LOG"
CODEX_EXTRACTION_NER_TIER="$TIER" \
  uv run python scripts/z1/seed_store.py --bg-model "$MODEL" >> "$LOG" 2>&1
seed_exit=$?
echo "SEED_EXIT=$seed_exit" | tee -a "$LOG"

# Asserted, not hoped for: this call was swallowed by a try/except on
# 2026-08-17 and an arm finished with ZERO summaries, which read as
# "the summariser is broken" for three days.
echo "=== batch summaries (asserted) ===" | tee -a "$LOG"
CODEX_EXTRACTION_NER_TIER="$TIER" \
  uv run python scripts/z1/drain_batch_summaries.py --expect-min 1 >> "$LOG" 2>&1
drain_exit=$?
echo "DRAIN_EXIT=$drain_exit" | tee -a "$LOG"
[ "$drain_exit" -ne 0 ] && echo "⚠ NO BATCH SUMMARIES — do NOT score summary_synthesis" | tee -a "$LOG"

echo "=== snapshot ===" | tee -a "$LOG"
uv run python scripts/z1/snapshot.py save --arm "$ARM" >> "$LOG" 2>&1
echo "SNAPSHOT_EXIT=$?" | tee -a "$LOG"

date -u +"end   %Y-%m-%dT%H:%M:%SZ" | tee -a "$LOG"
echo "RESEED_DONE arm=$ARM seed=$seed_exit drain=$drain_exit" | tee -a "$LOG"
