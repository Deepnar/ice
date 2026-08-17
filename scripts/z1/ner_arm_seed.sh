#!/bin/sh
# Z1 / A9b: seed the same 293 turns twice, varying ONLY the codex-grounding NER.
#
# ⚑ WHAT THIS ANSWERS, AND WHAT IT DOES NOT.
# A9b moved TWO of the four `extract_entities` call sites to the background tier
# (NuNER Zero) on 2026-08-03 — `clustering.py:119` and `turn_density.py:85` —
# and left the codex-extraction call site on the micro-NER. The evidence for
# that split was **ten turns**. This run settles it at 293.
#
#   arm A  codex grounding = micro-NER   (`preflight`, the shipped default)
#   arm B  codex grounding = NuNER Zero  (`background`)
#
# Everything else is identical in both arms, and that is the point:
#   · same background model, qwen3:4b-instruct
#   · clustering and key-term extraction use NuNER in BOTH arms (unchanged)
#   · the pre-flight request path keeps the micro-NER in BOTH arms (A9c, parked
#     — it is synchronous and NuNER's 50–150 ms would land on every request)
# ⇒ This is NOT "NuNER everywhere vs micro-NER everywhere". One slot differs.
#
# ⚠ ARM B TESTS NuNER **AS CONFIGURED** (user decision, 2026-08-17).
# `ner_utils._background_labels()` excludes `concept` and `object` — they are
# junk magnets on conversational text and were dropped for CLUSTERING's sake.
# For codex grounding they are load-bearing (an abstract noun is often the real
# subject of a fact). The exclusion stays for this run so the question has ONE
# variable; if arm B loses, the label list is the next thing to test, not a
# confound to argue about afterwards.
#
# ⚠ READING ARM B: a lower triplet count is NOT the metric. The confirmed-entity
# list is a whitelist, and an ungrounded triplet is written at
# `codex_conf_rejected` (0.35) rather than deleted — see codex_extractor.py:893.
# The micro-NER emits ~52.7 entities a turn to NuNER's ~15.6, so arm B is
# expected to shift confidence DOWN. Whether that costs anything is what the
# probes decide. The extraction shape-rule A/B (`e2c8cca`) was rejected for
# exactly the mistake of reading -16% triplets as a loss on its own.
#
# ── CODEX_NODE_PROMOTION ─────────────────────────────────────────────────────
# Exported true, matching `two_arm_seed.sh`. It is NOT the config default
# (False) and that default is deliberate — promotion merges two entity
# identities and must never run unattended in production. It is set here for
# ONE reason: arm 1 (`fixed-qwen3-4b-instruct`, the "before" snapshot these arms
# are compared against) was seeded with it on. Turning it off would put a second
# variable into the comparison. Per-run only; config.py is untouched.
#
# Run:
#   sh scripts/z1/ner_arm_seed.sh
# ~50 min per arm. Both arms are snapshotted, so either can be re-examined
# without re-seeding.
set -u

cd "$(dirname "$0")/../.." || exit 1
LOG="${NER_ARM_LOG:-/tmp/ice_ner_arm_seed.log}"
: > "$LOG"
echo "log: $LOG"

MODEL="${NER_ARM_MODEL:-qwen3:4b-instruct}"
export CODEX_NODE_PROMOTION=true

run_arm() {
  tier="$1"; arm="$2"
  echo "=== ARM $arm (ner tier=$tier, bg model=$MODEL) — clean ===" | tee -a "$LOG"
  uv run python scripts/z1/seed_store.py --clean >> "$LOG" 2>&1 || {
    echo "ARM_FAILED_CLEAN=$arm" | tee -a "$LOG"; return 1; }

  echo "=== ARM $arm — seeding 293 turns ===" | tee -a "$LOG"
  date -u +"start %Y-%m-%dT%H:%M:%SZ" >> "$LOG"
  CODEX_EXTRACTION_NER_TIER="$tier" \
    uv run python scripts/z1/seed_store.py --bg-model "$MODEL" >> "$LOG" 2>&1
  seed_exit=$?
  echo "SEED_EXIT_$arm=$seed_exit" | tee -a "$LOG"
  date -u +"end   %Y-%m-%dT%H:%M:%SZ" >> "$LOG"

  # ⚑ NOT optional, and not a repeat of what seed_store already did.
  # seed_store calls batch_summarize() inside a try/except; on 2026-08-17 that
  # call failed once, transiently, was swallowed, and the arm finished with
  # ZERO summaries — which made `summary_synthesis` unscoreable and was misread
  # as "the summariser is broken". This asserts the row count instead.
  echo "=== ARM $arm — batch summaries (asserted) ===" | tee -a "$LOG"
  CODEX_EXTRACTION_NER_TIER="$tier" \
    uv run python scripts/z1/drain_batch_summaries.py --expect-min 1 \
      >> "$LOG" 2>&1
  drain_exit=$?
  echo "DRAIN_EXIT_$arm=$drain_exit" | tee -a "$LOG"
  if [ "$drain_exit" -ne 0 ]; then
    echo "⚠ ARM $arm HAS NO BATCH SUMMARIES — do NOT score summary_synthesis" \
      | tee -a "$LOG"
  fi

  echo "=== ARM $arm — snapshot ===" | tee -a "$LOG"
  uv run python scripts/z1/snapshot.py save --arm "$arm" >> "$LOG" 2>&1
  echo "ARM_DONE=$arm seed=$seed_exit drain=$drain_exit" | tee -a "$LOG"
}

run_arm "preflight"  "ner-a-micro"
run_arm "background" "ner-b-nuner"

echo "BOTH_ARMS_DONE" | tee -a "$LOG"
echo
echo "Next: restore an arm, then answer/score/judge it:"
echo "  uv run python scripts/z1/snapshot.py restore --arm ner-a-micro"
echo "  uv run python scripts/z1/answer_probes.py   # model: gemma4:26b-a4b-it-q4_K_M"
echo "  uv run python scripts/z1/score_typed.py --tag ner-a-micro"
