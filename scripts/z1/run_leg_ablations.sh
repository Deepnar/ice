#!/bin/sh
# The five per-leg ablations, run in sequence against the live store.
#
# ⚑ RE-RUN PENDING. The 2026-08-22 attempt was STOPPED after one arm, and the
# reason is worth keeping: `legoff-batch_summary` came back with
# episodic_lookup HIGHER than the baseline (0.754 vs 0.634). Turning a leg off
# should not improve a recall score, and that asymmetry exposed a defect in the
# scorer, not in ICE — batch-summary fragments had been stamped with the turns
# they span, and one summary spans up to 33 turns, so returning it credited any
# gold turn inside that range. Fixed in 3037c9e; every arm needs re-running
# against a baseline taken with the corrected metric.
#
# ⇒ The lesson: an ablation is also a test OF THE INSTRUMENT. A leg whose
# removal improves the score is telling you about your metric.
#
# Run AFTER re-scoring a fresh baseline, and use the SAME store for all arms —
# the 2026-08-20 set had no same-commit, same-store control (its only baseline
# was recorded `dirty: True`).
set -u
cd "$(dirname "$0")/../.." || exit 1
LOG="logs/leg_ablations.log"
: > "$LOG"
for leg in batch_summary procedural codex vector rrf; do
  echo "=== ABLATE $leg ===" | tee -a "$LOG"
  uv run python scripts/z1/score_typed.py --tag "legoff-$leg" --ablate-leg "$leg" >> "$LOG" 2>&1
  echo "ABLATION_DONE=$leg exit=$?" | tee -a "$LOG"
done
echo "ALL ABLATIONS DONE" | tee -a "$LOG"
