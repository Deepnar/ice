#!/usr/bin/env bash
# FINAL private histories (A, SW, LAP, ICE2; B reuses Z2's; D dropped 2026-10-28): Z2's S1-S5 chain, then the maintainer's review.
# Cloud, inside Z2's cloud block (Z3 D13): DeepSeek V4.1 Flash writes, MiMo V2.6 Flash checks;
# NO Muse anywhere in the ground truth — a disagreement between the two is not settled by a
# third model: the entry stays unconfirmed (never used for questions) and goes to the review;
# the forward walk is MiMo's. Size it from measured per-call cost first (TRAPS 148). Every
# step resumes when rerun.
#   experiments/final/run_private_questions.sh <history>
# After the review packet: the maintainer writes logs/final-questions/<history>/review-decisions.json
# ({"reviewed": true, "drop": [...]}) and then runs
#   uv run --no-sync python experiments/z2/questions.py finalize-final --history <history> --out logs/final-questions
set -euo pipefail
h=$1
cd "$(dirname "$0")/../.."
UV=$(command -v uv)
export TIE_BREAKER=none VERIFIER=mimo
echo "[$h] S1/S2 ledger $(date +%T)"
"$UV" run --no-sync python experiments/z2/run_ledger.py --out logs/final-ledger --history "$h" --tie-breaker none
echo "[$h] S1b detail pass $(date +%T)"
"$UV" run --no-sync python experiments/z2/ledger_detail.py --history "$h" --ledger "logs/final-ledger/$h" --out "logs/final-ledger-detail/$h" --tie-breaker none
experiments/z2/run_questions.sh "$h" logs/final-ledger-detail logs/final-questions
echo "[$h] review packet: logs/final-questions/$h/review-packet.md — waiting for the maintainer's review"
