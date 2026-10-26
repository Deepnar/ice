#!/usr/bin/env bash
# FINAL private histories (A, D, SW, LAP): Z2's S1-S5 chain, then the maintainer's review.
# Cloud (writers DeepSeek + MiMo, Muse ties/judge) — run after the plan renews, sized from
# measured per-call cost first (TRAPS 148). Every step resumes when rerun.
#   experiments/final/run_private_questions.sh <history>
# After the review packet: the maintainer writes logs/final-questions/<history>/review-decisions.json
# ({"reviewed": true, "drop": [...]}) and then runs
#   uv run --no-sync python experiments/z2/questions.py finalize-final --history <history> --out logs/final-questions
set -euo pipefail
h=$1
cd "$(dirname "$0")/../.."
UV=$(command -v uv)
echo "[$h] S1/S2 ledger $(date +%T)"
"$UV" run --no-sync python experiments/z2/run_ledger.py --out logs/final-ledger --history "$h"
echo "[$h] S1b detail pass $(date +%T)"
"$UV" run --no-sync python experiments/z2/ledger_detail.py --history "$h" --ledger "logs/final-ledger/$h" --out "logs/final-ledger-detail/$h"
experiments/z2/run_questions.sh "$h" logs/final-ledger-detail logs/final-questions
echo "[$h] review packet: logs/final-questions/$h/review-packet.md — waiting for the maintainer's review"
