#!/usr/bin/env bash
# Z2 S3/S4 question chain for one history, then the S5 review packet.
# Waits for that history's detail pass (unit ice-z2-detail-<h>) when one is running.
#   experiments/z2/run_questions.sh <history> [ledger-dir] [out-dir]
# TIE_BREAKER (default muse; none = disagreements are not changes) and VERIFIER (forward
# walk, default muse) — FINAL sets none/mimo: Muse writes no ground truth (Z3 D13).
set -euo pipefail
h=$1
L=${2:-logs/z2-ledger-detail-2026-10-18-r2}
O=${3:-logs/z2-questions-final}
cd "$(dirname "$0")/../.."
UV=$(command -v uv)
TIE_BREAKER=${TIE_BREAKER:-muse}
VERIFIER=${VERIFIER:-muse}
while systemctl --user is-active --quiet "ice-z2-detail-$h"; do sleep 60; done
[ -f "$L/$h/ledger.json" ] || { echo "[$h] no ledger at $L/$h/ledger.json"; exit 1; }
grep -q '"detail_pass"' "$L/$h/ledger.json" || { echo "[$h] ledger has no detail pass yet"; exit 1; }
for step in threads changes pools phrase occurrences; do
  echo "[$h] $step start $(date +%T)"
  extra=(); [ "$step" = changes ] && extra=(--tie-breaker "$TIE_BREAKER")
  "$UV" run --no-sync python experiments/z2/questions.py "$step" --history "$h" --ledger "$L" --out "$O" "${extra[@]}"
  echo "[$h] $step done $(date +%T)"
done
echo "[$h] walk start $(date +%T)"
"$UV" run --no-sync python experiments/z2/walk_audit.py --history "$h" --ledger "$L" --out "$O" --verifier "$VERIFIER"
"$UV" run --no-sync python experiments/z2/questions.py review --history "$h" --ledger "$L" --out "$O"
echo "[$h] review packet written $(date +%T)"
