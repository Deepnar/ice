#!/usr/bin/env bash
# FINAL's private ground truth for A, SW and LAP (Z3 D13: no Muse in the ground truth) — the Z2
# S1-S5 chain, cloud only: DeepSeek V4.1 Flash and MiMo-V2.6-Flash write and check each other,
# disputes are not changes (--tie-breaker none), the forward walk is verified by MiMo, and the
# embeddings run on the GPU (maintainer, 2026-10-28: on the CPU they lagged the laptop). Resumable: every model
# call is durable, so rerunning the same command continues where it stopped.
#
#   experiments/final/private_chain.sh            # A SW LAP, side by side
#   experiments/final/private_chain.sh SW         # one history
#
# Per history, in order: ledger (both writers, merge) -> logs/final-ledger/<h>/; detail pass over
# long replies -> logs/final-questions/<h>/ledger.json; question chain -> the review packet in
# logs/final-questions/<h>/. Then the maintainer reviews, and
#   uv run --no-sync python experiments/z2/questions.py finalize-final --history <h> --out logs/final-questions
# writes logs/final-questions/<h>/asks.json, which FINAL's runner waits for.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Ctrl-C stops everything this started: background jobs of a script ignore it on their own
trap 'trap - INT TERM; kill 0' INT TERM
export TIE_BREAKER=none VERIFIER=mimo         # embeddings on the GPU (default device): a ~1.2 GB encoder, seconds not minutes
UV=$(command -v uv)
mkdir -p logs/final-ledger logs/final-questions

chain() {
  local h=$1
  # a history whose review packet exists is finished: a restart of the cloud prep re-ran SW from
  # its ledger (2026-10-28), new calls and a new packet over the one awaiting review
  if [ -f "logs/final-questions/$h/review-packet.md" ] && [ -z "${REDO:-}" ]; then
    echo "[$h] already has its review packet — skipped (REDO=1 to run it again)"
    return
  fi
  if pgrep -f -- "--history $h( |$)" >/dev/null; then
    echo "[$h] already running elsewhere — left to it"
    return
  fi
  echo "[$h] ledger start $(date +%T)"
  "$UV" run --no-sync python experiments/z2/run_ledger.py --out logs/final-ledger --history "$h" --tie-breaker none
  echo "[$h] detail pass start $(date +%T)"
  "$UV" run --no-sync python experiments/z2/ledger_detail.py --history "$h" --ledger "logs/final-ledger/$h" \
      --out "logs/final-questions/$h" --tie-breaker none
  echo "[$h] questions start $(date +%T)"
  experiments/z2/run_questions.sh "$h" logs/final-questions logs/final-questions
  echo "[$h] DONE $(date +%T) — review packet in logs/final-questions/$h/"
}

for h in "${@:-A SW LAP}"; do
  for one in $h; do
    chain "$one" >> "logs/final-questions/chain-$one.log" 2>&1 &   # appended: a restart keeps the earlier log
    echo "[$one] running — log: logs/final-questions/chain-$one.log"
  done
done
wait
grep -h "DONE\|Error\|Traceback" logs/final-questions/chain-*.log || true
