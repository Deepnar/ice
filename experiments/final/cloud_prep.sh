#!/usr/bin/env bash
# All of FINAL's cloud preparation in one go (maintainer, 2026-10-28): the nine synthetic
# conversations (experiments/synth/generate_all.sh) and the private ground truth for A, SW and
# LAP (experiments/final/private_chain.sh), side by side. Neither uses the GPU; both are
# resumable — rerun the same command after a stop or a quota pause.
#
#   experiments/final/cloud_prep.sh
#
# Logs: logs/final-synth/<id>.log and logs/final-questions/chain-<h>.log. Progress of the
# conversations: experiments/synth/generate_all.sh status.
set -uo pipefail
cd "$(dirname "$0")/../.."
# Ctrl-C stops everything this started: background jobs of a script ignore it on their own
trap 'trap - INT TERM; kill 0' INT TERM
experiments/synth/generate_all.sh > logs/cloud-prep-synth.log 2>&1 &
echo "synthetic conversations running — log: logs/cloud-prep-synth.log"
experiments/final/private_chain.sh > logs/cloud-prep-private.log 2>&1 &
echo "private ground truth (A, SW, LAP) running — log: logs/cloud-prep-private.log"
wait
echo "== synthetic"; tail -11 logs/cloud-prep-synth.log
echo "== private"; tail -5 logs/cloud-prep-private.log
