#!/usr/bin/env bash
# All of FINAL's cloud preparation in one go (maintainer, 2026-10-28): the nine synthetic
# conversations (experiments/synth/generate_all.sh) and the private ground truth for A, SW and
# LAP (experiments/final/private_chain.sh), side by side. Neither uses the GPU; both are
# resumable — rerun the same command after a stop or a quota pause.
#
#   experiments/final/cloud_prep.sh
#   experiments/final/cloud_prep.sh status        # what is done / running / stopped / waiting for review
#
# Logs: logs/final-synth/<id>.log and logs/final-questions/chain-<h>.log. Progress of the
# conversations: experiments/synth/generate_all.sh status.
set -uo pipefail
cd "$(dirname "$0")/../.."
# Ctrl-C stops everything this started: background jobs of a script ignore it on their own
trap 'trap - INT TERM; kill 0' INT TERM
if [ "${1:-}" = status ]; then                  # what is done, running, stopped — nothing is started
  uv run --no-sync python - <<'PY'
import json, re
from pathlib import Path
from experiments.synth.plans import PLANS
print("SYNTHETIC CONVERSATIONS (turns written / planned)")
for sid in [f"S{i}" for i in range(1, 10)]:
    d, plan = Path(f"logs/final-synth/{sid}"), json.loads((PLANS / f"{sid}.json").read_text())
    total = sum(s["pairs"] for s in plan["sessions"])
    rows = sum(1 for _ in open(d / "turns.jsonl")) if (d / "turns.jsonl").exists() else 0
    stop = json.loads((d / "stopped.json").read_text()) if (d / "stopped.json").exists() else None
    state = "FINISHED" if (d / "originals.json").exists() else (f"STOPPED at turn {stop['turn']}: {stop['problems'][0][:70]}" if stop else "in progress")
    print(f"  {sid}: {rows:5}/{total:<5} {state}")
print("PRIVATE GROUND TRUTH (A, SW, LAP)")
for h in ("A", "SW", "LAP"):
    q, log = Path(f"logs/final-questions/{h}"), Path(f"logs/final-questions/chain-{h}.log")
    steps = [l for l in log.read_text().splitlines() if re.match(r"\[\w+\] ", l)] if log.exists() else []
    if (q / "asks.json").exists():
        state = "DONE and reviewed: questions ready for FINAL"
    elif (q / "review-packet.md").exists():
        reviewed = json.loads((q / "review-decisions.json").read_text()).get("reviewed")
        state = "packet reviewed, needs finalize-final" if reviewed else "REVIEW PACKET READY: waiting for the maintainer"
    else:
        state = "running: " + (steps[-1].split("] ", 1)[1][:60] if steps else "not started")
    print(f"  {h}: {state}")
PY
  exit 0
fi
experiments/synth/generate_all.sh >> logs/cloud-prep-synth.log 2>&1 &
echo "synthetic conversations running — log: logs/cloud-prep-synth.log"
experiments/final/private_chain.sh >> logs/cloud-prep-private.log 2>&1 &
echo "private ground truth (A, SW, LAP) running — log: logs/cloud-prep-private.log"
wait
echo "== synthetic"; tail -11 logs/cloud-prep-synth.log
echo "== private"; tail -5 logs/cloud-prep-private.log
