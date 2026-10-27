#!/usr/bin/env bash
# All nine FINAL synthetic conversations, side by side, on the cloud (MiMo-V2.6-Flash writes the
# user, DeepSeek V4.1 Flash the assistant; generate.py's defaults). Each writes
# logs/final-synth/<id>/ — turns.jsonl as it goes, originals.json when finished — where FINAL's
# runner looks (its generate stages then count as done). Resumable: rerun the same command.
#
#   experiments/synth/generate_all.sh            # S1..S9
#   experiments/synth/generate_all.sh S4 S9      # some
#   experiments/synth/generate_all.sh status     # turns written, any conversation stopped for repair, cost
#   LIMIT=200 experiments/synth/generate_all.sh  # a stage: stop each conversation after turn 200 (rerun
#                                                # without LIMIT, or with a larger one, to continue)
#
# A turn that never verifies stops its conversation (generate.py exit 4); it is relaunched up to
# 3 times (8 fresh tries each), then left for repair: rerun later, or write
# logs/final-synth/<id>/repair-<turn>.json by hand (see stopped.json for the failed attempt).
set -euo pipefail
cd "$(dirname "$0")/../.."
UV=$(command -v uv)
mkdir -p logs/final-synth

if [ "${1:-}" = status ]; then
  "$UV" run --no-sync python - <<'PY'
import json, glob, os
total = cost = 0
stopped = []
for d in sorted(glob.glob("logs/final-synth/S*/")):
    sid = os.path.basename(d.rstrip("/"))
    rows = [json.loads(l) for l in open(d + "turns.jsonl")] if os.path.exists(d + "turns.jsonl") else []
    usage = [json.loads(l) for l in open(d + "usage.jsonl")] if os.path.exists(d + "usage.jsonl") else []
    c = sum(((u["usage"] or {}).get("opencode") or {}).get("allowance") or 0 for u in usage)
    done = os.path.exists(d + "originals.json")
    stop = json.load(open(d + "stopped.json")) if os.path.exists(d + "stopped.json") else None
    total += len(rows); cost += c
    if stop:
        stopped.append(sid)
    note = " (finished)" if done else f" — STOPPED at turn {stop['turn']}: {stop['problems'][:1]}" if stop else ""
    print(f"{sid}: {len(rows)} turns{note}, ${c:.2f}")
print(f"all: {total} turns, ${cost:.2f}" + (f" — needs repair: {', '.join(stopped)}" if stopped else ""))
PY
  exit 0
fi

one_conversation() {
  local sid=$1 tries=0
  until "$UV" run --no-sync python experiments/synth/generate.py --plan "$sid" --out "logs/final-synth/$sid" \
      ${LIMIT:+--limit "$LIMIT"} >> "logs/final-synth/$sid.log" 2>&1 && [ ! -f "logs/final-synth/$sid/stopped.json" ]; do
    tries=$((tries + 1))
    [ "$tries" -ge 3 ] && { echo "[$sid] stopped 3 times — left for repair (status shows the turn)"; return; }
    echo "[$sid] stopped at a turn — relaunching ($tries/3)"
  done
}

for sid in "${@:-S1 S2 S3 S4 S5 S6 S7 S8 S9}"; do
  for one in $sid; do
    one_conversation "$one" &
    echo "[$one] running — log: logs/final-synth/$one.log"
  done
done
wait
"$0" status
