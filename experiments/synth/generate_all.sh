#!/usr/bin/env bash
# All nine FINAL synthetic conversations, side by side, on the cloud (MiMo-V2.6-Flash writes the
# user, DeepSeek V4.1 Flash the assistant; generate.py's defaults). Each writes
# logs/final-synth/<id>/ — turns.jsonl as it goes, originals.json when finished — where FINAL's
# runner looks (its generate stages then count as done). Resumable: rerun the same command.
#
#   experiments/synth/generate_all.sh            # S1..S9
#   experiments/synth/generate_all.sh S4 S9      # some
#   experiments/synth/generate_all.sh status     # turns written, unverified so far, cost so far
set -euo pipefail
cd "$(dirname "$0")/../.."
UV=$(command -v uv)
mkdir -p logs/final-synth

if [ "${1:-}" = status ]; then
  "$UV" run --no-sync python - <<'PY'
import json, glob, os
total = cost = bad = 0
for d in sorted(glob.glob("logs/final-synth/S*/")):
    sid = os.path.basename(d.rstrip("/"))
    rows = [json.loads(l) for l in open(d + "turns.jsonl")] if os.path.exists(d + "turns.jsonl") else []
    usage = [json.loads(l) for l in open(d + "usage.jsonl")] if os.path.exists(d + "usage.jsonl") else []
    c = sum(((u["usage"] or {}).get("opencode") or {}).get("allowance") or 0 for u in usage)
    u = sum(not r["verified"] for r in rows)
    done = os.path.exists(d + "originals.json")
    total += len(rows); cost += c; bad += u
    print(f"{sid}: {len(rows)} turns{' (finished)' if done else ''}, unverified {u}, ${c:.2f}")
print(f"all: {total} turns, unverified {bad}, ${cost:.2f}")
PY
  exit 0
fi

for sid in "${@:-S1 S2 S3 S4 S5 S6 S7 S8 S9}"; do
  for one in $sid; do
    "$UV" run --no-sync python experiments/synth/generate.py --plan "$one" --out "logs/final-synth/$one" \
        > "logs/final-synth/$one.log" 2>&1 &
    echo "[$one] running — log: logs/final-synth/$one.log"
  done
done
wait
"$0" status
