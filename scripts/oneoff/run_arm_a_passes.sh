#!/bin/sh
# Arm A's four measurement passes, chained behind arm B's answer run.
#
# ⚑ THE RESTORE IS ASSERTED, NOT ASSUMED. Every pass below reads the LIVE store.
# If the restore silently fails, all four run against arm B and get written under
# arm A's name — an adjacent system reported under this one's name, which is the
# failure shape this session hit three times. The entity count is the check:
# arm A has 8,470, arm B has 6,271. It aborts rather than measuring.
set -u
cd /home/deepnar/Programs/ice || exit 1
L=logs/arm_a_passes.log; : > "$L"
say() { echo "$@" | tee -a "$L"; }

say "=== waiting for arm B answer run (pid ${1:-none}) ==="
if [ -n "${1:-}" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
fi
say "arm B answer run finished $(date +%H:%M:%S)"

say "=== restoring arm A snapshot ==="
uv run python scripts/z1/snapshot.py restore --arm ner-a-micro >> "$L" 2>&1
rc=$?
[ "$rc" -eq 0 ] || { say "⛔ RESTORE FAILED (exit $rc) — aborting, nothing measured"; exit 1; }

n=$(uv run python -c "
from src.api.db import SessionLocal
from sqlalchemy import text
s=SessionLocal(); print(s.execute(text('select count(*) from codex_entities')).scalar()); s.close()" 2>/dev/null | tail -1)
say "live store entities after restore: $n (arm A must be 8470; arm B is 6271)"
if [ "$n" != "8470" ]; then
  say "⛔ WRONG STORE LOADED — aborting before anything is measured under the wrong name"
  exit 1
fi

say "=== 1/4 typed retrieval score (all 444) ==="
uv run python scripts/z1/score_typed.py --tag ner-a-micro >> "$L" 2>&1
say "score_typed exit=$?"

say "=== 2/4 graph shape ==="
uv run python scripts/oneoff/probe_graph_shape.py --arm "ARM A (ner-a-micro)" >> "$L" 2>&1
say "graph_shape exit=$?"

say "=== 3/4 codex quality judge (n=200, same seed as arm B) ==="
uv run python scripts/z1/judge_codex.py --arm ner-a-micro --n 200 --seed 20260820 >> "$L" 2>&1
say "judge_codex exit=$?"

say "=== 4/4 answer the same 120 probes ==="
uv run python scripts/z1/answer_probes.py \
  --probes experiments/curation_files/typed_probes.stratified120.json \
  --n 120 --tag ner-a-micro --answer-model gemma4:26b-a4b-it-q4_K_M >> "$L" 2>&1
say "answer_probes exit=$?"

say "ARM_A_PASSES_DONE $(date +%H:%M:%S)"
