"""BEAM trial on one conversation (2026-10-31): the FINAL runner's own stages (BEAM export, 30-turn
replay, read) on its trial run, holding its lock. The full --limit 30 trial replays all 20 128K
conversations (~8 h, measured on LoCoMo's 10); one conversation exercises the same path.

    uv run --no-sync python scripts/oneoff/beam_one_trial.py [BEAM-100K-1]
"""
import fcntl, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.final import run_final as F
from experiments.z2 import run_z2 as Z

def flatten(stages):
    out = []
    for s in stages:
        out += [s] + (flatten(s.members) if isinstance(s, F.Group) else [])
    return out

conv = sys.argv[1] if len(sys.argv) > 1 else "BEAM-100K-1"
ctx = Z.Context("trial", directory=ROOT / "logs" / "final-run-trial")
assert ctx.state.get("limit") == 30, ctx.state.get("limit")
lock = ctx.lock_path.open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
stages = flatten(F.final_stages(30)); ctx.stages = stages
by = {s.key: s for s in stages}
for key in ("prep-beam-export", f"local-replay-{conv}", f"local-read-{conv}"):
    s = by[key]
    if s.complete(ctx):
        print(key, "already complete", flush=True); continue
    print(key, "running", flush=True)
    outcome = s.run(ctx)
    print(key, outcome, s.done(ctx), "/", s.total(ctx), flush=True)
    if outcome != "done":
        raise SystemExit(1)
