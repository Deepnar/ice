"""FINAL's live trials (maintainer, 2026-10-31): a small run of a real FINAL path on the actual data, to
see it work end to end before the long run — a check of the path, never a result.

  lme    LongMemEval-S: the first 2 questions of the pinned slice, replayed and read by oracle.py
         (database ice_final_trial_lmes, out logs/final-run-trial/lmes-trial)
  beam   BEAM: one 128K conversation's first 30 turns — the FINAL runner's own export, replay and read
         stages on run "trial" (logs/final-run-trial), driven by its own loop. The runner's --only takes
         whole groups, so `run_final.py run --name trial --limit 30` would replay all 20 128K
         conversations (~8 h at the LoCoMo trial's measured pace); one conversation runs the same path.

Both run as stages on the runner (its meter, lock and stop handler; 2026-11-01: one runner for every
run). Ctrl-C stops cleanly and the same command resumes: lme redoes only the question in flight; beam
stops its stage at the next safe point (the runner starts each stage in its own session, so a stop must
come through the runner's handler, never the terminal's signal alone).

    uv run --no-sync python experiments/final/trial.py lme
    uv run --no-sync python experiments/final/trial.py beam [--conversation BEAM-100K-1]
"""

import argparse
import fcntl
import os
import signal
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LIMIT = 30                         # turns per history in run "trial" (its state is pinned to it)
RUN = ROOT / "logs" / "final-run-trial"
LME = ["experiments/z2/oracle.py", "run", "--dataset", "s", "--slice", "experiments/lme/final_slice.json",
       "--limit", "2", "--database", "ice_final_trial_lmes", "--out", "logs/final-run-trial/lmes-trial"]


def members(stages):
    """Every stage, groups and their members alike, by key."""
    from experiments.final import run_final as F
    out = {}
    for s in stages:
        out[s.key] = s
        if isinstance(s, F.Group):
            out.update(members(s.members))
    return out


def lme_stage():
    """The LongMemEval-S trial as a runner stage (the runner's own Oracle stage on 2 slice questions)."""
    from experiments.z2 import run_z2 as Z

    class Trial(Z.Oracle):
        def __init__(self):
            Z.Stage.__init__(self, key="trial-lme", title="LongMemEval-S trial (2 questions)", unit="questions",
                             estimate=4000.0)

        def out(self, ctx):
            return RUN / "lmes-trial"

        def database(self, ctx):
            return "ice_final_trial_lmes"

        def total(self, ctx):
            from experiments.z2 import oracle
            return oracle.progress(self.out(ctx), total=2)[1]

        def extra_args(self, ctx):
            return LME[LME.index("--dataset"):LME.index("--database")]

    return Trial()


def lme():
    return drive([lme_stage()])


def beam(conversation):
    from experiments.final import run_final as F
    by = members(F.final_stages(LIMIT))
    keys = ["prep-beam-export", f"local-replay-{conversation}", f"local-read-{conversation}"]
    if any(k not in by for k in keys):
        print(f"unknown conversation {conversation}; BEAM 128K: "
              f"{sorted(k.removeprefix('local-replay-') for k in by if k.startswith('local-replay-BEAM-100K-'))}")
        return 2
    return drive([by[k] for k in keys])


def drive(stages):
    """Run *stages* in order on run "trial" with the runner's lock, meter and stop handler."""
    from experiments.z2 import run_z2 as Z
    ctx = Z.Context("trial", directory=RUN)
    if ctx.state.setdefault("limit", LIMIT) != LIMIT:
        print(f"run 'trial' is a {ctx.state['limit']}-turn slice, not {LIMIT}")
        return 2
    keys = [s.key for s in stages]
    lock = ctx.lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another runner holds {ctx.lock_path}")
        return 1
    ctx.stages, ctx.owned = list(stages), set(keys)

    def request_stop(signum, _frame):
        if not ctx.stopping:
            ctx.stopping = True
            ctx.log("stop requested", signal=signal.Signals(signum).name)
            print("\nStopping cleanly — the stage finishes its safe point; the same command resumes.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    ctx.log("trial started", only=keys, pid=os.getpid())
    outcomes = Z.drive(ctx, ctx.stages)
    ctx.render(force=True)
    print("\nOutcome: " + ", ".join(f"{k} {v}" for k, v in outcomes.items()))
    return 0 if all(v in ("done", "stopped") for v in outcomes.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("lme", help="LongMemEval-S, 2 questions")
    b = sub.add_parser("beam", help="BEAM, one conversation's first 30 turns")
    b.add_argument("--conversation", default="BEAM-100K-1")
    args = parser.parse_args()
    return lme() if args.command == "lme" else beam(args.conversation)


if __name__ == "__main__":
    raise SystemExit(main())
