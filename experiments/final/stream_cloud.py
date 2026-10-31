"""FINAL's streaming cloud runner (maintainer, 2026-10-31): answers and grades run beside the local phase.

The local phase is unchanged — each history is replayed once with a memory snapshot at every checkpoint,
and every checkpoint's questions are read from its restored snapshot (LSREP's independent reconstruction).
This runner polls the reads and sends work to the cloud as soon as it exists, per history and per
checkpoint, instead of waiting for the whole local phase:

- **answers**: a checkpoint is answered once every condition has read every one of its questions
  (`read_passes.complete_checkpoints`) — whole checkpoints only, so prompts sharing a cache session (the
  long-context prompts of one checkpoint) are always answered together and reuse the provider's cache.
  A history whose reads are not checkpointed (LongMemEval-S, the memory systems) is answered when its
  read stage is complete.
- **grades**: grade.py runs with --partial on what is answered (at most every GRADE_EVERY seconds, and
  only when new answers arrived); every judge call is durable and keyed by its content, so the final run
  re-asks nothing. The final, non-partial run — the one that writes report.json and marks the stage
  finished — happens once the history's answers and the grade stage's other inputs are complete.
  Other graders (BEAM, LoCoMo, LongMemEval-S) run once, when their inputs are complete.

Separation: every history keeps its own read, answer and grade folders (the FINAL runner's own stage
folders, so its meter shows this progress and its cloud phase later finds the stages done); nothing is
shared between histories, and a checkpoint's answers are never written before that checkpoint is read.
A stage another runner is running is left alone. Ctrl-C stops; the same command resumes.

    uv run --no-sync python experiments/final/stream_cloud.py                   # loop until all answered and graded
    uv run --no-sync python experiments/final/stream_cloud.py --once            # one pass
    uv run --no-sync python experiments/final/run_final.py status --watch       # the meter (same stages)
"""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.final import run_final as F  # noqa: E402
from experiments.z2 import run_z2 as Z  # noqa: E402

POLL = 300                 # seconds between passes
GRADE_EVERY = 1800         # at most one partial grading pass per history in this long


def context(name):
    """This runner's context: its own lock and state, the run's own slice (--limit) from the main state."""
    ctx = Z.Context(name, directory=ROOT / "logs" / f"final-run-{name}", lane="stream")
    main = ctx.dir / "state.json"
    if main.exists() and json.loads(main.read_text()).get("limit"):
        ctx.state["limit"] = json.loads(main.read_text())["limit"]
    return ctx


def flatten(stages):
    out = []
    for s in stages:
        out += flatten(s.members) if isinstance(s, F.Group) else [s]
    return out


def plan(ctx):
    """[(answers stage, its read stage, [grade stages])] for every answered history."""
    stages = flatten(F.final_stages(ctx.state.get("limit")))
    by_key = {s.key: s for s in stages}
    items = []
    for s in stages:
        if isinstance(s, Z.Answers) and s.units_key in by_key:
            name = s.key.removeprefix("cloud-answers-")           # a group's members carry no needs of their own
            grades = [g for g in stages if isinstance(g, F.FinalGrade)
                      and (s.key in g.needs or g.key == f"grade-{name}")]
            items.append((s, by_key[s.units_key], grades))
    return items


def busy_elsewhere(ctx, key):
    """True when another runner of this run is running the stage (never two writers on one folder)."""
    for path in ctx.dir.glob("state*.json"):
        try:
            if json.loads(path.read_text())["stages"].get(key, {}).get("status") == "running":
                return True
        except (ValueError, KeyError, OSError):
            continue
    return False


def ready_checkpoints(ctx, read):
    """None = the whole history is read; a set = these checkpoints are fully read; empty = nothing yet."""
    if read.complete(ctx):
        return None
    if isinstance(read, F.FinalRead):
        from experiments.z2 import read_passes as R
        asks = read.asks_for(ctx)
        if asks.exists() and read.out(ctx).exists():
            return R.complete_checkpoints(read.out(ctx), asks, R.select_conditions(read.preset))
    return set()


def run(cmd, log, say):
    with open(log, "a") as handle:
        handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
        handle.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT).returncode
    say(f"    exit {rc} — log {Z._rel(log)}")
    return rc


def one_pass(ctx, memo, say=print):
    """Answer and grade what is ready. Returns True when every answered history is fully graded."""
    all_done = True
    for answers, read, grades in plan(ctx):
        a_done, a_total = answers.done(ctx), answers.total(ctx)
        a_complete = answers.complete(ctx)
        done_grades = [g.complete(ctx) for g in grades]
        if a_complete and all(done_grades):
            continue
        all_done = False
        ready = ready_checkpoints(ctx, read)
        if not a_complete and ready != set() and not busy_elsewhere(ctx, answers.key):
            key = answers.key
            todo = None if ready is None else sorted(ready - set(memo.get(key, [])), key=str)
            if ready is None or todo:
                say(f"{key}: answering {'all' if ready is None else f'checkpoints {todo}'} ({a_done}/{a_total} so far)")
                cmd = [sys.executable, "experiments/z2/answers.py", "--units", str(Z._rel(answers.units(ctx))),
                       "--out", str(Z._rel(answers.out(ctx))), "--answerer", answers.answerer] \
                    + (["--conditions", ",".join(answers.conditions)] if answers.conditions else []) \
                    + ([] if ready is None else ["--checkpoints", ",".join(sorted(ready, key=str))])
                if run(cmd, ctx.dir / f"{key}.stream.log", say) == 0 and ready is not None:
                    memo[key] = sorted(ready, key=str)
                memo.setdefault("new_answers", []).append(key)
        a_complete = answers.complete(ctx)
        for g, finished in zip(grades, done_grades):
            if finished or busy_elsewhere(ctx, g.key):
                continue
            inputs_done = all(ctx.need_status(n) == "done" or (n == answers.key and a_complete) for n in g.needs)
            cmd = [sys.executable, *g.cmd_args(ctx, g.out(ctx)), "--judge", g.judge]
            if inputs_done and a_complete:
                say(f"{g.key}: final grading")
                if run(cmd, ctx.dir / f"{g.key}.stream.log", say) == 0 and not (g.out(ctx) / "status.json").exists():
                    (g.out(ctx) / "status.json").write_text(json.dumps(dict(finished=True)) + "\n")
            elif cmd[1].endswith("experiments/z2/grade.py") and answers.done(ctx) and \
                    answers.key in memo.get("new_answers", []) and time.time() - memo.get(f"graded:{g.key}", 0) >= GRADE_EVERY:
                say(f"{g.key}: grading what is answered so far")
                run(cmd + ["--partial"], ctx.dir / f"{g.key}.stream.log", say)
                memo[f"graded:{g.key}"] = time.time()
                memo["new_answers"] = [k for k in memo.get("new_answers", []) if k != answers.key]
    return all_done


def save(path, memo):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(memo, indent=1) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", default="main", help="the FINAL run (logs/final-run-<name>)")
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args()
    ctx = context(args.name)
    lock = ctx.lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another streaming runner holds {ctx.lock_path}")
        return 1
    memo_path = ctx.dir / "stream-memo.json"
    memo = json.loads(memo_path.read_text()) if memo_path.exists() else {}
    try:
        while True:
            print(f"--- pass {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
            finished = one_pass(ctx, memo, say=lambda m: print(m, flush=True))
            save(memo_path, memo)
            if finished:
                print("every answered history is answered and graded")
                return 0
            if args.once:
                return 3
            time.sleep(POLL)
    except KeyboardInterrupt:
        save(memo_path, memo)
        print("\nstopped; the same command resumes")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
