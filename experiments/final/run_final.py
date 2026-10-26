"""FINAL long-run driver (FINAL_experiments.md §2.1): every FINAL history behind one
resumable command, on Z2's runner (meter, retries, lock, clean stop, resume).

Phases, in order — each stage key starts with its phase:
  prep-*   B's FINAL asks from the sealed families, BEAM export (local); synthetic
           generation (cloud); the private histories' question chain (Z2's S1-S6, run by
           the maintainer with the review; this runner waits for its asks file)
  local-*  per history: replay, read passes (condition preset per dataset), local answers;
           plus the LongMemEval-S slice
  grade-*  cloud judging — not built yet (listed, so the meter shows what is left)

    uv run --no-sync python experiments/final/run_final.py run                    # everything, in order
    uv run --no-sync python experiments/final/run_final.py run --only <keys>      # some stages
    uv run --no-sync python experiments/final/run_final.py status --watch         # the meter
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402
from experiments.z2.histories import BEAM, HISTORIES, SYNTHETIC  # noqa: E402

PRIVATE = ["B", "A", "D", "SW", "LAP"]
PRESET = {"B": "final_main+final_ablation",                                       # D5
          "S9": "final_main+final_ablation+final_buildup"}                         # D5 + D10 (yes, 2026-10-26)
REPLAY_SECONDS = {"B": 35.0, "S3": 120.0}          # per pair; long pastes are slower (Z2's C ran ~160 s)


def safe(key):
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


class FinalReplay(Z.Replay):
    def __init__(self, history, limit=None, repeat=""):
        h = HISTORIES[history]
        super().__init__(history, h.turns, REPLAY_SECONDS.get(history, 35.0), limit)
        self.repeat = repeat
        self.key, self.title = f"local-replay-{history}{repeat}", f"replay {history}{repeat}"

    def database(self, ctx):
        return f"ice_final_{safe(ctx.name)}_{safe(self.history + self.repeat)}"

    def missing(self, ctx):
        path = HISTORIES[self.history].path("originals")
        return [] if path.exists() else [str(Z._rel(path))]


class FinalRead(Z.Stage):
    """Read passes of one history under its condition preset (experiments/z2/read_passes.py)."""

    def __init__(self, history, asks, replay, preset=None):
        repeat = getattr(replay, "repeat", "")
        super().__init__(key=f"local-read-{history}{repeat}", title=f"read passes {history}{repeat}",
                         needs=(replay.key,), unit="contexts", estimate=6.0)
        self.history, self.asks = history, Path(asks)
        self.preset = preset or PRESET.get(history, "final_main")
        self.replay = replay

    def out(self, ctx):
        return ctx.dir / self.key

    def missing(self, ctx):
        return [] if self.asks.exists() else [str(Z._rel(self.asks))]

    def _progress(self, ctx):
        from experiments.z2 import read_passes as R
        if not self.asks.exists():
            return 0, 0
        return R.progress(self.out(ctx), self.asks, R.select_conditions(self.preset))

    def total(self, ctx):
        return self._progress(ctx)[1]

    def done(self, ctx):
        return self._progress(ctx)[0]

    def complete(self, ctx):
        failures = self.out(ctx) / "failures.json"
        total = self.total(ctx)
        return bool(total) and self.done(ctx) >= total and failures.exists() and not json.loads(failures.read_text())

    def run(self, ctx):
        replay = self.replay
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/read_passes.py", "--history", self.history,
                   "--database", replay.database(ctx), "--replay", str(replay.out(ctx).relative_to(ROOT)),
                   "--asks", str(Z._rel(self.asks)), "--out", str(self.out(ctx).relative_to(ROOT)),
                   "--conditions", self.preset]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def answers(history, planned, repeat=""):
    return Z.Answers(history, key=f"local-answers-{history}{repeat}", title=f"answers {history}{repeat}",
                     units_key=f"local-read-{history}{repeat}", needs=(f"local-read-{history}{repeat}",),
                     planned=planned)


class Group(Z.Stage):
    """Several stages shown as one meter line, run in order (the 40 BEAM conversations)."""

    def __init__(self, key, title, members, unit, needs=()):
        super().__init__(key=key, title=title, unit=unit, needs=tuple(needs),
                         estimate=members[0].estimate if members else 0.0)
        self.members = members

    def total(self, ctx):
        return sum(m.total(ctx) for m in self.members)

    def done(self, ctx):
        return sum(m.done(ctx) for m in self.members)

    def complete(self, ctx):
        return all(m.complete(ctx) for m in self.members)

    def missing(self, ctx):
        return [x for m in self.members for x in m.missing(ctx)]

    def rate(self, ctx):
        rates = [r for r in (m.rate(ctx) for m in self.members) if r]
        return sorted(rates)[len(rates) // 2] if rates else None

    def run(self, ctx):
        for m in self.members:
            if m.complete(ctx):
                continue
            outcome = m.run(ctx)
            if outcome != "done":
                return outcome
        return "done"


class Once(Z.Stage):
    """A quick local preparation step, complete when its outputs exist."""

    def __init__(self, key, title, outputs, action, inputs=()):
        super().__init__(key=key, title=title, unit="files")
        self.outputs, self.action, self.inputs = [Path(o) for o in outputs], action, [Path(i) for i in inputs]

    def total(self, ctx):
        return len(self.outputs)

    def done(self, ctx):
        return sum(o.exists() for o in self.outputs)

    def complete(self, ctx):
        return self.done(ctx) == len(self.outputs)

    def missing(self, ctx):
        return [str(Z._rel(i)) for i in self.inputs if not i.exists()]

    def run(self, ctx):
        self.action()
        return "done" if self.complete(ctx) else "stuck"


class WaitFor(Z.Stage):
    """Inputs produced outside this runner (the private question chain + maintainer review)."""

    def __init__(self, key, title, path):
        super().__init__(key=key, title=title, unit="files")
        self.path = Path(path)

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.path.exists())

    def complete(self, ctx):
        return self.path.exists()

    def missing(self, ctx):
        return [] if self.path.exists() else [str(Z._rel(self.path))]

    def run(self, ctx):
        return "done" if self.path.exists() else "stuck"


class SynthGenerate(Z.Stage):
    """Turn-by-turn generation of one synthetic conversation (experiments/synth/generate.py; cloud)."""

    def __init__(self, sid):
        h = HISTORIES[sid]
        super().__init__(key=f"prep-generate-{sid}", title=f"generate {sid}", unit="turns", estimate=20.0)
        self.sid, self.turns, self.out = sid, h.turns, ROOT / f"logs/final-synth/{sid}"

    def total(self, ctx):
        return self.turns

    def done(self, ctx):
        path = self.out / "turns.jsonl"
        return sum(1 for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0

    def complete(self, ctx):
        status = self.out / "status.json"
        return status.exists() and json.loads(status.read_text()).get("finished") is True

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/synth/generate.py", "--plan", self.sid, "--out", str(Z._rel(self.out))]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class LongMemEvalS(Z.Oracle):
    """The pinned 100-instance LongMemEval-S slice (experiments/z2/oracle.py --dataset s)."""

    SLICE = ROOT / "experiments/lme/final_slice.json"

    def __init__(self):
        super().__init__()
        self.key, self.title, self.estimate = "local-lmes", "LongMemEval-S slice (replay + reads)", 3_500.0

    def database(self, ctx):
        return f"ice_final_{safe(ctx.name)}_lmes"

    def total(self, ctx):
        return json.loads(self.SLICE.read_text())["size"]

    def complete(self, ctx):
        return self.done(ctx) >= self.total(ctx)

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/oracle.py", "run", "--dataset", "s", "--slice",
                   str(Z._rel(self.SLICE)), "--database", self.database(ctx), "--out", str(self.out(ctx).relative_to(ROOT))]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class FinalGrade(Z.Stage):
    """Cloud judging of one history's answers (experiments/z2/grade.py, or a dataset's own grader)."""

    def __init__(self, key, title, cmd_args, needs, judge="muse"):
        super().__init__(key=key, title=title, needs=tuple(needs), unit="calls", estimate=3.0)
        self.cmd_args, self.judge = cmd_args, judge

    def out(self, ctx):
        return ctx.dir / self.key

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.complete(ctx))

    def complete(self, ctx):
        status = self.out(ctx) / "status.json"
        return status.exists() and json.loads(status.read_text()).get("finished") is True

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, *self.cmd_args(ctx, self.out(ctx)), "--judge", self.judge]
            rc = ctx.run_child(cmd, log_path)
            if rc == 0 and not (self.out(ctx) / "status.json").exists():
                (self.out(ctx) / "status.json").write_text(json.dumps(dict(finished=True)) + "\n")
            return rc, Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def grade_stage(history, repeat="", judge="muse"):
    """grade.py with the references and ledger each kind of history uses."""
    rel = lambda path: str(Z._rel(path))
    read, ans = f"local-read-{history}{repeat}", f"local-answers-{history}{repeat}"
    if history in SYNTHETIC:
        plans = ROOT / "experiments/synth/plans"
        refs = ["--occurrences", rel(plans / f"{history}.final-occurrences.json"), "--ledger", rel(plans / f"{history}.ledger.json")]
    elif history == "B":
        refs = ["--questions", "logs/z2-questions-final", "--ledger", "logs/z2-ledger-detail-2026-10-18-r2/B/ledger.json",
                "--changes", "logs/z2-questions-final/B/changes.json"]
    else:
        refs = ["--questions", "logs/final-questions", "--ledger", f"logs/final-ledger-detail/{history}/ledger.json",
                "--changes", f"logs/final-questions/{history}/changes.json"]
    args = lambda ctx, out: ["experiments/z2/grade.py", "--history", history, "--units", rel(ctx.dir / read),
                             "--answers", rel(ctx.dir / ans), *refs, "--out", rel(out)]
    return FinalGrade(f"grade-{history}{repeat}", f"grade {history}{repeat}", args, (ans,), judge)


def beam_grade(key, judge="muse"):
    tier, cid = key.split("-")[1], key.split("-")[2]
    rel = lambda path: str(Z._rel(path))
    args = lambda ctx, out: ["experiments/beam/grade.py", "--units", rel(ctx.dir / f"local-read-{key}"),
                             "--answers", rel(ctx.dir / f"local-answers-{key}"),
                             "--asks", f"logs/final-beam/{tier}/{cid}/asks.json", "--out", rel(out)]
    return FinalGrade(f"grade-{key}", f"grade {key}", args, (), judge)


def final_stages(limit=None):
    b_asks = ROOT / "logs/final-questions/B/asks.json"
    z2q = ROOT / "logs/z2-questions-final/B"

    def write_b_asks():
        from experiments.z2.read_passes import build_final_asks
        b_asks.parent.mkdir(parents=True, exist_ok=True)
        b_asks.write_text(json.dumps(build_final_asks(z2q / "split", z2q / "occurrences.json"), indent=1) + "\n")

    def export_beam():
        from experiments.beam.adapter import export
        export(ROOT / "logs/final-beam")

    out = [Once("prep-asks-B", "B's FINAL asks (sealed families)", [b_asks], write_b_asks,
                inputs=[z2q / "split/final-sealed.json", z2q / "occurrences.json"]),
           Once("prep-beam-export", "BEAM export (40 conversations)",
                [HISTORIES[k].path("asks") for k in BEAM], export_beam),
           *[SynthGenerate(s) for s in SYNTHETIC],
           *[WaitFor(f"prep-questions-{k}", f"questions {k} (Z2 S1-S6 chain + review)", HISTORIES[k].path("asks"))
             for k in PRIVATE if k != "B"]]
    for k in PRIVATE + list(SYNTHETIC):
        h = HISTORIES[k]
        asks = b_asks if k == "B" else h.path("asks")
        replay = FinalReplay(k, limit)
        replay.needs = ((f"prep-generate-{k}",) if k in SYNTHETIC else ()) + (("prep-asks-B",) if k == "B" else ())
        out += [replay, FinalRead(k, asks, replay), answers(k, planned=max(1, h.planned_asks or 100) * 6)]
    # Run-to-run variance (Z3 D11): S1 replayed a second time into its own store, read under the
    # main conditions and answered; its grades are compared with the first replay's.
    again = FinalReplay("S1", limit, repeat="-repeat")
    again.needs = ("prep-generate-S1",)
    out += [again, FinalRead("S1", HISTORIES["S1"].path("asks"), again, preset="final_main"),
            answers("S1", planned=600, repeat="-repeat")]
    for tier in ("100K", "500K", "1M"):
        keys = [k for k in BEAM if k.startswith(f"BEAM-{tier}-")]
        replays = [FinalReplay(k, limit) for k in keys]
        rk, dk = f"local-replay-BEAM{tier}", f"local-read-BEAM{tier}"
        out += [Group(rk, f"replay BEAM {tier} ({len(keys)})", replays, "turns", needs=("prep-beam-export",)),
                Group(dk, f"read passes BEAM {tier}",
                      [FinalRead(k, HISTORIES[k].path("asks"), r) for k, r in zip(keys, replays)], "contexts", needs=(rk,)),
                Group(f"local-answers-BEAM{tier}", f"answers BEAM {tier}", [answers(k, planned=120) for k in keys],
                      "answers", needs=(dk,))]
    out += [LongMemEvalS(), Z.Answers("lmes", key="local-answers-lmes", title="answers LongMemEval-S",
                                     units_key="local-lmes", needs=("local-lmes",), planned=600, estimate=60.0)]
    # Cloud judging, last: one judge (Muse) now; a second judge, when chosen, is another pass
    # with --judge <role> into its own stage keys (Z3 D7, benched).
    for k in PRIVATE + list(SYNTHETIC):
        out.append(grade_stage(k))
    out.append(grade_stage("S1", repeat="-repeat"))
    for tier in ("100K", "500K", "1M"):
        keys = [k for k in BEAM if k.startswith(f"BEAM-{tier}-")]
        out.append(Group(f"grade-BEAM{tier}", f"grade BEAM {tier}", [beam_grade(k) for k in keys], "conversations",
                         needs=(f"local-answers-BEAM{tier}",)))
    lmes = lambda ctx, out: ["experiments/z2/oracle.py", "grade", "--dataset", "s",
                             "--units", str(Z._rel(ctx.dir / "local-lmes")), "--answers",
                             str(Z._rel(ctx.dir / "local-answers-lmes")), "--out", str(Z._rel(out))]
    out.append(FinalGrade("grade-lmes", "grade LongMemEval-S", lmes, ("local-answers-lmes",)))
    return out


if __name__ == "__main__":
    raise SystemExit(Z.main(final_stages, run_dir=lambda name: ROOT / "logs" / f"final-run-{name}", prog="run_final.py"))
