"""FINAL long-run driver (FINAL_experiments.md §2.1; Z3 D13): every FINAL history behind one
resumable command, on Z2's runner (meter, retries, lock, clean stop, resume).

ALL LOCAL FIRST, THEN ALL CLOUD (Z3 D13) — each stage key starts with its phase:
  prep-*    local: synthetic derived files, B's FINAL asks, BEAM export, synthetic generation
            (local models); waits for the private histories' asks (their question chain runs in
            Z2's cloud block, with the maintainer's review)
  local-*   per history: replay, the audit inventory of the final store (B, S9 — before any
            read pass restores a checkpoint into that database), read passes; the
            LongMemEval-S slice; the memory systems' ingestion
  cloud-*   answers (Muse 1.2) for every read context
  audit-*   the background-content audit of B and S9 (Muse 1.3 + DeepSeek second opinion)
  grade-*   Muse 1.3 grading: our histories by the LSREP measures (pre-registration r2 §2), each
            public benchmark by its own norm — BEAM nuggets, LoCoMo Mem0's protocol (J, F1,
            BLEU-1), LongMemEval-S its own judge
  subset-*  the cloud-answerer subset (a non-Meta answerer), graded by Muse 1.3
  crossjudge-*  the cross-family judge (CROSS_JUDGE, picked at the time; waits until set):
            verdicts on the subset contexts' Muse 1.2 answers and non-Meta answers — measures
            the judge's family leniency
  upper-*   the full-history upper bound on BEAM's 500K tier: whole conversations, answered by
            Muse 1.2 within its own 1,048,576-token window
LoCoMo and the three memory systems (Graphiti, Mem0, A-Mem; on LoCoMo only) run through the
same phases.

    uv run --no-sync python experiments/final/run_final.py run                    # everything, in order
    uv run --no-sync python experiments/final/run_final.py run --only <keys>      # some stages
    uv run --no-sync python experiments/final/run_final.py status --watch         # the meter
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.z2 import run_z2 as Z  # noqa: E402
from experiments.z2.histories import BEAM, HISTORIES, LOCOMO, SYNTHETIC  # noqa: E402

PRIVATE = ["B", "A", "SW", "LAP", "ICE2"]  # D dropped 2026-10-28: the v3 hyper-parameter tuning history; ICE2 added 2026-10-31
AUDITED = ("B", "S9")                       # fidelity audit (pre-registration r2 §4)
SUBSET = ("S9", "B", "S5")                  # experiments/final/cloud_subset.py HISTORIES
SUBSET_ANSWERER = "cloud:longcat"           # a non-Meta family; picked at the time (prereg §6)
# The cross-family judge (Z3 D13, revised 2026-10-27): a fresh model of a non-Meta family, picked at
# the time — a local model ("local:<ollama model>") or a cheap cloud role — qualified on the planted
# controls first, and run last. Until it is set here, its stages wait.
CROSS_JUDGE = None
MEMORY_SYSTEMS = ("graphiti", "mem0", "amem")    # on LoCoMo only — the benchmark all three report on
# Which conditions are ANSWERED per kind of data (maintainer, 2026-10-28). recent_only is read
# everywhere — it is the prompt frame bare vector and long context are built in — but on the
# public benchmarks it is not answered: each benchmark runs the way its own field runs it.
BEAM_ANSWERED = ("full_default", "bare_vector", "long_context")
LOCOMO_ANSWERED = ("full_default", "full_b2000", "bare_vector", "long_context",
                   "recent_only:control")     # the no-memory answer, for the controls only (revision 5: NMH)
LMES_ANSWERED = ("full", "bare_vector", "full_context", "all_evidence")    # oracle.py's ICE is `full`
# Pre-registration §6 (maintainer, 2026-10-28): the answerer itself, Muse 1.2, within its own
# 1,048,576-token window — one model answers every condition. Set to None to make the stages wait.
UPPER_BOUND = dict(answerer="muse12", window=1_048_576)
# BEAM after the 2026-10-28 trim (experiments/beam/select.py): 128K tier (20) and 500K tier (5). Full
# history only on the 500K tier: the 128K conversations (≤ 225K tokens) already fit long context's
# 256K window whole, so there long context IS full history.
BEAM_TIERS = tuple(t for t in ("100K", "500K", "1M") if any(k.startswith(f"BEAM-{t}-") for k in BEAM))
UPPER_TIERS = ("500K",)
# The parts test (switch one part off / add parts one by one) on S9 only — B's run dropped 2026-10-28
PRESET = {"S9": "final_main+final_ablation+final_buildup",                        # D5 + D10
          "S4": "final_main+final_decay"}            # decay on the longest history (Z2 F18, Z3 D14)
PRESET.update({k: "final_locomo" for k in LOCOMO})                               # prereg r2 §4
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

    SLICE_ASKS = 10

    def asks_for(self, ctx):
        """The asks to read. In a --limit slice, questions are asked at the end of the history, past
        the replayed turns, so none would be read (the GPU day's LoCoMo trial, 2026-10-30): the first
        SLICE_ASKS questions are asked at the slice's last checkpoint instead, against a store holding
        only turns up to it (a trial of the read path, not a result)."""
        if not ctx.state.get("limit") or not self.asks.exists():
            return self.asks
        cps = sorted(int(p.stem.removeprefix("cp-")) for p in (self.replay.out(ctx) / "checkpoints").glob("cp-*.sql"))
        if not cps:
            return self.asks
        path = self.out(ctx).with_name(self.out(ctx).name + "-slice-asks.json")
        if not path.exists():
            asks = [dict(a, checkpoint_turn=cps[-1], slice_trial=True)
                    for a in json.loads(self.asks.read_text()) if a.get("kind", "question") == "question"][:self.SLICE_ASKS]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asks, indent=1) + "\n")
        return path

    def _progress(self, ctx):
        from experiments.z2 import read_passes as R
        asks = self.asks_for(ctx)
        if not asks.exists():
            return 0, 0
        return R.progress(self.out(ctx), asks, R.select_conditions(self.preset))

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
                   "--asks", str(Z._rel(self.asks_for(ctx))), "--out", str(self.out(ctx).relative_to(ROOT)),
                   "--conditions", self.preset]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def answers(history, planned, repeat="", conditions=None):
    return Z.Answers(history, key=f"cloud-answers-{history}{repeat}", title=f"answers {history}{repeat}",
                     units_key=f"local-read-{history}{repeat}", needs=(f"local-read-{history}{repeat}",),
                     planned=planned, conditions=conditions)


def references(history):
    """grade.py / bg_audit.py arguments naming one history's questions, ledger and changes."""
    rel = lambda path: str(Z._rel(path))
    if history in SYNTHETIC:
        plans = ROOT / "experiments/synth/plans"
        return dict(questions=["--occurrences", rel(plans / f"{history}.final-occurrences.json")],
                    ledger=["--ledger", rel(plans / f"{history}.ledger.json")])
    if history == "B":
        return dict(questions=["--questions", "logs/z2-questions-final"],
                    ledger=["--ledger", "logs/z2-ledger-detail-2026-10-18-r2/B/ledger.json",
                            "--changes", "logs/z2-questions-final/B/changes.json"])
    return dict(questions=["--questions", "logs/final-questions"],
                ledger=["--ledger", f"logs/final-ledger-detail/{history}/ledger.json",
                        "--changes", f"logs/final-questions/{history}/changes.json"])


class Inventory(Z.Stage):
    """The audit inventory of a replay's final store, frozen locally (bg_audit --inventory-only)
    before the read passes restore checkpoints into the same database."""

    def __init__(self, history, replay):
        super().__init__(key=f"local-inventory-{history}", title=f"audit inventory {history}", unit="files",
                         needs=(replay.key,))
        self.history, self.replay = history, replay

    def frozen(self, ctx):
        return ctx.dir / f"audit-{self.history}" / "items.json"

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.frozen(ctx).exists())

    def complete(self, ctx):
        return self.frozen(ctx).exists()

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/bg_audit.py", "--history", self.history,
                   "--database", self.replay.database(ctx), "--replay", str(self.replay.out(ctx).relative_to(ROOT)),
                   *references(self.history)["ledger"], "--out", str(Z._rel(ctx.dir / f"audit-{self.history}")),
                   "--inventory-only"]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class FinalAudit(Z.Audit):
    """Cloud judging of the frozen inventory (bg_audit.py; the database is not read again)."""

    def __init__(self, history, replay):
        super().__init__(history)
        self.key, self.title = f"audit-{history}", f"background audit {history}"
        self.needs, self.replay = (f"local-inventory-{history}",), replay

    def missing(self, ctx):
        return []

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            (self.out(ctx) / "status.json").unlink(missing_ok=True)
            cmd = [sys.executable, "experiments/z2/bg_audit.py", "--history", self.history,
                   "--database", self.replay.database(ctx), "--replay", str(self.replay.out(ctx).relative_to(ROOT)),
                   *references(self.history)["ledger"], "--out", str(self.out(ctx).relative_to(ROOT))]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


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


class ControlCalls(Z.Stage):
    """No-memory controls written by a child process (experiments/final/no_memory_controls.py), metered
    by its cloud calls — one write and one check per control — so the meter moves while it works
    (FINAL revision 5)."""

    def __init__(self, key, title, cmd, outputs, calls_glob, planned, *, inputs=(), needs=()):
        super().__init__(key=key, title=title, unit="calls", estimate=3.0, needs=tuple(needs))
        self.cmd, self.outputs, self.calls_glob, self.planned = cmd, [Path(o) for o in outputs], calls_glob, planned
        self.inputs = [Path(i) for i in inputs]

    def _rows(self):
        rows = []
        for path in ROOT.glob(self.calls_glob):
            rows += [line for line in path.read_text().splitlines() if line.strip()]
        return rows

    def total(self, ctx):
        return self.planned

    def done(self, ctx):
        return self.planned if self.complete(ctx) else min(len(self._rows()), self.planned - 1)

    def rate(self, ctx):
        secs = [json.loads(r).get("seconds") for r in self._rows()[-30:]]
        secs = sorted(s for s in secs if s)
        return secs[len(secs) // 2] if len(secs) >= 5 else None

    def complete(self, ctx):
        return all(o.exists() for o in self.outputs)

    def missing(self, ctx):
        return [str(Z._rel(i)) for i in self.inputs if not i.exists()]

    def run(self, ctx):
        log = ctx.dir / f"{self.key}.log"
        attempt = lambda: (ctx.run_child([sys.executable, *self.cmd], log), Z._tail(log))
        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


def locomo_controls_stage():
    from experiments.final.no_memory_controls import PER
    outputs = [HISTORIES[k].path("asks").with_name("controls.json") for k in LOCOMO]
    return ControlCalls("prep-locomo-controls", f"LoCoMo no-memory controls ({PER} per conversation; cloud)",
                        ["experiments/final/no_memory_controls.py", "locomo"], outputs,
                        "logs/final-locomo/*/controls/*/questions-raw.jsonl", len(LOCOMO) * PER * 2,
                        inputs=[HISTORIES[k].path("asks") for k in LOCOMO])


def synth_controls_stage(sid):
    from experiments.synth.keys import checkpoints
    from experiments.synth.plans import PLANS
    points = len(checkpoints(json.loads((PLANS / f"{sid}.json").read_text())))
    h = HISTORIES[sid]
    return ControlCalls(f"prep-controls-{sid}", f"no-memory controls {sid} (one per checkpoint; cloud)",
                        ["experiments/final/no_memory_controls.py", "synth", "--id", sid],
                        [h.path("asks").with_name("controls.json")],
                        f"logs/final-synth/{sid}/controls/*/questions-raw.jsonl", points * 2,
                        inputs=[h.path("originals")], needs=(f"prep-generate-{sid}",))


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
        super().__init__(key=f"prep-generate-{sid}", title=f"generate {sid}", unit="turns", estimate=20.0,
                         needs=("prep-synth-derive",))
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
                   str(Z._rel(self.SLICE)), "--database", self.database(ctx), "--out", str(self.out(ctx).relative_to(ROOT)),
                   # questions side by side, as Z2's S12 (this stage built its own command and ran one
                   # at a time: ~3,500 s a question, the heaviest single local stage)
                   "--workers", os.environ.get("Z2_S12_WORKERS", "2")]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class FinalGrade(Z.Stage):
    """Cloud judging of one history's answers (experiments/z2/grade.py, or a dataset's own grader)."""

    def __init__(self, key, title, cmd_args, needs, judge="muse", waits=None):
        super().__init__(key=key, title=title, needs=tuple(needs), unit="calls", estimate=3.0)
        self.cmd_args, self.judge, self.waits = cmd_args, judge, waits

    def missing(self, ctx):
        return self.waits() if self.waits else []

    def out(self, ctx):
        return ctx.dir / self.key

    # the grader's own count of its work (meter.json: grade.py's judge calls per phase, or the
    # BEAM / LoCoMo graders' answers); done is what its durable results file holds
    PHASES = ("grade", "claims", "rank", "controls", "semantic", "answers")
    RESULTS = ("grade-raw.jsonl", "beam-grades.jsonl", "locomo-grades.jsonl")

    def total(self, ctx):
        meter = self.out(ctx) / "meter.json"
        if not meter.exists():
            return 1
        m = json.loads(meter.read_text())
        return max(1, sum(m.get(k, 0) for k in self.PHASES))

    def done(self, ctx):
        if self.complete(ctx):
            return self.total(ctx)
        for name in self.RESULTS:
            path = self.out(ctx) / name
            if path.exists():
                return min(sum(1 for line in path.open() if line.strip()), self.total(ctx))
        return 0

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


def grade_stage(history, repeat="", judge="muse", *, units=None, answers_dir=None, key=None, verdicts_only=False,
                needs=None, replay=True, waits=None):
    """grade.py with the references and ledger each kind of history uses; the scoring report
    gets the replay's timings and, for B and S9, the background audit (fidelity)."""
    rel = lambda path: str(Z._rel(path))
    read, ans = units or f"local-read-{history}{repeat}", answers_dir or f"cloud-answers-{history}{repeat}"
    refs = references(history)
    audited = history in AUDITED and not repeat and not verdicts_only

    def args(ctx, out):
        extra = (["--replay", rel(ctx.dir / f"local-replay-{history}{repeat}" / "progress.jsonl")] if replay else []) \
            + ["--no-ranking"]
        if audited:
            extra += ["--audit", rel(ctx.dir / f"audit-{history}" / "bg-audit.json")]
        if verdicts_only:
            extra += ["--verdicts-only"]
        return ["experiments/z2/grade.py", "--history", history, "--units", rel(ctx.dir / read),
                "--answers", rel(ctx.dir / ans), *refs["questions"], *refs["ledger"], *extra, "--out", rel(out)]
    key = key or f"grade-{history}{repeat}"
    return FinalGrade(key, key.replace("-", " "), args, needs or ((ans,) + ((f"audit-{history}",) if audited else ())),
                      judge, waits)


def locomo_grade(key, *, units=None, answers_dir=None, out_key=None, judge="muse"):
    """LoCoMo graded the way memory papers grade it — Mem0's J, F1, BLEU-1 (experiments/locomo/grade.py)
    — plus the no-memory controls against ICE's recent-only replies (revision 5: NMH)."""
    rel = lambda path: str(Z._rel(path))
    ice_units, ice_answers = f"local-read-{key}", f"cloud-answers-{key}"
    units, answers_dir = units or ice_units, answers_dir or ice_answers
    sid = key.removeprefix("LOCOMO-")
    args = lambda ctx, out: ["experiments/locomo/grade.py", "--units", rel(ctx.dir / units),
                             "--answers", rel(ctx.dir / answers_dir),
                             "--occurrences", f"logs/final-locomo/{sid}/occurrences.json", "--out", rel(out),
                             "--no-memory-units", rel(ctx.dir / ice_units), "--no-memory-answers", rel(ctx.dir / ice_answers)]
    out_key = out_key or f"grade-{key}"
    return FinalGrade(out_key, out_key.replace("-", " "), args, (), judge)


def beam_grade(key, judge="muse", *, units=None, answers_dir=None, out_key=None):
    tier, cid = key.split("-")[1], key.split("-")[2]
    rel = lambda path: str(Z._rel(path))
    units, answers_dir = units or f"local-read-{key}", answers_dir or f"cloud-answers-{key}"
    args = lambda ctx, out: ["experiments/beam/grade.py", "--units", rel(ctx.dir / units),
                             "--answers", rel(ctx.dir / answers_dir),
                             "--asks", f"logs/final-beam/{tier}/{cid}/asks.json", "--out", rel(out)]
    out_key = out_key or f"grade-{key}"
    return FinalGrade(out_key, out_key.replace("-", " "), args, (), judge)


class UpperBuild(Z.Stage):
    """Full-history prompts for one BEAM 500K-tier conversation (experiments/final/full_history.py),
    built for the chosen model's window; waits until UPPER_BOUND names it."""

    def __init__(self, key):
        super().__init__(key=f"upper-build-{key}", title=f"full-history prompts {key}", unit="files",
                         needs=(f"local-read-{key}",))
        self.history = key

    def out(self, ctx):
        return ctx.dir / f"upper-read-{self.history}"

    def missing(self, ctx):
        return [] if UPPER_BOUND["answerer"] and UPPER_BOUND["window"] else \
            ["UPPER_BOUND in experiments/final/run_final.py — the ≥ 256K-window model and its window (prereg §6)"]

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.complete(ctx))

    def complete(self, ctx):
        summary = self.out(ctx) / "summary.json"
        return summary.exists() and json.loads(summary.read_text()).get("window") == UPPER_BOUND["window"]

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/final/full_history.py", "--history", self.history,
                   "--read", str(Z._rel(ctx.dir / f"local-read-{self.history}")), "--out", str(Z._rel(self.out(ctx))),
                   "--window", str(UPPER_BOUND["window"])]
            return ctx.run_child(cmd, log_path), Z._tail(log_path)

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class Baseline(Z.Stage):
    """A published memory system, run in its own environment (experiments/final/baselines/
    run_baseline.py) — subclassed per dataset (LoCoMo only since 2026-10-28)."""

    BASE = ROOT / "experiments/final/baselines"

    def out(self, ctx):
        return ctx.dir / self.key

    def run(self, ctx):
        import os
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = ["uv", "run", "--no-sync", "--project", str(self.BASE), "python", str(self.BASE / "run_baseline.py"),
                   self.phase, "--system", self.system, *self.args(ctx), "--out", str(self.out(ctx))]
            os.environ.setdefault("MEM0_TELEMETRY", "False")
            env_before = os.environ.get("UV_PROJECT_ENVIRONMENT")
            os.environ["UV_PROJECT_ENVIRONMENT"] = str(self.BASE / ".venv")
            try:
                return ctx.run_child(cmd, log_path), Z._tail(log_path)
            finally:
                if env_before is None:
                    os.environ.pop("UV_PROJECT_ENVIRONMENT", None)
                else:
                    os.environ["UV_PROJECT_ENVIRONMENT"] = env_before

        return Z.run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class LoCoMoSearch(Baseline):
    """A memory system on one LoCoMo conversation: one store, every question searched. Nothing of
    ICE is read, so it runs before ICE's freeze (maintainer, 2026-10-31); a stop is safe anywhere
    (experiments/final/baselines/run_baseline.py: a half-ingested store is rebuilt, saved hits kept)."""

    phase = "search"

    def __init__(self, system, history):
        Z.Stage.__init__(self, key=f"local-baseline-search-{system}-{history}", title=f"baseline {system} {history} (search)",
                         unit="questions", estimate=60.0)
        self.system, self.history = system, history

    def total(self, ctx):
        return HISTORIES[self.history].planned_asks

    def done(self, ctx):
        return len(list((self.out(ctx) / "instances" / self.history / "hits").glob("*.json")))

    def complete(self, ctx):
        return (self.out(ctx) / "instances" / self.history / "done.json").exists()

    def args(self, ctx):
        return ["--history", self.history]


class LoCoMoBaseline(LoCoMoSearch):
    """The search's saved hits placed in ICE's recent_only envelope for the same question — after
    FINAL's LoCoMo read (the frame the freeze may still change). Minutes; its units are answered."""

    phase = "wrap"

    def __init__(self, system, history):
        Z.Stage.__init__(self, key=f"local-baseline-{system}-{history}", title=f"baseline {system} {history} (wrap)",
                         needs=(f"local-read-{history}", f"local-baseline-search-{system}-{history}"),
                         unit="questions", estimate=1.0)
        self.system, self.history = system, history

    def done(self, ctx):
        return len(list((self.out(ctx) / "units" / self.system).glob("*.json")))

    def args(self, ctx):
        return ["--history", self.history, "--search", str(ctx.dir / f"local-baseline-search-{self.system}-{self.history}"),
                "--ice-units", str(ctx.dir / f"local-read-{self.history}")]


def amem_target():
    import importlib.util
    spec = importlib.util.spec_from_file_location("amem_vendor", ROOT / "experiments/final/baselines/vendor.py")
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    return vendor.TARGET


def fetch_amem():
    import importlib.util
    spec = importlib.util.spec_from_file_location("amem_vendor", ROOT / "experiments/final/baselines/vendor.py")
    vendor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendor)
    vendor.fetch()


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

    def export_locomo():
        from experiments.locomo.adapter import export
        export(ROOT / "logs/final-locomo")

    def derive_synth():
        import subprocess
        subprocess.run([sys.executable, "experiments/synth/derive.py"], check=True, cwd=ROOT)

    from experiments.synth.derive import derived_paths
    out = [Once("prep-synth-derive", "synthetic derived files (regenerated, checked against pins)",
                derived_paths(), derive_synth),
           Once("prep-asks-B", "B's FINAL asks (sealed families)", [b_asks], write_b_asks,
                inputs=[z2q / "split/final-sealed.json", z2q / "occurrences.json"]),
           Once("prep-beam-export", f"BEAM export ({len(BEAM)} conversations)",
                [HISTORIES[k].path("asks") for k in BEAM], export_beam),
           Once("prep-locomo-export", "LoCoMo export (10 conversations)",
                [HISTORIES[k].path("asks") for k in LOCOMO], export_locomo),
           # FINAL revision 5: ~10 no-memory controls per conversation (cloud, minutes) — before the memory
           # systems' search (A-Mem's store lives in memory: a later control would re-ingest it)
           locomo_controls_stage(),
           Once("prep-amem-vendor", "A-Mem's pinned memory layer (AMEM.json)", [amem_target()], fetch_amem),
           *[SynthGenerate(s) for s in SYNTHETIC],
           *[synth_controls_stage(s) for s in SYNTHETIC if s != "S9"],
           *[WaitFor(f"prep-questions-{k}", f"questions {k} (Z2 S1-S6 chain + review)", HISTORIES[k].path("asks"))
             for k in PRIVATE if k != "B"]]
    local, cloud = [], []
    replays = {}
    for k in PRIVATE + list(SYNTHETIC):
        h = HISTORIES[k]
        asks = b_asks if k == "B" else h.path("asks")
        replay = replays[k] = FinalReplay(k, limit)
        replay.needs = ((f"prep-generate-{k}",) if k in SYNTHETIC else ()) + (("prep-asks-B",) if k == "B" else ())
        read = FinalRead(k, asks, replay)
        if k in SYNTHETIC and k != "S9":
            read.needs = tuple(read.needs) + (f"prep-controls-{k}",)
        if k in AUDITED:
            inventory = Inventory(k, replay)
            read.needs = (replay.key, inventory.key) + tuple(n for n in read.needs if n.startswith("prep-"))
            local += [replay, inventory, read]
        else:
            local += [replay, read]
        cloud.append(answers(k, planned=max(1, h.planned_asks or 100) * 6))
    # Run-to-run variance (Z3 D11): S1 replayed a second time into its own store, read under the
    # main conditions and answered; its grades are compared with the first replay's.
    again = FinalReplay("S1", limit, repeat="-repeat")
    again.needs = ("prep-generate-S1",)
    local += [again, FinalRead("S1", HISTORIES["S1"].path("asks"), again, preset="final_main")]
    cloud.append(answers("S1", planned=600, repeat="-repeat"))
    for tier in BEAM_TIERS:
        keys = [k for k in BEAM if k.startswith(f"BEAM-{tier}-")]
        beam_replays = [FinalReplay(k, limit) for k in keys]
        rk, dk = f"local-replay-BEAM{tier}", f"local-read-BEAM{tier}"
        local += [Group(rk, f"replay BEAM {tier} ({len(keys)})", beam_replays, "turns", needs=("prep-beam-export",)),
                  Group(dk, f"read passes BEAM {tier}",
                        [FinalRead(k, HISTORIES[k].path("asks"), r) for k, r in zip(keys, beam_replays)], "contexts",
                        needs=(rk,))]
        cloud.append(Group(f"cloud-answers-BEAM{tier}", f"answers BEAM {tier}",
                           [answers(k, planned=60, conditions=BEAM_ANSWERED) for k in keys], "answers", needs=(dk,)))
    lk = sorted(LOCOMO)
    loco_replays = [FinalReplay(k, limit) for k in lk]
    local += [Group("local-replay-LOCOMO", f"replay LoCoMo ({len(lk)})", loco_replays, "turns", needs=("prep-locomo-export",)),
              Group("local-read-LOCOMO", "read passes LoCoMo",
                    [FinalRead(k, HISTORIES[k].path("asks"), r) for k, r in zip(lk, loco_replays)], "contexts",
                    needs=("local-replay-LOCOMO", "prep-locomo-controls"))]
    cloud.append(Group("cloud-answers-LOCOMO", "answers LoCoMo",
                       [answers(k, planned=HISTORIES[k].planned_asks * len(LOCOMO_ANSWERED), conditions=LOCOMO_ANSWERED)
                        for k in lk], "answers", needs=("local-read-LOCOMO",)))
    for system in MEMORY_SYSTEMS:
        cloud.append(Group(f"cloud-answers-baseline-{system}-LOCOMO", f"answers baseline {system} LoCoMo",
                           [Z.Answers(f"baseline-{system}-{k}", key=f"cloud-answers-baseline-{system}-{k}",
                                      title=f"answers baseline {system} {k}", units_key=f"local-baseline-{system}-{k}",
                                      needs=(f"local-baseline-{system}-{k}",), planned=HISTORIES[k].planned_asks)
                            for k in lk], "answers", needs=(f"local-baseline-{system}-LOCOMO",)))
    local.append(LongMemEvalS())
    # The memory systems' ingest + search: no ICE in it, so it may run early, by name, in the GPU-free
    # weeks before the freeze (`run --only local-baseline-search-…`); last in the local order, so in
    # FINAL's own run an unfinished one resumes after ICE's local stages (maintainer, 2026-10-31).
    local += [Group(f"local-baseline-search-{system}-LOCOMO", f"baseline {system} LoCoMo (ingest + search)",
                    [LoCoMoSearch(system, k) for k in lk], "questions",
                    needs=("prep-locomo-export", "prep-locomo-controls") + (("prep-amem-vendor",) if system == "amem" else ()))
              for system in MEMORY_SYSTEMS]
    local += [Group(f"local-baseline-{system}-LOCOMO", f"baseline {system} LoCoMo (wrap in ICE's frame)",
                    [LoCoMoBaseline(system, k) for k in lk], "questions",
                    needs=("local-read-LOCOMO", f"local-baseline-search-{system}-LOCOMO"))
              for system in MEMORY_SYSTEMS]
    cloud.append(Z.Answers("lmes", key="cloud-answers-lmes", title="answers LongMemEval-S", units_key="local-lmes",
                           needs=("local-lmes",), planned=100 * len(LMES_ANSWERED), conditions=LMES_ANSWERED))

    # ── cloud: answers above, then audits, grading, the subset, the cross-family judge
    cloud += [FinalAudit(k, replays[k]) for k in AUDITED]
    cloud += [grade_stage(k) for k in PRIVATE + list(SYNTHETIC)]
    cloud.append(grade_stage("S1", repeat="-repeat"))
    for tier in BEAM_TIERS:
        keys = [k for k in BEAM if k.startswith(f"BEAM-{tier}-")]
        cloud.append(Group(f"grade-BEAM{tier}", f"grade BEAM {tier}", [beam_grade(k) for k in keys], "conversations",
                           needs=(f"cloud-answers-BEAM{tier}",)))
    cloud.append(Group("grade-LOCOMO", "grade LoCoMo (Mem0's protocol)", [locomo_grade(k) for k in lk], "conversations",
                       needs=("cloud-answers-LOCOMO",)))
    for system in MEMORY_SYSTEMS:
        cloud.append(Group(f"grade-baseline-{system}-LOCOMO", f"grade baseline {system} LoCoMo",
                           [locomo_grade(k, units=f"local-baseline-{system}-{k}", answers_dir=f"cloud-answers-baseline-{system}-{k}",
                                         out_key=f"grade-baseline-{system}-{k}")
                            for k in lk], "conversations", needs=(f"cloud-answers-baseline-{system}-LOCOMO",)))
    upper = sorted(k for k in BEAM if k.split("-")[1] in UPPER_TIERS)
    cloud += [Group("upper-build-BEAM", "full-history prompts BEAM " + " + ".join(UPPER_TIERS),
                    [UpperBuild(k) for k in upper], "files",
                    needs=tuple(f"local-read-BEAM{t}" for t in UPPER_TIERS if t in BEAM_TIERS)),
              Group("upper-answers-BEAM", "full-history answers BEAM " + " + ".join(UPPER_TIERS),
                    [Z.Answers(k, key=f"upper-answers-{k}", title=f"full-history answers {k}", units_key=f"upper-read-{k}",
                               needs=(f"upper-build-{k}",), planned=20,
                               answerer=UPPER_BOUND["answerer"] or "unset-upper-bound-model") for k in upper],
                    "answers", needs=("upper-build-BEAM",)),
              Group("upper-grade-BEAM", "grade full history BEAM " + " + ".join(UPPER_TIERS),
                    [beam_grade(k, units=f"upper-read-{k}", answers_dir=f"upper-answers-{k}", out_key=f"upper-grade-{k}")
                     for k in upper], "conversations", needs=("upper-answers-BEAM",))]
    lmes = lambda ctx, out: ["experiments/z2/oracle.py", "grade", "--dataset", "s",
                             "--units", str(Z._rel(ctx.dir / "local-lmes")), "--answers",
                             str(Z._rel(ctx.dir / "cloud-answers-lmes")), "--out", str(Z._rel(out)), "--upstream-only"]
    cloud.append(FinalGrade("grade-lmes", "grade LongMemEval-S", lmes, ("cloud-answers-lmes",)))
    subset = lambda ctx, out: ["experiments/final/cloud_subset.py", "--run", str(Z._rel(ctx.dir)),
                               "--answerer", SUBSET_ANSWERER]
    cloud.append(FinalGrade("subset", f"cloud-answerer subset ({SUBSET_ANSWERER})", subset,
                            tuple(f"grade-{h}" for h in SUBSET)))
    for h in SUBSET:
        for side, ans in (("main", f"cloud-answers-{h}"), ("subset", f"subset-answers-{h}")):
            cloud.append(grade_stage(h, judge=CROSS_JUDGE or "unset-cross-judge", units=f"subset-read-{h}",
                                     answers_dir=ans, key=f"crossjudge-{h}-{side}", verdicts_only=True,
                                     needs=("subset",), waits=lambda: [] if CROSS_JUDGE else
                                     ["CROSS_JUDGE in experiments/final/run_final.py — the fresh non-Meta judge "
                                      "(local:<model> or a cloud role), qualified first (Z3 D13)"]))
    return out + local + cloud


if __name__ == "__main__":
    raise SystemExit(Z.main(final_stages, run_dir=lambda name: ROOT / "logs" / f"final-run-{name}", prog="run_final.py",
                            description=__doc__.split("\n\n")[0], runs="logs/final-run-<name>"))
