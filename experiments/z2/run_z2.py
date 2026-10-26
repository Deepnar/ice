"""Z2 long-run driver: S7 → S8 → S9 → S12 replay + reads (all local), then the cloud
stages S10 answers → S12 answers → S11 → S12 grading, in one resumable command — every
local stage first, every cloud stage last (spec D6c; answerer Muse 1.2, D6d).

The maintainer launches the long runs (2026-10-18); this is the one command, and
the same command resumes. Every stage keeps its own durable, resumable units, so a
break, crash, stop or reboot loses at most the unit in flight:

    uv run --no-sync python experiments/z2/run_z2.py run                # everything, in order
    uv run --no-sync python experiments/z2/run_z2.py run --only s7-B    # one stage
    uv run --no-sync python experiments/z2/run_z2.py status --watch     # the meter, from any terminal

Errors never just stop the run:
- before every attempt the prerequisites are checked (Postgres, Ollama, disk); a
  missing one is WAITED for, with the reason on screen, not failed on;
- a stage process that exits badly is resumed after a back-off (1, 2, 5 minutes);
  only three failures in a row WITHOUT ANY PROGRESS mark the stage stuck — its
  error is kept in the run log, the stages that need it are blocked, and every
  independent stage still runs;
- Ctrl-C, closing the terminal or `systemctl stop` is a clean stop: the stage
  process is asked to stop, the state is saved, and the same command continues.

Stages not built yet are listed as such; the run ends there with everything so far
saved. Private outputs live under logs/z2-run-<name>/; replay stores in scratch
databases ice_z2_<name>_<history>.
"""

import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-runner-1"
BACKOFF = (60, 120, 300)            # seconds between resumes of a failed stage process
STRIKES = 3                         # failures in a row without progress → stuck
MIN_FREE_GB = 10
TTY_REFRESH, LOG_REFRESH = 10, 300  # meter refresh: terminal / non-terminal (seconds)


# ── run context ─────────────────────────────────────────────────────────────

class Stop(Exception):
    pass


class Context:
    def __init__(self, name, *, health=None, sleep=time.sleep, out=sys.stdout, directory=None):
        if not re.fullmatch(r"[a-z0-9_]{1,24}", name):
            raise ValueError("run name: 1-24 of a-z 0-9 _")
        self.name = name
        self.dir = Path(directory) if directory else ROOT / "logs" / f"z2-run-{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else \
            dict(policy=POLICY, name=name, stages={}, started=time.time())
        self.stopping = False
        self.child = None
        self._health = health or prerequisites
        self._sleep = sleep
        self.out = out
        self.stages = []
        self._last_render = 0.0

    # state ------------------------------------------------------------------
    def stage_state(self, key):
        return self.state["stages"].setdefault(key, dict(status="pending", failures=[]))

    def save(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.state_path)

    def log(self, message, **fields):
        line = json.dumps(dict(at=time.strftime("%Y-%m-%d %H:%M:%S"), msg=message, **fields), default=str)
        with (self.dir / "runner.log").open("a") as handle:
            handle.write(line + "\n")

    # waiting ------------------------------------------------------------------
    def sleep(self, seconds, why):
        left = seconds
        while left > 0:
            if self.stopping:
                raise Stop()
            self.render(note=f"{why} — retrying in {int(left)} s")
            step = min(5, left)
            self._sleep(step)
            left -= step

    def wait_healthy(self):
        while True:
            problems = self._health()
            if not problems:
                return
            self.log("waiting for prerequisites", problems=problems)
            self.sleep(30, "waiting: " + "; ".join(problems))

    def run_child(self, cmd, log_path):
        """Run one stage process to its end, keeping the meter fresh; Stop on request."""
        with open(log_path, "a") as log:
            log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(map(str, cmd))}\n")
            log.flush()
            self.child = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                          start_new_session=True)   # our signals, not the terminal's
        try:
            while self.child.poll() is None:
                if self.stopping:
                    self._stop_child()
                    raise Stop()
                self.render()
                self._sleep(5)
            return self.child.returncode
        finally:
            self.child = None

    def _stop_child(self, grace=120):
        child = self.child
        child.send_signal(signal.SIGINT)              # the stage stops at its next safe point
        deadline = time.time() + grace
        while child.poll() is None and time.time() < deadline:
            time.sleep(1)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                child.kill()

    # meter ----------------------------------------------------------------
    def render(self, note=None, force=False):
        tty = self.out.isatty()
        now = time.time()
        if not force and now - self._last_render < (TTY_REFRESH if tty else LOG_REFRESH):
            return
        self._last_render = now
        text = meter(self, note)
        self.out.write(("\x1b[H\x1b[J" if tty else "\n") + text + "\n")
        self.out.flush()


def run_with_retries(ctx, key, attempt, done_count, complete, *, backoff=BACKOFF, strikes=STRIKES):
    """Drive one stage to completion through failures. Returns done / stuck / stopped."""
    st = ctx.stage_state(key)
    misses = 0
    while True:
        if complete():
            return "done"
        ctx.wait_healthy()
        before = done_count()
        st["attempts"] = st.get("attempts", 0) + 1
        ctx.save()
        rc, detail = attempt()
        if complete():
            return "done"
        progressed = done_count() > before
        misses = 0 if progressed else misses + 1
        failure = dict(at=time.time(), rc=rc, progressed=progressed, detail=detail)
        st["failures"] = (st["failures"] + [failure])[-20:]
        ctx.log("stage process ended without finishing", stage=key, failure=failure)
        ctx.save()
        if misses >= strikes:
            return "stuck"
        ctx.sleep(backoff[min(misses, len(backoff) - 1)] if misses else backoff[0],
                  f"{key} ended (exit {rc}); resuming")


# ── stages ──────────────────────────────────────────────────────────────────

@dataclass
class Stage:
    key: str
    title: str
    needs: tuple = ()
    built: bool = True
    unit: str = "units"
    estimate: float = 0.0           # planning seconds per unit until measured
    notes: list = field(default_factory=list)

    def total(self, ctx):
        return 0

    def done(self, ctx):
        return 0

    def rate(self, ctx):
        return None                 # measured seconds per unit, if any

    def complete(self, ctx):
        return False

    def missing(self, ctx):
        return []                   # input files this stage needs that do not exist yet

    def run(self, ctx):
        raise NotImplementedError


class Replay(Stage):
    """S7: the semi-LSREP replay of one history (experiments/z2/replay.py)."""

    def __init__(self, history, turns, estimate, limit=None):
        super().__init__(key=f"s7-{history}", title=f"S7 replay {history}", unit="turns", estimate=estimate)
        self.history, self.turns, self.limit = history, turns, limit

    def out(self, ctx):
        return ctx.dir / self.key

    def database(self, ctx):
        return f"ice_z2_{ctx.name}_{self.history.lower()}"

    def total(self, ctx):
        return min(self.turns, self.limit) if self.limit else self.turns

    def _progress(self, ctx):
        path = self.out(ctx) / "progress.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def done(self, ctx):
        turns = [p["turn"] for p in self._progress(ctx) if p.get("turn")]
        return min(max(turns, default=0), self.total(ctx))

    def rate(self, ctx):
        walls = [p["wall"] for p in self._progress(ctx)][-31:]
        gaps = sorted(b - a for a, b in zip(walls, walls[1:]) if 0 < b - a < 3600)  # skip restarts
        return gaps[len(gaps) // 2] if len(gaps) >= 3 else None

    def complete(self, ctx):
        seed = self.out(ctx) / "seed.jsonl"
        if not seed.exists():
            return False
        with seed.open("rb") as handle:                 # the "complete" record is the last line
            handle.seek(max(0, seed.stat().st_size - 1_000_000))
            return b'"event": "complete"' in handle.read()

    def run(self, ctx):
        out, db = self.out(ctx), self.database(ctx)
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            fresh = not (out / "seed.jsonl").exists()
            if fresh:
                self._clear_aborted_start(ctx, out, db)
            cmd = [sys.executable, "experiments/z2/replay.py", "--history", self.history,
                   "--out", str(out.relative_to(ROOT)), "--database", db,
                   "--create-db" if fresh else "--resume"]
            if self.limit:
                cmd += ["--limit", str(self.limit)]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))

    def _clear_aborted_start(self, ctx, out, db):
        """A start that died before its first durable checkpoint: set the empty
        database and directory aside so the fresh start can run. Anything holding
        data is refused, never removed."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url
        from src.api.config import settings
        from src.memory.models import Base
        source = make_url(settings.database_url)
        admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
        with admin.connect() as conn:
            exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": db}).scalar()
        cleared = bool(exists) or out.exists()
        if exists:
            engine = create_engine(source.set(database=db))
            with engine.connect() as conn:
                rows = sum(conn.execute(text(f'SELECT count(*) FROM "{t}"')).scalar()
                           for t in Base.metadata.tables)
            engine.dispose()
            if rows:
                raise RuntimeError(f"{db} holds data but {out}/seed.jsonl is missing; inspect by hand")
            with admin.connect() as conn:
                conn.execute(text(f'DROP DATABASE "{db}"'))
        admin.dispose()
        if out.exists():
            if list(out.glob("checkpoints/*.sql")):
                raise RuntimeError(f"{out} holds checkpoints but no seed.jsonl; inspect by hand")
            out.rename(out.with_name(f"{out.name}.aborted-{int(time.time())}"))
        if cleared:
            ctx.log("cleared an aborted start", stage=self.key, database=db)


# Inputs fixed once the question chain has run on the final ledgers (S3/S4). A stage
# whose inputs are missing waits as "needs input", it is not attempted.
INPUTS = {
    "ledger": {h: ROOT / f"logs/z2-ledger-detail-2026-10-18-r2/{h}/ledger.json" for h in ("B", "C")},
    "changes": {h: ROOT / f"logs/z2-questions-final/{h}/changes.json" for h in ("B", "C")},
    "asks": {h: ROOT / f"logs/z2-questions-final/{h}/z2-asks.json" for h in ("B", "C")},
}


class Audit(Stage):
    """S8: background-content audit of one history's final store (experiments/z2/bg_audit.py)."""

    def __init__(self, history):
        super().__init__(key=f"s8-{history}", title=f"S8 background audit {history}", needs=(f"s7-{history}",),
                         unit="judgments", estimate=2.5)   # trial: ~45 calls in 107 s, 4 in flight
        self.history = history

    def out(self, ctx):
        return ctx.dir / self.key

    def missing(self, ctx):
        return [str(_rel(INPUTS[k][self.history])) for k in ("ledger", "changes")
                if not INPUTS[k][self.history].exists()]

    def total(self, ctx):
        from experiments.z2 import bg_audit
        # Before the inventory exists: ~11 kinds x 100 + 20% re-judged + 100 fact
        # matches + 100 recall checks (planning number, replaced once inventoried).
        return bg_audit.progress(self.out(ctx))[1] or 1_420

    def done(self, ctx):
        from experiments.z2 import bg_audit
        return bg_audit.progress(self.out(ctx))[0]

    def rate(self, ctx):
        from experiments.z2 import bg_audit
        per_call = bg_audit.call_seconds(self.out(ctx))
        return per_call / bg_audit.WORKERS if per_call else None     # calls run in parallel

    def complete(self, ctx):
        status = self.out(ctx) / "status.json"
        return status.exists() and json.loads(status.read_text()).get("finished") is True

    def run(self, ctx):
        replay = next(s for s in ctx.stages if s.key == f"s7-{self.history}")
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            (self.out(ctx) / "status.json").unlink(missing_ok=True)
            cmd = [sys.executable, "experiments/z2/bg_audit.py", "--history", self.history,
                   "--database", replay.database(ctx), "--replay", str(replay.out(ctx).relative_to(ROOT)),
                   "--ledger", str(INPUTS["ledger"][self.history].relative_to(ROOT)),
                   "--changes", str(INPUTS["changes"][self.history].relative_to(ROOT)),
                   "--out", str(self.out(ctx).relative_to(ROOT))]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class ReadPasses(Stage):
    """S9: every Z2 question read under every condition on its checkpoint store
    (experiments/z2/read_passes.py)."""

    def __init__(self, history):
        super().__init__(key=f"s9-{history}", title=f"S9 read passes {history}", needs=(f"s8-{history}",),
                         unit="contexts", estimate=6.0)
        self.history = history

    def out(self, ctx):
        return ctx.dir / self.key

    def missing(self, ctx):
        path = INPUTS["asks"][self.history]
        return [] if path.exists() else [str(_rel(path))]

    def total(self, ctx):
        from experiments.z2 import read_passes
        # Planning number until the asks exist: ~300/~120 Z2 occurrences plus one
        # control per checkpoint (45/13), times the conditions.
        planned = {"B": 345, "C": 133}[self.history] * len(read_passes.CONDITIONS)
        return read_passes.progress(self.out(ctx), INPUTS["asks"][self.history])[1] or planned

    def done(self, ctx):
        from experiments.z2 import read_passes
        return read_passes.progress(self.out(ctx), INPUTS["asks"][self.history])[0]

    def rate(self, ctx):
        stamps = sorted(p.stat().st_mtime for p in (self.out(ctx) / "units").glob("*/*.json"))[-60:]
        gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if 0 <= b - a < 600)
        return gaps[len(gaps) // 2] if len(gaps) >= 5 else None

    def complete(self, ctx):
        failures = self.out(ctx) / "failures.json"
        total = self.total(ctx)
        return bool(total) and self.done(ctx) >= total and failures.exists() \
            and not json.loads(failures.read_text())

    def run(self, ctx):
        replay = next(s for s in ctx.stages if s.key == f"s7-{self.history}")
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/read_passes.py", "--history", self.history,
                   "--database", replay.database(ctx), "--replay", str(replay.out(ctx).relative_to(ROOT)),
                   "--asks", str(INPUTS["asks"][self.history].relative_to(ROOT)),
                   "--out", str(self.out(ctx).relative_to(ROOT))]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


ANSWERER = "muse12"     # spec D6d


class Answers(Stage):
    """S10: the answerer (spec D6d: Muse 1.2, cloud) answers every S9 context, one per
    distinct prompt (experiments/z2/answers.py)."""

    def __init__(self, history, *, key=None, title=None, units_key=None, needs=None, planned=None,
                 estimate=10.0):
        # Muse 1.2, four workers: a planning guess until the stage measures its own rate.
        super().__init__(key=key or f"s10-{history}", title=title or f"S10 answers {history}",
                         needs=needs or (f"s9-{history}",), unit="answers", estimate=estimate)
        self.history = history
        self.units_key = units_key or f"s9-{history}"
        self.planned = planned

    def out(self, ctx):
        return ctx.dir / self.key

    def units(self, ctx):
        return ctx.dir / self.units_key

    def total(self, ctx):
        from experiments.z2 import answers
        # ~35 read conditions per question; identical prompts collapse to ~20 distinct
        planned = self.planned or {"B": 345, "C": 133}[self.history] * 20
        return answers.progress(self.units(ctx), self.out(ctx))[1] or planned

    def done(self, ctx):
        from experiments.z2 import answers
        return answers.progress(self.units(ctx), self.out(ctx))[0]

    def rate(self, ctx):
        path = self.out(ctx) / "answers.jsonl"
        if not path.exists():
            return None
        walls = [json.loads(line)["wall"] for line in path.read_text().splitlines()[-61:] if line.strip()]
        gaps = sorted(b - a for a, b in zip(walls, walls[1:]) if 0 <= b - a < 3600)
        return gaps[len(gaps) // 2] if len(gaps) >= 5 else None

    def complete(self, ctx):
        status = self.out(ctx) / "status.json"
        return status.exists() and json.loads(status.read_text()).get("finished") is True

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            (self.out(ctx) / "status.json").unlink(missing_ok=True)
            cmd = [sys.executable, "experiments/z2/answers.py", "--units", str(self.units(ctx).relative_to(ROOT)),
                   "--out", str(self.out(ctx).relative_to(ROOT)), "--answerer", ANSWERER]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class Grading(Stage):
    """S11: Muse grades, claim checks, ranking, controls, then the report (grade.py, report.py)."""

    def __init__(self, history):
        super().__init__(key=f"s11-{history}", title=f"S11 grading + report {history}", needs=(f"s10-{history}",),
                         unit="calls", estimate=3.0)
        self.history = history

    def out(self, ctx):
        return ctx.dir / self.key

    def total(self, ctx):
        answers = ctx.dir / f"s10-{self.history}" / "status.json"
        n = json.loads(answers.read_text()).get("contexts") if answers.exists() else {"B": 345, "C": 133}[self.history] * 6
        return int(n * 1.4)                       # grades + ranking (2 orders) + claim checks, roughly

    def done(self, ctx):
        path = self.out(ctx) / "grade-raw.jsonl"
        if not path.exists():
            return 0
        with path.open() as handle:
            return min(sum(1 for _ in handle), self.total(ctx))

    def rate(self, ctx):
        from experiments.z2 import bg_audit
        path = self.out(ctx) / "grade-raw.jsonl"
        if not path.exists():
            return None
        seconds = sorted(json.loads(line).get("seconds") or 0 for line in path.read_text().splitlines()[-50:])
        seconds = [x for x in seconds if x]
        return seconds[len(seconds) // 2] / bg_audit.WORKERS if len(seconds) >= 3 else None

    def complete(self, ctx):
        status = self.out(ctx) / "status.json"
        return status.exists() and json.loads(status.read_text()).get("finished") is True

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            (self.out(ctx) / "status.json").unlink(missing_ok=True)
            cmd = [sys.executable, "experiments/z2/grade.py", "--history", self.history,
                   "--units", str((ctx.dir / f"s9-{self.history}").relative_to(ROOT)),
                   "--answers", str((ctx.dir / f"s10-{self.history}").relative_to(ROOT)),
                   "--questions", str(INPUTS["asks"][self.history].parent.parent.relative_to(ROOT)),
                   "--ledger", str(INPUTS["ledger"][self.history].relative_to(ROOT)),
                   "--changes", str(INPUTS["changes"][self.history].relative_to(ROOT)),
                   "--audit", str((ctx.dir / f"s8-{self.history}" / "bg-audit.json").relative_to(ROOT)),
                   "--replay", str((ctx.dir / f"s7-{self.history}" / "progress.jsonl").relative_to(ROOT)),
                   "--out", str(self.out(ctx).relative_to(ROOT))]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class Oracle(Stage):
    """S12: every LongMemEval oracle question replayed in isolation and read under five
    conditions (experiments/z2/oracle.py run)."""

    def __init__(self):
        super().__init__(key="s12", title="S12 LongMemEval oracle (replay + reads)", unit="questions",
                         estimate=300.0)      # ~22 turns per question plus five reads

    def out(self, ctx):
        return ctx.dir / self.key

    def database(self, ctx):
        return f"ice_z2_{ctx.name}_lme"

    def total(self, ctx):
        return 500

    def done(self, ctx):
        from experiments.z2 import oracle
        return oracle.progress(self.out(ctx))[0]

    def rate(self, ctx):
        stamps = sorted(p.stat().st_mtime for p in (self.out(ctx) / "instances").glob("*/done.json"))[-30:]
        gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]) if 0 <= b - a < 3600)
        return gaps[len(gaps) // 2] if len(gaps) >= 3 else None

    def complete(self, ctx):
        return self.done(ctx) >= 500

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/oracle.py", "run", "--database", self.database(ctx),
                   "--out", str(self.out(ctx).relative_to(ROOT))]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class OracleGrade(Stage):
    """S12 grading: LongMemEval's own judge prompts and the qualified Muse grade."""

    def __init__(self):
        super().__init__(key="s12-grade", title="S12 grading (upstream + Muse)", needs=("s12-answers",),
                         unit="grades", estimate=1.5)

    def out(self, ctx):
        return ctx.dir / self.key

    def total(self, ctx):
        return 2500                             # 500 questions x 5 conditions

    def done(self, ctx):
        path = self.out(ctx) / "oracle-grade-raw.jsonl"
        if not path.exists():
            return 0
        with path.open() as handle:
            return min(sum(1 for _ in handle) // 2, 2500)   # two judge calls per answer

    def complete(self, ctx):
        return (self.out(ctx) / "oracle-report.json").exists()

    def run(self, ctx):
        log_path = ctx.dir / f"{self.key}.log"

        def attempt():
            cmd = [sys.executable, "experiments/z2/oracle.py", "grade", "--units", str((ctx.dir / "s12").relative_to(ROOT)),
                   "--answers", str((ctx.dir / "s12-answers").relative_to(ROOT)),
                   "--out", str(self.out(ctx).relative_to(ROOT))]
            rc = ctx.run_child(cmd, log_path)
            return rc, _tail(log_path)

        return run_with_retries(ctx, self.key, attempt, lambda: self.done(ctx), lambda: self.complete(ctx))


class NotBuilt(Stage):
    def __init__(self, key, title, needs):
        super().__init__(key=key, title=title, needs=needs, built=False)


def stages(limit=None):
    """The Z2 run, in order. Estimates are planning numbers (heat-soaked laptop) until
    a stage has measured its own rate: B 51.7 s/turn (S0 r5); C ~200 s/turn (slice)."""
    return [
        Replay("B", 1119, 52.0, limit),
        Replay("C", 325, 200.0, limit),
        Audit("B"),
        Audit("C"),
        ReadPasses("B"),
        ReadPasses("C"),
        Oracle(),
        # Every cloud stage last, on one plan period (spec D6c, D6d): answers, then grading.
        Answers("B"),
        Answers("C"),
        Answers("lme", key="s12-answers", title="S12 answers (Muse 1.2)", units_key="s12", needs=("s12",),
                planned=2000),
        Grading("B"),
        Grading("C"),
        OracleGrade(),
    ]


# ── driver ──────────────────────────────────────────────────────────────────

def drive(ctx, selected):
    """Run *selected* stages in order; returns the per-stage outcome."""
    outcomes = {}
    for stage in selected:
        st = ctx.stage_state(stage.key)
        if st["status"] == "done" and stage.built and stage.complete(ctx):
            outcomes[stage.key] = "done"
            continue
        blocked = [n for n in stage.needs if outcomes.get(n, ctx.stage_state(n)["status"]) != "done"]
        if not stage.built:
            st["status"] = "not_built"
            outcomes[stage.key] = "not_built"
            ctx.save()
            continue
        missing = stage.missing(ctx)
        if missing and not blocked:
            st.update(status="needs_input", missing=missing)
            outcomes[stage.key] = "needs_input"
            ctx.log("stage needs input files", stage=stage.key, missing=missing)
            ctx.save()
            continue
        if blocked:
            st["status"] = "blocked"
            outcomes[stage.key] = "blocked"
            ctx.log("stage blocked", stage=stage.key, needs=blocked)
            ctx.save()
            continue
        st.update(status="running", started=st.get("started") or time.time())
        ctx.save()
        ctx.log("stage started", stage=stage.key)
        try:
            result = stage.run(ctx)
        except Stop:
            st["status"] = "stopped"
            ctx.save()
            ctx.log("run stopped", stage=stage.key)
            outcomes[stage.key] = "stopped"
            return outcomes
        except Exception as exc:                      # a bug in the driver/stage setup
            result = "stuck"
            st["failures"] = (st["failures"] + [dict(at=time.time(), detail=f"{type(exc).__name__}: {exc}")])[-20:]
            ctx.log("stage raised", stage=stage.key, error=f"{type(exc).__name__}: {exc}")
        st["status"] = result
        if result == "done":
            st["finished"] = time.time()
        ctx.save()
        ctx.log(f"stage {result}", stage=stage.key)
        outcomes[stage.key] = result
    return outcomes


def prerequisites():
    """What is missing right now (empty = ready). Each is waited for, never failed on."""
    problems = []
    try:
        from sqlalchemy import create_engine, text
        from sqlalchemy.engine import make_url
        from src.api.config import settings
        engine = create_engine(make_url(settings.database_url).set(database="postgres"),
                               connect_args={"connect_timeout": 5})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
    except Exception as exc:
        problems.append(f"Postgres unreachable ({type(exc).__name__}); start it: "
                        "docker compose -f docker/docker-compose.yml up -d")
    try:
        import urllib.request
        from src.api.config import settings
        urllib.request.urlopen(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=5).read()
    except Exception as exc:
        problems.append(f"Ollama unreachable ({type(exc).__name__}); start it: ollama serve")
    free = shutil.disk_usage(ROOT / "logs").free / 1e9
    if free < MIN_FREE_GB:
        problems.append(f"only {free:.1f} GB free under logs/ (need {MIN_FREE_GB})")
    return problems


def _tail(path, lines=15):
    try:
        text = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    keep = [line for line in text if re.search(r"Error|Exception|Traceback|failed|raise", line)]
    return "\n".join((keep or text)[-lines:])[-2000:]


def _rel(path):
    path = Path(path)
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def _clock(seconds):
    if seconds is None:
        return "?"
    seconds = int(seconds)
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    return (f"{days}d " if days else "") + f"{hours}h{rest // 60:02d}m"


def meter(ctx, note=None):
    rows, left_total, estimated = [], 0.0, False
    for stage in ctx.stages:
        st = ctx.stage_state(stage.key)
        status = st["status"]
        if not stage.built:
            rows.append(f" [  not built] {stage.title}")
            continue
        total, done = stage.total(ctx), stage.done(ctx)
        if stage.complete(ctx):
            status, done = "done", total
        rate = stage.rate(ctx)
        per = rate or stage.estimate
        left = max(0, total - done) * per
        if status != "done":
            left_total += left
            estimated |= rate is None
        share = done / total if total else 0
        bar = "█" * int(share * 20) + "░" * (20 - int(share * 20))
        speed = (f"{rate:.0f} s/{stage.unit[:-1]}" if rate else f"~{stage.estimate:.0f} s/{stage.unit[:-1]} (estimate)")
        tail = "" if status == "done" else f"  {speed}  left {_clock(left)}"
        fails = len(st.get("failures", []))
        if status == "needs_input":
            tail = "  waiting for: " + ", ".join(stage.missing(ctx))
        rows.append(f" [{status:>11}] {stage.title:<34} {done:>5}/{total:<5} {stage.unit} {bar} {share:4.0%}"
                    f"{tail}{f'  ({fails} restarts)' if fails else ''}")
    elapsed = time.time() - ctx.state.get("started", time.time())
    head = (f"Z2 run '{ctx.name}'  {_rel(ctx.dir)}   elapsed {_clock(elapsed)}   "
            f"built stages left ~{_clock(left_total)}{' (partly estimated)' if estimated else ''}")
    lines = [head, *rows]
    if any(not s.built for s in ctx.stages):
        lines.append(" Stages not built yet are skipped; a stage that needs one waits. Everything done is saved.")
    if note:
        lines.append(f" ⏳ {note}")
    lines.append(f" Ctrl-C stops cleanly; the same command resumes.  Log: {_rel(ctx.dir / 'runner.log')}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run (or resume) the stages in order")
    status = sub.add_parser("status", help="show the meter of a run")
    for p in (run, status):
        p.add_argument("--name", default="main", help="run name (logs/z2-run-<name>)")
    run.add_argument("--only", help="comma-separated stage keys, e.g. s7-B,s7-C")
    run.add_argument("--limit", type=int, help="replay only the first N turns per history (a slice check)")
    status.add_argument("--watch", action="store_true")
    args = parser.parse_args()

    ctx = Context(args.name)
    ctx.stages = stages(getattr(args, "limit", None) or ctx.state.get("limit"))   # status: the run's own slice
    if args.command == "status":
        lock = (ctx.dir / "runner.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
            alive = "no runner is active (run the same `run` command to continue)"
        except BlockingIOError:
            alive = "runner active"
        while True:
            ctx.state = json.loads(ctx.state_path.read_text()) if ctx.state_path.exists() else ctx.state
            ctx.render(note=alive, force=True)
            if not args.watch:
                return 0
            time.sleep(TTY_REFRESH)

    lock = (ctx.dir / "runner.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"another runner holds {ctx.dir}; watch it with: run_z2.py status --name {ctx.name} --watch")
        return 1
    known = {s.key for s in ctx.stages}
    wanted = set(args.only.split(",")) if args.only else known
    if wanted - known:
        print(f"unknown stage(s) {sorted(wanted - known)}; known: {sorted(known)}")
        return 2
    if args.limit:
        ctx.state.setdefault("limit", args.limit)
        if ctx.state["limit"] != args.limit:
            print(f"this run was started with --limit {ctx.state['limit']}; use the same value or a new --name")
            return 2
    elif ctx.state.get("limit"):
        print(f"this run is a {ctx.state['limit']}-turn slice; pass --limit {ctx.state['limit']} or use a new --name")
        return 2

    def request_stop(signum, _frame):
        if ctx.stopping:
            return
        ctx.stopping = True
        ctx.log("stop requested", signal=signal.Signals(signum).name)
        print("\nStopping cleanly — the stage finishes its safe point; the same command resumes.", flush=True)

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, request_stop)
    ctx.log("runner started", only=sorted(wanted), pid=os.getpid())
    outcomes = drive(ctx, [s for s in ctx.stages if s.key in wanted])
    ctx.render(force=True)
    print("\nOutcome: " + ", ".join(f"{k} {v}" for k, v in outcomes.items()))
    for key, result in outcomes.items():
        if result == "needs_input":
            print(f"\n{key} waits for input files: {', '.join(ctx.stage_state(key).get('missing', []))}")
        if result == "stuck":
            last = ctx.stage_state(key)["failures"][-1:]
            print(f"\n{key} is stuck after {STRIKES} failures without progress. Last error:\n"
                  f"{last[0].get('detail', '') if last else ''}\nFull log: {ctx.dir / (key + '.log')}")
    if "stopped" in outcomes.values():
        return 0                                    # a requested stop is not a failure
    return 0 if all(v in ("done", "not_built") for v in outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
