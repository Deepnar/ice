"""FINAL's live trials (experiments/final/trial.py): the BEAM trial runs only its three stages through
the runner's own loop, and a Ctrl-C stops the stage's child (started in its own session, so the
terminal's signal never reaches it) instead of leaving it running."""
import os
import signal
import sys
import threading

from experiments.final import trial as T
from experiments.z2 import run_z2 as Z


class Sleeper(Z.Stage):
    """A stage whose work is a real child process, like a replay."""

    def __init__(self, key, needs=()):
        super().__init__(key=key, title=key, unit="steps", estimate=1.0, needs=needs)
        self.pids = []

    def total(self, ctx):
        return 1

    def done(self, ctx):
        return int(self.complete(ctx))

    def complete(self, ctx):
        return (ctx.dir / f"{self.key}.ok").exists()

    def run(self, ctx):
        cmd = [sys.executable, "-c", f"import time, pathlib; time.sleep({self.seconds}); "
               f"pathlib.Path({str(ctx.dir / (self.key + '.ok'))!r}).touch()"]
        original = Z.subprocess.Popen

        def popen(*a, **k):
            child = original(*a, **k)
            self.pids.append(child.pid)
            return child
        Z.subprocess.Popen = popen
        try:
            return "done" if ctx.run_child(cmd, ctx.dir / f"{self.key}.log") == 0 else "stuck"
        finally:
            Z.subprocess.Popen = original


def fake_run(tmp_path, monkeypatch, seconds):
    from experiments.final import run_final as F
    stages = [Sleeper("prep-beam-export"), Sleeper("local-replay-BEAM-100K-1"),
              Sleeper("local-read-BEAM-100K-1", needs=("local-replay-BEAM-100K-1",))]
    for s in stages:
        s.seconds = seconds
    monkeypatch.setattr(F, "final_stages", lambda limit=None: stages)
    monkeypatch.setattr(T, "RUN", tmp_path)
    monkeypatch.setattr(Z, "TTY_REFRESH", 0.1, raising=False)
    return stages


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:                                               # reaped or a zombie of ours: not running
        return os.waitpid(pid, os.WNOHANG) == (0, 0)
    except ChildProcessError:
        return True


def test_a_stop_ends_the_child_and_the_same_command_resumes(tmp_path, monkeypatch):
    stages = fake_run(tmp_path, monkeypatch, seconds=60)
    threading.Timer(2.0, os.kill, (os.getpid(), signal.SIGINT)).start()
    previous = signal.getsignal(signal.SIGINT)
    try:
        assert T.beam("BEAM-100K-1") == 0
    finally:
        signal.signal(signal.SIGINT, previous)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
    first = stages[0]
    assert first.pids and not any(alive(p) for p in first.pids)        # the child was stopped, not orphaned
    assert not first.complete(Z.Context("trial", directory=tmp_path))
    for s in stages:
        s.seconds = 0
    assert T.beam("BEAM-100K-1") == 0                                   # the same command resumes and finishes
    assert all(s.complete(Z.Context("trial", directory=tmp_path)) for s in stages)


def test_an_unknown_conversation_is_refused(tmp_path, monkeypatch):
    fake_run(tmp_path, monkeypatch, seconds=0)
    assert T.beam("BEAM-100K-99") == 2


def test_the_lme_trial_is_the_runners_oracle_stage_on_two_slice_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RUN", tmp_path)
    stage = T.lme_stage()
    assert isinstance(stage, Z.Oracle) and stage.total(None) == 2 and stage.done(None) == 0
    args = stage.extra_args(None)
    assert args[:2] == ["--dataset", "s"] and "--slice" in args and args[-2:] == ["--limit", "2"]
