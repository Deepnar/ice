"""Z2 long-run driver (experiments/z2/run_z2.py): resumes, survives failures, never
just stops, keeps independent stages going, and its meter reads from files."""

import io

from experiments.z2 import run_z2 as R


class Fake(R.Stage):
    """A stage whose process completes `step` units per attempt, failing as scripted."""

    def __init__(self, key, total, script, needs=()):
        super().__init__(key=key, title=key, needs=needs, estimate=10.0)
        self.n, self.script, self.calls, self.units = total, list(script), 0, 0

    def total(self, ctx):
        return self.n

    def done(self, ctx):
        return self.units

    def complete(self, ctx):
        return self.units >= self.n

    def run(self, ctx):
        def attempt():
            self.calls += 1
            step = self.script.pop(0) if self.script else 0
            if step == "stop":
                ctx.stopping = True
                raise R.Stop()
            self.units = min(self.n, self.units + step)
            return (0 if self.units >= self.n else 1), "boom"
        return R.run_with_retries(ctx, self.key, attempt, lambda: self.units, lambda: self.complete(ctx),
                                  backoff=(0, 0, 0))


def context(tmp_path, health=lambda: []):
    return R.Context("t", health=health, sleep=lambda s: None, out=io.StringIO(), directory=tmp_path)


def test_failures_with_progress_are_resumed_to_completion(tmp_path):
    ctx = context(tmp_path)
    stage = Fake("a", 10, [3, 3, 3, 3])          # three crashes, each after progress
    assert R.drive(ctx, [stage]) == {"a": "done"} and stage.calls == 4
    assert len(ctx.stage_state("a")["failures"]) == 3


def test_no_progress_three_times_is_stuck_and_only_dependents_are_blocked(tmp_path):
    ctx = context(tmp_path)
    stuck, later, dependent = Fake("a", 5, [0, 0, 0]), Fake("b", 2, [2]), Fake("c", 1, [1], needs=("a",))
    out = R.drive(ctx, [stuck, later, dependent])
    assert out == {"a": "stuck", "b": "done", "c": "blocked"} and stuck.calls == 3


def test_stop_saves_state_and_the_same_command_resumes(tmp_path):
    ctx = context(tmp_path)
    first, second = Fake("a", 4, [2, "stop"]), Fake("b", 1, [1])
    assert R.drive(ctx, [first, second]) == {"a": "stopped"}
    again = context(tmp_path)                    # a new process reads the saved state
    assert again.stage_state("a")["status"] == "stopped"
    first.script, first.calls = [2], 0
    assert R.drive(again, [first, second]) == {"a": "done", "b": "done"}
    third = context(tmp_path)
    assert R.drive(third, [first, second]) == {"a": "done", "b": "done"} and first.calls == 1  # done is skipped


def test_missing_prerequisites_are_waited_for_not_failed_on(tmp_path):
    answers = [["Ollama unreachable"], ["Ollama unreachable"], []]
    ctx = context(tmp_path, health=lambda: answers.pop(0))
    assert R.drive(ctx, [Fake("a", 1, [1])]) == {"a": "done"} and answers == []


def test_unbuilt_stages_are_listed_and_end_the_run_cleanly(tmp_path):
    ctx = context(tmp_path)
    ctx.stages = [Fake("a", 3, [3]), R.NotBuilt("s8", "S8 audit", ("a",))]
    assert R.drive(ctx, ctx.stages) == {"a": "done", "s8": "not_built"}
    text = R.meter(ctx)
    assert "not built] S8 audit" in text and "3/3" in text and "done" in text and "skipped" in text


def test_meter_shows_estimates_until_a_rate_is_measured(tmp_path):
    ctx = context(tmp_path)
    ctx.stages = [Fake("a", 100, [])]
    ctx.stages[0].units = 40
    text = R.meter(ctx)
    assert "40/100" in text and "(estimate)" in text and "0h10m" in text   # 60 left × 10 s


def test_every_local_stage_runs_before_any_cloud_stage():
    from experiments.z2.run_z2 import ANSWERER, stages
    keys = [s.key for s in stages()]
    cloud = ("s10-B", "s10-C", "s12-answers", "s11-B", "s11-C", "s12-grade")
    assert keys.index("s12") < min(keys.index(k) for k in cloud) and ANSWERER == "muse12"


def test_a_cloud_lane_keeps_its_own_lock_and_state_and_refuses_gpu_stages(tmp_path, monkeypatch):
    import fcntl
    main = R.Context("t", directory=tmp_path, out=io.StringIO())
    lane = R.Context("t", directory=tmp_path, out=io.StringIO(), lane="cloud")
    assert main.state_path.name == "state.json" and lane.state_path.name == "state-cloud.json"
    assert main.lock_path != lane.lock_path
    held = main.lock_path.open("a")
    fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)                 # the main runner is busy...
    other = lane.lock_path.open("a")
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)                # ...and the lane still starts
    lane.stage_state("s10-B")["status"] = "running"
    lane.save()
    assert "s10-B" not in (R.Context("t", directory=tmp_path, out=io.StringIO()).state["stages"])
    assert {"s9-B", "s12", "s7-C"} <= R.GPU_STAGES and "s10-B" not in R.GPU_STAGES


def test_embedding_device_can_be_pinned_to_cpu(monkeypatch):
    from experiments.z2 import questions
    monkeypatch.setenv("Z2_EMBED_DEVICE", "cpu")
    assert questions.default_device() == "cpu"


def test_a_lane_reads_its_prerequisites_from_the_main_runner(tmp_path):
    import json
    (tmp_path / "state.json").write_text(json.dumps(dict(stages={"s9-B": dict(status="done")})))
    lane = R.Context("t", directory=tmp_path, out=io.StringIO(), lane="cloud")
    lane.owned = {"s10-B"}
    assert lane.need_status("s9-B") == "done"                         # the main runner's
    lane.stage_state("s10-B")["status"] = "done"
    assert lane.need_status("s10-B") == "done"                        # its own
    assert lane.need_status("s8-B") == "pending"


def test_a_lane_trusts_a_stage_it_finished_in_an_earlier_run(tmp_path):
    """Z2, 2026-11-02: --only s11-B on the cloud lane, whose S10 was done in its own state file
    while the main runner's state.json still said stopped — S11 was blocked."""
    import json
    (tmp_path / "state.json").write_text(json.dumps(dict(stages={"s10-B": dict(status="stopped")})))
    (tmp_path / "state-cloud.json").write_text(json.dumps(dict(stages={"s10-B": dict(status="done")})))
    lane = R.Context("t", directory=tmp_path, out=io.StringIO(), lane="cloud")
    lane.owned = {"s11-B"}
    assert lane.need_status("s10-B") == "done"
    assert lane.need_status("s10-C") == "pending"


def test_a_stage_finished_by_another_runner_is_done_for_all(tmp_path):
    """FINAL's local lanes: the main runner never reruns what a lane finished, and a lane's
    prerequisite done by another lane counts."""
    import json
    from experiments.z2 import run_z2 as Z
    ctx = Z.Context("t", directory=tmp_path)
    (tmp_path / "state-local1.json").write_text(json.dumps(dict(stages={"local-replay-S2": dict(status="done")})))
    assert ctx.need_status("local-replay-S2") == "done" and ctx.need_status("local-replay-S3") == "pending"
    lane = Z.Context("t", directory=tmp_path, lane="local2")
    lane.owned = {"local-read-S3"}
    assert lane.need_status("local-replay-S2") == "done"
