"""The runner's meter shows the stages a runner was started with (--only), read back from its log
for `status`, and a cloud-call stage counts calls while its child works."""
import json

from experiments.z2 import run_z2 as Z


class S(Z.Stage):
    def total(self, ctx):
        return 1


def test_the_meter_shows_only_the_runners_stages(tmp_path):
    ctx = Z.Context("t", directory=tmp_path)
    ctx.stages = [S(key="a", title="stage a"), S(key="b", title="stage b"), S(key="c", title="stage c")]
    ctx.shown = {"b"}
    text = Z.meter(ctx)
    assert "stage b" in text and "stage a" not in text and "`status --all`" in text
    ctx.shown = None
    assert all(f"stage {k}" in Z.meter(ctx) for k in "abc")


def test_status_reads_the_latest_runners_only_for_its_lane(tmp_path):
    ctx = Z.Context("t", directory=tmp_path)
    ctx.stages = [S(key=k, title=k) for k in "abcd"]
    log = [dict(msg="runner started", only=["a", "b"], lane=None), dict(msg="runner started", only=["c"], lane="cloud"),
           dict(msg="runner started", only=["b"], lane=None)]
    (tmp_path / "runner.log").write_text("".join(json.dumps(r) + "\n" for r in log))
    assert Z.started_only(ctx) == {"b"}
    ctx.lane = "cloud"
    assert Z.started_only(ctx) == {"c"}


def test_the_controls_stage_counts_cloud_calls(tmp_path, monkeypatch):
    from experiments.final import run_final as F
    monkeypatch.setattr(F, "ROOT", tmp_path)
    calls = tmp_path / "x" / "questions-raw.jsonl"
    calls.parent.mkdir()
    calls.write_text("".join(json.dumps(dict(seconds=2.5)) + "\n" for _ in range(6)))
    stage = F.ControlCalls("k", "t", ["s.py"], [tmp_path / "done.json"], "x/questions-raw.jsonl", 20)
    assert stage.done(None) == 6 and stage.total(None) == 20 and stage.rate(None) == 2.5
    (tmp_path / "done.json").write_text("{}")
    assert stage.done(None) == 20


def test_the_meter_is_drawn_when_a_stage_begins(tmp_path):
    import io

    class Quick(S):
        def run(self, ctx):
            return "done"

        def complete(self, ctx):
            return False

    out = io.StringIO()
    ctx = Z.Context("t", directory=tmp_path, out=out)
    ctx.stages = [Quick(key="a", title="quick stage")]
    ctx.label = "FINAL"
    Z.drive(ctx, ctx.stages)
    assert "FINAL run 't'" in out.getvalue() and "quick stage" in out.getvalue()


def test_on_a_terminal_the_meter_fits_the_screen(tmp_path):
    """FINAL lists ~60 stages; in tmux every frame that scrolls stays in the history (maintainer,
    2026-11-04). Fitted: never taller or wider than the terminal, working stages always shown."""
    ctx = Z.Context("t", directory=tmp_path)
    ctx.stages = [S(key=f"k{i}", title=f"stage number {i} of a long run") for i in range(60)]
    for i in range(20):
        ctx.stage_state(f"k{i}")["status"] = "done"
    for i in (20, 21):
        ctx.stage_state(f"k{i}")["status"] = "running"
    ctx.stage_state("k22")["status"] = "needs_input"
    lines = Z.meter(ctx, note="runner active", fit=(80, 20)).splitlines()
    text = "\n".join(lines)
    assert len(lines) <= 19 and max(map(len, lines)) <= 79
    assert all(f"number {i} " in text for i in (20, 21, 22))          # the working stages, in full
    assert "20 of 60 stages" in text and "more pending" in text and "Ctrl-C" in text and "runner active" in text
    assert len(Z.meter(ctx).splitlines()) > 60                          # the full list still exists
