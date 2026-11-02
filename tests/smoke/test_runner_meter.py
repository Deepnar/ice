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
