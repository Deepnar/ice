"""FINAL report: per-history condition summaries with hallucination measures, BEAM per tier,
S1 run-to-run variance; never pooled across datasets."""

import json

from experiments.final import report as F


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def cell(score, graded, sup=0.0, inv=0.0, abst=None):
    return dict(graded=graded, mean_score=score, superseded_rate=sup, invented_rate=inv, abstention_accuracy=abst)


def test_report_tables(tmp_path):
    write(tmp_path / "grade-S1/report.json", {"by_condition_and_type": {
        "full": {"update": cell(4.0, 10, 0.1), "negative": cell(5.0, 2, abst=1.0)},
        "recent_only": {"update": cell(2.0, 10, 0.5)}}})
    write(tmp_path / "grade-S1-repeat/report.json", {"by_condition_and_type": {
        "full": {"update": cell(3.8, 10), "negative": cell(5.0, 2, abst=1.0)}, "recent_only": {"update": cell(2.1, 10)}}})
    write(tmp_path / "grade-BEAM-1M-12/beam-report.json", {"by_condition_and_type": {
        "full": {"knowledge_update": dict(graded=2, mean_nugget=0.5)}}})
    r = F.build(tmp_path)
    full = r["histories"]["S1"]["conditions"]["full"]
    assert full["graded"] == 12 and full["abstention_accuracy"] == 1.0 and full["mean_score"] == round((40 + 10) / 12, 3)
    assert r["run_to_run_variance_S1"]["recent_only"] == 0.1
    assert r["beam"]["1M"]["full"]["mean_nugget"] == 0.5
    assert "## S1" in F.markdown(r) and "never pooled" in r["label"]
