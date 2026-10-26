"""FINAL report (pre-registration r2): per-history r2 headline (M1, M2, M2 out of window,
hallucination rates, tokens, TUR) from the grade report's `scoring`, the primary tests, the 1–5
kept descriptive, BEAM per tier, S1 run-to-run variance in M1/M2, the cross-family judge check;
never pooled across datasets."""

import json

from experiments.final import report as F
from experiments.z2 import scoring as S


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def verdict(grade):
    return dict(status="graded", grade=grade, items_covered=[], items_total=0)


def history(full_correct, n=20):
    graded, units, occ = {}, {}, {}
    for i in range(n):
        ask = f"S1:F{i:03d}@50#0"
        occ[ask] = dict(checkpoint_turn=50, type="update", expected_abstention=False,
                        reference=dict(citations=[dict(turn=2, current=True), dict(turn=1, current=False)], required=["x"]))
        graded[f"full|{ask}"] = verdict("correct" if i < full_correct else "incorrect")
        graded[f"bare_vector|{ask}"] = verdict("correct" if i < 4 else "incorrect")
        for c in ("full", "bare_vector"):
            units[f"{c}|{ask}"] = dict(prompt_tokens=2000, source_turns=[2], recent_tokens=0, legs=[])
    prefix = S.turn_prefix([None] * 60, count=lambda t: 1000)
    return graded, S.score("S1", graded, units, occ, prefix)


def test_report_tables(tmp_path):
    _, first = history(full_correct=16)
    _, second = history(full_correct=14)
    descriptive = {"full": {"update": dict(graded=20, mean_score=4.2, superseded_rate=0.0, invented_rate=0.0)}}
    write(tmp_path / "grade-S1/report.json", dict(scoring=first, by_condition_and_type=descriptive))
    write(tmp_path / "grade-S1-repeat/report.json", dict(scoring=second, by_condition_and_type=descriptive))
    write(tmp_path / "grade-BEAM-1M-12/beam-report.json", {"by_condition_and_type": {
        "full": {"knowledge_update": dict(graded=2, mean_nugget=0.5)}}})
    r = F.build(tmp_path)
    head = r["histories"]["S1"]["headline"]["full"]
    assert head["m1"] == 0.8 and head["m2"] == 0.8 and head["m2_out_of_window"] == 0.8 and head["tur"] == 0.4
    assert r["histories"]["S1"]["out_of_window"] == 20                       # distance to the CURRENT turn: 47K
    assert r["histories"]["S1"]["primary_tests"]["P2 M1 all: full vs bare_vector"]["verdict"] == "win"
    assert r["histories"]["S1"]["descriptive_1_5"]["full"]["mean_score"] == 4.2
    assert r["run_to_run_variance_S1"]["full"]["m1"] == -0.1
    assert r["beam"]["1M"]["full"]["mean_nugget"] == 0.5
    md = F.markdown(r)
    assert "## S1" in md and "never pooled" in r["label"] and "| full | 20 | 80.0 | 80.0 | 80.0 |" in md


def test_cross_family_judge_section(tmp_path):
    keys = [f"full|S9:F{i:03d}@50#0" for i in range(30)]
    write(tmp_path / "grade-S9/grades.json", dict(graded={k: verdict("correct") for k in keys}))
    write(tmp_path / "crossjudge-S9-main/grades.json", dict(graded={k: verdict("correct" if i % 2 else "incorrect")
                                                                    for i, k in enumerate(keys)}))
    both = {k: verdict("correct" if i % 2 else "incorrect") for i, k in enumerate(keys)}
    write(tmp_path / "subset-grade-S9/grades.json", dict(graded=both))
    write(tmp_path / "crossjudge-S9-subset/grades.json", dict(graded=both))
    j = F.cross_judge(tmp_path)
    assert j["family_effect"] == 0.5 and j["on_other_family_answers"]["agreement"] == 1.0
