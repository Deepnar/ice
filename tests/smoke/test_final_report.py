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
        graded[f"full_default|{ask}"] = verdict("correct" if i < full_correct else "incorrect")
        graded[f"bare_vector|{ask}"] = verdict("correct" if i < 4 else "incorrect")
        for c in ("full_default", "bare_vector"):
            units[f"{c}|{ask}"] = dict(prompt_tokens=2000, source_turns=[2], recent_tokens=0, legs=[])
    prefix = S.turn_prefix([None] * 60, count=lambda t: 6000)          # current evidence 288K back: beyond 256K
    return graded, S.score("S1", graded, units, occ, prefix)


def test_report_tables(tmp_path):
    _, first = history(full_correct=16)
    _, second = history(full_correct=14)
    descriptive = {"full_default": {"update": dict(graded=20, mean_score=4.2, superseded_rate=0.0, invented_rate=0.0)}}
    write(tmp_path / "grade-S1/report.json", dict(scoring=first, by_condition_and_type=descriptive))
    write(tmp_path / "grade-S1-repeat/report.json", dict(scoring=second, by_condition_and_type=descriptive))
    write(tmp_path / "grade-BEAM-500K-12/beam-report.json", {"by_condition_and_type": {
        "full": {"knowledge_update": dict(graded=2, mean_nugget=0.5)}}})
    r = F.build(tmp_path)
    head = r["histories"]["S1"]["headline"]["full_default"]
    assert head["m1"] == 0.8 and head["m2"] == 0.8 and head["m2_out_of_window"] == 0.8 and head["tur"] == 0.4
    assert r["histories"]["S1"]["out_of_window"] == 20                       # distance to the CURRENT turn
    assert r["histories"]["S1"]["primary_tests"]["M1 all: full_default vs bare_vector"]["verdict"] == "win"
    assert r["histories"]["S1"]["descriptive_1_5"]["full_default"]["mean_score"] == 4.2
    assert r["run_to_run_variance_S1"]["full_default"]["m1"] == -0.1
    assert r["beam"]["500K"]["full"]["mean_nugget"] == 0.5
    md = F.markdown(r)
    assert "## S1" in md and "suites pooled first" in r["label"] and "| full_default | 20 | 80.0 | 80.0 | 80.0 |" in md


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


def test_locomo_and_upper_bound_sections(tmp_path):
    exports = tmp_path / "exports"
    occ = [dict(family=f"LOCOMO-conv-x:Q{i:03d}", checkpoint_turn=9, repeat_index=0, type=t, expected_abstention=t == "adversarial",
                reference=dict(required=[] if t == "adversarial" else ["Tofu"], answer="Tofu", citations=[]))
           for i, t in enumerate(["single_hop", "single_hop", "temporal", "adversarial"])]
    write(exports / "conv-x/occurrences.json", occ)
    ask = lambda i: f"LOCOMO-conv-x:Q{i:03d}@9#0"
    write(tmp_path / "run/grade-LOCOMO-conv-x/grades.json", dict(graded={
        f"full|{ask(0)}": verdict("correct"), f"full|{ask(1)}": verdict("correct"), f"full|{ask(2)}": verdict("incorrect"),
        f"full|{ask(3)}": dict(verdict("correct"), abstention_expected=True, abstained=True)}))
    write(tmp_path / "run/grade-baseline-mem0-LOCOMO-conv-x/grades.json", dict(graded={
        f"mem0|{ask(0)}": verdict("incorrect"), f"mem0|{ask(3)}": dict(verdict("incorrect"), abstention_expected=True, abstained=False)}))
    out = F.locomo(tmp_path / "run", exports=exports)
    assert out["full"]["m1_categories_1_4"] == round(2 / 3, 4) and out["full"]["confident_on_never_said"] == 0.0
    assert out["mem0"]["m1_categories_1_4"] == 0.0 and out["mem0"]["confident_on_never_said"] == 1.0
    rows = lambda pairs: "".join(json.dumps(dict(key=f"{c}|q{i}", condition=c, status="graded", nugget_mean=v)) + "\n"
                                 for i in range(4) for c, v in pairs)
    (tmp_path / "run/upper-grade-BEAM-500K-1").mkdir(parents=True)
    (tmp_path / "run/upper-grade-BEAM-500K-1/beam-grades.jsonl").write_text(rows((("full_history", 0.75),)))
    (tmp_path / "run/grade-BEAM-500K-1").mkdir(parents=True)
    (tmp_path / "run/grade-BEAM-500K-1/beam-grades.jsonl").write_text(rows((("full_default", 0.5), ("long_context", 0.25))))
    up = F.upper_bound(tmp_path / "run")
    assert up["mean_nugget"] == {"full_default": 0.5, "long_context": 0.25, "full_history": 0.75}
    assert up["full_default minus full_history"] == dict(questions=4, diff=-0.25, ci95=[-0.25, -0.25])
    assert up["full_default minus long_context"]["diff"] == 0.25



def test_suites_pool_their_histories_first(tmp_path):
    _, a = history(full_correct=16)
    _, s2 = history(full_correct=10)
    write(tmp_path / "grade-S1/report.json", dict(scoring=a, by_condition_and_type={}))
    write(tmp_path / "grade-S2/report.json", dict(scoring=s2, by_condition_and_type={}))
    r = F.build(tmp_path)
    syn = r["suites"]["LSREP synthetic — carries the claims (releasable)"]
    assert syn["histories"] == ["S1", "S2"] and syn["questions"] == 40 and "S9" in syn["missing"]
    assert syn["headline"]["full_default"]["m1"] == round(26 / 40, 4)
    md = F.markdown(r)
    assert md.index("# 1. Suites") < md.index("## S1")
