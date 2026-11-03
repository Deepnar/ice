"""Z3 judging: cloud stages wait for their inputs and never write into Z2's own folders."""
from experiments.z2 import z3_judging as J


def test_stages_wait_for_inputs_and_answer_only_the_order_and_summary_settings():
    stages = {s.key: s for s in J.stages()}
    assert set(stages) == {"r5-panel", "r5-judge-B", "r5-judge-C", "r5-aux-B", "r5-aux-C", "r5-beam", "r34-answers-B",
                           "r34-grade-B", "r34-answers-C", "r34-grade-C", "r10-answers", "r10-grade"}
    assert stages["r34-answers-B"].inputs == ("reads-B", J.APPROVED)     # read first, and the judge approved
    assert stages["r34-grade-C"].needs == ("r34-answers-C",)
    cmd = " ".join(stages["r34-answers-B"].cmd)
    assert "--conditions " + ",".join(J.ANSWERED) in cmd and "cand64" not in cmd
    # Z2 F27: the retest grades carry the fixed claim check (SCA needs it), never the ranking
    assert "--verdicts-only" not in stages["r34-grade-B"].cmd and "--no-ranking" in stages["r34-grade-B"].cmd
    assert stages["r5-aux-B"].inputs[0].name == "report.json"           # Z2's grading of B finished first
    for s in stages.values():
        assert "s11-" not in " ".join(s.cmd).split("--out", 1)[-1]       # never Z2's grading folders


def test_r10_answers_only_three_settings_and_is_graded_by_longmemevals_judge():
    stages = {s.key: s for s in J.stages()}
    cmd = " ".join(stages["r10-answers"].cmd)
    assert "--conditions full,recent_only,full_rescue_reranker" in cmd and "s12-r10" in cmd
    assert "--upstream-only" in stages["r10-grade"].cmd and stages["r10-grade"].needs == ("r10-answers",)


def test_r10_graded_report_answers_below_the_bar_from_no_memory(capsys):
    from experiments.z2 import r10_report as R
    rows = [dict(q="a", abstention=False, score=2.0), dict(q="b", abstention=False, score=-2.0)]
    g = lambda ok: dict(status="graded", upstream=ok)
    grades = {"recent_only|a": g(False), "full|a": g(False), "full_rescue_reranker|a": g(True),
              "recent_only|b": g(True), "full|b": g(False), "full_rescue_reranker|b": g(False)}
    R.graded(grades, rows)
    lines = {l.split("bar ")[-1].split()[0]: int(l.split()[-1]) for l in capsys.readouterr().out.splitlines()
             if "reranker rescue, bar " in l and "off" not in l}
    assert lines["+0"] == 2 and lines["+3"] == 1 and lines["-4"] == 1   # b's no-memory answer was right
