"""Z3 judging: cloud stages wait for their inputs and never write into Z2's own folders."""
from experiments.z2 import z3_judging as J


def test_stages_wait_for_inputs_and_answer_only_the_order_and_summary_settings():
    stages = {s.key: s for s in J.stages()}
    assert set(stages) == {"r5-panel", "r5-judge-B", "r5-judge-C", "r34-answers-B", "r34-grade-B",
                           "r34-answers-C", "r34-grade-C"}
    assert stages["r34-answers-B"].inputs == ("reads-B", J.APPROVED)     # read first, and the judge approved
    assert stages["r34-grade-C"].needs == ("r34-answers-C",)
    cmd = " ".join(stages["r34-answers-B"].cmd)
    assert "--conditions " + ",".join(J.ANSWERED) in cmd and "cand64" not in cmd
    assert "--verdicts-only" in stages["r34-grade-B"].cmd
    for s in stages.values():
        assert "s11-" not in " ".join(s.cmd).split("--out", 1)[-1]       # never Z2's grading folders
