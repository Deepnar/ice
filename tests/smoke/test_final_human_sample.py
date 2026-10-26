"""FINAL human check of the judge: a stratified, fixed sample, and kappa."""

from experiments.final import human_sample as H


def test_draw_is_stratified_and_fixed():
    rows = [dict(history="S1", key=f"full|q{i}", judge=g) for i, g in enumerate(["correct"] * 50 + ["partial"] * 5 + ["incorrect"] * 20)]
    a = H.draw(rows, 30)
    assert a == H.draw(list(reversed(rows)), 30)
    assert sum(r["judge"] == "partial" for r in a) == 5 and len(a) == 30


def test_kappa():
    assert H.kappa([("correct", "correct"), ("incorrect", "incorrect")]) == 1.0
    assert H.kappa([("correct", "incorrect"), ("incorrect", "correct")]) < 0
