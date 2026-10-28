"""Z2 R5 judge check: the draw is fixed, stratified and split between ICE and plain search."""
from experiments.z2 import judge_check as J


def verdict(grade, covered, total=2, reason=""):
    return dict(status="graded", grade=grade, items_covered=list(range(covered)), items_total=total,
                superseded_used=False, reason=reason)


def graded():
    g = {}
    for i in range(40):
        for cond in ("full_default", "bare_vector"):
            g[f"{cond}|q{i}"] = verdict(("partial", "incorrect", "correct", "partial")[i % 4], 2 if i % 4 != 3 else 1)
        g[f"recent_only|q{i}"] = verdict("incorrect", 0)
    return g


def test_draw_is_deterministic_stratified_and_skips_recent_only():
    a, b = J.draw(graded(), scale=0.1), J.draw(graded(), scale=0.1)
    assert a == b and a
    assert not [x for x in a if x[1] == "recent_only"]
    strata = {s for s, _, _ in a}
    assert "partial, every point covered" in strata and "control: correct" in strata
    assert {x[1] for x in a} == {"full_default", "bare_vector"}


def test_first_matching_stratum_wins_and_per_point_matches_rescoring():
    r = verdict("partial", 2)
    assert J.covered_all(r) and J.per_point(r)
    assert not J.per_point(verdict("partial", 1))
