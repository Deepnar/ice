"""Tuning verdicts follow the pre-registered rule against the repeat noise floor."""

import json

import pytest

from experiments.v3_repair import tuning_results as tr


def report(path, rows, complete=True):
    path.write_text(json.dumps(dict(complete=complete, records=[
        dict(arm=arm, occurrence={"q": q}, grade=g, status="graded" if g else "pending", prompt_tokens=t)
        for arm, q, g, t in rows])))
    return path


def test_noise_floor_and_verdicts(tmp_path):
    rows = []
    for q in range(10):
        rows += [("baseline:full", q, "correct" if q < 6 else "incorrect", 1000),
                 ("baseline-repeat:full", q, "correct" if q < 5 else "incorrect", 1000),   # 1 disagreement
                 ("better:full", q, "correct" if q < 9 else "incorrect", 1000),            # +3 net
                 ("cheaper:full", q, "correct" if q < 6 else "incorrect", 800),            # same, -20% tokens
                 ("noisy:full", q, "correct" if q < 7 else "incorrect", 1000)]             # +1 = floor
    result = tr.analyse(tr.load([report(tmp_path / "a.json", rows)]))
    verdicts = {r["arm"]: r["verdict"] for r in result["rows"]}
    assert result["noise"]["floor_net"] == 1
    assert verdicts["better:full"].startswith("WIN: quality")
    assert verdicts["cheaper:full"].startswith("WIN: same quality")
    assert verdicts["noisy:full"] == "keep current (within noise)"


def test_later_plan_baseline_is_a_reask_and_incomplete_refuses(tmp_path):
    first = report(tmp_path / "a.json", [("baseline:full", 1, "correct", 10)])
    second = report(tmp_path / "b.json", [("baseline:full", 1, "incorrect", 10)])
    arms = tr.load([first, second])
    assert set(arms) == {"baseline:full", "baseline-reask1:full"}
    with pytest.raises(ValueError, match="incomplete"):
        tr.load([report(tmp_path / "c.json", [("baseline:full", 1, None, 10)], complete=False)])
