"""FINAL runner: every stage key is unique, every prerequisite is a real top-level stage,
each dataset reads under its preset, and BEAM groups chain replay -> read -> answers."""

from experiments.final import run_final as F


def test_stage_list_is_consistent():
    stages = F.final_stages()
    keys = [s.key for s in stages]
    assert len(keys) == len(set(keys))
    for s in stages:
        assert all(n in keys for n in s.needs), (s.key, s.needs)
    reads = {s.history: s for s in stages if isinstance(s, F.FinalRead)}
    assert reads["B"].preset == reads["S9"].preset == "final_main+final_ablation"
    assert reads["S1"].preset == "final_main" and set(reads) >= {"A", "D", "SW", "LAP", "S8"}
    groups = {s.key: s for s in stages if isinstance(s, F.Group)}
    assert groups["local-read-BEAM1M"].needs == ("local-replay-BEAM1M",) and len(groups["local-read-BEAM1M"].members) == 10
    member = groups["local-read-BEAM100K"].members[0]
    assert member.replay is groups["local-replay-BEAM100K"].members[0]
    assert next(s for s in stages if s.key == "local-replay-S5").needs == ("prep-generate-S5",)
    grades = [s for s in stages if s.key.startswith("grade-")]
    assert {"grade-B", "grade-S9", "grade-S1-repeat", "grade-BEAM1M", "grade-lmes"} <= {g.key for g in grades}
    assert keys.index("grade-B") > keys.index("local-answers-lmes")          # every cloud stage last


def test_s1_is_replayed_twice_into_separate_stores():
    stages = {s.key: s for s in F.final_stages()}
    first, again = stages["local-replay-S1"], stages["local-replay-S1-repeat"]
    ctx = type("C", (), {"name": "main"})()
    assert first.database(ctx) != again.database(ctx)
    assert stages["local-read-S1-repeat"].replay is again and stages["local-read-S1-repeat"].preset == "final_main"
    assert stages["local-answers-S1-repeat"].needs == ("local-read-S1-repeat",)
