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
    assert reads["B"].preset == "final_main+final_ablation"
    assert reads["S9"].preset == "final_main+final_ablation+final_buildup"           # Z3 D10
    assert reads["S1"].preset == "final_main" and set(reads) >= {"A", "D", "SW", "LAP", "S8"}
    groups = {s.key: s for s in stages if isinstance(s, F.Group)}
    assert groups["local-read-BEAM1M"].needs == ("local-replay-BEAM1M",) and len(groups["local-read-BEAM1M"].members) == 10
    member = groups["local-read-BEAM100K"].members[0]
    assert member.replay is groups["local-replay-BEAM100K"].members[0]
    assert next(s for s in stages if s.key == "local-replay-S5").needs == ("prep-generate-S5",)
    grades = [s for s in stages if s.key.startswith("grade-")]
    assert {"grade-B", "grade-S9", "grade-S1-repeat", "grade-BEAM1M", "grade-lmes"} <= {g.key for g in grades}
    last_local = max(i for i, k in enumerate(keys) if k.startswith(("prep-", "local-")))
    first_cloud = min(i for i, k in enumerate(keys) if not k.startswith(("prep-", "local-")))
    assert last_local < first_cloud                                          # all local, then all cloud (Z3 D13)
    assert keys.index("cloud-answers-lmes") < keys.index("grade-B")


def test_s1_is_replayed_twice_into_separate_stores():
    stages = {s.key: s for s in F.final_stages()}
    first, again = stages["local-replay-S1"], stages["local-replay-S1-repeat"]
    ctx = type("C", (), {"name": "main"})()
    assert first.database(ctx) != again.database(ctx)
    assert stages["local-read-S1-repeat"].replay is again and stages["local-read-S1-repeat"].preset == "final_main"
    assert stages["cloud-answers-S1-repeat"].needs == ("local-read-S1-repeat",)


def test_memory_baselines_are_stages_after_the_lme_reads():
    stages = {s.key: s for s in F.final_stages()}
    for system in ("graphiti", "mem0"):
        assert stages[f"local-baseline-{system}"].needs == ("local-lmes",)
        assert stages[f"grade-baseline-{system}"].needs == (f"cloud-answers-baseline-{system}",)


def test_audited_histories_freeze_their_inventory_before_any_read_pass():
    stages = {s.key: s for s in F.final_stages()}
    for h in F.AUDITED:
        assert stages[f"local-read-{h}"].needs == (f"local-replay-{h}", f"local-inventory-{h}")
        assert stages[f"audit-{h}"].needs == (f"local-inventory-{h}",)
        assert f"audit-{h}" in stages[f"grade-{h}"].needs
        ctx = type("C", (), {"dir": F.ROOT / "logs/final-run-x", "name": "x"})()
        args = stages[f"grade-{h}"].cmd_args(ctx, ctx.dir / f"grade-{h}")
        assert "--no-ranking" in args and "--audit" in args and "--replay" in args


def test_subset_and_cross_family_judge_stages():
    stages = {s.key: s for s in F.final_stages()}
    assert stages["subset"].needs == tuple(f"grade-{h}" for h in F.SUBSET)
    ctx = type("C", (), {"dir": F.ROOT / "logs/final-run-x", "name": "x"})()
    for h in F.SUBSET:
        main, sub = stages[f"crossjudge-{h}-main"], stages[f"crossjudge-{h}-subset"]
        assert main.judge == sub.judge == F.CROSS_JUDGE and main.needs == ("subset",)
        args = main.cmd_args(ctx, ctx.dir / main.key)
        assert "--verdicts-only" in args and f"subset-read-{h}" in " ".join(args) and f"cloud-answers-{h}" in " ".join(args)
        assert f"subset-answers-{h}" in " ".join(sub.cmd_args(ctx, ctx.dir / sub.key))
