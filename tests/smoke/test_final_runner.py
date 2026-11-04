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
    assert reads["B"].preset == "final_main"                                        # B's parts run dropped 2026-10-28
    assert reads["S9"].preset == "final_main+final_ablation+final_buildup"           # Z3 D10
    assert reads["S4"].preset == "final_main+final_decay"                            # Z2 F18
    assert reads["S1"].preset == "final_main" and set(reads) >= {"A", "SW", "LAP", "ICE2", "S8"} and "D" not in reads
    groups = {s.key: s for s in stages if isinstance(s, F.Group)}
    assert groups["local-read-BEAM500K"].needs == ("local-replay-BEAM500K",) and len(groups["local-read-BEAM500K"].members) == 5
    assert "local-read-BEAM1M" not in groups and F.BEAM_TIERS == ("100K", "500K")                  # trimmed 2026-10-28
    member = groups["local-read-BEAM100K"].members[0]
    assert member.replay is groups["local-replay-BEAM100K"].members[0]
    assert next(s for s in stages if s.key == "local-replay-S5").needs == ("prep-generate-S5",)
    grades = [s for s in stages if s.key.startswith("grade-")]
    assert {"grade-B", "grade-S9", "grade-S1-repeat", "grade-BEAM500K", "grade-lmes"} <= {g.key for g in grades}
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


def test_memory_systems_run_on_locomo_only():
    stages = {s.key: s for s in F.final_stages()}
    for system in F.MEMORY_SYSTEMS:
        assert f"local-baseline-{system}" not in stages and f"grade-baseline-{system}" not in stages
        assert stages[f"local-baseline-{system}-LOCOMO"].needs[0] == "local-read-LOCOMO"
        assert f"grade-baseline-{system}-LOCOMO" in stages
    # ingest + search read nothing of ICE (they may run before the freeze); only the wrap waits for
    # ICE's LoCoMo read, and every search comes before its wrap in the local order
    keys = list(stages)
    for system in F.MEMORY_SYSTEMS:
        search, wrap = stages[f"local-baseline-search-{system}-LOCOMO"], stages[f"local-baseline-{system}-LOCOMO"]
        assert not any("read" in n or "replay" in n for n in search.needs)
        assert all(not m.needs for m in search.members) and search.members[0].phase == "search"
        assert set(wrap.needs) == {"local-read-LOCOMO", search.key} and wrap.members[0].phase == "wrap"
        assert keys.index(search.key) < keys.index(wrap.key)
        assert "--ice-units" not in search.members[0].args(None)


def test_each_kind_of_data_answers_only_its_own_conditions():
    from experiments.z2 import read_passes as R
    assert R.PRESETS["final_main"] == ["full_default", "recent_only", "bare_vector", "long_context"]
    assert "warm_vector" not in R.PRESETS["final_main"] and "full" not in R.PRESETS["final_main"]
    assert R.LONG_CONTEXT_PROMPT_TOKENS == 256_000
    stages = {s.key: s for s in F.final_stages()}
    assert stages["cloud-answers-B"].conditions is None                             # our data: all read conditions
    assert stages["cloud-answers-BEAM100K"].members[0].conditions == F.BEAM_ANSWERED
    assert stages["cloud-answers-LOCOMO"].members[0].conditions == F.LOCOMO_ANSWERED
    assert stages["cloud-answers-lmes"].conditions == F.LMES_ANSWERED
    assert "recent_only" not in F.BEAM_ANSWERED + F.LOCOMO_ANSWERED + F.LMES_ANSWERED


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
        assert main.judge == sub.judge and main.needs == ("subset",)
        if F.CROSS_JUDGE is None:
            assert main.judge == "unset-cross-judge" and "CROSS_JUDGE" in main.missing(ctx)[0]
        args = main.cmd_args(ctx, ctx.dir / main.key)
        assert "--verdicts-only" in args and f"subset-read-{h}" in " ".join(args) and f"cloud-answers-{h}" in " ".join(args)
        assert f"subset-answers-{h}" in " ".join(sub.cmd_args(ctx, ctx.dir / sub.key))


def test_upper_bound_is_muse_in_its_own_window_and_waits_when_unset(monkeypatch):
    assert F.UPPER_BOUND == dict(answerer="muse12", window=1_048_576)              # prereg §6, 2026-10-28
    stages = {s.key: s for s in F.final_stages()}
    ctx = type("C", (), {"dir": F.ROOT / "logs/final-run-x", "name": "x"})()
    assert stages["upper-build-BEAM"].members[0].missing(ctx) == []
    assert stages["upper-answers-BEAM"].members[0].answerer == "muse12"
    assert len(stages["upper-build-BEAM"].members) == 5 and F.UPPER_TIERS == ("500K",)   # 128K fits 256K whole
    monkeypatch.setattr(F, "UPPER_BOUND", dict(answerer=None, window=None))
    unset = {s.key: s for s in F.final_stages()}
    assert "UPPER_BOUND" in unset["upper-build-BEAM"].members[0].missing(ctx)[0]
    assert unset["upper-answers-BEAM"].members[0].answerer == "unset-upper-bound-model"


def test_locomo_and_amem_are_wired_local_then_cloud():
    stages = {s.key: s for s in F.final_stages()}
    keys = list(stages)
    assert len(stages["local-read-LOCOMO"].members) == 10
    assert stages["local-read-LOCOMO"].members[0].preset == "final_locomo"
    assert "prep-amem-vendor" in stages["local-baseline-search-amem-LOCOMO"].needs
    for k in ("local-baseline-amem-LOCOMO", "local-replay-LOCOMO"):
        assert keys.index(k) < keys.index("cloud-answers-LOCOMO") < keys.index("grade-LOCOMO")
    member = stages["grade-baseline-mem0-LOCOMO"].members[0]
    ctx = type("C", (), {"dir": F.ROOT / "logs/final-run-x", "name": "x"})()
    args = " ".join(member.cmd_args(ctx, ctx.dir / member.key))
    assert "--occurrences logs/final-locomo/" in args and "--replay" not in args and "local-baseline-mem0-LOCOMO-" in args



def test_a_local_model_can_be_a_judge_role():
    from experiments.z2.profiles import metadata, profile
    p = profile("local:qwen3.8:27b")
    assert p.model == "qwen3.8:27b" and p.endpoint == "chat_completions" and "11434" in p.resolved_base_url()
    assert metadata("local:qwen3.8:27b")["model"] == "qwen3.8:27b"
    assert profile("muse").model == "muse-spark-1.3-contributor"


def test_local_lanes_keep_each_history_together_and_balance_hours():
    from types import SimpleNamespace as NS
    from experiments.final import local_lanes as L
    stage = lambda key, n: NS(key=key, total=lambda ctx, n=n: n, estimate=3600.0)
    stages = [stage("prep-x", 1), stage("local-replay-B", 10), stage("local-read-B", 2), stage("local-replay-S1", 5),
              stage("local-read-S1", 1), stage("local-replay-LOCOMO", 4), stage("local-read-LOCOMO", 1),
              stage("local-baseline-mem0-LOCOMO", 1), stage("cloud-answers-B", 9)]
    prep, groups, load = L.plan(stages, None, 2)
    assert prep == ["prep-x"]
    assert groups[0] == ["local-replay-B", "local-read-B"] and load == [12.0, 12.0]
    assert groups[1] == ["local-replay-S1", "local-read-S1", "local-replay-LOCOMO", "local-read-LOCOMO",
                         "local-baseline-mem0-LOCOMO"]


def test_clusters_are_not_in_the_parts_test_and_decay_has_both_arms():
    # Audit A12 (2026-10-15): automatic cluster scoping scoped out 19/108 gold turns and ships off,
    # so the parts test's cluster switch changes nothing (Z2 F18); decay is tested live vs neutral.
    from experiments.z2 import read_passes as R
    for preset in ("final_ablation", "final_buildup"):
        assert not [c for c in R.select_conditions(preset) if "clusters" in c]
        # Z3 (2026-11-04, D9 rule 4): parts switched off by default leave the parts test
        assert not [c for c in R.select_conditions(preset)
                    if c.startswith(("loo_", "add_")) and c.split("_")[-1] in ("summaries", "notes", "procedural")]
    assert "addback_summaries" in R.select_conditions("final_ablation")              # read back on beside full
    assert {"full_default", "full"} <= set(R.select_conditions("final_main+final_decay"))
    assert {"full_default", "full"} <= set(R.select_conditions("final_main+final_ablation+final_buildup"))
    assert "warm_vector" in R.select_conditions("final_buildup")                      # step 0 (F13)


def test_final_scratch_databases_pass_the_write_guards(tmp_path):
    # The GPU day's LoCoMo trial (2026-10-30): every ice_final_* replay was refused by an ice_z2_-only guard.
    from experiments.final import run_final as F
    from experiments.z2 import profile_write_path as P
    from experiments.z2 import run_z2 as Z
    ctx = Z.Context("trial", directory=tmp_path)
    replay = next(s for s in F.final_stages(None) if s.key == "local-replay-S1")
    name = replay.database(ctx)
    assert name.startswith(P.SCRATCH_PREFIXES) and name != "ice_db"
    with __import__("pytest").raises(ValueError):
        P.create_database("ice_db")


def test_a_slice_reads_questions_at_its_last_checkpoint(tmp_path):
    # LoCoMo asks every question at the end; a 30-turn slice must still exercise the read path (2026-10-30).
    import json
    from experiments.final import run_final as F
    from experiments.z2 import run_z2 as Z
    ctx = Z.Context("trial", directory=tmp_path)
    read = next(s for s in F.final_stages(30) if s.key == "local-read-LOCOMO").members[0]
    cps = read.replay.out(ctx) / "checkpoints"
    cps.mkdir(parents=True)
    for n in (9, 30):
        (cps / f"cp-{n}.sql").write_text("")
    assert read.asks_for(ctx) == read.asks                                  # no slice: the real asks
    ctx.state["limit"] = 30
    asks = json.loads(read.asks_for(ctx).read_text())
    assert len(asks) == read.SLICE_ASKS and {a["checkpoint_turn"] for a in asks} == {30}
    assert all(a["slice_trial"] for a in asks)


def test_local_lanes_draw_one_meter_and_keep_the_lanes_output_in_files(tmp_path, monkeypatch, capsys):
    """2026-11-04: lanes drawing over one another in one terminal filled tmux's history. The
    launcher draws one meter of every lane (a stage's status from whichever runner has it
    furthest); each lane's own output goes to <run>/lane-localK.out."""
    import json
    from experiments.final import local_lanes as L
    from experiments.final import run_final as F
    from experiments.z2 import run_z2 as Z

    class S(Z.Stage):
        def total(self, ctx):
            return 1

    fake = tmp_path / "fake_runner.py"
    fake.write_text(
        "import json, sys, time, pathlib\n"
        "a = sys.argv\n"
        "d = pathlib.Path('logs') / ('final-run-' + a[a.index('--name') + 1])\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "if '--lane' not in a: sys.exit(0)\n"
        "lane, keys = a[a.index('--lane') + 1], a[a.index('--only') + 1].split(',')\n"
        "print('LANE NOISE', lane, flush=True)\n"
        "w = lambda s: (d / f'state-{lane}.json').write_text(json.dumps(dict(stages={k: dict(status=s) for k in keys})))\n"
        "w('running'); time.sleep(1.5); w('done')\n")
    monkeypatch.setattr(L, "ROOT", tmp_path)
    monkeypatch.setattr(L, "RUNNER", str(fake))
    monkeypatch.setattr(Z, "TTY_REFRESH", 0.3)
    monkeypatch.setattr(F, "final_stages", lambda limit: [S(key=k, title=f"stage {k}") for k in
                                                           ("prep-x", "local-replay-B", "local-replay-S1")])
    run = tmp_path / "logs" / "final-run-t"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps(dict(started=0, stages={"local-replay-B": dict(status="pending")})))
    monkeypatch.setattr("sys.argv", ["local_lanes.py", "--lanes", "2", "--name", "t"])
    assert L.main() == 0
    out = capsys.readouterr().out
    assert "LANE NOISE" not in out and "LANE NOISE" in (run / "lane-local0.out").read_text()
    last = out.rsplit("FINAL", 1)[-1] if "FINAL" in out else out.rsplit("Z2 run", 1)[-1]
    assert "stage local-replay-B" in last and "stage local-replay-S1" in last and "lanes finished" in last
    assert last.count("[       done]") == 2                       # both lanes' stages, merged
    assert json.loads((run / "state.json").read_text())["stages"]["local-replay-B"]["status"] == "pending"  # never written
