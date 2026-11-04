"""Scoring r2 (FINAL_PREREGISTRATION §2–§4; Z2 D6e): M1/M2 from the judge's verdict and the
claim chain, evidence distance and the window split, McNemar / Holm / detectable difference,
the primary tests, hallucination non-inferiority, and the fidelity states."""

from experiments.z2 import scoring as S


def verdict(grade, **kw):
    return dict(status="graded", grade=grade, items_covered=kw.pop("covered", []), items_total=kw.pop("total", 0), **kw)


def test_m1_and_m2_rules():
    o = S.outcome(verdict("correct"))
    assert o["m1"] and o["m2"]
    assert not S.outcome(verdict("partial"))["m1"]
    assert not S.outcome(verdict("uncertain"))["m1"] and S.outcome(verdict("uncertain"))["uncertain"]
    stale = S.outcome(verdict("correct", superseded_used=True))
    assert stale["m1"] and not stale["m2"] and stale["outdated"]
    invented = S.outcome(verdict("correct", invented=1, claims={"x": "confirmed_invented"}))
    assert invented["m1"] and not invented["m2"]
    assert S.outcome(verdict("incorrect", abstention_expected=True, abstained=False))["confident_never_said"]
    assert S.outcome(verdict("correct", covered=[0], total=2))["coverage"] == 0.5
    assert S.outcome(dict(status="incomplete")) is None
    assert S.outcome(verdict("partial"), "attribution")["attribution_wrong"]


def test_distance_bins_and_recent_window():
    prefix = S.turn_prefix([None] * 10, count=lambda t: 1000 * t)     # turn t has 1000·t tokens
    assert prefix[3] == 6000
    # question at checkpoint 10 (after turn 10): farthest cited turn 2 → turns 3..10 lie between
    assert S.distance([5, 2], 10, prefix) == prefix[10] - prefix[2] == 52_000
    assert S.distance([10], 10, prefix) == 0 and S.distance([11, 12], 10, prefix) is None
    assert S.bin_of(42_000) == "32K-128K" and S.bin_of(600_000) == ">512K" and S.bin_of(None) == "no cited turn"
    assert S.recent_turns(10, 17_000, prefix) == {10} and S.recent_turns(10, 19_000, prefix) == {9, 10}


def test_statistics():
    assert S.mcnemar(0, 0) == 1.0
    assert abs(S.mcnemar(0, 6) - 2 / 64) < 1e-12
    assert S.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    small, large = S.detectable(100), S.detectable(1000)
    assert small and large and large < small < 0.2


def _history(n_fam=30, full_m2=True, base_m2=False):
    graded, units, occ = {}, {}, {}
    for i in range(n_fam):
        ask = f"B:f{i}@50#0"
        occ[ask] = dict(checkpoint_turn=50, type="fact_recall", expected_abstention=False,
                        reference=dict(citations=[dict(turn=2)], required=["x"]))
        graded[f"full_default|{ask}"] = verdict("correct" if full_m2 else "incorrect")       # ICE as it runs
        graded[f"full|{ask}"] = verdict("correct" if full_m2 else "incorrect")               # the ablation reference
        graded[f"bare_vector|{ask}"] = verdict("correct" if base_m2 or i % 5 == 0 else "incorrect")
        graded[f"loo_codex|{ask}"] = verdict("incorrect")
        for c in ("full_default", "full", "bare_vector", "loo_codex"):
            ice = c in ("full_default", "full")
            units[f"{c}|{ask}"] = dict(prompt_tokens=1000, source_turns=[2] if ice else [], recent_tokens=0,
                                       legs=["facts"] if ice else [])
    return graded, units, occ


def test_primary_tests_window_split_and_fidelity():
    graded, units, occ = _history()
    prefix = S.turn_prefix([None] * 60, count=lambda t: 6000)       # turn 2 → question: 288K, beyond 256K
    prompts = {k: k for k in units}                                  # switching codex off changes every prompt
    out = S.score("B", graded, units, occ, prefix, prompts=prompts,
                  audit=dict(kinds=dict(codex_fact=dict(items=10, criteria=dict(faithful=dict(yes_rate=0.99))))))
    assert out["out_of_window"] == 30 and out["window_rule"].startswith("distance")
    m2 = out["primary_tests"]["M2 all: full_default vs bare_vector"]
    assert m2["better"] == 24 and m2["worse"] == 0 and m2["verdict"] == "win"
    p1 = out["primary_tests"]["P1 M2 out-of-window: full_default vs bare_vector"]
    assert p1["pairs"] == 30 and p1["verdict"].startswith("not tested") and "p_holm" not in p1   # < 100 pairs
    assert out["by_condition"]["full_default"]["m1"] == 1.0 and out["by_condition"]["full_default"]["tur"] == 1.0
    assert out["evidence_recall"]["full"]["all_cited_turns_reached"] == 1.0
    assert out["evidence_recall"]["bare_vector"]["all_cited_turns_reached"] == 0.0
    codex = out["fidelity"]["codex"]
    assert codex["state"] == "contributing" and codex["prompt_reach"] == 1.0 and codex["items"] == {"codex_fact": 10}
    assert out["fidelity"]["bm25"]["state"] == "not ablated here"
    assert out["hallucination_noninferiority"]["invented"]["non_inferior"] is True


def test_long_context_packed_turns_decide_the_window_and_unchanged_prompts_mean_unexercised():
    graded, units, occ = _history(n_fam=4)
    for ask in occ:
        units[f"long_context|{ask}"] = dict(prompt_tokens=37_000, source_turns=[1, 2, 3], recent_tokens=0, legs=[])
        graded[f"long_context|{ask}"] = verdict("correct")
    prefix = S.turn_prefix([None] * 60, count=lambda t: 2000)
    prompts = {f"{c}|{a}": ("p" if c in ("full", "loo_codex") else c + a) for c in ("full", "loo_codex") for a in occ}
    out = S.score("B", graded, units, occ, prefix, prompts=prompts)
    assert out["window_rule"] == "long_context's packed turns" and out["out_of_window"] == 0
    assert out["fidelity"]["codex"]["state"] == "unexercised"


def test_bare_vector_units_are_seen(tmp_path):
    import json
    from experiments.z2 import report
    for cond, stage in (("bare_vector", dict(selected=[dict(turn=7, tokens=5)], prompt_messages=[dict(role="user", content="q")])),
                        ("recent_only", dict(selected=[], prompt_tokens=50, context_ledger=dict(blocks=dict(recent_turns=900))))):
        d = tmp_path / "units" / cond
        d.mkdir(parents=True)
        (d / "a.json").write_text(json.dumps(dict(condition=cond, ask=dict(id="f@9#0", kind="question"), stage=stage)))
    units = report.unit_index(tmp_path, count_messages=lambda m: 42)
    assert units["bare_vector|f@9#0"]["source_turns"] == [7] and units["bare_vector|f@9#0"]["prompt_tokens"] == 42
    assert units["bare_vector|f@9#0"]["recent_tokens"] == 900


def test_judge_family_check_finds_leniency_only_on_its_own_family():
    def grades(correct):
        return {f"full|B:f{i}@50#0": verdict("correct" if c else "incorrect") for i, c in enumerate(correct)}
    n = 40
    muse_main = grades([True] * n)                       # Muse passes every Muse-written answer
    other_main = grades([i % 2 == 0 for i in range(n)])  # the other judge passes half
    muse_sub = grades([i % 2 == 0 for i in range(n)])    # on non-Meta answers they agree
    other_sub = grades([i % 2 == 0 for i in range(n)])
    out = S.judge_family_check(muse_main, other_main, muse_sub, other_sub, resamples=2000)
    assert out["family_effect"] == 0.5 and out["ci95"][0] > 0 and out["reading"].startswith("Muse grades")
    assert out["on_other_family_answers"]["agreement"] == 1.0 and out["on_other_family_answers"]["kappa"] == 1.0
    same = S.judge_family_check(muse_sub, other_sub, muse_sub, other_sub, resamples=2000)
    assert same["family_effect"] == 0 and same["reading"] == "no detectable family leniency"


def test_pool_keeps_histories_apart_and_tests_the_pooled_questions():
    graded, units, occ = _history(n_fam=20)
    prefix = S.turn_prefix([None] * 60, count=lambda t: 2000)
    a = S.score("A", graded, units, occ, prefix)
    b = S.score("B", graded, units, occ, prefix)
    pooled = S.pool({"A": a, "B": b})
    assert pooled["questions"] == 40 and pooled["histories"] == ["A", "B"]          # same ask ids, two histories
    p2 = pooled["primary_tests"]["M1 all: full_default vs bare_vector"]
    assert p2["pairs"] == 40 and p2["families"] == 40 and p2["verdict"] == "win"
    assert pooled["by_condition"]["full_default"]["m1"] == 1.0 and pooled["by_condition"]["full_default"]["prompt_tokens"] == 1000


def test_a_part_off_in_full_ice_is_scored_from_its_add_back_read():
    """Z3 (2026-11-04): turn summaries are off in full ICE; S9 reads them back on beside it, so the
    part's effect is the add-back minus full, with the same sign as a leave-one-out."""
    graded, units, occ = _history()
    for ask in occ:
        graded[f"addback_summaries|{ask}"] = verdict("incorrect")             # showing summaries hurts
        units[f"addback_summaries|{ask}"] = dict(prompt_tokens=1000, source_turns=[2], recent_tokens=0, legs=["facts"])
    prefix = S.turn_prefix([None] * 60, count=lambda t: 6000)
    out = S.score("B", graded, units, occ, prefix, prompts={k: k for k in units})
    summaries = out["fidelity"]["summaries"]
    assert summaries["state"] == "off in full ICE; added back: hurts" and summaries["prompts_changed_when_off"] == 30
    assert summaries["m2_full_minus_off"]["diff"] < 0 and summaries["m2_full_minus_off"]["worse"] == 30
