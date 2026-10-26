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
    # question before turn 10: farthest cited turn 2 → turns 3..9 lie between
    assert S.distance([5, 2], 10, prefix) == prefix[9] - prefix[2] == 42_000
    assert S.distance([10, 11], 10, prefix) is None
    assert S.bin_of(42_000) == "32K-128K" and S.bin_of(600_000) == ">512K" and S.bin_of(None) == "no cited turn"
    assert S.recent_turns(10, 17_000, prefix) == {8, 9}


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
        graded[f"full|{ask}"] = verdict("correct" if full_m2 else "incorrect")
        graded[f"bare_vector|{ask}"] = verdict("correct" if base_m2 or i % 5 == 0 else "incorrect")
        graded[f"loo_codex|{ask}"] = verdict("incorrect")
        for c in ("full", "bare_vector", "loo_codex"):
            units[f"{c}|{ask}"] = dict(prompt_tokens=1000, source_turns=[2] if c == "full" else [], recent_tokens=0,
                                       legs=["facts"] if c == "full" else [])
    return graded, units, occ


def test_primary_tests_window_split_and_fidelity():
    graded, units, occ = _history()
    prefix = S.turn_prefix([None] * 60, count=lambda t: 2000)       # turn 2 → question: 94K tokens away
    prompts = {k: k for k in units}                                  # switching codex off changes every prompt
    out = S.score("B", graded, units, occ, prefix, prompts=prompts,
                  audit=dict(kinds=dict(codex_fact=dict(items=10, criteria=dict(faithful=dict(yes_rate=0.99))))))
    assert out["out_of_window"] == 30 and out["window_rule"].startswith("distance")
    p1 = out["primary_tests"]["P1 M2 out-of-window: full vs bare_vector"]
    assert p1["better"] == 24 and p1["worse"] == 0 and p1["verdict"] == "win"
    assert out["by_condition"]["full"]["m1"] == 1.0 and out["by_condition"]["full"]["tur"] == 1.0
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
