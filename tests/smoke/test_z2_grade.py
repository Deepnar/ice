"""Z2 S11 grading (spec D11): graded once per identical (answer, reference),
incomplete answers ungraded, the invented-content chain, two-order ranking, controls."""

import json

import numpy as np

from experiments.z2 import grade as G


def ctx(cond, ask_id, text, *, kind="question", finish="complete", cp=50, question="what is my cat called"):
    unit = dict(condition=cond, checkpoint=cp, ask=dict(id=ask_id, kind=kind, question=question))
    occ = dict(family=ask_id.split("@")[0], reference=dict(required=["The cat is called Tofu."], superseded=["Miso"]),
               expected_abstention=False) if kind == "question" else None
    return unit, dict(finish=finish, text=text), occ


class Judge:
    def __init__(self, route):
        self.route, self.calls = route, []

    def __call__(self, system, prompt):
        self.calls.append((system, json.loads(prompt)))
        return self.route(system, json.loads(prompt))


def grading(system, body):
    good = "Tofu" in body.get("ANSWER", "")
    return dict(items=[dict(item=0, covered=good)], grade="correct" if good else "incorrect",
                score=5 if good else 1, superseded_used="Miso" in body.get("ANSWER", ""), abstained=False,
                unsupported_claims=[] if good else ["the cat sleeps in a red basket"])


def test_identical_answers_are_graded_once_and_incomplete_is_ungraded(tmp_path):
    contexts = [ctx("full", "f@50#0", "Tofu."), ctx("warm_vector", "f@50#0", "Tofu."),
                ctx("bare_vector", "f@50#0", "Miso, it sleeps in a red basket."),
                ctx("recent_only", "f@50#0", "", finish="incomplete")]
    judge = Judge(grading)
    results, pending = G.grade_answers(contexts, tmp_path, judge)
    assert len(judge.calls) == 2 and not pending
    assert results["full|f@50#0"]["score"] == 5 and results["warm_vector|f@50#0"]["grade"] == "correct"
    assert results["bare_vector|f@50#0"]["superseded_used"] is True
    assert results["recent_only|f@50#0"]["status"] == "ungraded"


LEDGER = [dict(id="v", value="the cat sleeps on the sofa", valid_from_turn=3, usable_for_questions=True),
          dict(id="u", value="the cat is afraid of the vacuum", valid_from_turn=4, usable_for_questions=False)]
CLAIMS = {"the cat sleeps on the sofa": "ledger", "the cat is afraid of the vacuum": "unconfirmed",
          "the cat was adopted in spring": "miss", "the cat speaks french": "invented"}


def batched(system, body):
    """A judge for CLAIMS_SYSTEM: per claim, the entries stating it and whether a passage does."""
    assert system is G.CLAIMS_SYSTEM
    hit = {"ledger": "the cat sleeps on the sofa", "unconfirmed": "the cat is afraid of the vacuum"}
    out = []
    for c in body["CLAIMS"]:
        want = CLAIMS[c["text"]]
        out.append(dict(id=c["id"], stated_by=[e["id"] for e in body["ENTRIES"] if e["text"] == hit.get(want)],
                        found=want == "miss"))
    return dict(claims=out)


def check(results, contexts, judge, tmp_path):
    import experiments.z2.questions as Q
    original = Q.embed
    Q.embed = lambda texts, device="cpu": np.ones((len(texts), 5))
    try:
        return G.check_claims(results, contexts, LEDGER, None, ([(t, "user", f"message {t}") for t in range(1, 6)], np.eye(5)),
                              judge, tmp_path)
    finally:
        Q.embed = original


def test_claim_chain_separates_ledger_unconfirmed_miss_and_invented(tmp_path):
    results = {f"full|q{i}": dict(status="graded", unsupported_claims=[c]) for i, c in enumerate(CLAIMS)}
    contexts = [(dict(condition="full", checkpoint=50, ask=dict(id=f"q{i}")), None, None) for i in range(4)]
    check(results, contexts, Judge(batched), tmp_path)
    got = [list(results[f"full|q{i}"]["claims"].values())[0] for i in range(4)]
    assert got == ["in_ledger", "unconfirmed_source", "ledger_miss", "confirmed_invented"]
    assert [results[f"full|q{i}"]["invented"] for i in range(4)] == [0, 0, 0, 1]


def test_one_call_per_answer_and_identical_answers_share_it(tmp_path):
    """All of an answer's claims in one call (was one or two per claim: B's 19,063 claims); the
    same answer in another condition at the same checkpoint costs nothing more."""
    both = dict(status="graded", unsupported_claims=list(CLAIMS))
    results = {"full|q0": dict(both), "warm_vector|q0": dict(both, unsupported_claims=list(reversed(CLAIMS)))}
    contexts = [(dict(condition=c, checkpoint=50, ask=dict(id="q0")), None, None) for c in ("full", "warm_vector")]
    judge = Judge(batched)
    check(results, contexts, judge, tmp_path)
    assert len(judge.calls) == 1 and len(judge.calls[0][1]["CLAIMS"]) == 4
    assert results["full|q0"]["invented"] == results["warm_vector|q0"]["invented"] == 1


def test_a_reply_missing_a_claim_is_retried_never_filled_in(tmp_path):
    results = {"full|q0": dict(status="graded", unsupported_claims=list(CLAIMS))}
    contexts = [(dict(condition="full", checkpoint=50, ask=dict(id="q0")), None, None)]
    short = lambda system, body: dict(claims=batched(system, body)["claims"][:2])
    verdicts, pending = check(results, contexts, Judge(short), tmp_path)
    assert pending == 1 and not verdicts and set(results["full|q0"]["claims"].values()) == {"pending"}


def test_ranking_runs_both_orders_and_maps_back_to_conditions(tmp_path):
    contexts = [ctx("full", "f@50#0", "Tofu."), ctx("bare_vector", "f@50#0", "Miso."), ctx("warm_vector", "f@50#0", "Tofu.")]
    def route(system, body):
        return dict(ranks={a["label"]: (1 if "Tofu" in a["text"] else 2) for a in body["ANSWERS"]})
    judge = Judge(route)
    ranks, _ = G.rank(contexts, judge, tmp_path)
    r = ranks["f@50#0"]
    assert len(judge.calls) == 2 and r["order_agreement"] is True
    assert r["conditions"] == {"full": 1.0, "bare_vector": 2.0, "warm_vector": 1.0}
    first, second = (c[1]["ANSWERS"] for c in judge.calls)
    assert [a["text"] for a in first] == list(reversed([a["text"] for a in second]))


def test_controls_compare_with_recent_only_in_both_orders(tmp_path):
    contexts = [ctx("recent_only", "c1", "Lima.", kind="control", question="capital of peru"),
                ctx("full", "c1", "Lima.", kind="control", question="capital of peru"),
                ctx("bare_vector", "c1", "Lima, Peru's capital, on the coast.", kind="control", question="capital of peru")]
    judge = Judge(lambda system, body: dict(better="A" if "coast" in body["A"] else "B"))
    out, _ = G.controls(contexts, judge, tmp_path)
    assert out == {"full|c1": "same_reply", "bare_vector|c1": "A"}


def test_report_pairs_by_family_and_bootstraps_over_families():
    from experiments.z2 import report as Rp
    units = {f"{c}|f{i}@50#0": dict(type="update", kind="question", prompt_tokens=2000, memory_tokens=500, fragments=4)
             for c in ("full", "bare_vector", "warm_vector") for i in range(6)}
    graded = {}
    for i in range(6):
        graded[f"full|f{i}@50#0"] = dict(status="graded", grade="correct", score=5, items_covered=[0], items_total=1,
                                         superseded_used=False, abstained=False, abstention_expected=False, claims={})
        for base in ("bare_vector", "warm_vector"):
            graded[f"{base}|f{i}@50#0"] = dict(status="graded", grade="partial", score=3, items_covered=[], items_total=1,
                                              superseded_used=i == 0, abstained=False, abstention_expected=False, claims={})
    ranks = {f"f{i}@50#0": dict(status="ranked", conditions={"full": 1, "bare_vector": 2, "warm_vector": 2}) for i in range(6)}
    r = Rp.build("B", units, dict(graded=graded, ranks=ranks, controls={}))
    assert r["by_condition_and_type"]["full"]["update"]["mean_score"] == 5
    assert r["by_condition_and_type"]["full"]["update"]["tur"] == 2.5
    p = r["paired"]["full"]["bare_vector"]["update"]
    assert p == dict(families=6, mean_diff=2.0, ci95_over_families=[2.0, 2.0])
    assert r["ranking_win_rates"]["full vs bare_vector"] == {"win": 6}


def test_semantic_credit_judges_only_fragments_citing_gold_turns(tmp_path):
    unit = dict(condition="full", checkpoint=50, ask=dict(id="f@50#0", kind="question", question="cat?"),
                stage=dict(selected=[dict(leg="facts", text="The cat is Tofu.", source_turns=[9]),
                                     dict(leg="vector", text="unrelated", source_turns=[2])]))
    occ = dict(reference=dict(required=["The cat is called Tofu."], citations=[dict(turn=9)]))
    seen = []
    def judge(system, prompt):
        body = json.loads(prompt)
        seen.append(body)
        return dict(carried={"0": ["F1"]}, wrong=[])
    result, pending = G.semantic_credit([(unit, None, occ)], judge, tmp_path)
    assert [f["text"] for f in seen[0]["FRAGMENTS"]] == ["The cat is Tofu."] and not pending
    assert result["full|f@50#0"] == dict(gold_fragments=1, items=1, items_carried=1,
                                         carried_by_kind={"0": ["facts"]}, wrong=0,
                                         fragments_total=2, fragments_carrying=1,
                                         tokens_total=0, tokens_carrying=0, tokens_wrong=0)


def test_context_precision_is_a_token_share_and_plain_search_can_be_judged(tmp_path):
    """Z2 F27: plain search's pieces carried `turn` but no `source_turns` or text, so its precision
    was always 0; and a share of pieces favours whoever packs fewer, larger pieces."""
    from experiments.z2 import scoring as S
    occ = dict(reference=dict(required=["The cat is called Tofu."], citations=[dict(turn=9)]))
    bare = dict(condition="bare_vector", checkpoint=50, ask=dict(id="f@50#0", kind="question", question="cat?"),
                stage=dict(selected=[dict(turn=9, role="user", tokens=20, source_turns=[9], text="[User, turn 9] Tofu")]
                           + [dict(turn=t, role="user", tokens=20, source_turns=[t], text="x") for t in range(1, 5)]))
    ice = dict(condition="full", checkpoint=50, ask=dict(id="f@50#0", kind="question", question="cat?"),
               stage=dict(selected=[dict(leg="vector", tokens=20, source_turns=[9], text="Tofu"),
                                    dict(leg="vector", tokens=80, source_turns=[2], text="y")]))
    judge = lambda system, prompt: dict(carried={"0": ["F1"]}, wrong=[])
    result, _ = G.semantic_credit([(bare, None, occ), (ice, None, occ)], judge, tmp_path)
    assert result["bare_vector|f@50#0"]["tokens_carrying"] == 20 and result["bare_vector|f@50#0"]["tokens_total"] == 100
    # same tokens carrying, same budget: equal token precision, though piece shares differ 1/5 vs 1/2
    totals = {c: S.collections.Counter() for c in ("bare_vector", "full")}
    for key, r in result.items():
        c = key.split("|")[0]
        totals[c]["tokens"] += r["tokens_total"]; totals[c]["carrying"] += r["tokens_carrying"]
    assert totals["bare_vector"]["carrying"] / totals["bare_vector"]["tokens"] == totals["full"]["carrying"] / totals["full"]["tokens"]


def test_plain_search_reads_record_source_turns_and_text():
    import inspect
    from experiments.z2 import read_passes as R
    src = inspect.getsource(R.run)
    assert "source_turns=[c.turn], text=c.render()" in src


def test_claim_vectors_are_kept_for_a_restart(tmp_path):
    """B's 19,063 claims took ~40 CPU minutes to embed: a restart reuses them; other claims re-embed."""
    import numpy as np
    from experiments.z2 import grade as G
    calls = []
    embed = lambda texts: calls.append(len(texts)) or np.ones((len(texts), 3), dtype="float32")
    first = G.claim_vectors(tmp_path, ["a", "b"], embed)
    again = G.claim_vectors(tmp_path, ["a", "b"], embed)
    other = G.claim_vectors(tmp_path, ["a", "c", "d"], embed)
    assert calls == [2, 3] and first.shape == again.shape == (2, 3) and other.shape == (3, 3)



def test_evidence_is_taken_in_turn_and_capped():
    assert G.in_turn([[1, 2, 3], [4, 5], [1, 6]], 4) == [1, 4, 2, 5]
    assert G.in_turn([[1], [1]], 5) == [1]


def test_planned_calls_count_each_phase_and_claims_replace_the_estimate(tmp_path):
    contexts = [ctx("full", "f@50#0", "Tofu."), ctx("warm_vector", "f@50#0", "Tofu."),
                ctx("bare_vector", "f@50#0", "Miso."), ctx("recent_only", "f@50#0", "", finish="incomplete")]
    for unit, _, _ in contexts:
        unit["stage"] = dict(selected=[])
    plan = G.planned_calls(contexts)
    assert plan["grade"] == 2 and plan["rank"] == 2 and plan["claims_estimated"]      # Tofu./Miso.: two to grade, rank twice
    G.write_meter(tmp_path, **plan)
    G.write_meter(tmp_path, claims=7, claims_estimated=False)
    meter = json.loads((tmp_path / "meter.json").read_text())
    assert meter["claims"] == 7 and meter["grade"] == 2 and not meter["claims_estimated"]


def test_sources_carry_the_date_the_system_saw():
    # Z2 F12: undated SOURCES made every dated answer look invented to the judge.
    from experiments.z2 import grade as G
    ref = {"citations": [{"turn": 7, "role": "user", "quote": "we picked amber"},
                         {"turn": 9, "role": "assistant", "quote": None}]}
    assert G.sources(ref, {7: "2025-01-03 10:15"}) == [
        {"turn": 7, "role": "user", "date": "2025-01-03 10:15", "text": "we picked amber"}]
    assert G.sources(ref) == [{"turn": 7, "role": "user", "text": "we picked amber"}]


def test_grade_rubric_decides_on_required_only():
    from experiments.z2 import grade as G
    assert G.POLICY == "ice-z2-grade-2"
    assert "never lower the grade" in G.GRADE_SYSTEM and "against REQUIRED only" in G.GRADE_SYSTEM


def test_missing_clock_warns_and_grades_undated(capsys):
    from experiments.z2 import grade as G
    assert G.turn_dates("no-such-history") == {}
    assert "no turn dates for no-such-history" in capsys.readouterr().out


def test_turn_dates_are_utc_like_icess_labels():
    # C's clock is written at +08:00; the judge must see the UTC time ICE's prompts show (2026-10-29)
    from experiments.z2 import grade as G
    assert G.turn_dates("C")[1] == "2026-06-04 14:34 UTC"


def test_claim_passages_carry_dates_and_a_dated_claim_sees_its_turn(tmp_path):
    """Z2 F27: the claim check saw undated passages, so 60 of ICE's 63 dated claims marked invented
    on B named the exact minute of a real turn. Each passage now carries its date, and a claim
    naming a turn's minute (or day) is shown that turn even when the search would not find it."""
    import experiments.z2.questions as Q
    rows = [(t, role, f"message {t} {role}") for t in range(1, 41) for role in ("user", "assistant")]
    dates = {t: f"2025-01-{1 + t // 10:02d} {t % 10:02d}:15 UTC" for t in range(1, 41)}
    claim = "you first proposed the garden plan on [2025-01-03T07:15:00Z]"            # turn 27's minute
    results = {"full|q0": dict(status="graded", unsupported_claims=[claim])}
    contexts = [(dict(condition="full", checkpoint=40, ask=dict(id="q0")), None, None)]
    judge = Judge(lambda system, body: dict(claims=[dict(id="C1", stated_by=[], found=True, passage=0)]))
    vectors = np.zeros((len(rows), 4))
    vectors[:, 0] = 1                                                  # every passage ties
    original, Q.embed = Q.embed, lambda texts, device="cpu": np.ones((len(texts), 4))
    try:
        G.check_claims(results, contexts, LEDGER, None, (rows, vectors), judge, tmp_path, dates=dates)
    finally:
        Q.embed = original
    passages = judge.calls[0][1]["PASSAGES"]
    assert all(p["date"] == dates[p["turn"]] for p in passages)
    assert [p["turn"] for p in passages[:2]] == [27, 27]                # both halves of turn 27, first
    assert "date" in G.CLAIMS_SYSTEM and "with that date" in G.CLAIMS_SYSTEM


def test_control_judge_never_counts_memory_as_invented():
    """Z2 F27: the controls judge never sees the memory, and 17 of full ICE's 21 losses on B called
    true references "fabricated". It rates how the reply serves MESSAGE, both ways."""
    assert "remembers its earlier conversations" in G.CONTROL_SYSTEM
    assert "never count a mention" in G.CONTROL_SYSTEM and "unrelated to MESSAGE" in G.CONTROL_SYSTEM



def test_aux_check_plants_a_false_twin_at_a_time_no_turn_has():
    """The claim-check plants (Z2 F27): a true dated claim and the same claim nine days later."""
    from experiments.z2 import aux_judge_check as A
    dates = {1: "2025-01-09 05:30 UTC", 2: "2025-01-09 05:33 UTC"}
    graded = {"full|q1": dict(status="graded", unsupported_claims=["x"],
                              claims={"On [2025-01-09T05:30:00Z] you said the garden plan was ready": "confirmed_invented",
                                      "you like tea": "confirmed_invented"})}
    plants = A.claim_plants(graded, dates, 1.0)
    assert [(k, c) for k, _, c, _ in plants] == [("true", "On [2025-01-09T05:30:00Z] you said the garden plan was ready"),
                                                 ("false", "On [2025-01-18T05:30:00Z] you said the garden plan was ready")]
    assert A.PASS == dict(true_supported=0.9, false_invented=0.9, intrusion_clean_wins=0.8)


def test_a_true_plant_needs_its_content_in_the_dated_turn():
    """Run 1 (2026-11-03): a real timestamp on content from elsewhere is a genuine error — 33 of ICE's
    49 dated "inventions" on B — so a real date alone no longer makes a claim a true plant."""
    from experiments.z2 import aux_judge_check as A
    dates = {1: "2025-01-09 05:30 UTC", 2: "2025-01-09 05:33 UTC"}
    originals = [dict(turn_number=1, prompt="the garden plan is ready for spring planting", response="great"),
                 dict(turn_number=2, prompt="unrelated", response="unrelated")]
    graded = {"full|q1": dict(status="graded", unsupported_claims=["x"], claims={
        "On [2025-01-09T05:30:00Z] you said the garden plan was ready for spring": "confirmed_invented",
        "On [2025-01-09T05:33:00Z] you said the garden plan was ready for spring": "confirmed_invented"})}
    plants = A.claim_plants(graded, dates, 1.0, originals=originals)
    trues = [c for k, _, c, _ in plants if k == "true"]
    assert trues == ["On [2025-01-09T05:30:00Z] you said the garden plan was ready for spring"]   # 05:33's turn lacks it
