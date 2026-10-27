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
                                         fragments_total=2, fragments_carrying=1)


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
