"""FINAL synthetic plans: the validator accepts a consistent plan and names each kind of
impossibility that would corrupt a computed answer key."""

import copy

from experiments.synth import plans as P

BRIEF = dict(pairs=50, quotas=dict(facts=2, changed_share=0.5, corrections=1))
TAGS = {"topic": {"Admin_&_Productivity"}, "intent": {"Strategic_Planning"}}


def plan():
    s = lambda i, d: dict(index=i, date=d, pairs=25, style="casual", topics=["Admin_&_Productivity"],
                          intents=["Strategic_Planning"], beats=["plans a move"])
    return dict(id="T1", persona=dict(name="Ana", situation="moving cities", voice="casual"),
                sessions=[s(1, "2025-03-01"), s(2, "2025-03-04")],
                facts=[dict(id="F1", about="moving budget", speaker="user", status="decided",
                            values=[dict(turn=3, value="2,000 euros", change="introduce"),
                                    dict(turn=30, value="2,500 euros", change="correction")], mentions=[40]),
                       dict(id="F2", about="moving company", speaker="assistant", status="suggested",
                            values=[dict(turn=5, value="TransEuropa", change="introduce")])],
                events=[dict(id="E1", what="viewed a flat", date="2025-02-26", told_turn=28,
                             relative_phrase="last wednesday")],
                no_memory_turns=[10, 20, 35, 45, 48])


def problems(p):
    return P.validate(p, BRIEF, TAGS)


def test_a_consistent_plan_passes():
    assert problems(plan()) == []


def test_each_impossibility_is_named():
    cases = {
        "values out of order": lambda p: p["facts"][0]["values"].reverse(),
        "suggested said by the user": lambda p: p["facts"][1].update(speaker="user"),
        "mention before introduce": lambda p: p["facts"][0].update(mentions=[2]),
        "pairs do not sum": lambda p: p["sessions"][1].update(pairs=20),
        "dates go backwards": lambda p: p["sessions"][1].update(date="2025-02-01"),
        "unknown style": lambda p: p["sessions"][0].update(style="shouty"),
        "too few quiet turns": lambda p: p.update(no_memory_turns=[1]),
        "quota: corrections": lambda p: p["facts"][0]["values"][1].update(change="update"),
    }
    for name, break_it in cases.items():
        p = copy.deepcopy(plan())
        break_it(p)
        assert problems(p), name


def test_style_twin_changes_only_the_style():
    src = plan()
    twin = P.style_twin(src, "T2", "polished")
    assert twin["id"] == "T2" and all(s["style"] == "polished" for s in twin["sessions"])
    assert twin["facts"] == src["facts"] and twin["events"] == src["events"]
    assert [s["date"] for s in twin["sessions"]] == [s["date"] for s in src["sessions"]]
    assert src["sessions"][0]["style"] == "casual"                    # the source is untouched


def test_keys_answer_as_of_each_checkpoint():
    from experiments.synth import keys as K
    p = plan()
    occ = K.occurrences(p)
    f1 = [o for o in occ if o["family"] == "T1:F1"]
    assert {o["type"] for o in f1} == {"update"}
    at = {o["checkpoint_turn"]: o["reference"]["required"] for o in f1}
    assert at[50] == ["2,500 euros"]                          # after the correction
    later = [o for o in f1 if o["checkpoint_turn"] == 50][0]
    assert later["reference"]["superseded"] == ["2,000 euros"]
    f2 = [o for o in occ if o["family"] == "T1:F2"]
    assert f2 and f2[0]["type"] == "attribution" and f2[0]["reference"]["speaker"] == "assistant"
    assert K.reference(p["facts"][0], 2)[1] is True           # asked before it is said -> abstain


def test_generator_plants_verifies_retries_and_resumes(tmp_path, monkeypatch):
    from experiments.synth import generate as G
    monkeypatch.setattr(G, "validate", lambda p: [])
    p = plan()
    calls = {"user": 0}
    def user_llm(system, prompt):
        import json as j
        calls["user"] += 1
        body = j.loads(prompt)
        vals = " ".join(i["value"] for i in body["MUST_SAY"])
        return f"hi {vals}" if calls["user"] != 3 else "forgot it"      # third call misses -> retry
    def assistant_llm(system, messages):
        import re
        assert messages[-1]["role"] == "user"                          # the whole conversation, latest last
        return "ok " + " ".join(re.findall(r"- [^:]+: (.+)", system))
    status = G.run(p, tmp_path, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None)
    assert status["turns"] == 50 and not status["unverified"]
    rows = j_rows = __import__("json").loads((tmp_path / "originals.json").read_text())
    assert "2,000 euros" in rows[2]["prompt"] and "TransEuropa" in rows[4]["response"]
    assert "TransEuropa" not in rows[4]["prompt"]                      # a suggestion is the assistant's
    before = calls["user"]
    G.run(p, tmp_path, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None)
    assert calls["user"] == before                                      # resumed: nothing regenerated


def test_a_rejection_happens_once_at_its_first_mention():
    from experiments.synth import generate as G
    p = plan()
    p["facts"][1].update(status="rejected", mentions=[9])
    d = G.due(p)
    rejects = [(t, i) for t, slot in d.items() for i in slot["user"] if i["change"] == "reject"]
    assert [t for t, _ in rejects] == [9]
    assert not any(i["change"] == "mention" and i["id"] == "F2" for s in d.values() for i in s["user"])


def test_a_style_twin_asks_exactly_its_sources_questions():
    from experiments.synth import keys as K
    src = plan()
    twin = P.style_twin(src, "T2")
    a = [(o["family"].split(":")[1], o["checkpoint_turn"], o["reference"]["required"]) for o in K.occurrences(src)]
    b = [(o["family"].split(":")[1], o["checkpoint_turn"], o["reference"]["required"]) for o in K.occurrences(twin)]
    assert a == b


def test_phrasing_check_catches_missing_leaking_and_said_negatives():
    from experiments.synth import keys as K, phrase as Q
    p = plan()
    occ = K.occurrences(p)
    good = {"questions": {Q.key(o): "what's the latest on that thing we talked about" for o in occ},
            "negatives": [{"question": f"did i ever mention my boat {i}", "about": f"boat {i}"} for i in range(8)]}
    assert Q.check(p, occ, good) == []
    leak = dict(good, questions={k: "is it still 2,500 euros?" for k in good["questions"]})
    assert any("contains its answer" in x for x in Q.check(p, occ, leak))
    said = dict(good, negatives=good["negatives"][:7] + [{"question": "the budget?", "about": "moving budget"}])
    assert any("does say" in x for x in Q.check(p, occ, said))
    asks = Q.asks(p, occ, good)
    assert sum(a["type"] == "negative" for a in asks) == 8 and all(a["expected_abstention"] for a in asks if a["type"] == "negative")


def test_plan_ledger_chains_changes_and_final_occurrences_add_negatives():
    from experiments.synth import keys as K, phrase as Q
    from experiments.z2 import ledger_merge as M
    p = plan()
    chained = {e["id"]: e for e in M.chain(Q.plan_ledger(p)["entries"])}
    assert chained["T1:F1.0"]["valid_until_turn"] == 30 and chained["T1:F1.0"]["superseded_by"] == "T1:F1.1"
    occ = K.occurrences(p)
    phr = {"questions": {Q.key(o): "how's that going" for o in occ},
           "negatives": [{"question": "boat?", "about": f"boat {i}"} for i in range(8)]}
    final = Q.final_occurrences(p, occ, phr)
    assert sum(o["type"] == "negative" for o in final) == 8 and all(o["question"] for o in final)


def test_briefing_holds_only_local_current_attributed_facts():
    from experiments.synth import generate as G
    p = plan()
    p["facts"].append(dict(id="F3", about="childhood town", speaker="user", status="decided",
                           values=[dict(turn=4, value="Ostmere", change="introduce")]))
    assert "TransEuropa" not in G.briefing(p, 6)                                 # said once, never again
    p["facts"][1]["mentions"] = [20]
    early = G.briefing(p, 6)
    assert "moving budget: 2,000 euros (the user decided)" in early
    assert "TransEuropa (you suggested)" in early and "Ostmere" not in early      # never-again drops out
    later = G.briefing(p, 31)
    assert "2,500 euros" in later and "2,000 euros" not in later                 # current value only
    assert "moving budget" not in G.briefing(p, 2)                               # nothing ahead


def test_drift_checks_catch_outdated_values_and_reappearing_retention_facts():
    from experiments.synth import generate as G
    p = plan()
    p["facts"].append(dict(id="F3", about="childhood town", speaker="user", status="decided",
                           values=[dict(turn=4, value="Ostmere", change="introduce")]))
    assert any("outdated" in x for x in G.drift_problems(p, 35, "ok", "your budget is 2,000 euros"))
    assert any("never-again" in x for x in G.drift_problems(p, 40, "back in Ostmere we", "nice"))
    assert G.drift_problems(p, 35, "ok", "your budget is 2,500 euros") == []
    msgs = G.recent_messages([dict(user=f"u{i}", assistant=f"a{i}") for i in range(12)], "now?")
    assert len(msgs) == 17 and msgs[0]["content"] == "u4" and msgs[-1]["content"] == "now?"
