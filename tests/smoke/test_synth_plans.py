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
    def assistant_llm(system, prompt):
        import json as j
        return "ok " + " ".join(i["value"] for i in j.loads(prompt)["MUST_SUGGEST"])
    status = G.run(p, tmp_path, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None)
    assert status["turns"] == 50 and not status["unverified"]
    rows = j_rows = __import__("json").loads((tmp_path / "originals.json").read_text())
    assert "2,000 euros" in rows[2]["prompt"] and "TransEuropa" in rows[4]["response"]
    assert "TransEuropa" not in rows[4]["prompt"]                      # a suggestion is the assistant's
    before = calls["user"]
    G.run(p, tmp_path, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None)
    assert calls["user"] == before                                      # resumed: nothing regenerated
