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
    plans = tmp_path / "plans"
    plans.mkdir()
    (plans / f"{p['id']}.asks.json").write_text("[]")                     # its questions (none needed here)
    monkeypatch.setattr(G, "PLANS", plans)
    tmp_path = tmp_path / "out"
    calls = {"user": 0}
    def user_llm(system, prompt):
        import json as j
        calls["user"] += 1
        must = prompt.split("THIS MESSAGE MUST SAY:\n")[1].split("\n\n")[0].splitlines()
        vals = " ".join(line.split(": ", 1)[1].rsplit(" (", 1)[0] for line in must if line.startswith("- "))
        hello = "".join(chr(97 + int(d)) for d in str(calls["user"]))   # never the same message twice
        return f"{hello} {vals}" if calls["user"] != 3 else "forgot it"  # third call misses -> retry
    def assistant_llm(system, messages):
        import re
        assert messages[-1]["role"] == "user"                          # the whole conversation, latest last
        return "ok " + " ".join(re.findall(r"- [^:]+: (.+)", messages[-2]["content"]))   # this turn's notes
    status = G.run(p, tmp_path, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None)
    assert status["turns"] == 50 and status["finished"]
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
    assert G.drift_problems(p, 35, "ok", "your budget was 2,000 euros before") == []    # the reply check's call now
    assert any("never-again" in x for x in G.drift_problems(p, 40, "back in Ostmere we", "nice"))
    assert G.drift_problems(p, 35, "ok", "your budget is 2,500 euros") == []
    msgs = G.assistant_messages([dict(user=f"u{i}", assistant=f"a{i}") for i in range(12)], "NOTES", "now?")
    assert msgs[0]["content"] == "u0" and msgs[-2] == dict(role="system", content="NOTES") and msgs[-1]["content"] == "now?"


def test_the_user_side_never_says_a_later_value_early_and_never_leaks_or_repeats():
    """A first value said early is how people talk; a changing fact's later value said before its
    turn would undo a planned change. Invented details are the checker's to judge."""
    from experiments.synth import generate as G
    plan = dict(facts=[dict(id="F5", values=[dict(turn=5, value="£58,000 a year")])])
    history = [dict(user="my salary is £44,800 a year", assistant="noted, 2 options")]
    assert G.invented_values(plan, 3, "offer is £58,000 a year!", history) == []        # a first value early: fine
    later = dict(facts=[dict(id="F7", values=[dict(turn=1, value="£40,000 a year"), dict(turn=8, value="£52,000 a year")])])
    assert G.invented_values(later, 3, "now its £52,000 a year", history)[0].startswith("user said F7's later value early")
    assert G.invented_values(plan, 3, "rent here is like 950 a month", history) == []    # a realistic detail


def test_future_values_are_hidden_from_the_beats():
    from experiments.synth import generate as G
    plan = dict(facts=[dict(id="F1", about="the game's name", values=[dict(turn=4, value="Ferryman's Ledger")])])
    beats = ["pitches the game, Ferryman's Ledger, to the shop"]
    assert G.hide_future(plan, 2, beats) == ["pitches the game, …, to the shop"]
    assert G.hide_future(plan, 5, beats) == beats



def test_the_persona_is_hidden_too_and_an_echoed_prompt_is_rejected():
    from experiments.synth import generate as G
    plan = dict(facts=[dict(id="F1", about="race day", values=[dict(turn=3, value="22 June 2025")])])
    persona = dict(name="Imogen", situation="training for a half on 22 June 2025", age=38)
    assert G.hide_future(plan, 1, persona) == dict(name="Imogen", situation="training for a half on …", age=38)
    assert G.invented_values(plan, 1, '{"PERSONA": {"name": "Imogen"}}', []) == ["user side echoed its instructions"]


def test_a_long_value_may_be_said_in_other_words_if_its_key_parts_are_there():
    from experiments.synth import generate as G
    v = "Ostmere, a lake town up north; she moved away at age 11"
    assert G.key_parts(v) == {"Ostmere", "11"}
    assert G.said("grew up in ostmere, little lake town, left when i was 11", v)
    assert not G.said("grew up in ostmere, left when i was 12", v)            # 11 is a key part
    assert G.said("allergic to penicillin, i get hives", "penicillin, which gives her hives")
    assert not G.said("i love cilantro", "dislikes coriander, picks it out of everything")
    assert not G.said("start is 1 February", "Monday 1 February 2027")        # every part, whole words
    assert not G.said("bristol", "Bristol city")                              # short values stay word for word



def test_a_repeated_user_message_is_regenerated():
    from experiments.synth import generate as G
    history = [dict(user="hey so i got a verbal offer, can u help me plan", assistant="sure")]
    assert G.invented_values(dict(facts=[]), 2, "hey so i got a verbal offer, can u help me plan", history) == [
        "user side repeated an earlier message"]



def test_json_replies_and_named_instructions_are_never_messages():
    from experiments.synth import generate as G
    for leaked in ('{"messages": [{"content": "ok so", "role": "user"}]}', '{"content": "ok"}',
                   "MUST_SAY is empty, could you paste the last reply", "WHO YOU ARE: Ana"):
        assert G.invented_values(dict(facts=[]), 2, leaked, []) == ["user side echoed its instructions"], leaked
    assert G.invented_values(dict(facts=[]), 2, "here's the log:\n```\n{\"a\": 1}\n```", []) == []


def test_the_checker_reports_problems_and_never_passes_an_unreadable_verdict(monkeypatch):
    from experiments.synth import generate as G
    replies = iter(['{"problems": [{"kind": "invented", "quote": "Tideways", "detail": "a title for the game\'s name", "is_problem": true}]}',
                    '{"problems": [{"kind": "not_a_message", "quote": "reversed? no", "detail": "unsure", "is_problem": true}]}',
                    '{"problems": [{"kind": "wrong", "quote": "ok", "detail": "is correct, no error", "is_problem": false}]}',
                    "not json"])
    monkeypatch.setattr(G, "cloud_caller", lambda role, **_: lambda system, prompt: next(replies))
    check = G.checker_caller("deepseek")
    assert check("name: Ana", [], [], "my game Tideways") == ['checker invented: "Tideways" a title for the game\'s name']
    assert check("name: Ana", [], [], "can u remind me friday") == []        # its quote is not in the message
    assert check("name: Ana", [], [], "ok") == []                            # its own verdict: fine
    assert check("name: Ana", [], [], "ok") == ["checker reply unreadable: JSONDecodeError"]



def test_numbers_of_future_values_are_hidden_from_the_persona_and_filler_is_not_a_key_part():
    from experiments.synth import generate as G
    plan = dict(facts=[dict(id="F6", about="Oaxaca budget", values=[dict(turn=9, value="2400 usd all in, flights included")])])
    persona = dict(situation="planning Oaxaca on a 2,400 USD budget, 3 friends")
    assert G.hide_future(plan, 2, persona) == dict(situation="planning Oaxaca on a … USD budget, 3 friends")
    assert G.key_parts("about 90 minutes per game") == {"90", "minutes", "game"}



def test_a_turn_that_never_verifies_stops_the_conversation_and_a_repair_is_used(tmp_path, monkeypatch):
    """Round-4 review: a failed turn left a blank reply the next turns talked past. Now it is never
    kept — the run stops there; rerunning tries again; a hand-written repair is used as the turn."""
    import json
    from experiments.synth import generate as G
    monkeypatch.setattr(G, "validate", lambda p: [])
    monkeypatch.setattr(G, "MAX_TRIES", 2)
    p = plan()
    calls = {"n": 0}

    def user_llm(system, prompt):
        calls["n"] += 1
        must = prompt.split("THIS MESSAGE MUST SAY:\n")[1].split("\n\n")[0].splitlines()
        vals = " ".join(line.split(": ", 1)[1].rsplit(" (", 1)[0] for line in must if line.startswith("- "))
        hello = "".join(chr(97 + int(d)) for d in str(calls["n"]))
        return "forgot it" if "2,000 euros" in vals else f"{hello} {vals}"    # turn 3 never verifies
    assistant_llm = lambda system, messages: "ok " + " ".join(__import__("re").findall(r"- [^:]+: (.+)", messages[-2]["content"]))
    out = tmp_path / "out"
    status = G.run(p, out, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None, rollback=0)
    assert status == dict(plan=p["id"], turns=50, stopped_at=3, problems=status["problems"], finished=False)
    kept = [json.loads(l)["turn"] for l in (out / "turns.jsonl").read_text().splitlines()]
    assert kept == [1, 2] and json.loads((out / "stopped.json").read_text())["turn"] == 3
    (out / "repair-3.json").write_text(json.dumps(dict(user="my budget is 2,000 euros", assistant="noted")))
    status = G.run(p, out, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None, rollback=0)
    assert status["finished"] and status["repaired"] == [3]
    rows = [json.loads(l) for l in (out / "turns.jsonl").read_text().splitlines()]
    assert rows[2]["repaired"] and rows[2]["user"] == "my budget is 2,000 euros" and not (out / "stopped.json").exists()


def test_an_item_said_in_other_words_goes_to_the_checker_and_replies_are_checked():
    """Round-4 review: a date in another format and a rejection in other words failed the exact
    check. Such items go to the checker, which judges meaning; then the reply is checked too."""
    from experiments.synth import generate as G
    seen = []

    def checker(known, recent, must_say, message, unconfirmed=(), reply=None, upcoming=(), suggest=()):
        seen.append("reply" if reply is not None else [i["value"] for i in unconfirmed])
        return []
    items = dict(user=[dict(id="F4", about="passport appointment", value="thursday 10 april 2025 at 9:10am", change="introduce",
                            status="decided", paste=False)], assistant=[], quiet=False)
    plan = dict(id="T", facts=[], persona=dict(name="Ana"), sessions=[dict(index=1, pairs=3, style="casual", beats=["a"], topics=["t"])])
    session = plan["sessions"][0]
    row = G.generate_turn(plan, 1, session, [], items, user_llm=lambda s, p: "passport is 10 April 2025, 9:10",
                          assistant_llm=lambda s, m: "got it", sessions={1: 1, 2: 1, 3: 1}, checker=checker)
    assert row["verified"] and seen == [["thursday 10 april 2025 at 9:10am"], "reply"]



def test_a_planned_suggestion_has_the_user_ask_about_its_topic_never_its_value():
    from experiments.synth import generate as G
    seen = {}

    def user_llm(system, prompt):
        seen["prompt"] = prompt
        return "any ideas for a company name?"
    items = dict(user=[], assistant=[dict(id="F5", about="company name Tidewater Tabletop", value="Tidewater Tabletop",
                                          change="introduce", status="rejected", paste=False)], quiet=False)
    plan = dict(id="T", facts=[], persona=dict(name="Ruth"), sessions=[dict(index=1, pairs=3, style="casual", beats=["a"], topics=["t"])])
    G.generate_turn(plan, 1, plan["sessions"][0], [], items, user_llm=user_llm,
                    assistant_llm=lambda s, m: "how about Tidewater Tabletop?", sessions={1: 1, 2: 1, 3: 1})
    assert "ALSO ASK THE ASSISTANT FOR ITS IDEAS ON: company name" in seen["prompt"]
    assert "Tidewater" not in seen["prompt"]



def test_the_history_window_moves_in_blocks_so_the_prompt_start_stays_cached(monkeypatch):
    """Round 6: every model sees ~HISTORY_TOKENS of the conversation; its start moves a block at a
    time, so between moves each prompt starts with the previous prompt's bytes."""
    from experiments.synth import generate as G
    monkeypatch.setattr(G, "HISTORY_TOKENS", 100)
    monkeypatch.setattr(G, "HISTORY_BLOCK", 5)
    pairs = [dict(user="x" * 20, assistant="y" * 20) for _ in range(40)]          # 10 tokens a pair
    assert len(G.window(pairs[:8])) == 8                                       # all of it fits
    starts = [len(pairs[:n]) - len(G.window(pairs[:n])) for n in range(8, 41)]
    assert starts == sorted(starts) and set(starts) <= {0, 5, 10, 15, 20, 25, 30}  # forward, a block at a time
    a, b = G.transcript(G.window(pairs[:21])), G.transcript(G.window(pairs[:22]))
    assert b.startswith(a)                                                     # same start: the cache holds


def test_a_suggestion_is_turned_down_only_once_the_user_has_and_placeholders_are_caught():
    from experiments.synth import generate as G
    f = dict(id="F5", status="rejected", values=[dict(turn=6, value="Tidewater Tabletop")], mentions=[7])
    assert G.attribution(f, 7) == "you suggested" and G.attribution(f, 8) == "you suggested and the user turned down"
    assert G.artifacts("that leaves [some weeks] before the move") == ["unfinished placeholder: [some weeks]"]
    assert G.artifacts("see [the docs](https://x.y) and ```\n[section]\n```") == []


def test_every_rejected_attempt_is_logged_with_its_reason(tmp_path, monkeypatch):
    import json
    from experiments.synth import generate as G
    monkeypatch.setattr(G, "validate", lambda p: [])
    p = plan()
    plans = tmp_path / "plans"; plans.mkdir(); (plans / f"{p['id']}.asks.json").write_text("[]")
    monkeypatch.setattr(G, "PLANS", plans)
    calls = {"n": 0}

    def user_llm(system, prompt):
        calls["n"] += 1
        must = prompt.split("THIS MESSAGE MUST SAY:\n")[1].split("\n\n")[0].splitlines()
        vals = " ".join(line.split(": ", 1)[1].rsplit(" (", 1)[0] for line in must if line.startswith("- "))
        hello = "".join(chr(97 + int(d)) for d in str(calls["n"]))
        return f"{hello} {vals} [some weeks]" if calls["n"] == 2 else f"{hello} {vals}"
    assistant_llm = lambda system, messages: "ok " + " ".join(__import__("re").findall(r"- [^:]+: (.+)", messages[-2]["content"]))
    G.run(p, tmp_path / "out", user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None, limit=3)
    log = [json.loads(l) for l in (tmp_path / "out" / "attempts.jsonl").read_text().splitlines()]
    assert log[0]["stage"] == "user" and log[0]["problems"] == ["unfinished placeholder: [some weeks]"]



def test_a_later_value_equal_to_one_already_due_is_not_early():
    """Batch 6: rent £1,150 (turn 34) -> £1,250 (96) -> £1,150 (121); saying £1,150 at 34 is right."""
    from experiments.synth import generate as G
    rent = dict(facts=[dict(id="F25", values=[dict(turn=34, value="£1,150 a month"), dict(turn=96, value="£1,250 a month"),
                                              dict(turn=121, value="£1,150 a month")])])
    assert G.invented_values(rent, 34, "budget is £1,150 a month", []) == []
    assert G.invented_values(rent, 40, "now £1,250 a month", [])[0].startswith("user said F25's later value early")



def test_a_described_suggestion_is_checked_by_its_key_parts():
    from experiments.synth import generate as G
    v = "assistant suggested brown rice for Kit's dinners; rejected, he won't eat it"
    assert G.key_parts(v) == {"brown", "rice", "Kit's", "dinners"}
    assert G.said("what about brown rice for Kit's dinners this week?", v)


def test_future_beats_stay_hidden_and_an_invite_never_names_the_suggestion():
    from experiments.synth import generate as G
    item = dict(about="Bugsnag suggested for Pebble crash reporting", value="assistant suggested Bugsnag; user picked Sentry")
    assert "bugsnag" not in G.invite_topic(item).lower() and "crash reporting" in G.invite_topic(item)
    seen = {}

    def user_llm(system, prompt):
        seen["p"] = prompt
        return "how far should my long runs go"
    plan = dict(id="T", persona=dict(name="Imogen"), facts=[dict(id="F9", about="long-run cap", speaker="user", status="decided",
                                                                values=[dict(turn=3, value="19 km peak"), dict(turn=9, value="16 km now")])],
                sessions=[dict(index=1, pairs=10, style="casual", topics=["t"], beats=["sets the long-run cap at 19 km peak",
                                                                                     "drops the cap to 16 km now after the physio"])])
    G.generate_turn(plan, 8, plan["sessions"][0], [dict(user="u", assistant="a")] * 7,
                    dict(user=[], assistant=[], quiet=False), user_llm=user_llm, assistant_llm=lambda s, m: "ok",
                    sessions={t: 1 for t in range(1, 11)})
    assert "drops the cap" not in seen["p"] and "sets the long-run cap at 19 km peak" in seen["p"]



def test_a_one_word_later_value_is_early_only_beside_its_topic():
    """S8: "£900" said of something else stopped the conversation; the furniture budget's later £900."""
    from experiments.synth import generate as G
    budget = dict(facts=[dict(id="F18", about="furniture budget", values=[dict(turn=195, value="£1,500"), dict(turn=229, value="£900")])])
    assert G.invented_values(budget, 97, "rent there is about £900 a month", []) == []
    assert G.invented_values(budget, 200, "furniture can be £900 tops now", [])[0].startswith("user said F18's later value early")



def test_untold_events_are_unsettled_topics_and_generic_words_are_not_a_topic():
    from experiments.synth import generate as G
    plan = dict(facts=[], events=[dict(id="E13", what="paid the Oaxaca flight deposit, 380 usd, on the travel card",
                                       told_turn=112, relative_phrase="tomorrow", date="2025-03-21")])
    assert G.upcoming(plan, 107) == ["paid the Oaxaca flight deposit, on the travel card"] and G.upcoming(plan, 113) == []
    budget = dict(facts=[dict(id="F18", about="furniture budget", values=[dict(turn=195, value="£1,500"), dict(turn=229, value="£900")])])
    assert G.invented_values(budget, 97, "the budget gives roughly £900 more headroom", []) == []



def test_full_run_stops_s1_s3_s7():
    from experiments.synth import generate as G
    bus = dict(id="F19", about="Bristol bus pass", values=[dict(turn=220, value="£60 a month bus pass"),
                                                          dict(turn=262, value="no bus pass, cycling instead")])
    assert G.invented_values(dict(facts=[bus]), 220, "do i even need a bus pass at all", []) == []
    assert G.invented_values(dict(facts=[bus]), 220, "no bus pass for me, cycling instead", [])[0].startswith("user said F19")
    ver = dict(id="F40", about="release version", speaker="user", status="decided", values=[dict(turn=180, value="v0.9.4")])
    due = dict(user=[dict(id="F41", about="release date", value="Friday 2026-10-09 for v0.9.4")], assistant=[])
    assert G.drift_problems(dict(facts=[ver]), 185, "the release date for v0.9.4", "ok", due) == []
    assert G.drift_problems(dict(facts=[ver]), 185, "back on v0.9.4", "ok", dict(user=[], assistant=[])) == []   # 5 turns on
    assert G.drift_problems(dict(facts=[ver]), 260, "back on v0.9.4", "ok", dict(user=[], assistant=[]))          # 80 turns on
    name = dict(id="F27", about="victim's name", values=[dict(turn=78, value="the victim is Enid Harlow"),
                                                         dict(turn=207, value="Edith Harlow, I typed Enid by mistake")])
    assert G.hide_future(dict(facts=[name]), 201, ["realises it is Edith Harlow, not Enid"]) == ["realises it is …, not Enid"]



def test_a_retry_is_told_why_and_never_given_a_later_value():
    from experiments.synth import generate as G
    fb = G.feedback(["checker contradiction: \"£1,150\" the rent is £1,250", "user said F19's later value early: no bus pass, cycling instead"])
    assert fb[0].startswith("contradiction:") and "cycling" not in fb[1]
    text = G.user_prompt(dict(name="Ana"), "casual", [], None, [], [], None, rejected=fb)
    assert "YOUR PREVIOUS DRAFT WAS NOT USED" in text and "the rent is £1,250" in text


def test_a_stuck_turn_steps_back_once_then_stops(tmp_path, monkeypatch):
    import json
    from experiments.synth import generate as G
    monkeypatch.setattr(G, "validate", lambda p: [])
    monkeypatch.setattr(G, "MAX_TRIES", 1)
    p = plan()
    plans = tmp_path / "plans"; plans.mkdir(); (plans / f"{p['id']}.asks.json").write_text("[]")
    monkeypatch.setattr(G, "PLANS", plans)
    calls = {"n": 0}

    def user_llm(system, prompt):
        calls["n"] += 1
        must = prompt.split("THIS MESSAGE MUST SAY:\n")[1].split("\n\n")[0].splitlines()
        vals = " ".join(line.split(": ", 1)[1].rsplit(" (", 1)[0] for line in must if line.startswith("- "))
        hello = "".join(chr(97 + int(d)) for d in str(calls["n"]))
        return "forgot it" if "TransEuropa" in prompt or "2,000 euros" in vals else f"{hello} {vals}"
    assistant_llm = lambda system, messages: "ok " + " ".join(__import__("re").findall(r"- [^:]+: (.+)", messages[-2]["content"]))
    out = tmp_path / "out"
    first = G.run(p, out, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None, rollback=2)
    assert first["rolled_back_to"] == 1 and (out / "turns.jsonl").read_text() == ""          # stuck at 3: back to 1
    assert json.loads((out / "rollbacks.jsonl").read_text())["stuck_turn"] == 3
    second = G.run(p, out, user_llm=user_llm, assistant_llm=assistant_llm, say=lambda *_: None, rollback=2)
    assert second["stopped_at"] == 3 and not second["finished"]                               # once per turn, then stop
