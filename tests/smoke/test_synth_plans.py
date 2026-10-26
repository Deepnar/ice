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
