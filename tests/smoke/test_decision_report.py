"""FINAL revision 5 report (experiments/final/decision_report.py): MDM, MDF, NMH and POS from the
read index — checked on a hand-built history."""
from experiments.final import decision_report as D


def row(kind, given, cp, tokens, mem=0, seconds=None, type_="fact_recall"):
    return dict(kind=kind, type=type_, memory_given=given, checkpoint_turn=cp, prompt_tokens=tokens,
                memory_tokens=mem, read_seconds=seconds)


INDEX = {
    "full_default|q1": row("question", True, 10, 1000, seconds=20.0),
    "full_default|q2": row("question", False, 15, 400, seconds=0.5, type_="temporal"),
    "full_default|q3": row("question", True, 95, 3000, seconds=22.0),
    "full_default|c1": row("control", True, 12, 2000, mem=1800, seconds=21.0),
    "full_default|c2": row("control", False, 90, 300, seconds=0.4),
    "bare_vector|q1": row("question", None, 10, 1500), "bare_vector|q2": row("question", None, 15, 1500),
    "bare_vector|q3": row("question", None, 95, 1500),
}


def test_misses_false_alarms_and_what_a_skipped_read_saves():
    d = D.decision(INDEX, "full_default")
    assert d["questions"] == 3 and d["mdm"] == round(1 / 3, 4) and d["mdm_by_type"]["temporal"]["mdm"] == 1.0
    assert d["controls"] == 2 and d["mdf"] == 0.5 and d["mdf_memory_tokens"] == 1800
    assert d["read_seconds_fetching"] == 21.0 and d["read_seconds_skipping"] == 0.5


def test_harm_is_the_share_where_the_no_memory_reply_won():
    h = D.harm({"full_default|c1": "B", "full_default|c2": "A", "mem0|c1": "B", "recent_only|c1": "A",
                "full_default|c3": "pending"})
    assert h["full_default"] == dict(n=2, nmh=0.5, better=0.5) and h["mem0"]["nmh"] == 1.0 and "recent_only" not in h


def test_position_pairs_tokens_by_quarter_of_the_history():
    outcomes = {("full_default", "q1"): dict(m2=True), ("full_default", "q2"): dict(m2=False),
                ("full_default", "q3"): dict(m2=True)}
    pos = D.position(INDEX, outcomes, 100, "full_default")
    first, last = pos["Q1"], pos["Q4"]
    assert first["questions"] == 2 and first["sca"] == 0.5 and first["mdm"] == 0.5 and first["mdf"] == 1.0
    assert first["tokens_vs_bare_vector"]["mean_delta"] == -800 and first["tokens_vs_bare_vector"]["ice_cheaper"] == 1.0
    assert last["tokens_vs_bare_vector"]["mean_delta"] == 1500 and last["mdf"] == 0.0
    assert first["tokens_vs_long_context"] is None
