"""G9: the leg-weight resolver's mechanism — deliberately NOT its tuned values.

Z1 stage 2 exists to change the numbers in `settings.retrieval_leg_*`. A test
that pinned them here would go red the moment Z1 does its job, and TRAPS #7 is
explicit that pinning ground a scheduled item is about to move is a mistake.
So this suite checks the blend arithmetic, the validation, and the invariants
that must hold at ANY setting — the frozen values live in
tests/test_settings_freeze.py, which is a G9 artifact with a defined end.

No DB, no model: pure settings in, weights out.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.retrieval import leg_weights  # noqa: E402


@pytest.fixture
def table(monkeypatch):
    """A small synthetic table, so the checks do not depend on tuned values."""
    monkeypatch.setattr(settings, "retrieval_leg_base_weights",
                        {"bm25": 1.0, "vector": 1.0, "codex": 1.0,
                         "procedural": 1.0, "timeline": 0.5})
    monkeypatch.setattr(settings, "retrieval_leg_profiles", {
        "Factual_Retrieval": {"vector": 2.0, "bm25": 0.0, "codex": 0.0, "procedural": 0.0},
        "Troubleshooting": {"vector": 0.0, "bm25": 0.0, "codex": 0.0, "procedural": 4.0},
    })
    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides",
                        {"Software_&_Tech": {"procedural": 1.0}})


def test_single_intent_uses_its_own_row(table):
    w = leg_weights.resolve(["Factual_Retrieval"], [])
    assert w["vector"] == 2.0
    assert w["procedural"] == 0.0


def test_two_intents_average(table):
    """Each active intent contributes equally — the halving is the whole design."""
    w = leg_weights.resolve(["Factual_Retrieval", "Troubleshooting"], [])
    assert w["vector"] == 1.0      # (2.0 + 0.0) / 2
    assert w["procedural"] == 2.0  # (0.0 + 4.0) / 2


def test_intent_without_a_row_contributes_base(table):
    w = leg_weights.resolve(["Ideation"], [])
    assert w["vector"] == 1.0
    assert w["bm25"] == 1.0


def test_no_intents_falls_back_to_base(table):
    assert leg_weights.resolve([], []) == {"bm25": 1.0, "vector": 1.0, "codex": 1.0,
                                           "procedural": 1.0, "timeline": 0.5}


def test_topic_override_is_cumulative_on_top(table):
    plain = leg_weights.resolve(["Factual_Retrieval"], [])
    bumped = leg_weights.resolve(["Factual_Retrieval"], ["Software_&_Tech"])
    assert bumped["procedural"] == plain["procedural"] + 1.0


def test_timeline_is_pinned_not_blended(table):
    """T4: its firing condition is the gate, so intent must not move its weight."""
    for intents in ([], ["Factual_Retrieval"], ["Troubleshooting", "Ideation"]):
        assert leg_weights.resolve(intents, [])["timeline"] == 0.5


def test_weights_never_go_negative(table, monkeypatch):
    """The clamp, in isolation from the all-zero guard.

    This used to drive `Factual_Retrieval`, whose row in the fixture above
    already zeroes every leg but vector — so negating vector collapsed the
    whole set and the case now belongs to the all-zero guard instead. Driving
    an intent with NO row (base weights across the board) leaves the other legs
    standing, which is what isolates the clamp. The control line is the point:
    without it this passes whether the clamp works or the guard silently
    replaced the entire weight set.
    """
    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides",
                        {"Software_&_Tech": {"vector": -99.0}})
    w = leg_weights.resolve(["Ideation"], ["Software_&_Tech"])
    assert w["vector"] == 0.0
    assert w["bm25"] == 1.0  # control: the untouched legs survive


def test_all_zero_is_refused_and_falls_back(table, monkeypatch):
    """An all-zero weight set is unrankable, not restrictive.

    `_apply_rrf` registers a leg's fragments regardless of its weight — a 0.0
    weight scores them 0.0, it does not drop them. So every leg at 0 admits the
    whole candidate set at score 0 and lets `_apply_bonuses` (recency) order it,
    which looks exactly like a bad retrieval rather than a disabled one. The
    resolver refuses that state.
    """
    monkeypatch.setattr(settings, "retrieval_leg_profiles", {
        "Casual_Banter": {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0},
    })
    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides",
                        {"Null_Noise": {"vector": -0.5, "bm25": -0.6,
                                        "codex": -0.5, "procedural": -0.2}})
    collapsed = leg_weights.resolve(["Casual_Banter"], ["Null_Noise"])

    # Falls back to the pre-override blend — the intent's own suppression is
    # kept, only the delta that collapsed it is discarded.
    assert collapsed["vector"] == 0.5
    assert collapsed["bm25"] == 0.2
    assert not leg_weights._unrankable(collapsed)


def test_a_partial_zero_is_left_alone(table, monkeypatch):
    """The positive control: only the ALL-zero case is refused.

    Individual legs at 0.0 are a legitimate, shipped configuration
    (`Casual_Banter` zeroes codex and procedural today), so a guard that fired
    on those would overwrite tuning it has no business touching.
    """
    monkeypatch.setattr(settings, "retrieval_leg_profiles", {
        "Casual_Banter": {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0},
    })
    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides", {})
    w = leg_weights.resolve(["Casual_Banter"], [])
    assert w["codex"] == 0.0 and w["procedural"] == 0.0
    assert w["vector"] == 0.5 and w["bm25"] == 0.2


def test_all_zero_guard_warns(table, monkeypatch):
    """It substitutes a default for a real answer, so it must say so.

    `structlog.testing.capture_logs` rather than `caplog` — structlog renders
    to stdout and never reaches the stdlib handlers caplog inspects, so a
    caplog assertion here passes vacuously in both directions.
    """
    import structlog

    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides",
                        {"Software_&_Tech": {"vector": -99.0, "bm25": -99.0,
                                             "codex": -99.0, "procedural": -99.0}})
    with structlog.testing.capture_logs() as cap:
        leg_weights.resolve(["Factual_Retrieval"], ["Software_&_Tech"])

    warned = [e for e in cap if e.get("event") == "leg_weight_all_zero"]
    assert len(warned) == 1
    assert warned[0]["log_level"] == "warning"
    # The label combination that caused it is the whole point of the warning —
    # without it the operator cannot find the offending override.
    assert warned[0]["topic_tags"] == ["Software_&_Tech"]
    assert "reason" in warned[0]


def test_a_healthy_resolve_stays_quiet(table):
    """The control: the guard must not fire on the shipped configuration."""
    import structlog

    with structlog.testing.capture_logs() as cap:
        leg_weights.resolve(["Factual_Retrieval"], ["Software_&_Tech"])
    assert not [e for e in cap if e.get("event") == "leg_weight_all_zero"]


def test_timeline_alone_does_not_count_as_rankable(table, monkeypatch):
    """`timeline` is pinned after the blend, so it can never be the survivor.

    Counting it would make the guard unsatisfiable — the exact way a check like
    this dies silently.
    """
    assert leg_weights._unrankable({"bm25": 0.0, "vector": 0.0, "codex": 0.0,
                                    "procedural": 0.0, "timeline": 0.6})


def test_unknown_leg_name_raises(table, monkeypatch):
    """A weight on a leg that does not exist is silently ignored by fusion."""
    monkeypatch.setattr(settings, "retrieval_leg_profiles",
                        {"Factual_Retrieval": {"vektor": 2.0}})
    with pytest.raises(ValueError, match="vektor"):
        leg_weights.resolve(["Factual_Retrieval"], [])


def test_unknown_intent_label_does_not_raise_and_falls_back(table, monkeypatch):
    """A label the live schema lacks must not take retrieval down.

    The v1 checkpoint is a supported rollback and it has a different intent
    head, so an unrecognised row warns and is ignored — the same outcome as
    today, where the row simply never matches.
    """
    monkeypatch.setattr(settings, "retrieval_leg_profiles",
                        {"Not_A_Real_Intent": {"vector": 9.0}})
    w = leg_weights.resolve(["Not_A_Real_Intent"], [])
    assert w["vector"] == 1.0  # base, not 9.0


def test_null_noise_is_a_topic_not_an_intent():
    """Pins the finding that made the validator worth building.

    The shipped profile table carries a `Null_Noise` row inherited from the
    pre-G9 code, where one override was keyed off {Casual_Banter, Null_Noise}.
    `Null_Noise` is a TOPIC label in both v1 and v2 — it has never been able to
    appear in intent_tags — so that half of the row has always been dead.
    ROADMAP G15 plans routing on this label as an intent and must not be built
    from that assumption.

    The row stays where it is on purpose (behaviour freeze, G15 line 520).
    Re-homing it onto the topic head is G15's decision, not this file's, and
    the measurement of what each candidate re-homing would do to the weights
    is recorded in G15's entry — including that moving the row VERBATIM
    inverts it, because the topic table holds cumulative deltas where the
    intent table holds absolute weights.
    """
    from src.classifier.schema import load_schema, load_v1_schema
    assert "Null_Noise" not in load_schema().head("intent").labels
    assert "Null_Noise" not in load_v1_schema().head("intent").labels
    assert "Null_Noise" in load_schema().head("topic").labels
