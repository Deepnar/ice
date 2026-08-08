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
    monkeypatch.setattr(settings, "retrieval_leg_topic_overrides",
                        {"Software_&_Tech": {"vector": -99.0}})
    w = leg_weights.resolve(["Factual_Retrieval"], ["Software_&_Tech"])
    assert w["vector"] == 0.0


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
    """
    from src.classifier.schema import load_schema, load_v1_schema
    assert "Null_Noise" not in load_schema().head("intent").labels
    assert "Null_Noise" not in load_v1_schema().head("intent").labels
    assert "Null_Noise" in load_schema().head("topic").labels
