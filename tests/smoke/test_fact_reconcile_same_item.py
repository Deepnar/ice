"""Fact replacement designs (codex_fact_reconcile_design; Z2 S8 supersession, Z2 F20).

g60: a newer fact replaces an earlier one only when the model also says both are about the
same specific thing. pre_g60: can-both-be-true decides (same_item ignored). context: each
fact's cited passage and date go to the model and "contradicts" decides."""

from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.memory import facts


def verdict(**answer):
    earlier = NS(text="The 20-prompt test showed 6 LTM.", modality="shared", mentioned_at=None)
    claim = {"text": "The 58-prompt test showed 21 LTM.", "modality": "shared"}
    return facts.reconcile(claim, earlier, ask=lambda e, n, *p: answer, when=None)


def test_g60_replaces_only_on_the_same_item(monkeypatch):
    monkeypatch.setattr(settings, "codex_fact_reconcile_design", "g60")
    assert verdict(same_information=False, both_can_be_true=False, same_item=True) == "replaces"
    assert verdict(same_information=False, both_can_be_true=False, same_item=False) == "both_true"
    assert verdict(same_information=True, both_can_be_true=True, same_item=True) == "same"
    assert verdict(same_information=False, both_can_be_true=False) == "both_true"   # missing -> keep both


def test_pre_g60_ignores_same_item_and_is_the_default():
    assert settings.codex_fact_reconcile_design == "pre_g60"
    assert verdict(same_information=False, both_can_be_true=False, same_item=False) == "replaces"
    assert verdict(same_information=False, both_can_be_true=True) == "both_true"


def test_context_design_sends_passages_and_contradicts_decides(monkeypatch):
    monkeypatch.setattr(settings, "codex_fact_reconcile_design", "context")
    seen = []
    earlier = NS(text="Rent is 1150.", modality="asserted", mentioned_at=None)
    claim = {"text": "Rent is 1250.", "modality": "asserted"}
    passages = dict(earlier=dict(said="2025-01-02T10:00", passage="rent is 1150"),
                    newer=dict(said="2025-02-02T10:00", passage="rent went up to 1250"))
    ask = lambda e, n, p: seen.append(p) or dict(about="x", same_information=False, contradicts=True)
    assert facts.reconcile(claim, earlier, ask=ask, when=None, passages=passages) == "replaces"
    assert seen == [passages]
    ask = lambda e, n, p: dict(about="x", same_information=False, contradicts=False)
    assert facts.reconcile(claim, earlier, ask=ask, when=None, passages=passages) == "both_true"


@pytest.mark.parametrize("design,expected", [("g60", [False, True, True]), ("pre_g60", [True, True, True])])
def test_screen_uses_same_item_only_under_g60(monkeypatch, design, expected):
    monkeypatch.setattr(settings, "codex_fact_reconcile_design", design)
    earlier = [NS(text="a"), NS(text="b"), NS(text="c")]
    answers = [dict(same_information=False, both_can_be_true=False, same_item=False),
               dict(same_information=False, both_can_be_true=False, same_item=True),
               dict(same_information=True, both_can_be_true=True, same_item=True)]
    assert facts.screen({"text": "newer"}, earlier, ask_many=lambda e, n: answers) == expected


def test_context_passage_variants():
    turn = "User: " + "intro " * 200 + "the landlord raised the rent to 1250 this month. " + "outro " * 200
    common = dict(quote="the landlord raised the rent to 1250 this month.", said="2025-02-02T10:00",
                  turn_text=turn, user_text="why is my rent different?",
                  fields=dict(speaker="user", modality="asserted", subject="rent", relation="is", object="1250"))
    q = facts.context_passage("quote", **common)
    assert q == {"said": "2025-02-02T10:00", "passage": common["quote"]}
    w = facts.context_passage("window", **common)
    assert common["quote"] in w["passage"] and len(w["passage"]) == facts.CONTEXT_WINDOW
    assert facts.context_passage("turn", **common)["user_message"] == "why is my rent different?"
    f = facts.context_passage("fields", **common)
    assert f["subject"] == "rent" and f["object"] == "1250" and "user_message" not in f
    assert facts.context_passage("window", quote="x", said=None)["passage"] == "x"      # no source turn
