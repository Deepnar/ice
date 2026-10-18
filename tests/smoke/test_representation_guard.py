"""Summary-for-raw guard: only the question's specific terms (names, numbers) must
survive in a summary; filler words never block it, and wording does not flip it."""
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.memory import representation
from src.retrieval import orchestrator as orch_mod
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


@pytest.fixture
def orch(monkeypatch):
    o = HybridRetrievalOrchestrator.__new__(HybridRetrievalOrchestrator)
    o.embedder = None
    o.degraded = []
    o._leg_degraded = lambda leg, err: o.degraded.append(leg)
    monkeypatch.setattr(orch_mod, "extract_entities",
                        lambda text, embedder: [w for w in ("Mira", "Tamsin") if w.lower() in text.lower()])
    return o


QUESTION = "ok so like can you just remind me what Mira's power was in chapter 3"


def test_specific_terms_are_names_and_numbers(orch, monkeypatch):
    monkeypatch.setattr(settings, "representation_guard_terms", "specific")
    assert orch._guard_terms(QUESTION) == {"mira", "3"}


def test_every_word_restores_the_old_guard(orch, monkeypatch):
    monkeypatch.setattr(settings, "representation_guard_terms", "every_word")
    terms = orch._guard_terms(QUESTION)
    assert {"like", "just", "remind", "mira"} <= terms


def test_wording_does_not_change_specific_terms(orch, monkeypatch):
    monkeypatch.setattr(settings, "representation_guard_terms", "specific")
    variants = [QUESTION, QUESTION.lower(), "hey " + QUESTION + " lol", "What was Mira's power in chapter 3?"]
    assert len({frozenset(orch._guard_terms(v)) for v in variants}) == 1


def test_ner_failure_degrades_to_numbers_and_says_so(orch, monkeypatch):
    monkeypatch.setattr(settings, "representation_guard_terms", "specific")
    monkeypatch.setattr(orch_mod, "extract_entities", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    assert orch._guard_terms(QUESTION) == {"3"} and orch.degraded == ["representation.guard_ner"]


def test_entities_computed_once_per_prompt(orch, monkeypatch):
    calls = []
    monkeypatch.setattr(orch_mod, "extract_entities", lambda text, embedder: calls.append(text) or ["Mira"])
    orch._prompt_entities(QUESTION)
    orch._prompt_entities(QUESTION)
    assert calls == [QUESTION]


def test_filler_no_longer_blocks_a_summary_but_a_missing_name_does(monkeypatch):
    monkeypatch.setattr(representation, "supported_current", lambda *a: True)
    monkeypatch.setattr(representation, "representation_source", lambda row: "src")
    row = NS(raw_text="User: like, Mira can bend light, I just decided that in chapter 3",
             summary_text="The user decided Mira can bend light (chapter 3).",
             summary_coverage=0.9, abstract_text=None, inject_raw=False,
             representation_verification={"summary": {}})
    cls = NS(intent_tags=["Open_Exploration"])
    assert representation.choose_representation(row, cls, {"mira", "3"})[0] == row.summary_text
    assert representation.choose_representation(row, cls, {"like", "just", "mira"})[0] == row.raw_text
    row.summary_text = "The user decided a character can bend light."
    assert representation.choose_representation(row, cls, {"mira", "3"})[0] == row.raw_text
