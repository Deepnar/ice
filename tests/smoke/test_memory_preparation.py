"""Final prompt and admission, rather than classifier prefix or candidate read."""
from types import SimpleNamespace

import pytest

from src.api import memory_preparation as mp
from src.api.config import settings
from src.api.source_need import SourceNeed
from src.retrieval.orchestrator import ContextFragment


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(settings, "memory_source_gate_enabled", True)
    monkeypatch.setattr(settings, "memory_source_rescue_enabled", True)
    fragment = ContextFragment("User: We chose port 7813.", "episodic", 1, 9)
    captures = dict(judged=[], exposed=[], retrieved=[], fragments=[fragment], removed=[])
    db = SimpleNamespace(query=lambda *_a: SimpleNamespace(
        filter_by=lambda **_k: SimpleNamespace(first=lambda: None)))
    classifier = SimpleNamespace(embedder=SimpleNamespace(encode=lambda *_a, **_k: [1.0]))
    monkeypatch.setattr(mp, "conversation_summary_block", lambda *_a, **_k: [])
    def assemble(**kw):
        text = "Standing preference blue; required constraint keep source identity."
        if kw["retrieved_fragments"] and not captures["removed"]:
            text += "\n" + "\n".join(f.text for f in kw["retrieved_fragments"])
        return SimpleNamespace(messages=[dict(role="system", content=text),
            dict(role="user", content=kw["user_message"])], removed=list(captures["removed"]),
            ledger=SimpleNamespace(fits=lambda: True))
    monkeypatch.setattr(mp, "assemble_budgeted_prompt", assemble)
    class Orchestrator:
        def __init__(self, *_a): self.recent_token_budget = 100
        def set_budget_from_turn_count(self, *_a, **_k): pass
        def retrieve(self, **kw):
            assert kw["defer_exposure"] is True
            assert kw["scope"] == {"conversation_id": "test-conversation"}
            captures["retrieved"].append(kw)
            return captures["fragments"]
        def record_exposure(self, fragments): captures["exposed"].extend(fragments)
    monkeypatch.setattr(mp, "HybridRetrievalOrchestrator", Orchestrator)
    verdicts = []
    def judge(messages, *, current_only=False):
        assert current_only == args["base_retrieve"]
        if current_only:
            assert messages == [dict(role="user", content=args["user_message"])]
        captures["judged"].append(messages)
        return verdicts.pop(0)
    monkeypatch.setattr(mp, "judge_source_need", judge)
    args = dict(db=db, classifier=classifier,
        classification=SimpleNamespace(context_reliance="Zero_Shot"),
        conversation_id="test-conversation", user_message="Which port did we choose?",
        scope={"conversation_id": "test-conversation"}, turn_count=6, total_tokens=500,
        total_budget=4000, serving_window=8192, base_retrieve=False,
        memory_slots=[], bookmarked_texts=[])
    return args, captures, verdicts


def test_source_rescue_admits_only_after_actual_retrieved_quote(harness):
    args, captures, verdicts = harness
    verdicts.extend([SourceNeed("older_memory"), SourceNeed("visible_evidence", "We chose port 7813.")])
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and result.action == "rescue"
    assert result.source_checked
    assert captures["exposed"] == result.fragments == captures["fragments"]
    assert "7813" not in captures["judged"][0][0]["content"]
    assert "7813" in captures["judged"][1][0]["content"]
    assert "Standing preference blue" in captures["judged"][0][0]["content"]


@pytest.mark.parametrize("decision", ["general_knowledge", "unknown"])
def test_rescue_failure_preserves_original_decision_and_credits_nothing(harness, decision):
    args, captures, verdicts = harness
    verdicts.extend([SourceNeed("older_memory"), SourceNeed(decision)])
    result = mp.prepare_memory_context(**args)
    assert not result.retrieve and not result.fragments and not captures["exposed"]
    assert args["classification"].context_reliance == "Zero_Shot"
    assert result.prepared.messages == captures["judged"][0]


def test_evicted_candidate_quote_cannot_admit_rescue(harness):
    args, captures, verdicts = harness
    captures["removed"] = ["evidence"]
    verdicts.append(SourceNeed("older_memory"))
    result = mp.prepare_memory_context(**args)
    assert not result.retrieve and not captures["exposed"]
    assert len(captures["judged"]) == 1


def test_current_fact_skips_fetch_without_removing_standing_context(harness):
    args, captures, verdicts = harness
    args.update(base_retrieve=True, user_message="Our port is 7813. Which port?")
    verdicts.append(SourceNeed("visible_evidence", "Our port is 7813."))
    result = mp.prepare_memory_context(**args)
    assert not result.retrieve and not captures["retrieved"]
    assert result.source_checked
    assert "Standing preference blue" in result.prepared.messages[0]["content"]


def test_partial_recent_quote_preserves_prior_positive(harness):
    args, captures, verdicts = harness
    args.update(base_retrieve=True, user_message="Continue printing our earlier equations.")
    verdicts.append(SourceNeed("visible_evidence", "Our earlier model began with x = y."))
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and result.action == "keep"
    assert len(captures["judged"]) == 1


def test_unqualified_rescue_is_not_run_by_default(harness, monkeypatch):
    args, captures, verdicts = harness
    monkeypatch.setattr(settings, "memory_source_rescue_enabled", False)
    result = mp.prepare_memory_context(**args)
    assert not result.retrieve
    assert not result.source_checked
    assert not captures["judged"] and not captures["retrieved"]


def test_disabled_refinement_preserves_positive_without_model_call(harness, monkeypatch):
    args, captures, _ = harness
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    args["base_retrieve"] = True
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and result.action == "keep"
    assert not result.source_checked and not captures["judged"]
    assert result.fragments == captures["fragments"] == captures["exposed"]
    assert "Standing preference blue" in result.prepared.messages[0]["content"]


def test_positive_evicted_candidates_receive_no_exposure(harness, monkeypatch):
    args, captures, _ = harness
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    args["base_retrieve"] = True
    captures["removed"] = ["evidence"]
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and not result.fragments
    assert not captures["judged"] and not captures["exposed"]
