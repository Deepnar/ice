"""Final prompt and admission, rather than classifier prefix or candidate read."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.api import memory_preparation as mp
from src.api.config import settings
from src.api.source_need import SourceNeed
from src.api.source_proof import SourceProof
from src.retrieval.orchestrator import ContextFragment


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(settings, "memory_source_gate_enabled", True)
    monkeypatch.setattr(settings, "memory_source_rescue_enabled", True)
    fragment = ContextFragment("User: We chose port 7813.", "episodic", 1, 9, covers_entire_source=True)
    captures = dict(frames=[], proved=[], proofs=[], judged=[], exposed=[],
                    retrieved=[], fragments=[fragment], removed=[], visible=None)
    db = SimpleNamespace(query=lambda *_a: SimpleNamespace(
        filter_by=lambda **_k: SimpleNamespace(first=lambda: None)))
    classifier = SimpleNamespace(embedder=SimpleNamespace(encode=lambda *_a, **_k: [1.0]))
    monkeypatch.setattr(mp, "conversation_summary_block", lambda *_a, **_k: [])
    def assemble(**kw):
        text = "Standing preference blue; required constraint keep source identity."
        visible = (list(captures["visible"])
                   if captures["visible"] is not None
                   else list(kw["retrieved_fragments"])
                   if "evidence" not in captures["removed"] else [])
        if visible:
            text += "\n" + "\n".join(f.text for f in visible)
        return SimpleNamespace(messages=[dict(role="system", content=text),
            dict(role="user", content=kw["user_message"])], removed=list(captures["removed"]),
            ledger=SimpleNamespace(fits=lambda: True), visible_fragments=visible)
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
    def judge(messages):
        assert not args["base_retrieve"], "a positive decision reached the source judge"
        captures["judged"].append(messages)
        return verdicts.pop(0)
    monkeypatch.setattr(mp, "judge_source_need", judge)
    frozen = object()
    def freeze(question):
        captures["frames"].append(question)
        return frozen
    monkeypatch.setattr(mp, "freeze_source_question", freeze)
    def prove(question, source, *, framed):
        assert framed is frozen
        captures["proved"].append((question, source))
        return captures["proofs"].pop(0)
    monkeypatch.setattr(mp, "prove_source_fact", prove)
    args = dict(db=db, classifier=classifier,
        classification=SimpleNamespace(context_reliance="Zero_Shot"),
        conversation_id="test-conversation", user_message="Which port did we choose?",
        scope={"conversation_id": "test-conversation"}, turn_count=6, total_tokens=500,
        total_budget=4000, serving_window=8192, base_retrieve=False,
        memory_slots=[], bookmarked_texts=[])
    return args, captures, verdicts


def test_source_rescue_admits_only_after_complete_original_proof(harness):
    args, captures, verdicts = harness
    verdicts.append(SourceNeed("older_memory"))
    captures["proofs"].append(SourceProof("supported", "We chose port 7813."))
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and result.action == "rescue"
    assert result.source_checked
    assert captures["exposed"] == result.fragments == captures["fragments"]
    assert "7813" not in captures["judged"][0][0]["content"]
    assert captures["proved"] == [(args["user_message"], captures["fragments"][0].text)]
    assert "Standing preference blue" in captures["judged"][0][0]["content"]


@pytest.mark.parametrize("decision", ["not_supplied", "unknown"])
def test_rescue_failure_preserves_original_decision_and_credits_nothing(harness, decision):
    args, captures, verdicts = harness
    verdicts.append(SourceNeed("older_memory"))
    captures["proofs"].append(SourceProof(decision))
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


@pytest.mark.parametrize("message", ["Our port is 7813. Which port?",
                                     "Which came first, the port change or the kiln test?"])
def test_positive_is_never_cancelled_by_its_own_wording(harness, message):
    # Z2 F1: a question naming what it asks about once "proved" itself and lost its memory.
    args, captures, _ = harness
    args.update(base_retrieve=True, user_message=message)
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and result.action == "keep" and captures["retrieved"]
    assert not result.source_checked and not captures["judged"]
    assert "Standing preference blue" in result.prepared.messages[0]["content"]


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


def test_part_already_in_active_summary_receives_no_retrieval_credit(harness, monkeypatch):
    args, captures, _ = harness
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    args["base_retrieve"] = True
    captures["visible"] = []  # final assembler contained this source elsewhere
    result = mp.prepare_memory_context(**args)
    assert result.retrieve and not result.fragments and not captures["exposed"]


@pytest.mark.parametrize("kind,complete", [("codex", True), ("episodic", False), ("procedural", True)])
def test_generated_or_partial_fragment_cannot_prove_itself(harness, kind, complete):
    args, captures, verdicts = harness
    captures["fragments"][0] = replace(captures["fragments"][0],
        source_type=kind, covers_entire_source=complete)
    verdicts.append(SourceNeed("older_memory"))
    result = mp.prepare_memory_context(**args)
    assert not result.retrieve and not captures["proved"] and not captures["exposed"]
    assert not captures["frames"]


def test_rescue_shares_question_and_packs_only_proven_original(harness):
    args, captures, verdicts = harness
    original = captures["fragments"][0]
    unrelated = replace(original, text="User: The other kiln used amber glaze.")
    captures["fragments"] = [unrelated, original]
    verdicts.append(SourceNeed("older_memory"))
    captures["proofs"] = [SourceProof("not_supplied"),
                          SourceProof("supported", "We chose port 7813.")]
    result = mp.prepare_memory_context(**args)
    assert captures["frames"] == [args["user_message"]]
    assert len(captures["proved"]) == 2
    assert result.retrieve and result.fragments == captures["exposed"] == [original]
    assert "7813" in result.prepared.messages[0]["content"]
    assert "amber" not in result.prepared.messages[0]["content"]


def _dry(rows, total, available):
    ledger = SimpleNamespace(serving_window=8192, safety_margin=1.0, total=lambda: total,
                             available=lambda: available, fits=lambda: True)
    return SimpleNamespace(ledger=ledger, artifact_blocks={"recent_turns": {"items": [
        dict(stored_artifact_ids=[("episodic_memory", r)]) for r in rows]}})


def test_recent_rows_are_excluded_only_when_the_recent_window_cannot_be_evicted():
    assert mp._shown_recent_rows(_dry(["r1", "r2"], total=1000, available=6000), 3000) == {"r1", "r2"}
    assert mp._shown_recent_rows(_dry(["r1"], total=4000, available=6000), 3000) == frozenset()
    assert mp._shown_recent_rows(_dry([], total=10, available=6000), 3000) == frozenset()
    assert mp._shown_recent_rows(SimpleNamespace(artifact_blocks=None), 3000) == frozenset()


def test_orchestrator_drops_fragments_of_shown_rows():
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator
    orch = object.__new__(HybridRetrievalOrchestrator)
    orch._exclude_rows = frozenset({"row-a"})
    a = ContextFragment("A", "episodic", 1, 5, "row-a")
    b = ContextFragment("B", "episodic", 1, 5, "row-b")
    fact = ContextFragment("F", "codex", 1, 5, "row-a")
    assert orch._without_shown([a, b, fact]) == [b, fact]
    orch._exclude_rows = frozenset()
    assert orch._without_shown([a, b]) == [a, b]


def test_turn_summaries_switch_off_reads_raw_only(monkeypatch):
    from src.memory import representation
    row = SimpleNamespace(raw_text="User: the port is 7813\n\nAssistant: noted", summary_text="Port 7813.",
                          summary_coverage=1.0, abstract_text="Port.", representation_verification={})
    monkeypatch.setattr(settings, "turn_summaries_in_reads", False)
    assert representation.choose_representation(row) == (row.raw_text, None, None)


def test_bm25_is_a_supported_ablation_flag():
    from src.retrieval.configurable_orchestrator import SUPPORTED_FLAGS
    assert "bm25" in SUPPORTED_FLAGS
