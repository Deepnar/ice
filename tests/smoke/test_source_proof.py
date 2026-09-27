"""Factual admission keeps question conditions and binds complete originals."""
import json
from types import SimpleNamespace

import pytest

from src.api import source_proof as sp
from src.api.config import settings

QUESTION = "Which glaze did Mira use for the earlier kiln test?"
SOURCE = "User: Mira used cobalt glaze for the earlier kiln test."
FRAMES = dict(frame="The glaze Mira used for the earlier kiln test was {{answer}}.",
              assertion_frame="Mira used the glaze {{answer}} for the earlier kiln test.",
              reason=None)
FILL = dict(answer="cobalt", evidence_quote="Mira used cobalt glaze for the earlier kiln test.")
GOOD = dict(entailment=.999, neutral=.0005, contradiction=.0005)


@pytest.fixture
def native(monkeypatch):
    state = dict(calls=[], scores=[], devices=[], replies=[FRAMES, FILL], verdicts=[GOOD, GOOD, GOOD, GOOD])
    def post(url, **kw):
        state["calls"].append((url, kw["json"]))
        value = state["replies"].pop(0)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: dict(
            done=True, done_reason="stop", prompt_eval_count=700, eval_count=55,
            message=dict(content=json.dumps(value))))
    def score(pairs, *, device):
        state["scores"].append(pairs)
        state["devices"].append(device)
        return state["verdicts"]
    monkeypatch.setattr(sp.httpx, "post", post)
    monkeypatch.setattr(sp, "score_pairs", score)
    return state


def test_question_is_frozen_before_complete_source_and_both_assertions_land(native):
    assert sp.prove_source_fact(QUESTION, SOURCE).status == "supported"
    first, second = [body for _, body in native["calls"]]
    assert json.loads(first["messages"][1]["content"]) == {"latest_user_request": QUESTION}
    assert "cobalt" not in first["messages"][1]["content"]
    fill_input = json.loads(second["messages"][1]["content"])
    assert fill_input == dict(frame=FRAMES["frame"], assertion_frame=FRAMES["assertion_frame"], source=SOURCE)
    assert native["scores"] == [[(premise, f.replace("{{answer}}", "cobalt"))
        for premise in (SOURCE, SOURCE[:SOURCE.index(FILL["evidence_quote"]) + len(FILL["evidence_quote"])])
        for f in (FRAMES["frame"], FRAMES["assertion_frame"])]]
    assert native["devices"] == ["cpu"]
    assert all(b["think"] is True and b["keep_alive"] == 0
               and b["options"]["num_predict"] == settings.memory_source_proof_output_tokens
               for _, b in native["calls"])


@pytest.mark.parametrize("second", [dict(entailment=.113, neutral=.877, contradiction=.01),
                                    dict(entailment=.001, neutral=.001, contradiction=.998)])
def test_weak_attribute_form_cannot_overrule_direct_qualifier_failure(native, second):
    native["verdicts"] = [GOOD, second, GOOD, GOOD]
    proof = sp.prove_source_fact(QUESTION, SOURCE)
    assert proof.status != "supported" and proof.evidence_quote is None
    assert len(native["scores"][0]) == 4


def test_source_quote_cannot_be_invented_or_borrowed_from_question(native):
    native["replies"][1] = dict(answer="amber", evidence_quote="Mira used amber glaze.")
    proof = sp.prove_source_fact(QUESTION + " Mira used amber glaze.", SOURCE)
    assert proof.status == "unknown" and not native["scores"]


@pytest.mark.parametrize("replies", [
    [dict(frame=None, assertion_frame=None, reason="whole original omitted")],
    [FRAMES, dict(answer=None, evidence_quote=None)],
])
def test_missing_material_is_normal_non_admission_without_nli(native, replies, caplog):
    native["replies"] = replies
    assert sp.prove_source_fact(QUESTION, SOURCE).status == "not_supplied"
    assert not native["scores"] and "memory_source_proof_unknown" not in caplog.text


def test_provider_truncation_cannot_prove_source(monkeypatch):
    monkeypatch.setattr(sp.httpx, "post", lambda *_a, **_k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: dict(done=True, done_reason="length",
            prompt_eval_count=700, message=dict(content=json.dumps(FRAMES)))))
    assert sp.prove_source_fact(QUESTION, SOURCE).reason == "incomplete_native_response"


def test_overlength_original_is_rejected_whole_not_clipped(native, monkeypatch):
    monkeypatch.setattr(settings, "memory_source_gate_input_tokens", 1)
    proof = sp.prove_source_fact(QUESTION, SOURCE)
    assert proof.status == "unknown" and not native["calls"]
    assert proof.reason == "complete_input_exceeds_proof_bound"


def test_verifier_capacity_error_preserves_unknown_without_source_logging(native, monkeypatch, caplog):
    def unavailable(_pairs, **_kwargs): raise ValueError("private source secret must not be logged")
    monkeypatch.setattr(sp, "score_pairs", unavailable)
    proof = sp.prove_source_fact(QUESTION, SOURCE)
    assert proof.status == "unknown" and proof.reason == "ValueError"
    assert "private source secret" not in caplog.text


def test_question_conditions_in_full_premise_cannot_overrule_cited_assertion(native):
    native["verdicts"] = [GOOD, GOOD,
        dict(entailment=.0002, neutral=.9996, contradiction=.0002),
        dict(entailment=.0002, neutral=.9996, contradiction=.0002)]
    proof = sp.prove_source_fact(QUESTION, SOURCE)
    assert proof.status == "unknown" and proof.evidence_quote is None


def test_citation_context_keeps_original_header_and_excludes_later_question(native):
    original = "[source recorded: 2025-02-09 UTC]\n" + SOURCE
    whole = original + "\nWhich glaze did Mira use in March?"
    assert sp.prove_source_fact(QUESTION, whole).status == "supported"
    assert [p for p, _ in native["scores"][0]] == [whole, whole, original, original]


def test_two_sources_share_one_frozen_interpretation_and_three_native_calls(native):
    native["replies"] = [FRAMES, dict(answer=None, evidence_quote=None), FILL]
    frozen = sp.freeze_source_question(QUESTION)
    assert sp.prove_source_fact(QUESTION, "User: An unrelated event.",
                                framed=frozen).status == "not_supplied"
    assert sp.prove_source_fact(QUESTION, SOURCE, framed=frozen).status == "supported"
    assert len(native["calls"]) == 3
    assert [body["format"] for _, body in native["calls"]] == [
        sp.FRAME_SCHEMA, sp.FILL_SCHEMA, sp.FILL_SCHEMA]


def test_frozen_interpretation_cannot_be_reused_for_another_question(native):
    frozen = sp.freeze_source_question(QUESTION)
    result = sp.prove_source_fact("Which glaze did Mira use for the later test?",
                                  SOURCE, framed=frozen)
    assert result.status == "unknown" and result.reason == "question_frame_mismatch"
    assert len(native["calls"]) == 1 and not native["scores"]
