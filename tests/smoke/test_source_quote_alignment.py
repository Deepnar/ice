"""Copying changes preserve original evidence; semantic proof remains required."""

import json

import pytest

from src.api import memory_preparation, source_need, source_proof
from test_temporal_axis_path import exercise_date_axis

QUESTION = "Which glaze did Mira use for the earlier kiln test?"
ORIGINAL = "Mira used cobalt glaze for the earlier\n\tkiln test."
SOURCE = "User: " + ORIGINAL
FRAMES = (
    "The glaze Mira used for the earlier kiln test was {{answer}}.",
    "Mira used the glaze {{answer}} for the earlier kiln test.",
)
GOOD = {"entailment": .999, "neutral": .0005, "contradiction": .0005}


@pytest.fixture
def proof_calls(monkeypatch):
    seen = []

    def score(pairs, *, device):
        assert device == "cpu"
        seen.append(pairs)
        return [GOOD] * len(pairs)

    monkeypatch.setattr(source_proof, "score_pairs", score)
    return seen


def fill(monkeypatch, quote):
    monkeypatch.setattr(source_proof, "_call", lambda *_a: {
        "answer": "cobalt", "evidence_quote": quote})


@pytest.mark.parametrize("quote", [ORIGINAL, ORIGINAL.lower(),
    "Mira used cobalt glaze for the earlier kiln test.",
    "mira used cobalt glaze for the earlier kiln test."])
def test_copy_restores_original_for_all_four_verifier_pairs(monkeypatch, proof_calls, quote):
    fill(monkeypatch, quote)
    framed = source_proof.SourceQuestion(QUESTION, FRAMES, "ready")
    result = source_proof.prove_source_fact(QUESTION, SOURCE, framed=framed)
    assert result.status == "supported" and result.evidence_quote == ORIGINAL
    claims = [frame.replace("{{answer}}", "cobalt") for frame in FRAMES]
    assert proof_calls == [[(SOURCE, claim) for _ in range(2) for claim in claims]]
    encoded = json.dumps(dict(required_information="The glaze.",
                              decision="visible_evidence", evidence_quote=quote))
    verdict = source_need.parse_verdict(encoded, "stop", [{"content": SOURCE}])
    assert verdict == source_need.SourceNeed("visible_evidence", ORIGINAL)
    assert source_need.source_action(True, verdict, SOURCE) == "skip"


@pytest.mark.parametrize("quote", ["Mira used amber glaze for the earlier kiln test.",
    "Mira did not use cobalt glaze for the earlier kiln test.",
    "Mira used cobalt glaze for the earlier, kiln test.", "Mira used a blue glaze."])
def test_invented_negated_punctuation_or_paraphrase_is_not_a_copy(monkeypatch, proof_calls, quote):
    fill(monkeypatch, quote)
    result = source_proof.prove_source_fact(
        QUESTION, SOURCE, framed=source_proof.SourceQuestion(QUESTION, FRAMES, "ready"))
    assert result.status == "unknown" and not proof_calls
    encoded = json.dumps(dict(required_information="The glaze.",
                              decision="visible_evidence", evidence_quote=quote))
    assert source_need.parse_verdict(encoded, "stop", [{"content": SOURCE}]).decision == "unknown"


def test_distinct_original_spellings_remain_ambiguous(monkeypatch, proof_calls):
    quote = "mira used cobalt glaze."
    fill(monkeypatch, quote)
    source = "User: Mira used cobalt glaze.\nUser: MIRA used cobalt glaze."
    result = source_proof.prove_source_fact(
        QUESTION, source, framed=source_proof.SourceQuestion(QUESTION, FRAMES, "ready"))
    assert result.status == "unknown" and not proof_calls
    encoded = json.dumps(dict(required_information="The glaze.",
                              decision="visible_evidence", evidence_quote=quote))
    messages = [{"content": "Mira used cobalt glaze."}, {"content": "MIRA used cobalt glaze."}]
    assert source_need.parse_verdict(encoded, "stop", messages).decision == "unknown"


def test_aligned_quote_cannot_override_contradiction(monkeypatch, proof_calls):
    fill(monkeypatch, ORIGINAL.lower())
    monkeypatch.setattr(source_proof, "score_pairs", lambda pairs, **_kw: [
        {"entailment": .001, "neutral": .001, "contradiction": .998}] * len(pairs))
    result = source_proof.prove_source_fact(
        QUESTION, SOURCE, framed=source_proof.SourceQuestion(QUESTION, FRAMES, "ready"))
    assert result.status == "not_supplied" and result.evidence_quote is None


def test_old_source_copy_survives_actual_sql_and_final_rescue(monkeypatch, proof_calls):
    def call(_instructions, schema, _payload):
        if schema is source_proof.FRAME_SCHEMA:
            return dict(frame="The employer Petra worked for in 2010 was {{answer}}.",
                        assertion_frame="Petra worked for {{answer}} in 2010.", reason=None)
        return dict(answer="Atlas", evidence_quote="petra worked at atlas in 2010.")

    monkeypatch.setattr(source_proof, "_call", call)
    monkeypatch.setattr(memory_preparation, "judge_source_need",
                        lambda *_a, **_kw: source_need.SourceNeed("older_memory"))
    result = exercise_date_axis(monkeypatch, actual_gate=True, base_retrieve=False)
    assert result["source_action"] == "rescue" and result["gold_in_final_prompt"]
    assert proof_calls and all("Atlas" in premise for pairs in proof_calls for premise, _ in pairs)
