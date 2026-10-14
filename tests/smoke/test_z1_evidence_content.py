"""Parent-source recall cannot stand in for the survival of its required value."""

import json
from types import SimpleNamespace

import pytest

from scripts.z1.evidence_content import (LITERAL_POLICY, audit_probe,
                                         content_status, rendered_text_units, sha)
from src.memory.representation import representation_source
from src.memory.source import chat_provenance


def fixture():
    original = "The approved port is 8391."
    fragment = {"source_row": "gold-row", "origin_batches": ["gold-batch"], "text": "The port was approved."}
    stage = dict(generated_by_leg={"vector": [{"source_row": "gold-row"}]},
                 ranked_candidates=[{"source_row": "gold-row"}], budgeted=[fragment], selected=[fragment],
                 prompt_messages=[{"role": "system", "content": fragment["text"]}],
                 gold_fragment_coverage={"selected": 1})
    probe = dict(probe_id="example", split_turn=80, preflight=stage, controls={},
                 gold_sources=[{"turn": 1, "prompt": original, "response": "Noted."}])
    start = original.index("8391")
    review = dict(probe_id="example", reviewed_through_turn=80, reason="Source-confirmed exact port.",
        requirements=[dict(id="port", kind="exact_literal", alternatives=[dict(
            turn=1, role="user", start=start, end=start+4, source_sha256=sha(original), literal="8391")])])
    return probe, review


def test_id_hit_with_missing_value_fails_content_and_semantic_support():
    probe, review = fixture()
    result = audit_probe(probe, review)
    assert result["source_id_coverage"]["selected"] == 1
    stages = result["requirements"][0]["stages"]
    assert stages["generated"]["literal"] == stages["ranked"]["literal"] == "unobserved"
    assert stages["selected"]["literal"] == stages["final_input"]["literal"] == "missing"
    assert stages["final_input"]["semantic_support"] == "not_supported"


def test_literal_in_a_negated_statement_does_not_pass_semantic_review():
    probe, review = fixture()
    probe["preflight"]["prompt_messages"][0]["content"] = "Port 8391 was refused."
    result = audit_probe(probe, review)["requirements"][0]["stages"]["final_input"]
    assert result["literal"] == "retained"
    assert result["semantic_support"] == "unreviewed"
    review["requirements"][0]["semantic_review"] = {"final_input": dict(
        status="not_supported", content_sha256=result["content_sha256"], reason="Wrong polarity.")}
    assert audit_probe(probe, review)["requirements"][0]["stages"]["final_input"]["semantic_support"] == "not_supported"
    probe["preflight"]["prompt_messages"][0]["content"] = "Port 8391 was approved."
    with pytest.raises(ValueError, match="bind"):
        audit_probe(probe, review)


@pytest.mark.parametrize("defect", ["source", "offset", "role", "cutoff"])
def test_unreviewed_or_stale_source_keys_cannot_score(defect):
    probe, review = fixture()
    span = review["requirements"][0]["alternatives"][0]
    if defect == "source": span["source_sha256"] = "changed"
    elif defect == "offset": span["start"] -= 1
    elif defect == "role": span["role"] = "assistant"
    else: review["reviewed_through_turn"] = 81
    with pytest.raises(ValueError):
        audit_probe(probe, review)


@pytest.mark.parametrize("body", [
    "First line\nSecond line\tValue.",
    'Quoted "value" and Unicode — café.',
    r"Path C:\new\tests; literal \n is code.",
])
def test_native_attributed_source_encoding_preserves_exact_body_without_grading(body):
    raw = f"User: {body}\n\nAssistant: Recorded reply."
    source = representation_source(SimpleNamespace(raw_text=raw, source_spans=chat_provenance(body, "Recorded reply.")))
    probe, review = fixture()
    probe["gold_sources"][0]["prompt"] = body
    review["requirements"][0]["alternatives"] = [dict(
        turn=1, role="user", start=0, end=len(body), source_sha256=sha(body), literal=body)]
    probe["preflight"]["prompt_messages"] = [dict(role="system", content="[Source evidence]\n" + source)]
    result = audit_probe(probe, review)
    assert result["literal_policy"] == LITERAL_POLICY
    assert result["requirements"][0]["stages"]["final_input"]["literal"] == "retained"
    assert result["requirements"][0]["stages"]["final_input"]["semantic_support"] == "unreviewed"


@pytest.mark.parametrize("body,literal", [("line\nnext", r"line\nnext"), (r"line\nnext", "line\nnext")])
def test_transport_escape_cannot_impersonate_source_code_or_newline(body, literal):
    text = "The user said: " + json.dumps(body)
    assert content_status([{"text": text}], [literal]) == "missing"
    assert content_status([{"text": text}], [body]) == "retained"


def test_decoding_is_not_recursive_and_does_not_join_roles_or_fragments():
    nested = 'The assistant said: "line\\nnext"'
    text = "The user said: " + json.dumps(nested)
    assert list(rendered_text_units(text)) == ["The user said: ", nested, ""]
    assert content_status([{"text": text}], ["line\nnext"]) == "missing"
    separate = 'The user said: "first"\nThe assistant said: "second"'
    assert content_status([{"text": separate}], ["first\nsecond"]) == "missing"
    assert content_status([{"text": "first"}, {"text": "second"}], ["firstsecond"]) == "missing"


@pytest.mark.parametrize("text", [
    r'Unattributed "line\nnext"',
    r'The user said: "line\nnext',
    r'The user said: "line\nnext\q"',
])
def test_unattributed_or_malformed_strings_are_not_unescaped(text):
    assert content_status([{"text": text}], ["line\nnext"]) == "missing"


def test_unknown_stages_and_negated_encoded_evidence_keep_existing_semantic_rules():
    assert content_status([{"text": None}], ["unknown"]) == "unobserved"
    probe, review = fixture()
    probe["preflight"]["prompt_messages"][0]["content"] = 'The user said: "Port 8391 was refused."'
    result = audit_probe(probe, review)["requirements"][0]["stages"]["final_input"]
    assert result["literal"] == "retained" and result["semantic_support"] == "unreviewed"
    review["requirements"][0]["semantic_review"] = {"final_input": dict(
        status="not_supported", content_sha256=result["content_sha256"], reason="Wrong polarity.")}
    assert audit_probe(probe, review)["requirements"][0]["stages"]["final_input"]["semantic_support"] == "not_supported"
