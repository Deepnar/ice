"""Parent-source recall cannot stand in for the survival of its required value."""

import pytest

from scripts.z1.evidence_content import audit_probe, sha


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
