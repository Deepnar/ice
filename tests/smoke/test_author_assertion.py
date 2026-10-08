import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.memory.assertion_support import AssertionInput, parse_decisions, review_assertions
from src.memory.claims import claim_representation
from src.memory.source import digest, single_provenance


def item(text="Mira works at Atlas."):
    return AssertionInput(text, "user", text, "Mira works at Atlas.", 0, len(text))


def response(body, *, finish="stop"):
    cases = json.loads(body["messages"][1]["content"])
    return dict(done=True, done_reason=finish, prompt_eval_count=600, eval_count=80,
                message=dict(content=json.dumps({"decisions": [
                    dict(id=c["id"], source_act="assertion", supports_candidate=True,
                         reason="Explicit source assertion.") for c in cases]})))


@pytest.mark.parametrize("change", [
    lambda d: d.update(id=True), lambda d: d.update(supports_candidate="true"),
    lambda d: d.update(source_act="uncertain"), lambda d: d.update(reason=""),
    lambda d: d.update(extra="not part of protocol"),
])
def test_invalid_or_uncertain_decisions_never_admit(change):
    decision = dict(id=1, source_act="assertion", supports_candidate=True, reason="Explicit.")
    change(decision)
    receipt = parse_decisions(json.dumps(dict(decisions=[decision])), "stop", [item()])[0]
    assert receipt["status"] == "unknown"


def test_reordered_complete_ids_preserve_mapping_and_duplicates_abstain():
    decisions = [dict(id=i, source_act="assertion", supports_candidate=i == 1,
                      reason="Distinct source decision.") for i in (2, 1)]
    result = parse_decisions(json.dumps(dict(decisions=decisions)), "stop", [item(), item()])
    assert [r["status"] for r in result] == ["supported", "not_asserted"]
    decisions[0]["id"] = 1
    assert all(r["status"] == "unknown" for r in parse_decisions(
        json.dumps(dict(decisions=decisions)), "stop", [item(), item()]))
    assert parse_decisions("{}", "length", [item()])[0]["status"] == "unknown"


@pytest.mark.parametrize("act", ["question", "hypothetical"])
def test_true_boolean_cannot_override_nonassertive_context(act):
    decision = dict(id=1, source_act=act, supports_candidate=True,
                    reason="Literal matching is not author commitment.")
    receipt = parse_decisions(json.dumps(dict(decisions=[decision])), "stop", [item()])[0]
    assert receipt["status"] == "not_asserted"
    assert receipt["model_supports_candidate"] is True
    assert receipt["reason"] == "non_assertive_source_act"


def test_native_reasoning_keeps_each_complete_source_and_receipt(monkeypatch):
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    calls = []

    def sender(body):
        calls.append(body)
        return response(body)

    original = "Mira works at Atlas.\n\nThis paragraph asks about a different relation."
    target = replace(item(original), quote="Mira works at Atlas.")
    result = review_assertions([target, replace(target, proposition="Mira works at Atlas.")], sender=sender)
    assert all(r["status"] == "supported" for r in result)
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert len(payload) == 2
    assert all(c["original"] == original for c in payload)
    assert all("source_quote" not in c for c in payload)
    assert calls[0]["think"] is True
    schema = calls[0]["format"]["properties"]["decisions"]
    assert schema["minItems"] == schema["maxItems"] == 2
    assert schema["items"]["properties"]["id"]["enum"] == [1, 2]
    assert result[0]["source_sha256"] == digest(original)
    assert result[0]["quote_sha256"] == digest(target.quote)
    assert result[0]["usage"]["finish_reason"] == "stop"


def test_source_bounds_unknown_finish_and_outages_do_not_create_support(monkeypatch):
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    assert review_assertions([replace(item(), quote="invented")], sender=response)[0]["status"] == "unknown"
    assert review_assertions([item()], sender=lambda b: response(b, finish="length"))[0]["status"] == "unknown"

    def outage(_body):
        raise RuntimeError("offline")

    assert review_assertions([item()], sender=outage)[0]["reason"] == "provider_RuntimeError"
    from src.workers.runtime import JobYielded

    with pytest.raises(JobYielded):
        review_assertions([item()], sender=lambda _: (_ for _ in ()).throw(JobYielded()))
    monkeypatch.setattr(settings, "codex_assertion_input_tokens", 1)
    calls = []
    assert all(r["status"] == "unknown" for r in review_assertions(
        [item(), item()], sender=lambda b: calls.append(b)))
    assert not calls


def test_dedicated_reasoning_is_explicitly_unqualified(monkeypatch):
    monkeypatch.setattr(settings, "background_model_mode", "dedicated")
    receipt = review_assertions([item()], sender=lambda b: pytest.fail("must not substitute"))[0]
    assert receipt["reason"] == "dedicated_reasoning_unqualified"


def test_withheld_relation_read_preserves_cross_paragraph_proposal_context():
    original = "Here is a possible alternate plot.\n\nMira works at Atlas."
    excerpt = "Mira works at Atlas."
    claim = SimpleNamespace(text=excerpt, sentence=excerpt, role="user",
        raw_sha256=digest(original), start=original.index(excerpt), end=len(original),
        verification={"needs_author_context": True})
    source = SimpleNamespace(raw_text=original, source_spans=single_provenance(original, "user"))
    assert claim_representation(claim, source=source) == original
    source.raw_text = "An edited unrelated source."
    assert claim_representation(claim, source=source) == excerpt


@pytest.mark.parametrize("metadata", [{}, {"author_assertions": [{"status": "supported"}]}])
def test_global_or_legacy_proof_cannot_supply_only_the_question_paragraph(metadata):
    question = "Does Mira work at Atlas?"
    original = question + "\n\nMira does work at Atlas."
    claim = SimpleNamespace(text=question, sentence=question, role="user",
        raw_sha256=digest(original), start=0, end=len(question), verification=metadata)
    source = SimpleNamespace(raw_text=original, source_spans=single_provenance(original, "user"))
    assert claim_representation(claim, source=source) == original
