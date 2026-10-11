"""Optional incomplete retry preserves first proof; unrelated failures propagate."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.api.config import settings
from src.memory.representation import choose_representation, verify_representations
from src.memory.source import chat_provenance
from src.memory.summary_support import SummaryCandidate
from src.memory.support import verify_support
from src.workers import post_flight, turn_density
from src.workers.completion_text import IncompleteCompletion
from src.workers.runtime import JobYielded

TERMS = dict(entities=["Atlas", "Orion"], figures=["8391"], identifiers=["build_v7"])


def failure():
    raise IncompleteCompletion(
        "completion is not complete: length", finish_reason="length",
        details={"completion_tokens": 900},
    )


def test_exact_first_candidate_and_attached_evidence_survive(monkeypatch):
    monkeypatch.setattr(settings, "turn_summary_coverage_threshold", .7)
    first = SummaryCandidate(
        {"summary": [{"text": "Atlas port8391", "sources": ["w0"]}], "abstract": []},
        [{"id": "w0", "sentinel": "original"}],
    )
    text, coverage = turn_density.retry_on_coverage_miss(first, TERMS, lambda missing: failure())
    assert text is first and coverage == .5
    assert text.assertions is first.assertions and text.windows is first.windows


@pytest.mark.parametrize("error", [
    PermissionError("access denied"), OSError("store inaccessible"),
    ValueError("unrelated validation"), RuntimeError("programming failure"), JobYielded("busy"),
])
def test_unrelated_failure_is_not_reclassified_as_incomplete(error):
    def retry(_missing):
        raise error

    with pytest.raises(type(error)) as raised:
        turn_density.retry_on_coverage_miss("Atlas", TERMS, retry)
    assert raised.value is error


def test_each_incomplete_retry_warns_with_native_metadata(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(turn_density, "logger", logger)
    for _ in range(2):
        turn_density.retry_on_coverage_miss("Atlas", TERMS, lambda missing: failure())
    assert logger.warning.call_count == 2
    for call in logger.warning.call_args_list:
        assert call.kwargs["finish_reason"] == "length"
        assert call.kwargs["provider_usage"] == {"completion_tokens": 900}
        assert call.kwargs["preserving_first_candidate"] is True
        assert call.kwargs["error_type"] == "IncompleteCompletion"
        assert "IncompleteCompletion" in call.kwargs["traceback"]
        assert "provider_usage" not in call.kwargs["traceback"]


def completion(content, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=40, completion_tokens=900, total_tokens=940),
    )


def source_case():
    prompt = "I selected Atlas on port8391. Orion and build_v7 remain undecided."
    response = "Acknowledged."
    raw = f"User: {prompt}\n\nAssistant: {response}"
    row = SimpleNamespace(
        id="controlled", raw_text=raw, source_spans=chat_provenance(prompt, response),
        summary_coverage=0., summary_text=None, abstract_text=None, inject_raw=False,
    )
    return prompt, response, row


def first_json():
    return json.dumps({
        "summary": [{"text": "The user selected Atlas on port8391.", "sources": ["w0"]}],
        "abstract": [{"text": "The user selected port8391.", "sources": ["w0"]}],
    })


@pytest.mark.parametrize("entailment", [.99, .2])
def test_actual_generation_verification_reader_keeps_independent_gates(monkeypatch, entailment):
    prompt, response, row = source_case()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return completion(first_json()) if len(calls) == 1 else completion("partial invalid port9010", "length")

    monkeypatch.setattr(post_flight.bg_client.chat.completions, "create", create)
    generated = post_flight.generate_summary(prompt, response, TERMS, source_row=row)
    summary, coverage, abstract = generated
    assert len(calls) == 2 and coverage == .5
    assert "9010" not in summary and "8391" in abstract
    assert generated.proof["summary"][0]["sources"] == ["w0"]

    def verifier(premise, claim):
        return verify_support(premise, claim, scorer=lambda pairs: [dict(
            entailment=entailment, neutral=1 - entailment, contradiction=0,
        )])

    row.summary_text = summary
    row.summary_coverage = coverage
    row.abstract_text = abstract
    row.representation_verification = verify_representations(
        row, summary, abstract, proof=generated.proof, verifier=verifier,
    )
    main, _, small = choose_representation(row)
    assert main == row.raw_text  # Low coverage still prevents the summary replacing raw.
    assert small == (abstract if entailment == .99 else None)
    if entailment == .99:
        row.raw_text += " A later change."
        assert choose_representation(row) == (row.raw_text, None, None)


def test_initial_incomplete_generation_still_has_no_candidate(monkeypatch):
    prompt, response, row = source_case()
    create = Mock(return_value=completion("partial invalid JSON", "length"))
    monkeypatch.setattr(post_flight.bg_client.chat.completions, "create", create)
    assert post_flight.generate_summary(prompt, response, TERMS, source_row=row) == ("", 0., None)
    assert create.call_count == 1


def test_actual_optional_retry_yields_to_runtime(monkeypatch):
    prompt, response, row = source_case()
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return completion(first_json())
        raise JobYielded("runtime lane busy")

    monkeypatch.setattr(post_flight.bg_client.chat.completions, "create", create)
    with pytest.raises(JobYielded):
        post_flight.generate_summary(prompt, response, TERMS, source_row=row)
    assert len(calls) == 2
