"""Long sources earn compression only through complete source-linked assertions."""

import json
import re
from types import SimpleNamespace as NS

import pytest

from src.memory.representation import choose_representation, verify_representations
from src.memory.source import single_provenance
from src.memory.summary_support import (
    assertion_text, original_units, parse_assertions, source_windows, verify_assertions,
)
from src.memory.support import verify_support


class Tokenizer:
    def __call__(self, text, claim=None, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        count = len(offsets) + (len(claim.split()) if claim else 0) + 3
        return {"offset_mapping": offsets, "input_ids": list(range(count))}


def source(text):
    return NS(id="one", raw_text=text, source_spans=single_provenance(text, "user"),
              summary_coverage=1., inject_raw=False, abstract_text=None)


def windows(row):
    return source_windows(original_units([row]), tokenizer=Tokenizer())


def supported(premise, claim):
    return verify_support(premise, claim, scorer=lambda pairs:
        [dict(entailment=.99, neutral=.005, contradiction=.005)])


def test_windows_cover_complete_original_without_crossing_speakers():
    row = source("context " * 1200 + "The approved port is 8391.")
    result = windows(row)
    assert len(result) > 1 and result[0]["start"] == 0
    assert result[-1]["end"] == len(row.raw_text)
    for previous, current in zip(result, result[1:]):
        assert previous["start"] < current["start"] < previous["end"]
        assert current["role"] == previous["role"] == "user"
    assert all(len(Tokenizer()(w["premise"])["input_ids"]) <= 256 for w in result)
    assert "8391" in result[-1]["premise"]


def test_long_source_can_substitute_only_its_current_supported_assertions():
    row = source("Background " * 1000 + "The user approved port 8391.")
    spans = windows(row)
    claims = [{"text": "The user approved port 8391.", "sources": [spans[-1]["id"]]}]
    row.summary_text = assertion_text(claims)
    proof = {"summary": claims, "abstract": [], "windows": spans}
    row.representation_verification = verify_representations(row, row.summary_text, None,
        proof=proof, verifier=supported)
    assert choose_representation(row)[0] == row.summary_text
    assert row.representation_verification["summary"]["completeness"].startswith("originals_retained")
    row.raw_text += " Correction: port 9010 instead."
    assert choose_representation(row)[0] == row.raw_text


@pytest.mark.parametrize("mutation", ["assertion", "source", "premise", "model", "policy"])
def test_mutated_proof_cannot_authorize_compression(mutation):
    row = source("The user approved port 8391.")
    spans = windows(row)
    claims = [{"text": "The user approved port 8391.", "sources": ["w0"]}]
    row.summary_text = assertion_text(claims)
    row.representation_verification = verify_representations(row, row.summary_text, None,
        proof={"summary": claims, "windows": spans}, verifier=supported)
    verdict = row.representation_verification["summary"]
    if mutation == "assertion": verdict["assertions"][0]["text"] = "The user approved port 9010."
    elif mutation == "source": verdict["assertions"][0]["sources"] = ["invented"]
    elif mutation == "premise": verdict["assertions"][0]["premise"] = "A different original."
    elif mutation == "model": verdict["verifier"] = "different"
    else: verdict["policy"]["pair_tokens"] -= 1
    assert choose_representation(row)[0] == row.raw_text


def test_one_unsupported_assertion_blocks_whole_summary():
    row = source("Approval was refused. Keep Redis disabled.")
    claims = [{"text": "Keep Redis disabled.", "sources": ["w0"]},
              {"text": "Approval was granted.", "sources": ["w0"]}]
    def verifier(premise, claim):
        return verify_support(premise, claim, scorer=lambda pairs:
            [dict(entailment=.01, neutral=.005, contradiction=.985)]
            if "granted" in claim else [dict(entailment=.99, neutral=.005, contradiction=.005)])
    result = verify_assertions(row.raw_text, assertion_text(claims), claims, windows(row), verifier=verifier)
    assert result["status"] == "contradicted"
    assert result["assertions"][0]["verification"]["status"] == "supported"


def test_real_batch_boundary_keeps_oversize_summary_unknown_and_abstract_independent(monkeypatch):
    from src.memory import support
    row = source("context " * 1200 + "The port is 8391.")
    spans = windows(row)
    calls = []
    monkeypatch.setattr(support, "support_tokenizer", lambda: Tokenizer())
    def score(pairs):
        calls.append(pairs)
        assert all(len(Tokenizer()(p, h)["input_ids"]) <= 512 for p, h in pairs)
        return [dict(entailment=.99, neutral=.005, contradiction=.005) for _ in pairs]
    monkeypatch.setattr(support, "score_pairs", score)
    claims = [{"text": "The port is 8391.", "sources": [w["id"] for w in spans]}]
    result = verify_assertions(row.raw_text, "The port is 8391.", claims, spans)
    assert result["status"] == "unknown" and result["reason"] == "SupportInputError"
    abstract = [{"text": "The port is 8391.", "sources": [spans[-1]["id"]]}]
    assert verify_assertions(row.raw_text, "The port is 8391.", abstract, spans)["status"] == "supported"
    assert calls[-1] and all(not pairs for pairs in calls[:-1])


def test_multiple_citations_are_one_complete_premise_not_or_votes():
    row = source("context " * 300 + "Only if approved, use Redis. Approval was refused.")
    spans = windows(row)
    calls = []
    def verifier(premise, claim):
        calls.append((premise, claim))
        return verify_support(premise, claim, scorer=lambda pairs:
            [dict(entailment=.01, neutral=.005, contradiction=.985)])
    claims = [{"text": "Use Redis.", "sources": [w["id"] for w in spans]}]
    result = verify_assertions(row.raw_text, "Use Redis.", claims, spans, verifier=verifier)
    assert len(calls) == 1 and "Approval was refused" in calls[0][0]
    assert result["status"] == "contradicted"


@pytest.mark.parametrize("bad", [
    {"summary": [{"text": "Fact.", "sources": ["invented"]}]},
    {"summary": [{"text": "Fact.", "sources": []}]},
    {"summary": [{"text": "Fact.", "sources": ["w0"], "extra": "Uncited assertion."}]},
    {"summary": [], "free_prose": "Uncited assertion."},
])
def test_unresolved_or_uncited_output_is_not_a_summary(bad):
    with pytest.raises(ValueError):
        parse_assertions(json.dumps(bad), windows(source("Original source.")))


def test_unknown_role_is_never_inferred_from_written_prefix():
    row = NS(id="unknown", raw_text="User: approved port 8391.")
    with pytest.raises(ValueError, match="roles_unknown"):
        original_units([row])


def test_source_changed_between_generation_and_verification_keeps_original():
    row = source("The approved port is 8391.")
    spans = windows(row)
    claims = [{"text": "The approved port is 8391.", "sources": ["w0"]}]
    row.raw_text = "The approved port is 9010."
    row.source_spans = single_provenance(row.raw_text, "user")
    row.summary_text = assertion_text(claims)
    def unexpected_verifier(*args):
        raise AssertionError("Old windows must not verify against a new source hash")
    row.representation_verification = verify_representations(row, row.summary_text, None,
        proof={"summary": claims, "windows": spans}, verifier=unexpected_verifier)
    assert row.representation_verification["summary"]["reason"] == "source_window_identity_changed"
    assert choose_representation(row)[0] == row.raw_text


@pytest.mark.parametrize("abstract", [[{"text": "The port changed."}], "Uncited prose.",
    [{"text": "First."}, {"text": "Second."}]])
def test_invalid_optional_abstract_does_not_discard_valid_summary(abstract):
    parsed = parse_assertions(json.dumps({"summary": [{"text": "The port is 8391.", "sources": ["w0"]}],
        "abstract": abstract}), windows(source("The port is 8391.")), abstract=True)
    assert parsed["summary"][0]["sources"] == ["w0"]
    assert parsed["abstract"] == []
