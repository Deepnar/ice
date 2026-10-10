"""Long sources earn compression only through complete source-linked assertions."""

import json
import re
from types import SimpleNamespace as NS

import pytest

from src.memory.representation import choose_representation, verify_representations
from src.memory.source import single_provenance
from src.memory.summary_support import (
    assertion_text, cited_premise, original_units, parse_assertions, proof_digest,
    source_windows, verify_assertions,
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


def cited_window(identifier, text, start=0, unit="u", role="user", unit_hash="original"):
    from src.memory.source import digest
    return dict(id=identifier, unit=unit, role=role, start=start, end=start + len(text),
        unit_sha256=unit_hash, text_sha256=digest(text),
        premise=f"The {role} said: " + json.dumps(text, ensure_ascii=False),
        recorded_at="2025-01-01", timestamp_provenance="original")


def test_cited_union_preserves_complete_correction_and_original_order():
    text = "Use port 8391. Correction: use port 9010 instead."
    first = cited_window("w0", text[:31])
    second = cited_window("w1", text[15:], 15)
    premise, spans = cited_premise([first, second], ["w1", "w0"])
    assert premise == "The user said: " + json.dumps(text)
    assert len(spans) == 1 and spans[0]["end"] == len(text)


def test_cited_union_keeps_gaps_roles_and_other_originals_separate():
    spans = [cited_window("w0", "Refused."), cited_window("w1", "Approved.", 50),
        cited_window("w2", "Refused.", unit="v", role="assistant"),
        cited_window("w3", "Refused.", unit="z")]
    premise, merged = cited_premise(spans, [w["id"] for w in reversed(spans)])
    assert len(merged) == 4 and premise.count("Refused.") == 3
    assert 'The assistant said: "Refused."' in premise and merged[1]["start"] == 50


@pytest.mark.parametrize("mutation", ["role", "offset", "hash", "overlap", "unit_identity"])
def test_cited_union_refuses_inconsistent_source_spans(mutation):
    from src.memory.source import digest
    first = cited_window("w0", "abcdef")
    second = cited_window("w1", "defghi", 3)
    if mutation == "role": second["role"] = "assistant"
    elif mutation == "offset": second["end"] += 1
    elif mutation == "hash": second["text_sha256"] = "different"
    elif mutation == "overlap":
        second["premise"] = 'The user said: "zzzghi"'
        second["text_sha256"] = digest("zzzghi")
    else: second["recorded_at"] = "2025-01-02"
    with pytest.raises(ValueError): cited_premise([first, second], ["w0", "w1"])


def test_cited_union_refuses_missing_or_duplicate_citations():
    for citations in [[], ["missing"], ["w0", "w0"]]:
        with pytest.raises(ValueError): cited_premise([cited_window("w0", "source")], citations)


def test_exact_union_fits_complete_pair_without_dropping_cited_text(monkeypatch):
    from src.memory import support
    text = "alpha " * 364
    spans = [cited_window("w0", text[:208 * 6]), cited_window("w1", text[156 * 6:], 156 * 6)]
    claim = "assertion " * 100
    old = "\n".join(w["premise"] for w in spans)
    assert len(Tokenizer()(old, claim)["input_ids"]) > 512
    monkeypatch.setattr(support, "support_tokenizer", lambda: Tokenizer())
    calls = []
    def score(pairs):
        calls.extend(pairs)
        assert all(len(Tokenizer()(p, h)["input_ids"]) <= 512 for p, h in pairs)
        return [dict(entailment=.99, neutral=.005, contradiction=.005) for _ in pairs]
    monkeypatch.setattr(support, "score_pairs", score)
    result = verify_assertions(text, claim.strip(), [dict(text=claim.strip(), sources=["w0", "w1"])], spans)
    assert len(calls) == 1 and text in calls[0][0]
    assert result["status"] == "supported" and len(result["assertions"][0]["cited_spans"]) == 1
    # This controlled scorer qualifies complete-input construction, not semantics.


def test_exact_supported_legacy_proof_stays_usable_without_promoting_unknown():
    row = source("The user approved port 8391.")
    spans = windows(row)
    row.summary_text = "The user approved port 8391."
    claims = [dict(text=row.summary_text, sources=["w0"])]
    result = verify_representations(row, row.summary_text, None,
        proof=dict(summary=claims, windows=spans), verifier=supported)
    legacy = result["summary"]
    legacy["method"] = legacy["policy"]["method"] = "source_window_assertions_v1"
    legacy["policy"].pop("premise_composition")
    for assertion in legacy["assertions"]: assertion.pop("cited_spans")
    legacy["proof_sha256"] = proof_digest(legacy)
    row.representation_verification = result
    assert choose_representation(row)[0] == row.summary_text
    legacy["status"] = "unknown"
    assert choose_representation(row)[0] == row.raw_text
    legacy["status"] = "supported"
    legacy["method"] = legacy["policy"]["method"] = "invented"
    legacy["proof_sha256"] = proof_digest(legacy)
    assert choose_representation(row)[0] == row.raw_text
