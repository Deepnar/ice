"""SUMMARY_CLAIMS_V3: quoted pointers, per-claim acceptance, pieces, scaled caps.

docs/specs/SUMMARY_CLAIMS_V3.md. Measured reason (2026-10-15, history C): 64% of
claims verified but 2/86 pieces passed all-or-nothing; 10.5% of claims cited
more source than the verifier reads; 1 in 3 short summaries was broken JSON.
No model is called here: NLI scores are injected.

Run:  uv run --no-sync pytest tests/smoke/test_summary_claims_v3.py -q
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings  # noqa: E402
from src.memory import summary_support as ss  # noqa: E402
from src.memory import support  # noqa: E402
from src.memory.source import chat_provenance, digest  # noqa: E402
from src.workers.completion_text import IncompleteCompletion  # noqa: E402

USER = "The project codename is Atlas and the service listens on port 8391."
ASSISTANT = "Noted: Atlas on port 8391. Should I also reserve port 8392 for staging?"


def _turn(user=USER, assistant=ASSISTANT, ident="t1"):
    raw = f"User: {user}\n\nAssistant: {assistant}"
    return NS(id=ident, raw_text=raw, source_spans=chat_provenance(user, assistant),
              timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), ts_provenance="recorded")


@pytest.fixture
def v3(monkeypatch):
    monkeypatch.setattr(settings, "summary_assertion_method", "v3")


@pytest.fixture
def nli(monkeypatch):
    """Entailment unless the claim contains 'UNSUPPORTED'."""
    def score(pairs, **_kw):
        return [{"entailment": 0.02, "neutral": 0.97, "contradiction": 0.01} if "UNSUPPORTED" in c
                else {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005} for _p, c in pairs]
    monkeypatch.setattr(support, "score_pairs", score)


def _windows(turn=None):
    return ss.source_windows(ss.original_units([turn or _turn()]))


def _role_window(windows, role):
    return next(w for w in windows if w["role"] == role)


def _claim(w, quote, speaker, text):
    return {"sources": [w["id"]], "quote": quote, "speaker": speaker, "text": text}


def test_schema_pins_window_ids_and_puts_the_pointer_first():
    ws = _windows()
    schema = ss.claim_schema(ws, abstract=True)
    row = schema["properties"]["summary"]["items"]
    assert list(row["properties"]) == ["sources", "quote", "speaker", "text"]
    assert row["properties"]["sources"]["items"]["enum"] == [w["id"] for w in ws]
    assert row["properties"]["sources"]["maxItems"] == 2
    assert schema["properties"]["abstract"]["maxItems"] == 1


def test_a_bad_claim_is_dropped_and_its_siblings_survive(v3):
    ws = _windows()
    u, a = _role_window(ws, "user"), _role_window(ws, "assistant")
    rows = [_claim(u, "codename is Atlas", "user", "The user said the codename is Atlas."),
            _claim(u, "codename is Zeus", "user", "The user said the codename is Zeus."),
            _claim(u, "port 8391", "assistant", "The assistant chose port 8391."),
            {"sources": ["w99"], "quote": "Atlas", "speaker": "user", "text": "Atlas."},
            _claim(a, "reserve   port\n8392", "assistant",
                   "The assistant asked whether to reserve port 8392 for staging.")]
    parsed = ss.parse_assertions(json.dumps({"summary": rows}), ws)
    assert [r["text"] for r in parsed["summary"]] == [rows[0]["text"], rows[4]["text"]], (
        "whitespace in a quote is not a fabrication; a wrong quote, speaker or id is")
    assert sorted(d["reason"] for d in parsed["dropped"]) == [
        "quote_not_in_source", "speaker_mismatch", "unresolved_source"]


def test_v2_still_rejects_the_whole_parse(monkeypatch):
    monkeypatch.setattr(settings, "summary_assertion_method", "v2")
    ws = _windows()
    bad = {"summary": [{"text": "x", "sources": ["w99"]}]}
    with pytest.raises(ValueError):
        ss.parse_assertions(json.dumps(bad), ws)


def test_per_claim_acceptance_keeps_only_supported_claims_and_binds_the_kept_text(v3, nli):
    turn = _turn()
    ws = _windows(turn)
    u = _role_window(ws, "user")
    rows = [_claim(u, "codename is Atlas", "user", "The user said the codename is Atlas."),
            _claim(u, "port 8391", "user", "UNSUPPORTED the user owns a boat.")]
    source = "source text"
    record = ss.verify_assertions(source, ss.assertion_text(rows), rows, ws,
                                  dropped=[{"kind": "summary", "reason": "speaker_mismatch"}])
    kept = ss.assertion_text(record["assertions"])
    assert kept == rows[0]["text"]
    assert record["status"] == "supported" and record["method"] == ss.METHOD_V3
    assert {d["reason"] for d in record["dropped"]} == {"speaker_mismatch", "unknown"}
    assert support.supported_current(record, source, kept)
    assert not support.supported_current(record, source, ss.assertion_text(rows)), (
        "the pre-filter text must not pass as verified")


def test_no_supported_claim_means_the_reader_uses_raw(v3, nli):
    ws = _windows()
    u = _role_window(ws, "user")
    rows = [_claim(u, "Atlas", "user", "UNSUPPORTED claim.")]
    record = ss.verify_assertions("s", ss.assertion_text(rows), rows, ws)
    assert record["status"] == "unknown" and record["assertions"] == []
    assert not support.supported_current(record, "s", "")


def test_a_v2_record_stays_readable_after_the_switch(monkeypatch, nli):
    monkeypatch.setattr(settings, "summary_assertion_method", "v2")
    ws = _windows()
    rows = [{"text": "The user said the codename is Atlas.", "sources": [_role_window(ws, "user")["id"]]}]
    record = ss.verify_assertions("s", ss.assertion_text(rows), rows, ws)
    monkeypatch.setattr(settings, "summary_assertion_method", "v3")
    assert support.supported_current(record, "s", rows[0]["text"])


def test_pieces_keep_every_window_in_order_and_never_loop():
    long_user = " ".join(f"Fact {i}: item{i} has value {i * 7}." for i in range(400))
    ws = _windows(_turn(user=long_user))
    pieces = ss.piece_windows(ws, budget=600)
    assert len(pieces) > 1
    assert [w["id"] for p in pieces for w in p] == [w["id"] for w in ws]
    assert ss.piece_windows(ws[:1], budget=1) == [ws[:1]], "an oversized window is its own piece"


def test_output_cap_scales_with_input_between_floor_and_ceiling(monkeypatch):
    monkeypatch.setattr(settings, "summary_output_ratio", 0.5)
    monkeypatch.setattr(settings, "summary_output_ceiling", 3000)
    assert ss.output_cap(100, 900) == 900
    assert ss.output_cap(4000, 900) == 2000
    assert ss.output_cap(100_000, 900) == 3000


def test_generate_claims_merges_pieces_and_survives_one_failed_piece(v3, monkeypatch):
    long_user = " ".join(f"Fact {i}: item{i} has value {i * 7}." for i in range(400))
    ws = _windows(_turn(user=long_user))
    monkeypatch.setattr(settings, "summary_piece_tokens", 600)
    pieces = ss.piece_windows(ws)
    seen = []

    def call(piece, cap, schema, abstract):
        seen.append((len(piece), cap, schema is not None, abstract))
        if len(seen) == 2:
            raise IncompleteCompletion("cut", finish_reason="length")
        w = piece[0]
        quote = json.loads(w["premise"].split(" said: ", 1)[1])[:30]
        return json.dumps({"summary": [_claim(w, quote, w["role"], f"Claim from {w['id']}.")]})
    parsed, failures = ss.generate_claims(call, ws, floor=900, abstract=True)
    assert len(seen) == len(pieces) and all(s[2] for s in seen), "v3 sends the schema every piece"
    assert not any(s[3] for s in seen), "a multi-piece source gets no abstract"
    assert len(parsed["summary"]) == len(pieces) - 1 and len(failures) == 1
    assert "abstract" not in parsed


def test_generate_claims_raises_when_every_piece_fails(v3):
    ws = _windows()

    def call(*_a):
        raise IncompleteCompletion("cut", finish_reason="length")
    with pytest.raises(IncompleteCompletion):
        ss.generate_claims(call, ws, floor=900)


def test_fold_splits_a_huge_single_turn_instead_of_freezing(v3, nli, monkeypatch):
    """2026-10-15: grouping never splits a turn, so one oversized turn failed
    the whole conversation's fold every cycle. Now it is noted in pieces."""
    from src.workers import conversation_summary as cs
    monkeypatch.setattr(settings, "summary_piece_tokens", 600)
    long_user = " ".join(f"Fact {i}: item{i} has value {i * 7}." for i in range(400))
    turn = _turn(user=long_user)
    source, known = cs._original_source(turn)
    calls = []

    def llm(prompt, max_tokens, schema=None):
        calls.append(max_tokens)
        w = next(w for w in ss.source_windows(ss.original_units([turn]))
                 if f"[{w['id']};" in prompt.source)
        quote = json.loads(w["premise"].split(" said: ", 1)[1])[:30]
        return json.dumps({"summary": [_claim(w, quote, w["role"], f"Claim from {w['id']}.")]})
    part = cs._source_note([turn], source, known, llm, support.verify_support)
    assert len(calls) > 1 and part["mode"] == "supported"
    assert part["text"].count("Claim from") == len(calls)


def test_fold_falls_back_to_source_when_a_group_cannot_fit(v3, monkeypatch):
    from src.workers import conversation_summary as cs
    turn = _turn()
    source, known = cs._original_source(turn)

    def llm(prompt, max_tokens, schema=None):
        raise IncompleteCompletion("too big", finish_reason="input_over_budget")
    part = cs._source_note([turn], source, known, llm, support.verify_support)
    assert part["mode"] == "source" and part["support_failure"] == "incomplete_generation"

    def transient(prompt, max_tokens, schema=None):
        raise IncompleteCompletion("server hiccup", finish_reason=None)
    with pytest.raises(IncompleteCompletion):
        cs._source_note([turn], source, known, transient, support.verify_support)


def test_a_quote_without_markdown_markers_still_matches_but_a_paraphrase_does_not(v3):
    """Real failures from history C, 2026-10-15: right words, formatting dropped."""
    user = "**Phase 2 is done.** - `sqlalchemy` is the Python library for talking to databases. It's ready."
    ws = _windows(_turn(user=user))
    u = _role_window(ws, "user")
    rows = [_claim(u, "Phase 2 is done.", "user", "The user said Phase 2 is done."),
            _claim(u, "sqlalchemy is the Python library", "user", "The user said sqlalchemy is a library."),
            _claim(u, "It\u2019s ready.", "user", "A curly apostrophe is formatting."),
            _claim(u, "Phase two is finished", "user", "A paraphrase is not a quote.")]
    parsed = ss.parse_assertions(json.dumps({"summary": rows}), ws)
    assert len(parsed["summary"]) == 3
    assert [d["reason"] for d in parsed["dropped"]] == ["quote_not_in_source"]


def test_entity_note_keeps_only_the_quoted_supported_claim(v3, nli):
    from src.memory import entity_notes
    turn = _turn()
    entity = NS(id="e1", canonical_name="Atlas")
    calls = []

    def llm(prompt, max_tokens, schema=None):
        calls.append(schema)
        ws = ss.source_windows(ss.original_units([turn]))
        u, a = _role_window(ws, "user"), _role_window(ws, "assistant")
        return json.dumps({"summary": [
            _claim(u, "port 8391", "user", "The user said Atlas listens on port 8391."),
            _claim(a, "reserve port 8392", "user", "The user reserved port 8392.")]})
    note, review = entity_notes.generate_note(entity, [turn], llm=llm)
    assert calls and calls[0] is not None, "v3 sends its schema"
    assert note == "The user said Atlas listens on port 8391."
    assert [d["reason"] for d in review["verification"]["dropped"]] == ["speaker_mismatch"]


def test_turn_summary_retries_dropped_key_terms_and_stores_quoted_claims(v3, monkeypatch):
    from src.workers import post_flight as pf
    from src.workers.turn_density import extract_key_terms
    calls = []

    def fake(prompt, response, model, terms, missing=None, *, windows, max_tokens, schema, abstract, **_):
        calls.append(missing)
        w = _role_window(windows, "user")
        claim = (_claim(w, "codename is Atlas", "user", "The user said the codename is Atlas and the port is 8391.")
                 if missing else _claim(w, "codename is Atlas", "user", "The user named a project."))
        return json.dumps({"summary": [claim], **({"abstract": []} if abstract else {})})
    monkeypatch.setattr(pf, "_summary_llm_call", fake)
    terms = {"entities": ["Atlas"], "figures": ["8391"], "identifiers": []}
    summary, coverage, _ = pf.generate_summary(USER, ASSISTANT, terms, "m", source_row=_turn())
    assert len(calls) == 2 and calls[1], "the retry names the dropped terms"
    assert "8391" in summary and coverage == 1.0
