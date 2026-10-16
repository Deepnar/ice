"""CODEX_V4 D4: v4 claims = v3 claims + modality and the source's subject/relation/object.

docs/specs/CODEX_V4.md §5.1. The instruction was qualified on 41 real pieces and
maintainer-graded; its text is pinned here so an edit cannot ship unqualified.
No model is called: NLI scores are injected.

Run:  uv run --no-sync pytest tests/smoke/test_codex_v4_claims.py -q
"""

import hashlib
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
from src.memory.source import chat_provenance, single_provenance  # noqa: E402

USER = "I moved to Lisbon in March and I work at Acme now."
ASSISTANT = "You could also look at coworking spaces near Alfama."
# sha256 of the qualified instruction ("m5", abstract=False). Changing the text means
# re-running the CODEX_V4 §5.1 qualification, then updating this value.
INSTRUCTION_SHA256 = "f793ca5f6b99024eec9e4799337d19368950f8a4530ac9a6c25fc579d9864f02"


def _turn(user=USER, assistant=ASSISTANT):
    raw = f"User: {user}\n\nAssistant: {assistant}"
    return NS(id="t1", raw_text=raw, source_spans=chat_provenance(user, assistant),
              timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc), ts_provenance="recorded")


@pytest.fixture
def v4(monkeypatch):
    monkeypatch.setattr(settings, "summary_assertion_method", "v4")


@pytest.fixture
def nli(monkeypatch):
    def score(pairs, **_kw):
        return [{"entailment": 0.02, "neutral": 0.97, "contradiction": 0.01} if "UNSUPPORTED" in c
                else {"entailment": 0.99, "neutral": 0.005, "contradiction": 0.005} for _p, c in pairs]
    monkeypatch.setattr(support, "score_pairs", score)


def _fact(w, quote, speaker, text, modality="asserted", subject="user", relation="moved to",
          obj="Lisbon"):
    return {"sources": [w["id"]], "quote": quote, "speaker": speaker, "modality": modality,
            "subject": subject, "relation": relation, "object": obj, "text": text}


def test_instruction_is_the_qualified_text(v4):
    text = ss.assertion_instruction(abstract=False)
    assert hashlib.sha256(text.encode()).hexdigest() == INSTRUCTION_SHA256
    assert "'shared'" in text and "figurative" in text


def test_v4_schema_and_v3_schema(v4, monkeypatch):
    windows = ss.source_windows(ss.original_units([_turn()]))
    row = ss.claim_schema(windows)["properties"]["summary"]["items"]
    assert row["required"] == list(ss.FACT_FIELDS)
    assert row["properties"]["modality"]["enum"] == list(ss.MODALITIES)
    monkeypatch.setattr(settings, "summary_assertion_method", "v3")
    row = ss.claim_schema(windows)["properties"]["summary"]["items"]
    assert row["required"] == list(ss.CLAIM_FIELDS) and "modality" not in row["properties"]


def test_speaker_enum_offers_only_roles_present():
    doc = "The service listens on port 8391."
    turn = NS(id="d1", raw_text=doc, source_spans=single_provenance(doc, "document"),
              timestamp=None, ts_provenance="x")
    windows = ss.source_windows(ss.original_units([turn]))
    assert ss.claim_schema(windows)["properties"]["summary"]["items"]["properties"]["speaker"]["enum"] == ["document"]
    chat = ss.source_windows(ss.original_units([_turn()]))
    assert ss.claim_schema(chat)["properties"]["summary"]["items"]["properties"]["speaker"]["enum"] == ["user", "assistant"]


def test_v4_parse_keeps_fields_and_drops_bad_rows(v4):
    windows = ss.source_windows(ss.original_units([_turn()]))
    user = next(w for w in windows if w["role"] == "user")
    good = _fact(user, "I moved to Lisbon in March", "user", "The user said they moved to Lisbon in March.")
    bad_modality = dict(good, modality="certain")
    missing_field = {k: v for k, v in good.items() if k != "relation"}
    parsed = ss.parse_assertions(json.dumps({"summary": [good, bad_modality, missing_field]}), windows)
    assert parsed["summary"] == [good]
    assert [d["reason"] for d in parsed["dropped"]] == ["invalid_claim_shape", "invalid_claim_shape"]


def test_v4_mechanical_checks_still_apply(v4):
    windows = ss.source_windows(ss.original_units([_turn()]))
    user = next(w for w in windows if w["role"] == "user")
    invented = _fact(user, "I moved to Porto", "user", "The user said they moved to Porto.")
    wrong_speaker = _fact(user, "I moved to Lisbon in March", "assistant", "The assistant said Lisbon.")
    parsed = ss.parse_assertions(json.dumps({"summary": [invented, wrong_speaker]}), windows)
    assert parsed["summary"] == []
    assert sorted(d["reason"] for d in parsed["dropped"]) == ["quote_not_in_source", "speaker_mismatch"]


def test_v4_verified_records_keep_fact_fields(v4, nli):
    turn = _turn()
    windows = ss.source_windows(ss.original_units([turn]))
    user = next(w for w in windows if w["role"] == "user")
    asst = next(w for w in windows if w["role"] == "assistant")
    kept = _fact(user, "I moved to Lisbon in March", "user", "The user said they moved to Lisbon in March.")
    dropped = _fact(asst, "You could also look at coworking spaces", "assistant",
                    "UNSUPPORTED The assistant said coworking.", modality="suggested")
    claims = [kept, dropped]
    proof = ss.verify_assertions(turn.raw_text, ss.assertion_text(claims), claims, windows)
    assert proof["method"] == ss.METHOD_V4
    assert len(proof["assertions"]) == 1
    row = proof["assertions"][0]
    assert {k: row[k] for k in ("modality", "subject", "relation", "object")} == {
        "modality": "asserted", "subject": "user", "relation": "moved to", "object": "Lisbon"}
    assert ss.proof_current(proof, turn.raw_text, ss.assertion_text(proof["assertions"]))


def test_opening_header_only_for_pieces_inside_a_message(v4):
    long_user = "Here is my spec: " + " ".join(f"rule {i} keeps the decay at 0.97." for i in range(900))
    turn = _turn(user=long_user)
    windows = ss.source_windows(ss.original_units([turn]))
    pieces = ss.piece_windows(windows, 1024)
    assert len(pieces) > 1
    first, later = ss._with_opening(pieces[0], windows), ss._with_opening(pieces[1], windows)
    assert "continues" not in first[0]
    assert later[0]["continues"].startswith("Here is my spec:")
    assert ss.window_prompt(later).startswith("[context, not a source:")
    assert "continues" not in pieces[1][0]                  # stored windows untouched
    assert "context, not a source" not in json.dumps([w["premise"] for w in windows])


def test_no_header_under_v3(monkeypatch):
    monkeypatch.setattr(settings, "summary_assertion_method", "v3")
    long_user = "Here is my spec: " + " ".join(f"rule {i} keeps the decay at 0.97." for i in range(900))
    windows = ss.source_windows(ss.original_units([_turn(user=long_user)]))
    seen = []

    def call(piece, cap, schema, abstract):
        seen.append(ss.window_prompt(piece))
        raise ValueError("stop")
    with pytest.raises(ValueError):
        ss.generate_claims(call, windows, floor=900)
    assert seen and not any(p.startswith("[context") for p in seen)
