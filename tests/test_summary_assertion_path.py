"""Focused v3 actual E4B + pinned NLI + database writer/reader qualification.

Run explicitly through tests/support/disposable_database.py. Synthetic source
controls validate this seam; they are not a benchmark or a broad quality claim.
"""

import os
import uuid
from types import SimpleNamespace as NS

import pytest
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_assembler import get_recent_turns
from src.memory.models import Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services.retrieval_svc import recent_turns

assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
        and make_url(settings.database_url).database == os.environ["ICE_TEST_DATABASE"])


@pytest.mark.parametrize("correction", [False, True])
def test_actual_summary_writer_and_readers_preserve_value_and_late_correction(monkeypatch, correction):
    from src.workers import post_flight as worker

    tail = ("Initially we considered port 8391 for Atlas. Final correction: use port 9010, "
            "not 8391. Keep Redis persistence disabled." if correction else
            "The Atlas service uses PostgreSQL. The approved port is 8391.")
    user = "Background discussion about the documentation. " * 200 + tail
    assistant = ("Port 9010 is the approved current value. Redis persistence stays disabled." if correction
                 else "Use the approved PostgreSQL port 8391.")
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch,
            raw_text=f"User: {user}\n\nAssistant: {assistant}",
            source_spans=chat_provenance(user, assistant), context_reliance="Long_Term_Memory",
            idempotency_key=str(uuid.uuid4())))
        db.commit()
    terms = (dict(entities=["Atlas", "Redis"], figures=["8391", "9010"], identifiers=[]) if correction else
             dict(entities=["Atlas", "PostgreSQL"], figures=["8391"], identifiers=[]))
    monkeypatch.setattr(worker, "extract_key_terms", lambda *a: terms)
    monkeypatch.setattr(worker, "compute_entropy", lambda *a: .1)
    for name in ("extract_codex", "extract_procedural", "run_chunk_turn"):
        monkeypatch.setattr(worker, name, lambda *a, **k: None)
    worker.evaluate_turn(str(batch), user, assistant, str(cid), "recorded-answer")
    with SessionLocal() as db:
        row = db.query(EpisodicMemory).filter_by(batch_id=batch).one()
        verdict = row.representation_verification["summary"]
        assert verdict["method"] == "source_window_assertions_v1"
        assert verdict["status"] == "supported", verdict["reason"]
        assert not row.inject_raw
        assert len(row.summary_text) < len(row.raw_text) // 4
        if correction:
            assert "9010" in row.summary_text and "8391" in row.summary_text
            assert "disabled" in row.summary_text.lower()
        else:
            assert "8391" in row.summary_text and "PostgreSQL" in row.summary_text
        prompt = "\n".join(m["content"] for m in get_recent_turns(db, str(cid), max_tokens=4000))
        explicit = recent_turns(db, conversation_id=str(cid))[0]["text"]
        query = NS(prompt="Atlas", topic_tags=[], intent_tags=[])
        lexical = HybridRetrievalOrchestrator(db, None)._bm25_episodic(query, None, str(cid))
        assert all(row.summary_text in text for text in (prompt, explicit, "\n".join(f.text for f in lexical)))
        # Editing an authoritative source invalidates the previous approval in
        # every reader without needing another generation/model call.
        row.raw_text += "\nA later correction changed the source."
        db.commit()
        assert recent_turns(db, conversation_id=str(cid))[0]["text"] == row.raw_text
        messages = get_recent_turns(db, str(cid), max_tokens=8000)
        assert any(user in m["content"] and m["role"] == "user" for m in messages)
        assert any(assistant + "\nA later correction changed the source." in m["content"]
                   and m["role"] == "assistant" for m in messages)


@pytest.mark.parametrize("provider", ["conversation", "batch"])
def test_actual_conversation_note_uses_same_bounded_source_contract(provider):
    from src.workers.conversation_summary import _default_llm, _original_source, _source_note
    from src.memory.support import verify_support

    user = "Background discussion about the documentation. " * 200 + "The approved Atlas port is 8391."
    assistant = "Use the approved Atlas port 8391."
    row = NS(id="note-control", raw_text=f"User: {user}\n\nAssistant: {assistant}",
             source_spans=chat_provenance(user, assistant), timestamp=None, ts_provenance="unknown")
    original, known = _original_source(row)
    llm = _default_llm
    if provider == "batch":
        from src.workers.batch_summarizer import _batch_llm
        llm = _batch_llm
    from src.memory.summary_support import SourceNotePrompt, note_messages
    def real_client(prompt, **kwargs):
        assert isinstance(prompt, SourceNotePrompt)
        messages = note_messages(prompt)
        assert messages[0]["role"] == "system" and "Return only JSON" in messages[0]["content"]
        assert "8391" in messages[1]["content"] and messages[1]["role"] == "user"
        return llm(prompt, **kwargs)
    note = _source_note([row], original, known, real_client, verify_support, max_tokens=900)
    if note["mode"] == "supported":
        assert note["verification"]["method"] == "source_window_assertions_v1"
        assert "8391" in note["text"] and len(note["text"]) < len(original) // 4
    else:
        assert note["text"] == original
        if note["verification"] is None:
            assert note["support_failure"] == "invalid_summary_assertions"
        else:
            assert note["verification"]["method"] == "source_window_assertions_v1"
            assert note["verification"]["status"] != "supported"
    print("v3_actual_note", provider, "mode", note["mode"], "reason",
          (note["verification"] or {}).get("reason", note.get("support_failure")))


def test_actual_postflight_refuses_false_approval_even_with_full_term_coverage(monkeypatch):
    from src.workers import post_flight as worker
    from src.memory.summary_support import SummaryCandidate

    user = "Background discussion. " * 350 + "Use Redis only if approved. Approval was refused."
    assistant = "Keep Redis disabled."
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch,
            raw_text=f"User: {user}\n\nAssistant: {assistant}",
            source_spans=chat_provenance(user, assistant), context_reliance="Long_Term_Memory",
            idempotency_key=str(uuid.uuid4())))
        db.commit()
    monkeypatch.setattr(worker, "extract_key_terms", lambda *a:
        dict(entities=["Redis"], figures=[], identifiers=[]))
    monkeypatch.setattr(worker, "compute_entropy", lambda *a: .1)
    for name in ("extract_codex", "extract_procedural", "run_chunk_turn"):
        monkeypatch.setattr(worker, name, lambda *a, **k: None)
    def proposal(*args, **kwargs):
        windows = kwargs["windows"]
        cited = [w["id"] for w in windows if w["role"] == "user"][-1]
        return SummaryCandidate({"summary": [{"text": "The user approved Redis.", "sources": [cited]}],
                                 "abstract": []}, windows)
    monkeypatch.setattr(worker, "_summary_llm_call", proposal)
    worker.evaluate_turn(str(batch), user, assistant, str(cid), "recorded-answer")
    with SessionLocal() as db:
        row = db.query(EpisodicMemory).filter_by(batch_id=batch).one()
        assert row.summary_coverage == 1.0
        assert row.representation_verification["summary"]["status"] in {"contradicted", "unknown"}
        assert row.inject_raw
        assert recent_turns(db, conversation_id=str(cid))[0]["text"] == row.raw_text
        assert "Approval was refused" in "\n".join(m["content"] for m in
                                                     get_recent_turns(db, str(cid), max_tokens=4000))
