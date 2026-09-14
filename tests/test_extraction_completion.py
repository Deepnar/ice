"""v3 extraction completion and retry behavior through the real DB write path.

Controlled model responses establish mechanics, not model quality. Run from the
repo root: uv run python tests/test_extraction_completion.py
"""

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (
    CodexEdge,
    CodexEntity,
    CodexEvent,
    Conversation,
    EpisodicMemory,
    IdempotencyKey,
)
from src.workers import codex_extractor as cx
from src.workers import post_flight as pf
from src.workers.extraction_result import ExtractionOutputError
from src.workers.idempotency import job_key
from src.workers.runtime import JobYielded


@pytest.fixture
def turn(monkeypatch):
    monkeypatch.setattr(settings, "codex_sentence_claims", False)
    cid, batch = uuid.uuid4(), uuid.uuid4()
    names = [f"repair_{uuid.uuid4().hex}", f"repair_{uuid.uuid4().hex}"]
    with SessionLocal() as db:
        db.add(Conversation(id=cid))
        db.flush()
        db.add(
            EpisodicMemory(
                conversation_id=cid,
                batch_id=batch,
                raw_text=f"{names[0]} uses {names[1]}.",
                idempotency_key=f"repair-test:{batch}",
                context_reliance="Long_Term_Memory",
                lossless_flag=False,
                inject_raw=True,
                is_private=False,
            )
        )
        db.commit()
    monkeypatch.setattr(settings, "codex_extraction_chunk_adaptive", False)
    monkeypatch.setattr(cx, "known_relations", lambda: [])
    monkeypatch.setattr(cx, "extract_entities", lambda *a, **kw: names)
    monkeypatch.setattr(cx, "make_llm_reconciler", lambda: None)
    yield SimpleNamespace(cid=cid, batch=batch, names=names)
    with SessionLocal() as db:
        ids = [
            e.id
            for e in db.query(CodexEntity).filter(CodexEntity.canonical_name.in_(names))
        ]
        db.query(CodexEdge).filter(CodexEdge.source_batch == batch).delete(
            synchronize_session=False
        )
        db.query(CodexEvent).filter(CodexEvent.batch_source == batch).delete(
            synchronize_session=False
        )
        if ids:
            db.query(CodexEntity).filter(CodexEntity.id.in_(ids)).delete(
                synchronize_session=False
            )
        db.query(IdempotencyKey).filter(
            IdempotencyKey.key.in_(
                [job_key("codex", batch), job_key("post_flight", batch)]
            )
        ).delete(synchronize_session=False)
        db.query(EpisodicMemory).filter_by(batch_id=batch).delete(
            synchronize_session=False
        )
        db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
        db.commit()


def response(monkeypatch, content, finish_reason="stop"):
    def create(**kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )

    monkeypatch.setattr(
        cx,
        "bg_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )


def state(turn):
    with SessionLocal() as db:
        return (
            db.query(IdempotencyKey)
            .filter_by(key=job_key("codex", turn.batch))
            .count(),
            db.query(CodexEdge).filter_by(source_batch=turn.batch).count(),
        )


def test_failed_nonlossless_turn_is_retryable_and_healthy_retry_writes_graph(
    turn, monkeypatch
):
    response(monkeypatch, '{"facts": [')
    with pytest.raises(ExtractionOutputError):
        cx.extract_codex(str(turn.batch))
    assert state(turn) == (0, 0)
    response(
        monkeypatch,
        json.dumps(
            {
                "facts": [
                    {
                        "subject": turn.names[0],
                        "relation": "uses",
                        "object": turn.names[1],
                    }
                ]
            }
        ),
    )
    cx.extract_codex(str(turn.batch))
    assert state(turn) == (1, 1)
    with SessionLocal() as db:
        edge = db.query(CodexEdge).filter_by(source_batch=turn.batch).one()
        strength = edge.strength
    # Completed work never invokes generation or reinforces a second time.
    response(monkeypatch, "broken")
    cx.extract_codex(str(turn.batch))
    assert state(turn) == (1, 1)
    with SessionLocal() as db:
        assert (
            db.query(CodexEdge).filter_by(source_batch=turn.batch).one().strength
            == strength
        )


def test_valid_empty_completes_but_private_turn_never_extracts(turn, monkeypatch):
    with SessionLocal() as db:
        db.query(EpisodicMemory).filter_by(batch_id=turn.batch).update(
            {"is_private": True}
        )
        db.commit()
    response(monkeypatch, "broken")
    cx.extract_codex(str(turn.batch))
    assert state(turn) == (0, 0)
    with SessionLocal() as db:
        db.query(EpisodicMemory).filter_by(batch_id=turn.batch).update(
            {"is_private": False}
        )
        db.commit()
    response(monkeypatch, '{"facts": []}')
    cx.extract_codex(str(turn.batch))
    assert state(turn) == (1, 0)


def test_later_chunk_failure_does_not_commit_prefix(turn, monkeypatch):
    monkeypatch.setattr(cx, "_chunk_text", lambda *a, **kw: ["first", "second"])
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ConnectionError("test transport interruption")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            [
                                {
                                    "subject": turn.names[0],
                                    "relation": "uses",
                                    "object": turn.names[1],
                                }
                            ]
                        )
                    ),
                    finish_reason="stop",
                )
            ]
        )

    monkeypatch.setattr(
        cx,
        "bg_client",
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        ),
    )
    with pytest.raises(ConnectionError):
        cx.extract_codex(str(turn.batch))
    assert len(calls) == 2
    assert state(turn) == (0, 0)


def test_yield_reaches_runtime_without_false_completion(turn, monkeypatch):
    import src.workers.runtime as runtime

    def pause(*args):
        raise JobYielded("test user activity")

    monkeypatch.setattr(runtime, "yield_if_user_active", pause)
    response(monkeypatch, "[]")
    with pytest.raises(JobYielded):
        cx.extract_codex(str(turn.batch))
    assert state(turn) == (0, 0)


def test_postflight_retry_reaches_short_turn_extraction(turn, monkeypatch):
    # Evaluation already completed on a previous attempt; derivatives must run.
    with SessionLocal() as db:
        db.add(
            IdempotencyKey(
                key=job_key("post_flight", turn.batch),
                processed_at=datetime.now(timezone.utc),
            )
        )
        db.commit()
    monkeypatch.setattr(pf, "extract_procedural", lambda **kw: None)
    response(monkeypatch, "")
    with pytest.raises(ExtractionOutputError):
        pf.evaluate_turn(
            str(turn.batch), "A short correction", "Recorded", str(turn.cid)
        )
    assert state(turn) == (0, 0)
    response(monkeypatch, "[]")
    pf.evaluate_turn(str(turn.batch), "A short correction", "Recorded", str(turn.cid))
    assert state(turn) == (1, 0)


def test_emotion_and_reflexive_facts_are_not_silently_filtered(turn, monkeypatch):
    monkeypatch.setattr(cx, "canonical_relation", lambda raw, **kw: raw)
    response(
        monkeypatch,
        json.dumps(
            [
                {"subject": turn.names[0], "relation": "feels", "object": "happy"},
                {
                    "subject": turn.names[0],
                    "relation": "monitors",
                    "object": turn.names[0],
                },
            ]
        ),
    )
    facts = cx.extract_triplets("A service monitors itself; its owner feels happy.")
    assert len(facts) == 2
    assert {fact["relation"] for fact in facts} == {"feels", "monitors"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
