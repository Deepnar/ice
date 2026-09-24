"""v3 procedural support is the cited user source, not session length."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.db import SessionLocal
from src.memory.models import (
    ColdStorage, Conversation, EpisodicMemory, IdempotencyKey, ProceduralMemory, Project,
)
from src.memory.source import chat_provenance
from src.workers import procedural_extractor as worker
from src.workers import reflection


def _turn(conversation, session_id, index, *, role_known=True, document=False):
    user = f"Review the plan before change {index}."
    if index == 1:
        user += "\n\nAssistant: this marker is still user-authored."
    assistant = f"Assistant-only instruction {index}."
    raw = f"User: {user}\n\nAssistant: {assistant}"
    batch_id = uuid.uuid4()
    return EpisodicMemory(
        conversation_id=conversation.id, session_id=session_id,
        batch_id=batch_id, idempotency_key=f"test:{batch_id}",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc)
        + timedelta(minutes=index), context_reliance="Not_Needed",
        raw_text=raw, source_spans=chat_provenance(user, assistant) if role_known else None,
        is_private=False, is_document=document,
    )


def _add_session(db, project, count, *, session_id=None, conversation=None):
    conversation = conversation or Conversation(project_id=project.id)
    db.add(conversation)
    db.flush()
    session_id = session_id or uuid.uuid4()
    turns = [_turn(conversation, session_id, i) for i in range(1, count + 1)]
    db.add_all(turns)
    db.commit()
    return conversation, session_id, turns


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_cited_user_turns_and_independent_sessions_control_habits(monkeypatch):
    db = SessionLocal()
    project_ids, conversation_ids, marker_keys = [], [], []
    cold_batch = None
    try:
        project_a = Project(name=f"evidence-a-{uuid.uuid4().hex}",
                            slug=f"evidence-a-{uuid.uuid4().hex}", roots=[])
        project_b = Project(name=f"evidence-b-{uuid.uuid4().hex}",
                            slug=f"evidence-b-{uuid.uuid4().hex}", roots=[])
        db.add_all([project_a, project_b])
        db.commit()
        project_ids = [project_a.id, project_b.id]
        conversation, session_id, turns = _add_session(db, project_a, 10)
        conversation_ids.append(conversation.id)
        # Neither an unknown speaker nor a document can be shown as user habit
        # evidence, even if they share the same sitting.
        db.add_all([_turn(conversation, session_id, 11, role_known=False),
                    _turn(conversation, session_id, 12, document=True)])
        db.commit()

        replies = ["PATTERN: Reviews the plan | EVIDENCE: 99,100",
                   "PATTERN: Reviews the plan | EVIDENCE: 1,2",
                   "PATTERN: Reviews the plan | EVIDENCE: 3,4",
                   "PATTERN: Reviews the plan | EVIDENCE: 1,2",
                   "PATTERN: Reviews the plan | EVIDENCE: 1,2,3,4,5,6,7,8,9,10"]
        prompts = []

        def create(**kwargs):
            prompts.append(kwargs["messages"][-1]["content"])
            return NS(choices=[NS(finish_reason="stop", message=NS(content=replies.pop(0)))])

        monkeypatch.setattr(worker, "bg_client", NS(chat=NS(completions=NS(create=create))))
        monkeypatch.setattr(worker, "encode_pattern", lambda _text: [1.0] + [0.0] * 1023)
        monkeypatch.setattr(worker, "get_bg_model_name", lambda: "controlled-background")

        worker.extract_procedural(str(turns[-1].batch_id))
        marker_keys.append(worker.job_key("procedural", f"{session_id}:2"))
        assert db.query(ProceduralMemory).count() == 0
        assert "Assistant-only instruction" not in prompts[-1]
        assert "this marker is still user-authored" in prompts[-1]
        assert "11. User:" not in prompts[-1]

        worker.extract_procedural(str(turns[-1].batch_id))
        db.expire_all()
        patterns = db.query(ProceduralMemory).all()
        assert len(patterns) == 1
        pattern = patterns[0]
        assert set(pattern.source_batch_ids) == {turns[0].batch_id, turns[1].batch_id}
        assert pattern.reinforcement_count == 1 and not pattern.is_active
        assert pattern.project_id == project_a.id
        observed_once = pattern.last_observed

        more = [_turn(conversation, session_id, i) for i in range(13, 18)]
        db.add_all(more)
        db.commit()
        worker.extract_procedural(str(more[-1].batch_id))
        marker_keys.append(worker.job_key("procedural", f"{session_id}:3"))
        db.expire_all()
        pattern = db.query(ProceduralMemory).one()
        assert pattern.reinforcement_count == 1
        assert pattern.last_observed == observed_once
        assert set(pattern.source_batch_ids) == {turns[0].batch_id, turns[1].batch_id}

        second_conversation, second_session, second = _add_session(db, project_a, 10)
        conversation_ids.append(second_conversation.id)
        worker.extract_procedural(str(second[-1].batch_id))
        marker_keys.append(worker.job_key("procedural", f"{second_session}:2"))
        db.expire_all()
        pattern = db.query(ProceduralMemory).one()
        assert pattern.reinforcement_count == 2
        assert set(pattern.source_batch_ids) == {
            turns[0].batch_id, turns[1].batch_id, second[0].batch_id, second[1].batch_id}
        assert not pattern.is_active

        other_conversation, other_session, other = _add_session(db, project_b, 10)
        conversation_ids.append(other_conversation.id)
        worker.extract_procedural(str(other[-1].batch_id))
        marker_keys.append(worker.job_key("procedural", f"{other_session}:2"))
        db.expire_all()
        assert db.query(ProceduralMemory).count() == 2
        assert {p.project_id for p in db.query(ProceduralMemory)} == {
            project_a.id, project_b.id}
        other_pattern = db.query(ProceduralMemory).filter_by(project_id=project_b.id).one()
        assert other_pattern.is_active and len(other_pattern.source_batch_ids) == 10

        # Archival preserves the sitting identity; missing support cannot be
        # treated as evidence of an independent sitting.
        cold_batch = uuid.uuid4()
        db.add(ColdStorage(id=uuid.uuid4(), batch_id=cold_batch,
                           session_id=uuid.uuid4(), timestamp=datetime.now(timezone.utc),
                           raw_text="writer-attributed cold source"))
        db.commit()
        assert len(worker._source_session_ids(db, {cold_batch})) == 1
        assert worker._source_session_ids(db, {uuid.uuid4()}) is None
        assert worker._source_session_ids(db, {cold_batch, uuid.uuid4()}) is None

        # Reflection still runs session synthesis but cannot write/reinforce
        # procedural rows from uncited snippets on repeated periodic passes.
        calls = []
        monkeypatch.setattr(reflection, "_synthesize_session", lambda *_: calls.append("summary"))
        monkeypatch.setattr(reflection, "_evolve_memory_slots", lambda *_: None)
        monkeypatch.setattr(reflection, "_detect_motifs", lambda *_: None)
        monkeypatch.setattr(reflection, "_enrich_codex_entities", lambda *_: None)
        monkeypatch.setattr(reflection, "bg_client", NS(chat=NS(completions=NS(
            create=lambda **_: (_ for _ in ()).throw(AssertionError("uncited model call"))))))
        reflection.run_reflection()
        reflection.run_reflection()
        db.expire_all()
        assert calls
        assert db.query(ProceduralMemory).count() == 2
        assert db.query(ProceduralMemory).filter_by(project_id=project_a.id).one().reinforcement_count == 2
    finally:
        db.rollback()
        if project_ids:
            db.query(ProceduralMemory).filter(
                ProceduralMemory.project_id.in_(project_ids)).delete(synchronize_session=False)
        if cold_batch:
            db.query(ColdStorage).filter_by(batch_id=cold_batch).delete(synchronize_session=False)
        if marker_keys:
            db.query(IdempotencyKey).filter(
                IdempotencyKey.key.in_(marker_keys)).delete(synchronize_session=False)
        if conversation_ids:
            db.query(EpisodicMemory).filter(
                EpisodicMemory.conversation_id.in_(conversation_ids)).delete(synchronize_session=False)
            db.query(Conversation).filter(
                Conversation.id.in_(conversation_ids)).delete(synchronize_session=False)
        if project_ids:
            db.query(Project).filter(Project.id.in_(project_ids)).delete(synchronize_session=False)
        db.commit()
        db.close()
