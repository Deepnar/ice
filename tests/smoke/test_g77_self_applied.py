"""G77 — background decisions apply themselves, journaled and undoable.

Each test fails on the pre-G77 code: stale-slot rewrites and tier-2 decision
conflicts waited in the queue with nothing able to apply them, there was no
undo, and approving a motif proposal created an empty cluster. Reflection's
conversation slots are tested in test_g77_reflection_slots.py.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.db import SessionLocal
from src.memory.models import (
    ContextCluster, Conversation, Decision, EpisodicMemory, MemorySlot, Project, ReviewQueue,
)
from src.memory.source import chat_provenance
from src.services import review, slots
from src.services.errors import ConflictError, ValidationError
from src.workers import decision_extractor, maintenance_agent

MARK = f"g77-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def db():
    assert os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
    session = SessionLocal()
    session.info["g77_conversations"] = []
    rows_before = {r.id for r in session.query(ReviewQueue.id)}
    try:
        yield session
    finally:
        session.rollback()
        convs = session.info["g77_conversations"]
        session.query(ReviewQueue).filter(~ReviewQueue.id.in_(rows_before)).delete(
            synchronize_session=False)
        if convs:
            for model in (MemorySlot, EpisodicMemory):
                session.query(model).filter(model.conversation_id.in_(convs)).delete(
                    synchronize_session=False)
            session.query(Conversation).filter(Conversation.id.in_(convs)).delete(
                synchronize_session=False)
        session.commit()
        session.close()


def _conversation(db, exchanges):
    conv = Conversation()
    db.add(conv)
    db.flush()
    db.info["g77_conversations"].append(conv.id)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns = []
    for i, (user, assistant) in enumerate(exchanges):
        raw = f"User: {user}\n\nAssistant: {assistant}"
        turn = EpisodicMemory(conversation_id=conv.id, batch_id=uuid.uuid4(),
                              context_reliance="Zero_Shot", raw_text=raw,
                              source_spans=chat_provenance(user, assistant),
                              timestamp=start + timedelta(minutes=i),
                              idempotency_key=f"{MARK}-{uuid.uuid4()}")
        db.add(turn)
        turns.append(turn)
    db.commit()
    return conv, turns


def _conv_slot(db, conv_id, name):
    return db.query(MemorySlot).filter_by(slot_name=name, scope_tier="conversation",
                                          conversation_id=conv_id).one_or_none()


def test_journaled_undo_refuses_over_a_later_edit(db):
    conv, _ = _conversation(db, [("hello", "hi")])
    first = slots.write_journaled(db, "conversation_focus", f"{MARK} one", updated_by="reflection",
                                  scope_tier="conversation", conversation_id=conv.id)
    second = slots.write_journaled(db, "conversation_focus", f"{MARK} two", updated_by="reflection",
                                   scope_tier="conversation", conversation_id=conv.id)
    with pytest.raises(ConflictError):
        review.undo(db, first["review_item_id"])
    review.undo(db, second["review_item_id"])
    db.expire_all()
    assert _conv_slot(db, conv.id, "conversation_focus").content == f"{MARK} one"


def test_stale_slot_applies_pure_removals_only(db):
    conv, _ = _conversation(db, [("x", "y")])
    slot = slots.write_journaled(db, "pending_items", f"- {MARK} a\n- {MARK} b\n- {MARK} c",
                                 updated_by="user", scope_tier="conversation",
                                 conversation_id=conv.id)
    row = db.get(MemorySlot, uuid.UUID(slot["id"]))

    def age():
        row.last_updated = datetime.now(timezone.utc) - timedelta(days=60)
        db.commit()

    def mine():
        return [i for i in maintenance_agent._detect_stale_slot(db, 50)
                if i.payload["slot_id"] == slot["id"]]

    age()
    item, = mine()
    assert item.payload["conversation_id"] == str(conv.id)
    counters = {"llm_decisions": 0, "applications": 0, "proposals": 0}
    run = uuid.uuid4()
    removal = lambda prompt, n: {"suggestion": f"{MARK} a\n  -  {MARK} c"}  # noqa: E731
    assert maintenance_agent._process(db, item, removal, run, counters) == "applied"
    db.expire_all()
    assert row.content == f"- {MARK} a\n- {MARK} c"
    applied = db.query(ReviewQueue).filter(
        ReviewQueue.item_content["agent_run_id"].astext == str(run)).one()
    assert applied.status == "applied" and applied.item_content["removed_lines"] == [f"{MARK} b"]
    review.undo(db, str(applied.id))
    db.expire_all()
    assert row.content == f"- {MARK} a\n- {MARK} b\n- {MARK} c"

    age()
    item, = mine()
    addition = lambda prompt, n: {"suggestion": f"{MARK} a\n{MARK} brand new"}  # noqa: E731
    assert maintenance_agent._process(db, item, addition, uuid.uuid4(), counters) == "dismissed"
    db.expire_all()
    assert row.content == f"- {MARK} a\n- {MARK} b\n- {MARK} c"
    assert mine() == []          # the dismissal stands until the slot changes


def _project_with_decision(db, files):
    project = Project(name=f"{MARK}-{uuid.uuid4().hex[:4]}", slug=f"{MARK}-{uuid.uuid4().hex[:4]}",
                      roots=["/tmp"])
    db.add(project)
    db.flush()
    vec = [1.0] + [0.0] * 1023
    old = Decision(project_id=project.id, decision=f"{MARK}: use sqlite for storage",
                   files_affected=files, embedding=vec)
    db.add(old)
    db.commit()
    return project, old, vec


def _extracted(files):
    return {"decision": f"{MARK}: use postgres for storage", "rationale": "", "type": "decision",
            "alternatives_rejected": [], "files_affected": files}


@pytest.mark.parametrize("answer,status", [
    ({"same_information": False, "both_can_be_true": False}, "superseded"),
    ({"same_information": False, "both_can_be_true": True}, "both_stand"),
    (RuntimeError("model down"), "both_stand"),
])
def test_tier2_decision_is_decided_not_queued(db, answer, status):
    project, old, vec = _project_with_decision(db, ["a.py"])

    def ask(earlier, newer):
        assert earlier == old.decision and newer.endswith("use postgres for storage")
        if isinstance(answer, Exception):
            raise answer
        return answer

    try:
        out = decision_extractor.reconcile_and_insert(db, project.id, _extracted(["b.py"]),
                                                      vec, ask=ask)
        assert out["status"] == status
        row = db.query(ReviewQueue).filter(
            ReviewQueue.item_content["old_id"].astext == str(old.id)).one()
        db.expire_all()
        if status == "superseded":
            assert row.status == "applied" and old.superseded_by == uuid.UUID(out["id"])
            review.undo(db, str(row.id))
            db.expire_all()
            assert old.valid_until is None and old.superseded_by is None
        else:
            assert row.status == "resolved" and old.valid_until is None
            assert (row.item_content["error"] == "RuntimeError") == isinstance(answer, Exception)
    finally:
        db.rollback()
        db.query(Decision).filter_by(project_id=project.id).delete()
        db.query(Project).filter_by(id=project.id).delete()
        db.commit()


def test_motif_cluster_approval_is_refused(db):
    item = ReviewQueue(item_type="new_cluster_proposal", item_content={"cluster_name": MARK})
    db.add(item)
    db.commit()
    with pytest.raises(ValidationError):
        review.approve(db, str(item.id))
    db.rollback()
    assert db.query(ContextCluster).filter_by(name=MARK).count() == 0
