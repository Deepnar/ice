"""v3 conflict mechanics with synthetic SQL fixtures and controlled decisions."""

import uuid
from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity, CodexEvent, ReviewQueue
from src.services import review
from src.services.errors import ValidationError
from src.workers import codex_extractor as cx
from src.workers import maintenance_agent as ma


@pytest.fixture
def graph():
    with SessionLocal() as db:
        a, b = [CodexEntity(id=uuid.uuid4(), canonical_name=f"control_{uuid.uuid4().hex}")
                for _ in range(2)]
        db.add_all([a, b])
        db.flush()
        yield db, a, b
        db.rollback()


@pytest.mark.parametrize("old,new", [("buys", "sells"), ("teaches", "learns_from"),
                                      ("parent_of", "child_of")])
def test_converses_are_separate_but_not_expiry_evidence(graph, monkeypatch, old, new):
    db, a, b = graph
    edge = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation=old)
    db.add(edge)
    db.flush()
    assert cx._is_inverse_pair(old, new)
    assert cx.check_conflict(db, a.id, new, b.id, "Both relationships hold.") is None
    assert edge.valid_until is None
    monkeypatch.setattr(cx, "get_or_create_entity", lambda db, name, **kw:
                        a if name == a.canonical_name else b)
    monkeypatch.setattr(cx, "_regenerate_context_payload", lambda *args: None)
    cx.handle_triplet(db, a.canonical_name, new, b.canonical_name, uuid.uuid4(),
                      turn_text="Both relationships hold.")
    db.flush()
    assert edge.valid_until is None
    live = db.query(CodexEdge).filter_by(source_id=a.id, target_id=b.id,
                                       valid_until=None).all()
    assert {e.relation for e in live} == {old, new}


def test_opposition_requires_source_decision(graph):
    db, a, b = graph
    edge = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="friend")
    db.add(edge)
    db.flush()
    conflict = cx.check_conflict(db, a.id, "enemy", b.id, "A calls B an enemy.")
    assert conflict
    # Missing source cannot expire, even if a supplied callable would say so.
    assert cx.reconcile_conflict(db, conflict, a, "enemy", b, uuid.uuid4(),
                                 None, lambda _: "expire_old")
    assert edge.valid_until is None
    assert cx.reconcile_conflict(db, conflict, a, "enemy", b, uuid.uuid4(),
                                 "Both are true in different contexts.", lambda _: "keep_both")
    assert edge.valid_until is None


def test_background_candidates_propose_once_without_expiring(graph, monkeypatch):
    db, a, b = graph
    old = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="friend")
    new = CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id, relation="enemy")
    db.add_all([old, new])
    db.flush()
    # Keep this fixture transaction local even through the production helper.
    monkeypatch.setattr(db, "commit", db.flush)
    items = [i for i in ma._detect_contradictions(db, 20)
             if i.payload["source_id"] == str(a.id)]
    assert len(items) == 1 and items[0].tier == 2
    counters = {"applications": 0, "proposals": 0, "llm_decisions": 0}
    assert ma._process(db, items[0], None, uuid.uuid4(), counters) == "proposed"
    assert counters["proposals"] == 1 and counters["applications"] == 0
    assert ma._apply_contradiction(db, items[0], uuid.uuid4()) == "noop"
    assert old.valid_until is None and new.valid_until is None
    assert db.query(ReviewQueue).filter_by(item_type="codex_contradiction").count() == 1


@pytest.mark.parametrize("keep_count", [0, 1, 2])
def test_review_requires_explicit_choice_and_applies_it(graph, monkeypatch, keep_count):
    db, a, b = graph
    edges = [CodexEdge(source_batch=uuid.uuid4(), source_id=a.id, target_id=b.id,
                       relation=rel) for rel in ("friend", "enemy")]
    db.add_all(edges)
    db.flush()
    pair = [str(e.id) for e in edges]
    item = ReviewQueue(item_type="codex_contradiction", item_content={"edge_ids": pair})
    db.add(item)
    db.flush()
    monkeypatch.setattr(db, "commit", db.flush)
    monkeypatch.setattr(cx, "_regenerate_context_payload", lambda *args: None)
    with pytest.raises(ValidationError):
        review.approve(db, str(item.id))
    assert item.status == "pending"
    with pytest.raises(ValidationError):
        review.approve(db, str(item.id), keep_edge_ids=[str(uuid.uuid4())])
    review.approve(db, str(item.id), keep_edge_ids=pair[:keep_count])
    assert item.status == "approved"
    assert sum(e.valid_until is None for e in edges) == keep_count
    assert db.query(CodexEvent).filter_by(entity_id=a.id,
                                         event_type="edge_expired").count() == 2 - keep_count


@pytest.mark.parametrize("content,finish,expected", [
    ("expire_old", "stop", "expire_old"),
    ("do not expire_old", "stop", "review"),
    ("keep_both or expire_old", "stop", "review"),
    ("expire_old", "length", "review"),
    ("keep_both", "stop", "keep_both"),
])
def test_reconciler_exact_complete_output(monkeypatch, content, finish, expected):
    captured = []
    def create(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason=finish)])
    monkeypatch.setattr(cx, "bg_client", SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    source = "Earlier context. " * 60 + "The final correction must be included."
    ctx = dict(subject="Atlas", relation="enemy", object="Beacon", old_relation="friend",
               old_object="Beacon", turn=source)
    assert cx.make_llm_reconciler()(ctx) == expected
    assert source in captured[0]["messages"][1]["content"]
    captured.clear()
    monkeypatch.setattr(settings, "codex_reconcile_input_tokens", 1)
    assert cx.make_llm_reconciler()(ctx) == "review"
    assert not captured
