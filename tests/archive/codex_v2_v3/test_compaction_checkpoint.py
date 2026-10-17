"""A graph checkpoint contains full state across real expiration/compaction rounds."""

import io
import json
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import event

from scripts.z1 import seed_v3
from src.api.config import settings
from src.api.db import SessionLocal, engine
from src.memory.models import CodexEdge, CodexEntity, CodexEvent, CodexSnapshot
from src.workers import compaction
from src.workers.codex_extractor import _expire_edge


@pytest.fixture
def graph(monkeypatch):
    assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
    monkeypatch.setattr(settings, 'compaction_event_threshold', 2)
    with SessionLocal() as db:
        source, target = CodexEntity(canonical_name=str(uuid.uuid4())), CodexEntity(canonical_name=str(uuid.uuid4()))
        db.add_all([source, target]); db.commit()
        ids = (source.id, target.id)
        yield db, source, target
        db.rollback()
        db.query(CodexSnapshot).filter(CodexSnapshot.entity_id.in_(ids)).delete(synchronize_session=False)
        db.query(CodexEvent).filter(CodexEvent.entity_id.in_(ids)).delete(synchronize_session=False)
        db.query(CodexEdge).filter(CodexEdge.source_id.in_(ids)).delete(synchronize_session=False)
        db.query(CodexEntity).filter(CodexEntity.id.in_(ids)).delete(synchronize_session=False); db.commit()


def add(db, source, target, relation, *, negated=False):
    edge = CodexEdge(source_id=source.id, target_id=target.id, relation=relation,
        negated=negated, source_batch=uuid.uuid4(), confidence='pending',
        extraction_confidence=.73, observed_batches=[uuid.uuid4()])
    db.add(edge); db.flush()
    db.add(CodexEvent(entity_id=source.id, event_type='edge_added', batch_source=edge.source_batch,
        payload=dict(edge_id=str(edge.id), relation=relation, target_id=str(target.id), negated=negated)))
    db.commit()
    return edge


def test_two_rounds_preserve_unaffected_identity_polarity_and_journal(graph, monkeypatch):
    db, source, target = graph
    a, b = add(db, source, target, 'uses'), add(db, source, target, 'works_at')
    # Legacy snapshot remains historical, never silently re-certified.
    legacy = CodexSnapshot(entity_id=source.id, last_event_id=uuid.uuid4(),
        full_state={'active_edges': ['legacy']})
    db.add(legacy); db.commit(); legacy_id = legacy.id
    first = compaction.compact_entities()
    assert first['snapshots'] == 1 and first['events_compacted'] == 2
    db.expire_all()
    old = db.query(CodexSnapshot).filter_by(entity_id=source.id).order_by(CodexSnapshot.snapshot_ts.desc()).first()
    old_state = old.full_state
    assert set(old_state['active_edges']) == {str(a.id), str(b.id)}
    _expire_edge(db, a.id, uuid.uuid4(), 'synthetic correction')
    db.commit()
    c = add(db, source, target, 'prefers', negated=True)
    events = {str(e.id): (e.event_type, e.payload, e.batch_source, e.timestamp)
              for e in db.query(CodexEvent).filter_by(entity_id=source.id)}
    expiration = next(v for v in events.values() if v[0] == 'edge_expired')
    assert expiration[1]['edge_id'] == str(a.id) and 'relation' not in expiration[1]
    monkeypatch.setattr(settings, 'maintenance_intervals', {'compaction': 7200})
    sink = io.StringIO()
    seed_v3.due_maintenance(db, sink, datetime.now(timezone.utc), uuid.uuid4(), {})
    event = json.loads(sink.getvalue())
    assert event['result'] == dict(method='graph_projection_checkpoint_v2', snapshots=1, events_compacted=2)
    db.expire_all()
    latest = db.query(CodexSnapshot).filter_by(entity_id=source.id).order_by(CodexSnapshot.snapshot_ts.desc()).first()
    state = latest.full_state
    assert state['version'] == 2 and state['basis'] == 'current_graph_projection'
    assert set(state['active_edges']) == {str(b.id), str(c.id)}
    records = {e['id']: e for e in state['edges']}
    assert set(records) == {str(a.id), str(b.id), str(c.id)}
    assert records[str(a.id)]['valid_until'] is not None
    assert records[str(c.id)]['negated'] is True
    assert records[str(b.id)]['observed_batches'] == [str(v) for v in b.observed_batches]
    assert records[str(b.id)]['confidence'] == 'pending' and records[str(b.id)]['extraction_confidence'] == .73
    assert state['context_payload'] == source.context_payload
    assert state['properties'] == source.properties and state['aliases'] == source.aliases
    assert db.get(CodexSnapshot, old.id).full_state == old_state
    assert db.get(CodexSnapshot, legacy_id).full_state == {'active_edges': ['legacy']}
    assert {str(e.id): (e.event_type, e.payload, e.batch_source, e.timestamp)
            for e in db.query(CodexEvent).filter_by(entity_id=source.id)} == events
    assert all(e.compacted for e in db.query(CodexEvent).filter_by(entity_id=source.id))
    assert compaction.compact_entities()['snapshots'] == 0
    assert db.query(CodexSnapshot).filter_by(entity_id=source.id).count() == 3


def test_concurrent_new_event_stays_outside_consistent_checkpoint(graph):
    db, source, target = graph
    a, b = add(db, source, target, 'uses'), add(db, source, target, 'works_at')
    later = []
    def insert_after_view(_conn, _cursor, statement, _parameters, _context, _many):
        if not later and 'FROM codex_edges' in statement and statement.lstrip().startswith('SELECT'):
            later.append(None)  # Prevent the helper's own queries from triggering it.
            with SessionLocal() as writer:
                s, t = writer.get(CodexEntity, source.id), writer.get(CodexEntity, target.id)
                later[0] = add(writer, s, t, 'prefers').id
    event.listen(engine, 'before_cursor_execute', insert_after_view)
    try:
        result = compaction.compact_entities()
    finally:
        event.remove(engine, 'before_cursor_execute', insert_after_view)
    assert later and result['events_compacted'] == 2
    db.expire_all()
    snapshot = db.query(CodexSnapshot).filter_by(entity_id=source.id).one()
    assert set(snapshot.full_state['active_edges']) == {str(a.id), str(b.id)}
    assert {e['id'] for e in snapshot.full_state['edges']} == {str(a.id), str(b.id)}
    pending = db.query(CodexEvent).filter_by(entity_id=source.id, compacted=False).one()
    assert pending.payload['edge_id'] == str(later[0])
    assert db.get(CodexEdge, later[0]) is not None


def test_checkpoint_commit_failure_rolls_back_journal_and_snapshot(graph, monkeypatch):
    db, source, target = graph
    add(db, source, target, 'uses'); add(db, source, target, 'works_at')
    failed = SessionLocal()
    def fail():
        raise RuntimeError('controlled checkpoint commit failure')
    monkeypatch.setattr(failed, 'commit', fail)
    monkeypatch.setattr(compaction, 'SessionLocal', lambda: failed)
    with pytest.raises(RuntimeError, match='controlled checkpoint'):
        compaction.compact_entities()
    db.expire_all()
    assert db.query(CodexSnapshot).filter_by(entity_id=source.id).count() == 0
    assert not any(e.compacted for e in db.query(CodexEvent).filter_by(entity_id=source.id))
    assert db.query(CodexEdge).filter_by(source_id=source.id).count() == 2
