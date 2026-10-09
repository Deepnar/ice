"""Actual import, graph writer, periodic job and reader; inference controlled."""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.ingestion.importer import _store_turn
from src.memory.claims import store_claims
from src.memory.models import (
    CodexClaim, CodexClaimLink, CodexEdge, CodexEntity, CodexEvent,
    Conversation, EpisodicMemory, ReviewQueue,
)
from src.memory.source_order import sequence_receipt, verified_sequence
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor as writer
from src.workers import maintenance_agent as maintenance

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from author_assertion_case import Encoder, SCORES


def exercise_ordered_update(monkeypatch, *, original=False, periodic=False,
                            ordered=True, decision='expire_old', actual=False, new_source=None):
    if not os.getenv('ICE_TEST_DATABASE'):
        pytest.skip('Requires disposable_database.py; never uses the working store.')
    encoder, cid, batches, calls = Encoder(), uuid.uuid4(), [], []
    monkeypatch.setattr(writer, 'embedder', encoder)
    monkeypatch.setattr(settings, 'agent_max_scanned', 6)
    monkeypatch.setattr(maintenance, 'DETECTORS', {
        'reconciliation_leftover': maintenance._detect_reconciliation_leftovers})
    reconciler = writer.make_llm_reconciler() if actual else lambda ctx: decision
    decider = maintenance.make_llm_decider() if actual else lambda *_a, **_kw: {'decision': decision}

    def inline(ctx):
        calls.append(ctx)
        return reconciler(ctx)

    def periodic_decider(prompt, *args, **kwargs):
        calls.append(prompt)
        return decider(prompt, *args, **kwargs)

    with SessionLocal() as db:
        initial_entities = {r[0] for r in db.query(CodexEntity.id).all()}
        db.add(Conversation(id=cid)); db.commit()
        try:
            old = None
            for position, (source, target) in enumerate([
                ('Petra uses Atlas as her primary database.', 'atlas'),
                ('Petra still uses Atlas as her primary database.', 'atlas'),
                (new_source or 'Correction: Petra no longer uses Atlas. Her primary database is now Nimbus.', 'nimbus'),
            ], 1):
                timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=position)
                raw = f'User: {source}\n\nAssistant: Noted.'
                stored = _store_turn(db, cid, source, 'Noted.', timestamp, str(uuid.uuid4()),
                    'fresh', timestamp, 'original' if original else 'synthetic_raw_import', None,
                    encoder, source_sequence=sequence_receipt(raw, 'a' * 64, position) if ordered else None)
                batch = stored[0]; batches.append(batch)
                row = db.query(EpisodicMemory).filter_by(batch_id=batch).one()
                assert (verified_sequence(row) is not None) == ordered
                claims = store_claims(db, row, [source], encoder=encoder,
                    verifier=lambda s, c: verify_support(s, c, scorer=lambda _p: [SCORES]))
                edge = writer.handle_triplet(db, 'petra', 'uses', target, str(batch),
                    turn_text=raw, reconciler=None if periodic else inline, source_claims=claims)
                assert edge is not None
                for claim in claims:
                    db.add(CodexClaimLink(claim_id=claim.id, edge_id=edge.id))
                db.commit()
                if old is None:
                    old = edge
            if periodic:
                result = maintenance.run_maintenance_agent(db, llm_decider=periodic_decider)
                assert result['source_authority_counts'] == {
                    'eligible' if (original or ordered) else 'source_order_unknown': 1}
                assert result['llm_decisions'] == int(original or ordered)
                again = maintenance.run_maintenance_agent(db, llm_decider=periodic_decider)
                assert again['llm_decisions'] == 0
            db.refresh(old)
            admitted = original or ordered
            expired = admitted and decision == 'expire_old'
            assert (old.valid_until is not None) == expired
            assert len(calls) == int(admitted)
            events = db.query(CodexEvent).filter_by(event_type='edge_expired').all()
            matching = [e for e in events if e.payload.get('edge_id') == str(old.id)]
            assert len(matching) == int(expired)
            if expired:
                receipt = matching[0].payload['source_ordering']
                assert receipt['basis'] == ('original_calendar' if original else 'conversation_sequence')
                if ordered:
                    assert (receipt['old_position'], receipt['new_position']) == (2, 3)
                if not original:
                    assert receipt['old_recorded_at'] is receipt['new_recorded_at'] is None
                assert old.valid_until.year >= 2026  # operation date, not invented source event date
            orchestrator = HybridRetrievalOrchestrator(db, encoder)
            orchestrator._prime_edge_times([edge])
            rendered = orchestrator._fact_line(db.get(CodexEntity, edge.source_id), edge,
                                               db.get(CodexEntity, edge.target_id))
            assert (new_source or 'Correction: Petra no longer uses Atlas.') in rendered
            if ordered:
                assert 'Verified conversation order: 3' in rendered
            if not original:
                assert 'source time unknown' in rendered
            return dict(original=original, ordered=ordered, periodic=periodic, decision=decision,
                        model_calls=len(calls), expired=expired, original_claims=len(batches))
        finally:
            db.rollback()
            entity_ids = {r[0] for r in db.query(CodexEntity.id).all()} - initial_entities
            edges = [r[0] for r in db.query(CodexEdge.id).filter(CodexEdge.source_id.in_(entity_ids))]
            db.query(CodexClaimLink).filter(CodexClaimLink.edge_id.in_(edges)).delete(synchronize_session=False)
            db.query(CodexEvent).filter(CodexEvent.entity_id.in_(entity_ids)).delete(synchronize_session=False)
            for review in db.query(ReviewQueue).filter_by(item_type='codex_reconciliation').all():
                if review.item_content.get('new_batch_id') in {str(b) for b in batches}:
                    db.delete(review)
            db.query(CodexEdge).filter(CodexEdge.id.in_(edges)).delete(synchronize_session=False)
            db.query(CodexEntity).filter(CodexEntity.id.in_(entity_ids)).delete(synchronize_session=False)
            db.query(CodexClaim).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()


@pytest.mark.parametrize('original,ordered,periodic,decision', [
    (False, True, False, 'expire_old'), (False, True, True, 'expire_old'),
    (True, False, False, 'expire_old'), (True, True, True, 'expire_old'),
    (False, False, False, 'expire_old'), (False, False, True, 'expire_old'),
    (False, True, False, 'keep_both'), (False, True, True, 'keep_both'),
])
def test_shared_ordered_source_authority(monkeypatch, original, ordered, periodic, decision):
    exercise_ordered_update(monkeypatch, original=original, ordered=ordered,
                            periodic=periodic, decision=decision)
