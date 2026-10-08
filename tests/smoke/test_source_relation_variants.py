"""A vocabulary rewrite cannot borrow a source predicate's support verdict."""

import json
import os
import sys
import uuid
from pathlib import Path

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import support
from src.memory.assertion_support import review_assertions
from src.memory.claims import store_claims
from src.memory.models import (CodexClaim, CodexClaimLink, CodexEdge, CodexEntity,
                               CodexEvent, Conversation, EpisodicMemory, IdempotencyKey)
from src.memory.relation_support import verify_relation_pairs
from src.memory.source import digest, single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor as worker
from src.workers.idempotency import job_key

# Use the existing synthetic encoder; no extra inference model in this seam.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from author_assertion_case import Encoder, SCORES


def exercise_variant(monkeypatch, outcome='raw', *, actual=False, real_nli=False):
    source = ('Petra worked at Atlas last year. She now works at Nimbus.'
              if outcome != 'canonical' else 'Petra utilizes Atlas.')
    canonical, raw = ('works_at', 'worked_at') if outcome != 'canonical' else ('uses', 'utilizes')
    monkeypatch.setattr(settings, 'codex_sentence_claims', True)
    monkeypatch.setattr(settings, 'codex_assertion_check_enabled', True)
    monkeypatch.setattr(settings, 'background_model_mode', 'shared')
    monkeypatch.setattr(worker, 'embedder', Encoder())
    monkeypatch.setattr(worker, 'make_llm_reconciler', lambda: None)
    monkeypatch.setattr(worker, 'extract_triplets', lambda _text, *, source_sentences, **_kw:
        source_sentences.append(source) or [dict(subject='petra', relation=canonical,
            _source_relation=raw, object='atlas', source_sentence=source, confidence=.9)])
    monkeypatch.setattr(worker, 'store_claims', lambda db, row, sentences, *, encoder:
        store_claims(db, row, sentences, encoder=encoder, verifier=lambda s, c:
            support.verify_support(s, c, scorer=lambda _pairs: [SCORES])))
    if not real_nli:
        monkeypatch.setattr(worker, 'verify_relation_pairs', lambda pairs:
            verify_relation_pairs(pairs, scorer=lambda p: [SCORES] * len(p)))
    calls = []

    def reviewer(items):
        calls.append(items)
        if real_nli:
            assert 1 <= len(items) <= 2
        else:
            assert len(items) == 2
        if len(items) == 2:
            assert items[0].proposition != items[1].proposition
        if actual:
            receipts = review_assertions(items)
            print(json.dumps([dict(proposition=item.proposition, status=r['status'],
                                  source_act=r['source_act'], reason=r['reason'])
                              for item, r in zip(items, receipts)]))
            return receipts
        from src.memory.assertion_support import parse_decisions
        selected = {'raw': [False, True], 'canonical': [True, True], 'none': [False, False]}[outcome]
        return parse_decisions(json.dumps(dict(decisions=[dict(id=i+1, source_act='assertion',
            supports_candidate=chosen, reason='Synthetic predicate support.')
            for i, chosen in enumerate(selected)])), 'stop', items)

    monkeypatch.setattr(worker, 'review_assertions', reviewer)
    batch, cid = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        original_entities = {row[0] for row in db.query(CodexEntity.id).all()}
        db.add(Conversation(id=cid));db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch, raw_text=source,
            source_spans=single_provenance(source, 'user'), context_reliance='Long_Term_Memory',
            idempotency_key=str(batch)));db.commit()
    try:
        worker.extract_codex(str(batch));worker.extract_codex(str(batch))
        assert len(calls) == 1
        with SessionLocal() as db:
            assert db.query(CodexClaim).filter_by(source_batch=batch).count() == 1
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            assert len(edges) == (0 if outcome == 'none' else 1)
            if edges:
                edge = edges[0]
                chosen = raw if outcome == 'raw' else canonical
                assert edge.relation == chosen
                assert {str(edge.source_batch)} | set(edge.observed_batches or []) == {str(batch)}
                assert edge.confidence == 'pending'  # two candidates are not corroboration
                link = db.query(CodexClaimLink).filter_by(edge_id=edge.id).one()
                proof = link.relation_verification
                assert proof['claim_sha256'] == digest(f'petra {chosen.replace("_", " ")} atlas.')
                assert proof['author_assertion']['status'] == 'supported'
                assert proof['canonicalization']['selected_relation'] == chosen
                assert proof['canonicalization']['selection'] == ('source_fallback_supported'
                    if outcome == 'raw' else 'canonical_supported')
            assert db.query(IdempotencyKey).filter_by(key=job_key('codex', batch)).count() == 1
            fragments = HybridRetrievalOrchestrator(db, None)._codex_claims(
                'Petra Atlas', None, {'conversation_id': str(cid)})
            assert fragments and any(source in f.text for f in fragments)
            return dict(outcome=outcome, chosen_relation=edges[0].relation if edges else None,
                        edges=len(edges), claims=1, repeated_calls=0, original_retrievable=True)
    finally:
        with SessionLocal() as db:
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            edge_ids = [e.id for e in edges]
            nodes = {n for e in edges for n in (e.source_id, e.target_id)
                     if n not in original_entities}
            if edge_ids:
                db.query(CodexClaimLink).filter(CodexClaimLink.edge_id.in_(edge_ids)).delete(synchronize_session=False)
            if nodes:
                db.query(CodexEvent).filter(CodexEvent.entity_id.in_(nodes)).delete(synchronize_session=False)
            db.query(CodexEdge).filter_by(source_batch=batch).delete(synchronize_session=False)
            if nodes:
                db.query(CodexEntity).filter(CodexEntity.id.in_(nodes)).delete(synchronize_session=False)
            db.query(CodexClaim).filter_by(source_batch=batch).delete(synchronize_session=False)
            db.query(IdempotencyKey).filter_by(key=job_key('codex', batch)).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(batch_id=batch).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False);db.commit()


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='requires disposable PostgreSQL')
@pytest.mark.parametrize('outcome', ['raw', 'canonical', 'none'])
def test_writer_selects_only_independently_supported_relation(monkeypatch, outcome):
    exercise_variant(monkeypatch, outcome)
