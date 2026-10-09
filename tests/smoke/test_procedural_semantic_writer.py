"""Citations affect activation only after complete source-semantic review."""

import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, IdempotencyKey, ProceduralMemory, Project, StoreMeta
from src.memory.procedural_support import review_sources
from src.memory.source import chat_provenance, digest
from src.workers import procedural_extractor as worker
from test_procedural_support import fake_review


def exercise_source_writer(monkeypatch, case, *, actual=False):
    if not os.getenv('ICE_TEST_DATABASE'):
        raise RuntimeError('requires disposable PostgreSQL')
    pattern = 'The user asks for a plan review before changes.'
    db = SessionLocal()
    project = Project(name=str(uuid.uuid4()), slug=str(uuid.uuid4()), roots=[])
    db.add(project); db.flush()
    conversation = Conversation(project_id=project.id)
    db.add(conversation); db.flush()
    sid, batches = uuid.uuid4(), []
    for i in range(10):
        user = ('Can you review the plan before making changes?' if i < 2 and case != 'unsupported'
                else 'Plan review dashboards can show change counts. This is a topic note.')
        b = uuid.uuid4(); batches.append(b)
        db.add(EpisodicMemory(conversation_id=conversation.id, batch_id=b, session_id=sid,
            raw_text=f'User: {user}\n\nAssistant: Noted.', source_spans=chat_provenance(user, 'Noted.'),
            context_reliance='Long_Term_Memory', timestamp=datetime.now(timezone.utc),
            idempotency_key=str(uuid.uuid4())))
    db.commit()
    originals = {r.batch_id: digest(r.raw_text) for r in db.query(EpisodicMemory).filter_by(conversation_id=conversation.id)}
    key = worker.job_key('procedural', f'{sid}:2')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(
            content=f'PATTERN: {pattern} | EVIDENCE: 1,2,3,4,5,6,7,8,9,10'))])
    try:
        monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=create))))
        monkeypatch.setattr(worker, 'encode_pattern', lambda _: [1.] + [0.] * 1023)
        monkeypatch.setattr('src.model_registry.registry.get_model_context_window', lambda _: 32768)
        monkeypatch.setattr('src.model_registry.runtime_probe.serving_window', lambda *_: 32768)
        if not actual:
            def reviewer(p, sources):
                if case == 'unknown':
                    def unavailable(_):
                        raise RuntimeError('controlled outage')
                    return review_sources(p, sources, sender=unavailable)
                return fake_review(p, sources, supports=lambda s: s.startswith('Can you'))
            monkeypatch.setattr(worker, 'review_sources', reviewer)
        if case == 'unknown':
            from src.workers.completion_text import IncompleteCompletion
            with pytest.raises(IncompleteCompletion):
                worker.extract_procedural(str(batches[-1]))
        else:
            worker.extract_procedural(str(batches[-1]))
        db.expire_all()
        rows = db.query(ProceduralMemory).filter_by(project_id=project.id).all()
        proof = db.get(StoreMeta, 'procedural_review:' + key).value['proof']
        from scripts.z1.seed_v3 import procedural_source_review
        counted = procedural_source_review(db, batches[-1])
        assert counted['status'] == proof['status'] and counted['receipt_present']
        if case == 'supported':
            assert len(rows) == 1 and not rows[0].is_active and rows[0].confidence_score == .3
            assert set(rows[0].source_batch_ids) == set(batches[:2])
            assert proof['kind'] == 'observed_behavior'
            assert counted['source_counts'] == dict(supported=2, unsupported=8, unknown=0)
        else:
            assert not rows and proof['status'] == case
        assert bool(db.get(IdempotencyKey, key)) == (case != 'unknown')
        assert {r.batch_id: digest(r.raw_text) for r in db.query(EpisodicMemory).filter_by(
            conversation_id=conversation.id)} == originals
        if case != 'unknown':
            worker.extract_procedural(str(batches[-1]))
            assert len(calls) == 1
        return dict(case=case, cited=10, supporting_sources=len(rows[0].source_batch_ids) if rows else 0,
                    active=any(r.is_active for r in rows), proof_status=proof['status'], originals_unchanged=True,
                    actual_source_model=actual)
    finally:
        db.rollback()
        db.query(ProceduralMemory).filter_by(project_id=project.id).delete(synchronize_session=False)
        db.query(IdempotencyKey).filter_by(key=key).delete(synchronize_session=False)
        db.query(StoreMeta).filter_by(key='procedural_review:' + key).delete(synchronize_session=False)
        db.query(EpisodicMemory).filter_by(conversation_id=conversation.id).delete(synchronize_session=False)
        db.query(Conversation).filter_by(id=conversation.id).delete(synchronize_session=False)
        db.query(Project).filter_by(id=project.id).delete(synchronize_session=False)
        db.commit(); db.close()


@pytest.mark.skipif(not os.getenv('ICE_TEST_DATABASE'), reason='disposable PostgreSQL required')
@pytest.mark.parametrize('case', ['supported', 'unsupported', 'unknown'])
def test_semantics_controls_support_count_and_completion(monkeypatch, case):
    exercise_source_writer(monkeypatch, case)
