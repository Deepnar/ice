"""v3 procedural request sizing and durable negative bucket receipts."""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, IdempotencyKey, ProceduralMemory
from src.memory.source import chat_provenance
from src.workers import procedural_extractor as worker
from src.workers.completion_text import IncompleteCompletion
from src.workers.idempotency import job_key

pytestmark = pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"), reason="requires disposable database")


def add_turns(db, cid, sid, first, last):
    batch = None
    for i in range(first, last + 1):
        user = f'Please explain item {i} before implementing it.'
        batch = uuid.uuid4()
        db.add(EpisodicMemory(conversation_id=cid, session_id=sid, batch_id=batch,
                             raw_text=f'User: {user}\n\nAssistant: Noted.',
                             source_spans=chat_provenance(user, 'Noted.'), context_reliance='Long_Term_Memory',
                             timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i),
                             idempotency_key=str(uuid.uuid4())))
    db.commit()
    return batch


@pytest.fixture
def source():
    cid, sid = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        batch = add_turns(db, cid, sid, 1, 6)
    yield cid, sid, batch
    with SessionLocal() as db:
        keys = [job_key('procedural', f'{sid}:{bucket}') for bucket in range(11)]
        db.query(IdempotencyKey).filter(IdempotencyKey.key.in_(keys)).delete(synchronize_session=False)
        db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
        db.query(Conversation).filter_by(id=cid).delete()
        db.commit()


def set_client(monkeypatch, content='NONE', finish='stop', calls=None):
    calls = [] if calls is None else calls
    def create(**kw):
        calls.append(kw)
        return NS(choices=[NS(finish_reason=finish, message=NS(content=content))])
    monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(worker, 'get_bg_model_name', lambda: 'controlled-background')
    from src.model_registry import registry, runtime_probe
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *_a: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *_a: 32768)
    return calls


def test_complete_none_skips_same_bucket_and_allows_new_bucket(monkeypatch, source):
    cid, sid, batch = source
    calls = set_client(monkeypatch)
    worker.extract_procedural(str(batch)); worker.extract_procedural(str(batch))
    assert len(calls) == 1
    assert calls[0]['max_tokens'] == settings.procedural_extraction_max_tokens == 512
    with SessionLocal() as db:
        assert db.query(IdempotencyKey).filter_by(key=job_key('procedural', f'{sid}:1')).count() == 1
        assert not any(str(batch) in {str(b) for b in (p.source_batch_ids or [])}
                       for p in db.query(ProceduralMemory))
        new_batch = add_turns(db, cid, sid, 7, 10)
    worker.extract_procedural(str(new_batch))
    assert len(calls) == 2
    with SessionLocal() as db:
        assert db.query(IdempotencyKey).filter_by(key=job_key('procedural', f'{sid}:2')).count() == 1


@pytest.mark.parametrize('finish, content', [('length', 'NONE'), ('length', 'PATTERN: One habit. | EVIDENCE: 1,2'),
                                            ('stop', 'NONETHELESS the user has no habits'), ('stop', 'NONE. More output')])
def test_incomplete_or_nonliteral_none_never_stamps_empty_success(monkeypatch, source, finish, content):
    _cid, sid, batch = source
    set_client(monkeypatch, content, finish)
    if finish == 'length':
        with pytest.raises(IncompleteCompletion):
            worker.extract_procedural(str(batch))
    else:
        worker.extract_procedural(str(batch))
    with SessionLocal() as db:
        assert db.query(IdempotencyKey).filter_by(key=job_key('procedural', f'{sid}:1')).count() == 0


def test_over_capacity_complete_sources_are_not_sent_or_stamped(monkeypatch, source):
    _cid, sid, batch = source
    calls = set_client(monkeypatch)
    from src.model_registry import runtime_probe
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *_a: 520)
    with pytest.raises(IncompleteCompletion, match='capacity'):
        worker.extract_procedural(str(batch))
    assert calls == []
    with SessionLocal() as db:
        assert db.query(IdempotencyKey).filter_by(key=job_key('procedural', f'{sid}:1')).count() == 0
