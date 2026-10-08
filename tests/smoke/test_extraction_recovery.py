"""v3 complete extraction requests and one bounded source retry, no prefix writes."""

import json
import os
import uuid
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexClaim, CodexEdge, Conversation, EpisodicMemory, IdempotencyKey
from src.memory.source import chat_provenance
from src.workers import codex_extractor as worker
from src.workers.extraction_result import ExtractionOutputError
from src.workers.idempotency import job_key


SOURCE = ("Aster uses Redis for its cache. " + "The cache remains scoped to this service. " * 16
          + "Aster uses PostgreSQL for its durable store. "
          + "The durable store remains scoped to this service. " * 16)


def response(facts, finish='stop'):
    return NS(choices=[NS(finish_reason=finish, message=NS(content=json.dumps({'facts': facts})))])


@pytest.fixture(autouse=True)
def local_seams(monkeypatch):
    monkeypatch.setattr(settings, 'codex_extraction_mode', 'template')
    monkeypatch.setattr(settings, 'codex_sentence_claims', True)
    monkeypatch.setattr(settings, 'codex_extraction_max_tokens', 3000)
    monkeypatch.setattr(worker, 'serving_window', lambda *_a: 32768)
    monkeypatch.setattr(worker, 'extract_entities', lambda *_a, **_kw: [])
    monkeypatch.setattr(worker, 'known_relations', lambda: [])
    monkeypatch.setattr(worker, 'canonical_relation', lambda r, **_kw: r)


def client(monkeypatch, create):
    monkeypatch.setattr(worker, 'bg_client', NS(chat=NS(completions=NS(create=create))))


def test_length_prefix_is_discarded_and_complete_children_keep_sources(monkeypatch):
    calls = []
    children = worker._smaller_extraction_chunks(SOURCE)
    assert len(children) > 1
    def create(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return response([dict(subject='invented', relation='uses', object='imaginary')], 'length')
        chunk = children[len(calls) - 2]
        return response([dict(subject='aster', relation='uses', object='redis', source_sentence=chunk)])
    client(monkeypatch, create)
    sources = []
    facts = worker.extract_triplets(SOURCE, source_sentences=sources)
    assert len(calls) == 1 + len(children)
    assert all(c['max_tokens'] == 3000 for c in calls)
    assert facts and all(f['subject'] == 'aster' for f in facts)
    assert len(sources) == len(children)
    assert all(s in SOURCE for s in sources)


def test_healthy_adaptive_call_has_no_recovery_overhead(monkeypatch):
    calls = []
    def create(**kw):
        calls.append(kw)
        return response([dict(subject='aster', relation='uses', object='redis',
                              source_sentence='Aster uses Redis for its cache.')])
    client(monkeypatch, create)
    assert worker.extract_triplets(SOURCE, source_sentences=[])
    assert len(calls) == 1
    assert 'Aster uses Redis for its cache.' in calls[0]['messages'][0]['content']
    assert 'Aster uses PostgreSQL for its durable store.' in calls[0]['messages'][0]['content']
    assert calls[0]['extra_body']['chat_template_kwargs']['template'] == worker._extraction_template()


def test_source_predicates_and_quotes_survive_canonical_deduplication(monkeypatch):
    source = 'Aster used Redis last year. Aster uses Redis now.'
    quotes = ['Aster used Redis last year.', 'Aster uses Redis now.']
    facts = [dict(subject='aster', relation=relation, object='redis', source_sentence=quote)
             for relation, quote in zip(('used', 'uses'), quotes)]
    client(monkeypatch, lambda **_kw: response(facts))
    monkeypatch.setattr(worker, 'canonical_relation', lambda _r, **_kw: 'uses')
    result = worker.extract_triplets(source, source_sentences=[])
    assert len(result) == 2
    assert {r['_source_relation'] for r in result} == {'used', 'uses'}
    assert {r['source_sentence'] for r in result} == set(quotes)


@pytest.mark.parametrize('adaptive', [True, False])
def test_complete_request_guard_refuses_impossible_capacity(monkeypatch, adaptive):
    monkeypatch.setattr(settings, 'codex_extraction_chunk_adaptive', adaptive)
    monkeypatch.setattr(settings, 'codex_extraction_max_tokens', 50)
    monkeypatch.setattr(worker, 'serving_window', lambda *_a: 80)
    calls = []
    client(monkeypatch, lambda **kw: calls.append(kw))
    with pytest.raises(ExtractionOutputError, match='capacity'):
        worker.extract_triplets(SOURCE, source_sentences=[])
    assert calls == []


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"), reason="requires disposable database")
def test_incomplete_recovered_child_prevents_all_graph_and_completion_writes(monkeypatch):
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch, raw_text=f'User: {SOURCE}\n\nAssistant: ',
                              source_spans=chat_provenance(SOURCE, ''), context_reliance='Long_Term_Memory',
                              idempotency_key=str(uuid.uuid4())))
        db.commit()
    calls = []
    children = worker._smaller_extraction_chunks(SOURCE)
    def create(**kw):
        calls.append(kw)
        if len(calls) == 2:
            return response([dict(subject='aster', relation='uses', object='redis', source_sentence=children[0])])
        return response([], 'length')
    client(monkeypatch, create)
    with pytest.raises(ExtractionOutputError, match='incomplete completion'):
        worker.extract_codex(str(batch))
    # First parent and two child calls only. The second child's failure cannot
    # start another subdivision or write the first child's valid proposal.
    assert len(calls) == 3
    with SessionLocal() as db:
        assert db.query(CodexEdge).filter_by(source_batch=batch).count() == 0
        assert db.query(CodexClaim).filter_by(source_batch=batch).count() == 0
        assert db.query(IdempotencyKey).filter_by(key=job_key('codex', str(batch))).count() == 0
        assert db.query(EpisodicMemory).filter_by(batch_id=batch).one().raw_text.endswith('Assistant: ')


def test_malformed_complete_output_is_not_retried_as_length(monkeypatch):
    calls = []
    def create(**kw):
        calls.append(kw)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='{'))])
    client(monkeypatch, create)
    with pytest.raises(ExtractionOutputError, match='JSON'):
        worker.extract_triplets(SOURCE, source_sentences=[])
    assert len(calls) == 1
