"""Entity accounts require complete originals, complete output and current proof."""
import io
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import entity_notes, support
from src.memory.models import CodexEntity, CodexEvent, ColdStorage, Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import bg_client_factory, reflection
from src.workers.codex_extractor import _regenerate_context_payload
from test_summary_assertions import Tokenizer


@pytest.fixture
def created():
    assert os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_')
    rows = []
    yield rows
    with SessionLocal() as db:
        for cid, eid in rows:
            db.query(CodexEvent).filter_by(entity_id=eid).delete(synchronize_session=False)
            db.query(CodexEntity).filter_by(id=eid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
        db.commit()


def setup(db, *, old=False, known=True, created=None):
    cid, bid = uuid.uuid4(), uuid.uuid4()
    db.add(Conversation(id=cid)); db.flush()
    name = 'Atlas-' + uuid.uuid4().hex
    user = f'We are evaluating {name} ports. ' + ('Background discussion. ' * 60) + 'I refused port 9123; use port 7813.'
    assistant = 'Understood.'
    source = EpisodicMemory(conversation_id=cid, batch_id=bid,
        raw_text=f'User: {user}\n\nAssistant: {assistant}', context_reliance='Long_Term_Memory',
        source_spans=chat_provenance(user, assistant) if known else None, idempotency_key=str(uuid.uuid4()))
    entity = CodexEntity(canonical_name=name, description='previous account' if old else '',
                         last_updated=datetime.now(timezone.utc) - timedelta(days=30))
    db.add_all([source, entity]); db.flush()
    if created is not None:
        created.append((cid, entity.id))
    db.add_all([CodexEvent(entity_id=entity.id, event_type='edge_added', batch_source=bid)
                for _ in range(3 if old else 1)])
    db.commit()
    return source, entity


def renderer(db, entity):
    reader = HybridRetrievalOrchestrator(db, None)
    result = []
    reader._render_codex_entity(entity, 0, [], [], None, result)
    return '\n'.join(result)


@pytest.mark.parametrize('mode', ['supported', 'contradicted', 'length', 'malformed', 'unknown_roles'])
def test_actual_sql_complete_provider_and_source_receipt(monkeypatch, mode, created):
    assert os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_')
    monkeypatch.setattr(support, 'support_tokenizer', lambda: Tokenizer())
    score = dict(entailment=.005, neutral=.005, contradiction=.99) if mode == 'contradicted' else dict(entailment=.99, neutral=.005, contradiction=.005)
    monkeypatch.setattr(support, 'score_pairs', lambda pairs: [score for _ in pairs])
    calls = []
    candidate = 'The user approved Atlas port 9123.' if mode == 'contradicted' else 'The user refused Atlas port 9123 and chose port 7813.'
    def create(**kwargs):
        calls.append(kwargs)
        assert kwargs['max_tokens'] == 900
        assert kwargs['response_format']['json_schema']['schema'] == entity_notes.SCHEMA
        assert 'refused port 9123' in kwargs['messages'][1]['content'], 'complete tail was lost'
        content = 'malformed' if mode == 'malformed' else json.dumps(dict(summary=[dict(text=candidate, sources=['w0'])]))
        return NS(choices=[NS(finish_reason='length' if mode == 'length' else 'stop', message=NS(content=content))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda: NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'fixture')
    from src.model_registry import registry, runtime_probe
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *_: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *_: 32768)
    with SessionLocal() as db:
        source, entity = setup(db, old=True, known=mode != 'unknown_roles', created=created)
        candidate = candidate.replace('Atlas', entity.canonical_name)
        raw = source.raw_text
        receipt = reflection._enrich_codex_entities(db)
        db.commit(); db.expire_all()
        entity = db.get(CodexEntity, entity.id)
        assert receipt['selected'] == receipt['attempted'] == 1
        assert len(calls) == (0 if mode == 'unknown_roles' else 1)
        assert db.get(EpisodicMemory, source.id).raw_text == raw
        if mode == 'supported':
            assert entity.description == candidate and receipt['supported'] == 1
            assert entity_notes.current_note_support(db, entity)
            assert '[Source-supported account]' in renderer(db, entity)
            _regenerate_context_payload(entity, db)
            assert '_description_support' not in entity.context_payload and '_description_review' not in entity.context_payload
            assert db.query(CodexEvent).filter_by(entity_id=entity.id, event_type='description_updated').count() == 1
            from scripts.z1 import seed_v3, report_v3_replay
            monkeypatch.setattr(settings, 'maintenance_intervals', {'reflection': 7200})
            sink = io.StringIO()
            seed_v3.due_maintenance(db, sink, datetime.now(timezone.utc), source.conversation_id, {})
            event = json.loads(sink.getvalue())
            assert event['result']['entity_enrichment']['selected'] == 0
            with TemporaryDirectory() as temporary:
                path = Path(temporary) / 'seed.jsonl'
                path.write_text(json.dumps(dict(event='run', meta={'extra': {}})) + '\n' + sink.getvalue())
                observed = report_v3_replay.summarize(path)['entity_enrichment']['counts']
                assert observed['observed_calls'] == 1 and observed['selected'] == 0
        else:
            assert entity.description == 'previous account'
            assert receipt['supported'] == 0 and receipt['errors'] == 0
            if mode == 'malformed':
                assert entity.properties[entity_notes.REVIEW_KEY]['generated_candidate'] == 'malformed'
            assert not entity_notes.current_note_support(db, entity)
            assert '[Unverified stored note]' in renderer(db, entity)
            assert db.query(CodexEvent).filter_by(entity_id=entity.id, event_type='description_updated').count() == 0


def test_current_proof_survives_cold_copy_but_not_changed_source_or_description(monkeypatch, created):
    assert os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_')
    monkeypatch.setattr(support, 'support_tokenizer', lambda: Tokenizer())
    monkeypatch.setattr(support, 'score_pairs', lambda pairs: [dict(entailment=.99, neutral=.005, contradiction=.005) for _ in pairs])
    with SessionLocal() as db:
        source, entity = setup(db, created=created)
        candidate = f'The user chose {entity.canonical_name} port 7813.'
        note, proof = entity_notes.generate_note(entity, [source], llm=lambda *a, **k:
            json.dumps(dict(summary=[dict(text=candidate, sources=['w0'])])))
        assert note == candidate
        entity.description = note
        entity.properties = {entity_notes.PROOF_KEY: proof}
        db.commit()
        assert entity_notes.current_note_support(db, entity)
        cold = ColdStorage(id=source.id, batch_id=source.batch_id, conversation_id=source.conversation_id,
            session_id=source.session_id, raw_text=source.raw_text, source_spans=source.source_spans,
            timestamp=source.timestamp, ts_provenance=source.ts_provenance,
            context_reliance=source.context_reliance, idempotency_key=source.idempotency_key)
        db.add(cold); db.delete(source); db.commit()
        assert entity_notes.current_note_support(db, entity) and '[Source-supported account]' in renderer(db, entity)
        entity.description = 'Different unsupported claim'
        assert not entity_notes.current_note_support(db, entity)
        entity.description = candidate
        cold.raw_text += '\nA later correction.'
        db.commit()
        assert not entity_notes.current_note_support(db, entity)
        assert '[Unverified stored note]' in renderer(db, entity)
        assert db.get(ColdStorage, cold.id).raw_text.endswith('A later correction.')
