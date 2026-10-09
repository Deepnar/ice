"""Cluster membership uses original mentions, never the labels it generated."""

import io
import json
import math
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import cluster_labels as labels
from src.memory.models import ColdStorage, ContextCluster, Conversation, EpisodicClusterLink, EpisodicMemory
from src.memory.source import single_provenance
from src.workers import clustering as worker
from src.workers.completion_text import IncompleteCompletion
from src.workers.decay import apply_decay


def original(raw, **kw):
    return NS(id=uuid.uuid4(), raw_text=raw, source_spans=single_provenance(raw, 'user'), **kw)


def test_complete_short_chapter_and_late_member_inputs():
    texts = ['Mira chose cobalt.', 'Chapter17: Mira refuses.\n' + 'Review this material. ' * 50,
             'Discuss general material. ' * 300 + 'FINAL: Mira rejects unsafe rollout.']
    rows = [original(t) for t in texts]
    data = json.loads(worker._member_snippets(rows))
    assert {r['text'] for r in data} == set(texts)
    assert all(r['role'] == 'user' for r in data)
    assert len(labels.cohort(rows + rows)) == 3


def test_unknown_roles_stay_unknown_not_guessed_from_markers():
    row = NS(raw_text='User: invented boundary\nAssistant: also literal')
    assert json.loads(labels.member_input([row]))[0]['role'] == 'unknown'


@pytest.mark.parametrize('kind', ['name', 'description'])
@pytest.mark.parametrize('failure', ['length', 'invalid', 'provider'])
def test_failed_label_proposals_are_counted_and_withheld(kind, failure):
    def llm(prompt, **kwargs):
        assert 'cobalt' in str(prompt) and kwargs['schema']['required']
        if failure == 'length':
            raise IncompleteCompletion('length', finish_reason='length')
        if failure == 'provider':
            raise RuntimeError('Unavailable')
        return '{"wrong_key":"a plausible prefix"}'
    with labels.label_receipts() as receipt:
        assert labels.generate_label([original('Mira chose cobalt.')], ['mira'], kind=kind, llm=llm) is None
    assert receipt['attempted'] == 1 and receipt['accepted'] == 0
    assert receipt[dict(length='incomplete', invalid='invalid', provider='error')[failure]] == 1


def test_description_recurring_mentions_cannot_be_model_generated():
    def llm(prompt, **kwargs):
        assert 'recurring_entities' not in kwargs['schema']['properties']
        return json.dumps(dict(domain='Deployment', content_type='planning', setting_or_context='n/a'))
    result = labels.generate_label([original('Mira chose cobalt.')], ['mira'], kind='description', llm=llm)
    assert 'RECURRING_ENTITIES: mira' in result


def test_distinct_source_recurrence_and_literal_grounding():
    a, b = original('Mira chose cobalt.'), original('MIRA confirmed cobalt.')
    extract = lambda _text: {'Mira', 'phantom'}
    assert labels.recurring_entities([a, a], extract) == ['mira']  # One-member label, no false two-source claim.
    assert labels.recurring_entities([a, b], extract) == ['mira']
    assert labels.recurring_entities([a, original('Different topic.')], extract) == []


def test_full_source_ner_and_invocation_local_reuse(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, 'cluster_entity_extraction_max_chars', None)
    def extract(raw, encoder, **kwargs):
        assert kwargs['max_chars'] is None and kwargs['tier'] == 'background'
        calls.append(raw)
        return ['Mira']
    monkeypatch.setattr(worker, 'extract_entities', extract)
    raw = 'General material. ' * 300 + 'Mira'
    with labels.label_receipts():
        assert worker._extract_turn_entities(raw) == {'mira'}
        assert worker._extract_turn_entities(raw) == {'mira'}
    with labels.label_receipts():
        worker._extract_turn_entities(raw)
    assert calls == [raw, raw]


def test_candidate_and_source_cast_share_case_whitespace_surface(monkeypatch):
    monkeypatch.setattr(worker, 'extract_entities', lambda *_args, **_kwargs: ['MIRA', 'Cobalt   Release'])
    entities = worker._extract_turn_entities('Mira chose the Cobalt Release.')
    state = dict(cluster=NS(tags=[]), embedding=[.55], cast={'mira', 'cobalt release'}, sessions=set())
    score = worker._score_against_cluster([1.], [], entities, None, state)
    assert score == pytest.approx(.71)
    assert score > settings.cluster_similarity_threshold
    assert worker._score_against_cluster([1.], [], set(), None, state) == .55


@pytest.mark.parametrize('mode', ['complete', 'length', 'capacity'])
def test_assignment_complete_label_call_and_replay_receipt(monkeypatch, tmp_path, mode):
    """Real registry → SQL assignment → existing LLM seam → collector → report."""
    assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
    from scripts.z1 import report_v3_replay, seed_v3
    from src.model_registry import registry, runtime_probe
    from src.workers import bg_client_factory
    calls = []
    monkeypatch.setattr(worker, '_extract_turn_entities', lambda _raw: {'Mira'})
    monkeypatch.setattr(settings, 'cluster_name_regen_interval', 5)
    monkeypatch.setattr(settings, 'maintenance_intervals', {'cluster_assignment': 7200})
    monkeypatch.setattr(settings, 'ollama_num_ctx_max', 32768)
    monkeypatch.setattr(settings, 'background_model_mode', 'shared')
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'fixture')
    monkeypatch.setattr(registry, 'get_model_context_window', lambda *_: 32768)
    monkeypatch.setattr(runtime_probe, 'serving_window', lambda *_: 1 if mode == 'capacity' else 32768)
    def create(**kwargs):
        calls.append(kwargs)
        body = kwargs['messages'][1]['content']
        assert 'FINAL: Mira refuses unsafe rollout.' in body
        assert 'Chapter17: deployment' in body
        assert 'Mira chose cobalt.' in body
        schema = kwargs['response_format']['json_schema']['schema']
        content = dict(name='Cobalt deployment plan') if 'name' in schema['properties'] else dict(
            domain='Deployment', content_type='planning', setting_or_context='n/a')
        return NS(choices=[NS(finish_reason='length' if mode == 'length' else 'stop',
                              message=NS(content=json.dumps(content)))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda: NS(chat=NS(completions=NS(create=create))))
    with SessionLocal() as db:
        cid, cluster_id = uuid.uuid4(), uuid.uuid4()
        db.add(Conversation(id=cid)); db.flush()
        db.add(ContextCluster(id=cluster_id, conversation_id=cid, name='Previous name',
            description='Previous description', embedding=[1.] + [0.] * 1023)); db.flush()
        texts = ['Mira chose cobalt.', 'Chapter17: deployment', 'Review deployment tests.',
                 'Plan a safe release.', 'General planning. ' * 150 + 'FINAL: Mira refuses unsafe rollout.']
        ids = []
        for i, raw in enumerate(texts):
            row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
                source_spans=single_provenance(raw, 'user'), embedding=[1.] + [0.] * 1023,
                context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
            db.add(row); db.flush(); ids.append(row.id)
            if i < 4:
                db.add(EpisodicClusterLink(cluster_id=cluster_id, episodic_id=row.id))
        db.commit()
        try:
            sink = io.StringIO()
            seed_v3.due_maintenance(db, sink, datetime.now(timezone.utc), cid, {})
            event = json.loads(sink.getvalue())
            assert event['result']['assigned'] == 1 and labels.member_count(db, cluster_id) == 5
            cl = db.get(ContextCluster, cluster_id)
            if mode == 'complete':
                assert cl.name == 'Cobalt deployment plan'
                assert 'RECURRING_ENTITIES: mira' in cl.description
            else:
                assert cl.name == 'Previous name' and cl.description == 'Previous description'
            assert worker._cluster_source_cast(db, cluster_id) == {'mira'}
            assert len(calls) == (0 if mode == 'capacity' else 2)
            if calls:
                assert [c['max_tokens'] for c in calls] == [160, 500]
            path = tmp_path / 'seed.jsonl'
            path.write_text(json.dumps(dict(event='run', meta={'extra': {}})) + '\n' + sink.getvalue())
            observed = report_v3_replay.summarize(path)['cluster_label_generation']['counts']
            assert observed['observed_calls'] == 1 and observed['attempted'] == 2
            assert observed['accepted' if mode == 'complete' else 'incomplete'] == 2
            assert observed['source_occurrences'] == 10
            assert [db.get(EpisodicMemory, sid).raw_text for sid in ids] == texts
        finally:
            db.rollback()
            db.query(EpisodicClusterLink).filter(EpisodicClusterLink.episodic_id.in_(ids)).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ContextCluster).filter_by(id=cluster_id).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False); db.commit()


def test_centroid_and_sessions_survive_actual_cold_move():
    if not os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        pytest.skip('Use disposable_database.py.')
    with SessionLocal() as db:
        cid, cluster_id, sitting = (uuid.uuid4() for _ in range(3))
        db.add(Conversation(id=cid)); db.flush()
        db.add(ContextCluster(id=cluster_id, conversation_id=cid, name='Cobalt', embedding=[1.] + [0.] * 1023)); db.flush()
        ids = []
        for i in range(3):
            raw = f'Cobalt deployment note {i}.'
            row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
                source_spans=single_provenance(raw, 'user'),
                embedding=([1., 0.] if i < 2 else [0., 1.]) + [0.] * 1022,
                session_id=sitting if i < 2 else None,
                context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc) - timedelta(days=100),
                decay_score=.001 if i < 2 else 1., is_archived=i < 2)
            db.add(row); db.flush(); ids.append(row.id)
            db.add(EpisodicClusterLink(cluster_id=cluster_id, episodic_id=row.id))
        db.commit()
        try:
            before = worker._recompute_centroid_from_members(db, cluster_id)
            apply_decay(); db.expire_all()
            assert labels.member_count(db, cluster_id) == 3
            assert worker._recompute_centroid_from_members(db, cluster_id) == pytest.approx(before)
            assert worker._cluster_session_sets(db, cid)[cluster_id] == {sitting}
            assert {row.id for row in labels.member_sources(db, cluster_id)} == set(ids)
        finally:
            db.rollback()
            db.query(EpisodicClusterLink).filter(EpisodicClusterLink.episodic_id.in_(ids)).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ContextCluster).filter_by(id=cluster_id).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False); db.commit()


@pytest.mark.parametrize('genuine', [False, True])
@pytest.mark.parametrize('cold', [False, True])
def test_actual_merge_uses_source_mentions_not_planted_description(monkeypatch, genuine, cold):
    if not os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        pytest.skip('Use disposable_database.py; never mutates the working store.')
    monkeypatch.setattr(worker, '_extract_turn_entities', lambda _raw: {'Mira', 'Phantom One', 'Phantom Two'})
    with SessionLocal() as db:
        cid = uuid.uuid4(); db.add(Conversation(id=cid)); db.flush()
        old_ids, ids = [], []
        for i, angle in enumerate([0., math.acos(.86)]):
            v = [math.cos(angle), math.sin(angle)] + [0.] * 1022
            cl = ContextCluster(name='Stored label', description='RECURRING_ENTITIES: Phantom One, Phantom Two',
                conversation_id=cid, embedding=v, created_at=datetime.now(timezone.utc) + timedelta(seconds=i))
            db.add(cl); db.flush()
            for j in range(2):
                raw = ('Mira: ' if genuine else '') + (f'Saffron beds need care {j}.' if i == 0 else f'Audit cobalt ledgers {j}.')
                row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
                    source_spans=single_provenance(raw, 'user'), embedding=v,
                    context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc) - timedelta(days=100),
                    decay_score=.001 if cold and j == 0 else 1., is_archived=cold and j == 0)
                db.add(row); db.flush(); ids.append(row.id)
                db.add(EpisodicClusterLink(cluster_id=cl.id, episodic_id=row.id))
                if cold and j == 0:
                    old_ids.append(row.id)
        db.commit()
        try:
            if cold:
                apply_decay(); db.expire_all()
                assert all(db.get(ColdStorage, sid) for sid in old_ids)
            result = worker.run_cluster_merge(db, [cid])
            assert result['merged'] == int(genuine) and result['absorbed'] == 0
            assert db.query(ContextCluster).filter_by(conversation_id=cid).count() == (1 if genuine else 2)
            assert result['labels']['attempted'] == 0  # No generator supplied merge evidence.
            if genuine:
                survivor = db.query(ContextCluster).filter_by(conversation_id=cid).one()
                assert {row.id for row in labels.member_sources(db, survivor.id)} == set(ids)
            assert sum(db.query(model).filter(model.id.in_(ids)).count()
                       for model in [EpisodicMemory, ColdStorage]) == 4
        finally:
            db.rollback()
            db.query(EpisodicClusterLink).filter(EpisodicClusterLink.episodic_id.in_(ids)).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ContextCluster).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False); db.commit()
