"""v3 stable artifact identity follows actual selected native memory bodies."""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch

import httpx
import numpy as np
import pytest
from transformers import AutoTokenizer

from src.api import memory_preparation
from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory import tokens
from src.memory.models import (BatchNote, BatchSummary, CodexEntity, CodexFact,
    CodexFactEntity, CodexFactSource, ColdChunk, ColdStorage, Conversation,
    ConversationNote, ConversationSummary, EpisodicChunk, EpisodicMemory)
from src.memory.source import chat_provenance, digest
from src.memory.summary_snapshot import bind_snapshot, compose_parts, source_snapshot
from src.retrieval import orchestrator as old
from scripts.z1.historical_clock import historical_clock

VEC = [1.] + [0.] * 1023
NOW = datetime.now(timezone.utc)
candidate = old

class Encoder:
    def encode(self, value, **_kwargs):
        return np.array([VEC for _ in value] if isinstance(value, list) else VEC)

@pytest.fixture
def store(empty_replay_database, monkeypatch):
    from src.api.db import SessionLocal
    assert os.environ['ICE_TEST_DATABASE'].startswith('ice_test_')
    monkeypatch.setattr(tokens, '_tokenizer', AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-Embedding-0.6B', local_files_only=True))
    monkeypatch.setattr(tokens, '_load_attempted', True)
    monkeypatch.setattr(settings, 'memory_source_gate_enabled', False)
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', False)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    attempts = []
    def deny(*_args, **_kwargs):
        attempts.append(1)
        raise AssertionError('model HTTP in metadata controls')
    with SessionLocal() as db, historical_clock(NOW, db.get_bind()), patch.object(httpx.Client, 'request', deny), patch.object(
            httpx.AsyncClient, 'request', deny):
        yield db
        db.rollback()
    assert not attempts

def native_fragments(db, method, *args, **kwargs):
    reader = candidate.HybridRetrievalOrchestrator(db, Encoder())
    values = getattr(reader, method)(*args, **kwargs)
    for budget in (1, 512):
        selected = reader._enforce_token_budget(values, max_tokens=budget)
        assert all(fragment.stored_artifact_ids for fragment in selected)
        assert sum(fragment.token_count for fragment in selected) <= budget
    return values


def source(db, prompt='Rhea lives in harbor city.', *, document=False, cid=None, cold=False):
    cid = cid or uuid.uuid4()
    if db.get(Conversation, cid) is None:
        db.add(Conversation(id=cid, memory_scope_type='global')); db.flush()
    raw = f'User: {prompt}\n\nAssistant: Recorded.'
    cls = ColdStorage if cold else EpisodicMemory
    row = cls(id=uuid.uuid4(), conversation_id=cid, batch_id=uuid.uuid4(),
        raw_text=raw, source_spans=chat_provenance(prompt, 'Recorded.'),
        embedding=VEC, timestamp=NOW - timedelta(days=1), ts_provenance='original',
        context_reliance='Long_Term_Memory', is_document=document,
        **({} if cold else {'idempotency_key': str(uuid.uuid4()),
                                                 'inject_raw': True, 'decay_score': 1.}))
    if cold: row.cluster_ids = []
    db.add(row); db.flush()
    return row

def test_original_chunk_selection_and_same_text_distinct_rows(store):
    db = store
    row = source(db, document=True)
    chunks = [EpisodicChunk(id=uuid.uuid4(), turn_id=row.id, chunk_index=i,
        chunk_text=body, embedding=VEC) for i, body in enumerate([
            'Opening introduction.', 'harbor marker alpha.', 'harbor marker alpha.', 'Unselected tail.'])]
    db.add_all(chunks); db.flush()
    reader = candidate.HybridRetrievalOrchestrator(db, Encoder())
    ids = []
    selected = reader._relevant_doc_chunks(row.id, {'harbor'}, artifact_ids=ids)
    assert selected == 'harbor marker alpha.\n[…]\nharbor marker alpha.'
    assert ids == [('episodic_chunks', str(c.id)) for c in chunks[1:3]]
    ids = []
    assert reader._relevant_doc_chunks(row.id, set(), artifact_ids=ids) == chunks[0].chunk_text
    assert ids == [('episodic_chunks', str(chunks[0].id))]
    lexical = native_fragments(db, '_rows_to_fragments', [row], 'episodic', prompt_text='harbor')
    assert lexical[0].stored_artifact_ids == tuple(('episodic_chunks', str(c.id)) for c in chunks[1:3])
    vector = native_fragments(db, '_vector_chunks', VEC, None, str(row.conversation_id))
    assert len(vector) == 3
    assert len({f.stored_artifact_ids for f in vector}) == 3
    row.is_document = False
    db.flush()
    original = native_fragments(db, '_rows_to_fragments', [row], 'episodic')
    assert original[0].stored_artifact_ids == (('episodic_memory', str(row.id)),)
    assert reader._relevant_doc_chunks(uuid.uuid4(), {'harbor'}, artifact_ids=[]) is None

@pytest.mark.parametrize('embedding', [None, VEC])
def test_cold_parent_and_chosen_chunks_preserve_packing(store, embedding):
    db = store
    parent = source(db, cold=True)
    chunks = [ColdChunk(id=uuid.uuid4(), turn_id=parent.id, chunk_index=i,
        chunk_text=body, embedding=VEC) for i, body in enumerate([
            'harbor first.', 'harbor second.', 'harbor third.', 'Unrelated fourth.'])]
    db.add_all(chunks); db.flush()
    values = native_fragments(db, '_cold_chunk_candidates', parent, {'harbor'}, embedding)
    assert {f.stored_artifact_ids for f in values} == {(('cold_chunks', str(c.id)),) for c in chunks[:3]}
    values = native_fragments(db, '_cold_lookup', {'harbor'}, str(parent.conversation_id), None, embedding)
    assert values[0].stored_artifact_ids == (('cold_storage', str(parent.id)),)
    assert all(f.stored_artifact_ids for f in values)

def notes(db, kind, mode):
    row = source(db)
    if kind == 'batch':
        root = BatchSummary(id=uuid.uuid4(), conversation_id=row.conversation_id,
                            start_turn_index=0, end_turn_index=1, summary_text='pending', embedding=VEC)
        db.add(root); db.flush()
        row.batch_summary_id = root.id; db.flush()
    else:
        root = ConversationSummary(conversation_id=row.conversation_id, summary_text='pending',
                                   covers_turns=1, embedding=VEC)
        db.add(root); db.flush()
    part = dict(source_ids=[str(row.id)], mode='source', text=row.raw_text, recorded_range='unknown')
    root.summary_text = compose_parts([part])
    root.source_manifest = bind_snapshot(source_snapshot(db, row.conversation_id,
        **({'batch_summary_id': root.id} if kind == 'batch' else {})), root.summary_text, parts=[part])
    if mode != 'missing':
        params = dict(ordinal=1, text=part['text'] if mode == 'indexed' else 'Stale part body.',
            mode='source', recorded_range='unknown', source_ids=part['source_ids'],
            batch_ids=[str(row.batch_id)], embedding=VEC)
        note = BatchNote(summary_id=root.id, **params) if kind == 'batch' else ConversationNote(
            conversation_id=row.conversation_id, **params)
        db.add(note)
    db.flush()
    return row, root

@pytest.mark.parametrize('kind', ['batch', 'conversation'])
@pytest.mark.parametrize('mode', ['indexed', 'missing', 'mismatch'])
def test_note_parts_use_composite_keys_and_fallback_roots(store, kind, mode):
    db = store
    row, root = notes(db, kind, mode)
    values = native_fragments(db, '_batch_summary_lookup', VEC,
        str(row.conversation_id) if kind == 'batch' else None, include_cross=kind != 'batch')
    assert len(values) == 1
    parent = str(root.id) if kind == 'batch' else str(row.conversation_id)
    table = f'{kind}_notes' if mode == 'indexed' else f'{kind}_summaries'
    key = json.dumps([parent, 1], separators=(',', ':')) if mode == 'indexed' else parent
    assert values[0].stored_artifact_ids == ((table, key),)
    if kind == 'conversation':
        db.get(Conversation, row.conversation_id).memory_scope_type = 'none'; db.flush()
        assert native_fragments(db, '_batch_summary_lookup', VEC, None) == []

def graph(db):
    """A CODEX_V4 fact from one source turn, linked to its subject and object."""
    row = source(db)
    rhea, harbor = [CodexEntity(id=uuid.uuid4(), canonical_name=name, entity_type='person',
        context_payload='', description='', source='conversation', embedding=VEC)
        for name in ('rhea', 'harbor city')]
    db.add_all([rhea, harbor]); db.flush()
    sentence = 'The user said Rhea lives in harbor city.'
    fact = CodexFact(id=uuid.uuid4(), text=sentence, text_sha256=digest(sentence), speaker='user',
        modality='asserted', subject='Rhea', relation='lives in', object='harbor city', embedding=VEC,
        mentioned_at=row.timestamp, valid_from=row.timestamp, confidence=.99)
    db.add(fact); db.flush()
    db.add(CodexFactSource(fact_id=fact.id, source_batch=row.batch_id, window_sha256=digest('w'),
        episodic_id=row.id, conversation_id=row.conversation_id, window={},
        quote='Rhea lives in harbor city.', recorded_at=row.timestamp, verification={}))
    db.add_all([CodexFactEntity(fact_id=fact.id, entity_id=rhea.id, role='subject'),
                CodexFactEntity(fact_id=fact.id, entity_id=harbor.id, role='object')])
    db.flush()
    return row, rhea, harbor, fact

def test_native_fact_ids_and_private_source_hides_the_fact(store, monkeypatch):
    db = store
    row, _rhea, _harbor, fact = graph(db)
    monkeypatch.setattr(old, 'extract_entities', lambda *_a, **_k: ['rhea'])
    scope = {'conversation_id': str(row.conversation_id)}
    values = native_fragments(db, '_facts_leg', 'Where does rhea live?', VEC, scope)
    assert values and values[0].stored_artifact_ids == (('codex_facts', str(fact.id)),)
    row.is_private = True; db.flush()
    reader = candidate.HybridRetrievalOrchestrator(db, Encoder())
    reader._resolve_exclusion_sets(scope)
    assert not any(('codex_facts', str(fact.id)) in f.stored_artifact_ids
                   for f in reader._facts_leg('Where does rhea live?', VEC, scope))

def test_final_native_preparation_retains_source_identity_and_exact_current_question(store, monkeypatch):
    db = store
    row, _anchor, _harbor, _fact = graph(db)
    monkeypatch.setattr(candidate, 'extract_entities', lambda *_a, **_k: ['rhea'])
    classification = ClassificationResult([], ['Factual_Retrieval'], 'Long_Term_Memory',
        [0.] * 27, 1., prompt='Where does rhea live?', p_ltm=1.,
        head_confidences={'topic': 1., 'intent': 1.})
    value = memory_preparation.prepare_memory_context(db=db, classifier=NS(embedder=Encoder()),
        classification=classification, conversation_id=row.conversation_id,
        user_message=classification.prompt, scope={'conversation_id': str(row.conversation_id)},
        turn_count=1, total_tokens=100, total_budget=2048, serving_window=8192,
        base_retrieve=True, memory_slots=[], bookmarked_texts=[])
    assert value.retrieve and value.prepared.ledger.fits() and value.fragments
    assert value.prepared.messages[-1] == dict(role='user', content=classification.prompt)
    assert any(('episodic_memory', str(row.id)) in f.stored_artifact_ids for f in value.fragments)
    assert all(f.text in value.prepared.messages[-3]['content'] for f in value.fragments)
    from scripts.z1.seed_v3 import fragment
    assert all(fragment(f, include_text=True)['stored_artifact_ids'] for f in value.fragments)
