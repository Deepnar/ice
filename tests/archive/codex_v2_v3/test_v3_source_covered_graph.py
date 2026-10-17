"""v3 graph quotations share context only with complete retained originals.

SQL evidence and final native packing are real; model seams are controlled.
Actual installed-model and before/after parity receipts are qualified separately.
"""
import hashlib
import os
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import httpx
import pytest
from sqlalchemy import event
from transformers import AutoTokenizer

from scripts.z1.historical_clock import historical_clock
from src.api import prompt_budget
from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory import tokens
from src.memory.claims import store_claims
from src.memory.models import (CodexClaimLink, CodexEdge, CodexEntity,
                               Conversation, EpisodicMemory)
from src.memory.source import chat_provenance
from src.memory.support import verify_support
from src.retrieval import orchestrator

NOW = datetime(2025, 1, 2, tzinfo=timezone.utc)


class Encoder:
    max_seq_length = 0

    def tokenizer(self, text, **kwargs):
        return {'input_ids': [1]}


@pytest.fixture
def store(empty_replay_database, monkeypatch):
    from src.api.db import SessionLocal

    assert os.environ['ICE_TEST_DATABASE'].startswith('ice_test_')
    monkeypatch.setattr(tokens, '_tokenizer', AutoTokenizer.from_pretrained(
        'Qwen/Qwen3-Embedding-0.6B', local_files_only=True))
    monkeypatch.setattr(tokens, '_load_attempted', True)
    monkeypatch.setattr(settings, 'retrieval_strengthen_writes', False)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    monkeypatch.setattr(settings, 'memory_source_gate_enabled', False)
    attempts = []

    def deny(*args, **kwargs):
        attempts.append(1)
        raise AssertionError('HTTP in controlled source/packing check')

    with SessionLocal() as db, historical_clock(
            NOW + timedelta(days=1), db.get_bind()), patch.object(
            httpx.Client, 'request', deny), patch.object(
            httpx.AsyncClient, 'request', deny):
        assert db.query(Conversation).count() == 0
        yield db
        db.rollback()
    assert not attempts


def cohort(db):
    conversation = Conversation(id=uuid.uuid4(), memory_scope_type='global')
    anchor = CodexEntity(id=uuid.uuid4(), canonical_name='atlas',
                        entity_type='entity', source='conversation')
    db.add_all([conversation, anchor])
    db.flush()
    originals, edges = [], []
    for index, port in enumerate(('8391', '9010')):
        user = (
            'For the sandbox project I selected Atlas as the service name and '
            'port8391. This is the current selection.' if index == 0 else
            'Correction: the sandbox project now uses port9010; Atlas remains '
            'the service name.')
        assistant = ('Understood: the sandbox service is Atlas on port8391.'
                     if index == 0 else 'Atlas now uses port9010.')
        original = EpisodicMemory(
            id=uuid.uuid4(), conversation_id=conversation.id,
            batch_id=uuid.uuid4(), timestamp=NOW + timedelta(minutes=index),
            ts_provenance='original',
            raw_text=f'User: {user}\n\nAssistant: {assistant}',
            source_spans=chat_provenance(user, assistant),
            idempotency_key=str(uuid.uuid4()), lossless_flag=True,
            inject_raw=True, decay_score=1., context_reliance='Long_Term_Memory')
        target = CodexEntity(id=uuid.uuid4(), canonical_name='port' + port,
                             entity_type='entity', source='conversation')
        db.add_all([original, target])
        db.flush()
        edge = CodexEdge(
            id=uuid.uuid4(), source_id=anchor.id, target_id=target.id,
            relation='uses', source_batch=original.batch_id,
            observed_batches=[original.batch_id], strength=1.,
            extraction_confidence=.9, confidence='pending',
            learned_at=original.timestamp, valid_from=original.timestamp,
            source='conversation', negated=False)
        db.add(edge)
        db.flush()
        claims = store_claims(
            db, original, [assistant], encoder=Encoder(),
            verifier=lambda premise, hypothesis: verify_support(
                premise, hypothesis, scorer=lambda pairs: [dict(
                    entailment=.99, neutral=.005, contradiction=.005)]))
        claim = next(claim for claim in claims if claim.role == 'assistant')
        db.add(CodexClaimLink(claim_id=claim.id, edge_id=edge.id,
                             relation_verification={'status': 'supported'}))
        db.flush()
        originals.append(original)
        edges.append(edge)
    return conversation, anchor, originals, edges


def graph(db, *, coverage=True):
    classification = ClassificationResult(
        [], ['Factual_Retrieval'], 'Long_Term_Memory', [0.] * 27, 1.,
        prompt='Which port does Atlas use?', p_ltm=1.)
    reader = orchestrator.HybridRetrievalOrchestrator(db, Encoder())
    reader.use_fuzzy_match = False
    with patch.object(orchestrator, 'extract_entities', lambda *args, **kw: ['atlas']):
        if not coverage:
            with patch.object(reader, '_graph_quote_requirements', return_value=()):
                return reader._codex_graph(classification)
        return reader._codex_graph(classification)


def originals(rows):
    return [orchestrator.ContextFragment(
        row.raw_text, 'episodic', 1., tokens.count(row.raw_text),
        source_batch_id=str(row.id), conversation_id=str(row.conversation_id),
        covers_entire_source=True) for row in rows]


def pack(fragments, **kwargs):
    return prompt_budget.assemble_budgeted_prompt(
        memory_slots=[], retrieved_fragments=fragments, user_message='Which port?',
        serving_window=kwargs.pop('serving_window', 20000),
        generation_reserve=100, **kwargs)


def without_coverage(**kwargs):
    """Existing assembly with only the new sharing decision disabled."""
    with patch.object(prompt_budget, '_graph_quotes_covered', return_value=False):
        return prompt_budget.assemble_budgeted_prompt(**kwargs)


def packed_value(value):
    return dict(messages=value.messages, ledger=value.ledger.snapshot(),
                removed=value.removed, item_counts=value.item_counts,
                visible_fragments=[asdict(f) for f in value.visible_fragments],
                artifact_blocks=value.artifact_blocks)


def test_native_metadata_preserves_graph_text_scores_and_queries(store):
    _, _, rows, _ = cohort(store)
    observations, statements = [], []

    def observe(*args):
        statements.append(args[2])

    event.listen(store.get_bind(), 'before_cursor_execute', observe)
    try:
        for enabled in (False, True):
            statements.clear()
            values = graph(store, coverage=enabled)
            observations.append(([asdict(f) for f in values], list(statements)))
    finally:
        event.remove(store.get_bind(), 'before_cursor_execute', observe)
    assert observations[0][1] == observations[1][1]
    old, new = observations[0][0], observations[1][0]
    metadata = {'source_quote_requirements', 'source_quote_text_sha256'}
    assert [{k: v for k, v in f.items() if k not in metadata} for f in old] == [
        {k: v for k, v in f.items() if k not in metadata} for f in new]
    assert len(new) == 1 and len(new[0]['source_quote_requirements']) == 2
    assert {row_id for row_id, text in new[0]['source_quote_requirements']} == {
        str(row.id) for row in rows}
    assert new[0]['source_quote_text_sha256'] == hashlib.sha256(
        new[0]['text'].encode()).hexdigest()


@pytest.mark.parametrize('mode', [
    'note', 'typed_preview', 'distinct_learned', 'distinct_valid', 'legacy_relation'])
def test_unique_graph_information_remains_visible(store, mode):
    _, anchor, rows, edges = cohort(store)
    if mode == 'note':
        anchor.description = 'A separate retained account.'
    if mode == 'typed_preview':
        store.get(CodexEntity, edges[0].target_id).entity_type = 'network_port'
    if mode == 'distinct_learned':
        edges[0].learned_at = NOW + timedelta(days=1)
    if mode == 'distinct_valid':
        edges[0].valid_from = NOW - timedelta(days=1)
    if mode == 'legacy_relation':
        store.query(CodexClaimLink).filter_by(edge_id=edges[0].id).delete()
    store.flush()
    values = graph(store)
    assert values and all(not f.source_quote_requirements for f in values)
    assert any(f.source_type == 'codex' for f in pack(
        originals(rows) + values).visible_fragments)


@pytest.mark.parametrize('graph_first', [False, True])
def test_complete_originals_share_context_without_duplicate_graph(store, graph_first):
    _, _, rows, _ = cohort(store)
    values, raw = graph(store), originals(rows)
    arguments = dict(
        memory_slots=[], retrieved_fragments=values + raw if graph_first else raw + values,
        user_message='Which port?', serving_window=20000, generation_reserve=100)
    unshared = without_coverage(**arguments)
    result = prompt_budget.assemble_budgeted_prompt(**arguments)
    assert result.ledger.fits() and result.visible_fragments == raw
    assert result.ledger.total() < unshared.ledger.total()
    body = '\n'.join(message['content'] for message in result.messages)
    assert all(row.raw_text in body for row in rows)
    assert 'source_quote_requirements' not in body
    assert result.ledger.total() == tokens.count_messages(result.messages)


@pytest.mark.parametrize('mode', [
    'missing_source', 'wrong_id', 'partial', 'degraded', 'changed_graph', 'legacy_metadata'])
def test_incomplete_coverage_keeps_graph(store, mode):
    _, _, rows, _ = cohort(store)
    values, raw = graph(store), originals(rows)
    if mode == 'missing_source':
        raw.pop()
    if mode == 'wrong_id':
        raw[0] = replace(raw[0], source_batch_id=str(uuid.uuid4()))
    if mode == 'partial':
        raw[0] = replace(raw[0], text='Only a partial source.')
    if mode == 'degraded':
        raw[0] = replace(raw[0], covers_entire_source=False)
    if mode == 'changed_graph':
        values = [replace(values[0], text=values[0].text + ' Unique information.')]
    if mode == 'legacy_metadata':
        values = [replace(values[0], source_quote_requirements=())]
    result = pack(raw + values)
    assert all(f in result.visible_fragments for f in values)


def test_source_sharing_avoids_unnecessary_whole_evidence_eviction(store):
    _, _, rows, _ = cohort(store)
    values, raw = graph(store), originals(rows)
    limit = pack(raw).ledger.total() + 100
    arguments = dict(
        memory_slots=[], retrieved_fragments=raw + values, user_message='Which port?',
        serving_window=limit, generation_reserve=100)
    assert 'evidence' in without_coverage(**arguments).removed
    result = prompt_budget.assemble_budgeted_prompt(**arguments)
    assert 'evidence' not in result.removed
    assert result.visible_fragments == raw and result.ledger.fits()


def test_source_note_replacement_rechecks_returning_originals(store):
    conversation, _, rows, _ = cohort(store)
    values, raw = graph(store), originals(rows)
    whole = '\n\n'.join(row.raw_text for row in rows) + '\nExtra history. ' * 100
    short = 'Other history.'
    arguments = dict(
        memory_slots=[], retrieved_fragments=raw + values, user_message='Which port?',
        conversation_id=str(conversation.id))
    small = prompt_budget.assemble_budgeted_prompt(
        **arguments, conversation_summary_text=short,
        serving_window=20000, generation_reserve=100)
    result = prompt_budget.assemble_budgeted_prompt(
        **arguments, conversation_summary_text=whole,
        conversation_summary_options=[whole, short],
        conversation_summary_source_ids={whole: [str(row.id) for row in rows]},
        serving_window=small.ledger.total() + 100, generation_reserve=100)
    assert result.ledger.fits() and result.visible_fragments == raw
    assert any(entry['block'] == 'conversation_summary_part'
               for entry in result.ledger.evictions)


def test_graph_only_and_legacy_paths_preserve_final_packing(store):
    _, _, rows, _ = cohort(store)
    values = graph(store)
    arguments = dict(
        memory_slots=[], retrieved_fragments=values, user_message='Which port?',
        serving_window=20000, generation_reserve=100)
    assert packed_value(without_coverage(**arguments)) == packed_value(
        prompt_budget.assemble_budgeted_prompt(**arguments))
    arguments['retrieved_fragments'] = originals(rows) + [replace(
        f, source_quote_requirements=(), source_quote_text_sha256=None) for f in values]
    assert packed_value(without_coverage(**arguments)) == packed_value(
        prompt_budget.assemble_budgeted_prompt(**arguments))


def test_resolved_name_absent_from_original_quotes_keeps_graph(store):
    _, anchor, rows, _ = cohort(store)
    anchor.canonical_name = 'nebula gateway'
    anchor.aliases = ['atlas']
    store.flush()
    values = graph(store)
    assert values and 'nebula gateway' in values[0].text
    assert not values[0].source_quote_requirements
    assert values[0] in pack(originals(rows) + values).visible_fragments
