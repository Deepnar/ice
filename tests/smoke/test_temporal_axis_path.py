"""Actual dated SQL, claim/cold readers and final preparation; ranking controlled."""

import os
import sys
import uuid
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from src.api import memory_preparation
from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_assembler import get_recent_turns
from src.api.source_need import SourceNeed
from src.classifier.schemas import ClassificationResult
from src.memory.claims import store_claims
from src.memory.models import ColdStorage, CodexClaim, Conversation, EpisodicMemory
from src.memory.source import chat_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import detect_timescope, to_scope_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'support'))
from author_assertion_case import Encoder, SCORES


def exercise_date_axis(monkeypatch, *, axis='unknown', cold=False, actual_gate=False,
                       base_retrieve=True):
    if not os.getenv('ICE_TEST_DATABASE'):
        pytest.skip('Requires disposable_database.py; never uses the working store.')
    encoder, cid, sitting = Encoder(), uuid.uuid4(), uuid.uuid4()
    question = 'Where did Petra work in 2010?'
    classification = ClassificationResult(prompt=question, topic_tags=['Software_&_Tech'],
        intent_tags=['Factual_Retrieval'], context_reliance='Long_Term_Memory',
        raw_probs=[0.] * 27, max_confidence=1., p_ltm=1., p_temporal=1.,
        head_confidences={'topic': 1., 'intent': 1.})
    ts = replace(detect_timescope(question, now=datetime(2026, 1, 1, tzinfo=timezone.utc),
                                  p_ltm=1., p_temporal=1.), axis=axis)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    monkeypatch.setattr(HybridRetrievalOrchestrator, '_codex_graph', lambda *_a, **_kw: [])
    monkeypatch.setattr(settings, 'memory_source_gate_enabled', True)
    if not actual_gate:
        monkeypatch.setattr(memory_preparation, 'judge_source_need',
                            lambda *_a, **_kw: SourceNeed('not_supplied'))

    def source(user, stamp, session=None):
        return EpisodicMemory(conversation_id=cid, raw_text=f'User: {user}\n\nAssistant: Noted.',
            source_spans=chat_provenance(user, 'Noted.'), timestamp=stamp, ts_provenance='original',
            session_id=session or uuid.uuid4(), batch_id=uuid.uuid4(), decay_score=1.,
            embedding=encoder.encode(user).tolist(), topic_tags=['Software_&_Tech'],
            intent_tags=['Factual_Retrieval'], context_reliance='Long_Term_Memory',
            inject_raw=True, idempotency_key=str(uuid.uuid4()))

    with SessionLocal() as db:
        db.add(Conversation(id=cid, memory_scope_type='none')); db.flush()
        gold = source('Petra worked at Atlas in 2010.', datetime(2020, 6, 1, tzinfo=timezone.utc))
        wrong_year = source('Petra worked at Nimbus in 2011.', datetime(2010, 6, 1, tzinfo=timezone.utc))
        db.add_all([gold, wrong_year]); db.flush()
        store_claims(db, gold, ['Petra worked at Atlas in 2010.'], encoder=encoder,
                     verifier=lambda s, c: verify_support(s, c, scorer=lambda _p: [SCORES]))
        for i in range(42):
            db.add(source(f'Unrelated weather note number {i}.',
                datetime(2021, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i), sitting))
        db.commit()
        gold_id, raw = gold.id, gold.raw_text
        try:
            if cold:
                copied = {c.name: getattr(gold, c.name) for c in ColdStorage.__table__.columns
                          if hasattr(gold, c.name)}
                copied['decay_score'] = .09
                db.add(ColdStorage(**copied)); db.delete(gold); db.commit()
            scope = {'conversation_id': str(cid), 'timescope': to_scope_dict(ts)}
            recent = get_recent_turns(db, str(cid), max_tokens=4000, max_count=40)
            assert 'Atlas' not in str(recent)
            o = HybridRetrievalOrchestrator(db, encoder)
            o._active_timescope = ts
            claims = o._codex_claims('Petra work 2010', None, scope)
            assert any('Atlas' in f.text for f in claims) == (axis != 'source_record')
            if cold:
                candidates = o._cold_lookup(set(), str(cid), scope, encoder.encode(question).tolist())
            else:
                candidates = o._bm25_episodic(classification, scope, str(cid))
            assert any(f.source_batch_id == str(gold_id) for f in candidates) == (axis != 'source_record')
            memory = memory_preparation.prepare_memory_context(db=db, classifier=NS(embedder=encoder),
                classification=classification, conversation_id=str(cid), user_message=question,
                scope=scope, turn_count=44, total_tokens=3000, total_budget=16000,
                serving_window=32768, base_retrieve=base_retrieve, memory_slots=[], bookmarked_texts=[])
            assert memory.retrieve and memory.prepared.ledger.fits()
            final = '\n'.join(m['content'] for m in memory.prepared.messages)
            present = 'Petra worked at Atlas in 2010.' in final
            assert present == (axis != 'source_record')
            if present:
                assert 'source recorded: 2020-06-01T00:00:00Z' in final
            outcome = dict(axis=axis, cold=cold, turns=44, recent_speaker_messages=len(recent),
                golden_outside_recent=True, gold_in_final_prompt=present,
                claim_reader_finds_gold=any('Atlas' in f.text for f in claims),
                source_checked=memory.source_checked, source_action=memory.action,
                fits=memory.prepared.ledger.fits(), final_fragments=len(memory.fragments))
            if actual_gate and present and base_retrieve:
                from dataclasses import asdict
                from src.api import source_proof
                from src.api.source_proof import freeze_source_question, prove_source_fact
                scores = []
                original_scorer = source_proof.score_pairs
                def capture(pairs, **kwargs):
                    result = original_scorer(pairs, **kwargs)
                    scores.append(dict(pairs=pairs, scores=result))
                    return result
                monkeypatch.setattr(source_proof, 'score_pairs', capture)
                framed = freeze_source_question(question)
                proof = prove_source_fact(question, raw, framed=framed)
                opponent = prove_source_fact(question, wrong_year.raw_text, framed=framed)
                outcome['answer_support'] = dict(gold=proof.status, wrong_year=opponent.status)
                outcome['proof_frame'] = asdict(framed)
                outcome['proof_scores'] = scores
            return outcome
        finally:
            db.rollback()
            db.query(CodexClaim).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()


@pytest.mark.parametrize('axis,cold', [
    ('unknown', False), ('source_record', False), ('event', False),
    ('unknown', True), ('source_record', True),
])
def test_referenced_event_survives_source_date_difference(monkeypatch, axis, cold):
    exercise_date_axis(monkeypatch, axis=axis, cold=cold)
