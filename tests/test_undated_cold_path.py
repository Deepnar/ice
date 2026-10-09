"""One v3 cold-recall control through actual local encoding/reranking/preparation."""

import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scripts.z1.run_meta import file_digest, run_meta
from src.api import memory_preparation
from src.api.config import settings
from src.api.db import SessionLocal
from src.classifier.schemas import ClassificationResult
from src.memory.embedder import get_embedder
from src.memory.models import ColdStorage, Conversation, EpisodicMemory
from src.memory.source import chat_provenance


def main():
    if not os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        raise SystemExit('Use tests/support/disposable_database.py; never uses the working database.')
    encoder, cid = get_embedder(), uuid.uuid4()
    originals = [
        'My recovery phrase is cobalt harbor.',
        'My bicycle has a red basket.',
        'I bought new tea cups last month.',
    ]
    source_time = datetime.now(timezone.utc) - timedelta(days=500)
    question = 'What was my recovery phrase?'
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__,
        'src/retrieval/orchestrator.py', 'src/retrieval/reranker.py',
        'src/api/memory_preparation.py']], settings_keys=[
        'embedding_model_name', 'retrieval_rerank_model', 'timescope_cold_limit'],
        extra={'version': 'v3', 'classifier_source_need_cloud': 'not tested',
               'embedding_ner_reranker_sql_preparation': 'actual', 'cloud_calls': 0})
    with SessionLocal() as db, pytest.MonkeyPatch.context() as patches:
        patches.setattr(settings, 'memory_source_gate_enabled', False)
        patches.setattr(settings, 'retrieval_rerank_enabled', True)
        patches.setattr(settings, 'retrieval_strengthen_writes', False)
        db.add(Conversation(id=cid, memory_scope_type='none')); db.flush()
        rows = [ColdStorage(id=uuid.uuid4(), conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=f'User: {user}\n\nAssistant: Noted.',
            source_spans=chat_provenance(user, 'Noted.'), timestamp=source_time,
            embedding=encoder.encode(user).tolist(), ts_provenance='original', cluster_ids=[])
            for user in originals]
        db.add_all(rows); db.commit()
        try:
            results = []
            for tier, score, archived in [('cold', None, None), ('warm_archived', .08, True),
                                          ('warm_probation', .12, False)]:
                if tier == 'warm_archived':
                    warm = [EpisodicMemory(id=row.id, conversation_id=cid, batch_id=row.batch_id,
                        raw_text=row.raw_text, source_spans=row.source_spans, timestamp=row.timestamp,
                        ts_provenance=row.ts_provenance, embedding=list(row.embedding),
                        decay_score=score, is_archived=archived, inject_raw=True,
                        context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
                        for row in rows]
                    for row in rows:
                        db.delete(row)
                    db.flush(); db.add_all(warm); db.commit(); rows = warm
                elif tier == 'warm_probation':
                    for row in rows:
                        row.decay_score, row.is_archived = score, archived
                    db.commit()
                for weak in (False, True):
                    results.append(exercise(db, encoder, rows, cid, question, weak, tier))
            print(json.dumps(dict(meta=meta, results=results), indent=2))
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(ColdStorage).filter_by(conversation_id=cid).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()


def exercise(db, encoder, rows, cid, question, weak, tier):
    cls = ClassificationResult(prompt=question, topic_tags=[],
        intent_tags=['Factual_Retrieval'], context_reliance='Long_Term_Memory',
        raw_probs=[0.] * 27, max_confidence=1., p_ltm=1.,
        head_confidences={'topic': .01 if weak else 1., 'intent': .01 if weak else 1.})
    result = memory_preparation.prepare_memory_context(db=db, classifier=NS(embedder=encoder),
        classification=cls, conversation_id=str(cid), user_message=question,
        scope={'conversation_id': str(cid)}, turn_count=3, total_tokens=100,
        total_budget=16000, serving_window=32768, base_retrieve=True,
        memory_slots=[], bookmarked_texts=[])
    final = '\n'.join(m['content'] for m in result.prepared.messages)
    gold = any(f.source_batch_id == str(rows[0].id) for f in result.fragments)
    assert gold and 'cobalt harbor' in final and result.prepared.ledger.fits()
    model = ColdStorage if tier == 'cold' else EpisodicMemory
    assert all(db.get(model, row.id).raw_text == row.raw_text for row in rows)
    return dict(tier=tier, weak_confidence=weak, gold_in_final_prompt=gold,
        fragments=len(result.fragments), ledger_fits=True, original_sources_intact=True)


if __name__ == '__main__':
    main()
