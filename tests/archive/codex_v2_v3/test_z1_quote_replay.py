"""One actual local-model source-unit replay of the v3 turn54 quote failure.

Disposable database only; no full campaign, original-store mutation or cloud.
The private request/response receipt belongs under ignored logs/.
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.engine import make_url

from scripts.z1.seed_v3 import load_plan
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexClaim, CodexEdge, Conversation, EpisodicMemory, IdempotencyKey
from src.memory.source import single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor as cx
from src.workers.idempotency import job_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_')
    assert make_url(settings.database_url).database == os.environ['ICE_TEST_DATABASE']
    assert args.receipt.resolve().is_relative_to(Path('logs').resolve())
    source = load_plan(include_typed=False, capture_unlabeled_native=True)[0]['bb558b5f'][53]['prompt']
    whitespace_span = re.search(r'(\w+)[ \t]{2,}(\w+)', source)
    assert whitespace_span is not None
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid));db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch, raw_text=source,
            source_spans=single_provenance(source, 'user'),
            idempotency_key=str(uuid.uuid4()), context_reliance='Long_Term_Memory'))
        db.commit()
    original = cx.bg_client.chat.completions.create
    calls = []
    def record(**kwargs):
        result = original(**kwargs)
        choice = result.choices[0]
        calls.append(dict(request=kwargs, content=choice.message.content,
                          finish_reason=choice.finish_reason))
        return result
    cx.bg_client.chat.completions.create = record
    start = time.perf_counter()
    cx.extract_codex(str(batch))
    with SessionLocal() as db:
        claims = db.query(CodexClaim).filter_by(source_batch=batch).all()
        assert claims and all(c.sentence in source and c.role == 'user' for c in claims)
        assert any(whitespace_span.group(0) in c.sentence for c in claims)
        assert db.query(CodexEdge).filter_by(source_batch=batch).count() == 0
        assert db.query(IdempotencyKey).filter_by(key=job_key('codex',batch)).count() == 1
        fragments = HybridRetrievalOrchestrator(db,None)._codex_claims(whitespace_span.group(2), None,
            {'conversation_id':str(cid)})
        assert fragments and any(whitespace_span.group(0) in f.text for f in fragments)
        receipt = dict(version='v3', scope='one actual source-unit writer/reader control',
            elapsed_seconds=time.perf_counter()-start, claims=len(claims), graph_edges=0,
            original_whitespace_preserved=True, calls=calls)
    args.receipt.write_text(json.dumps(receipt,indent=2))
    print('v3 actual turn54 source quote writer/retriever passed; no cloud or full campaign')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
