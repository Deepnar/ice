"""Bounded local cluster label check; run only through disposable_database.py.

Exercises real NuNER, E4B and SQL creation on two synthetic coherent originals.
This verifies the local path, not population naming/coherence or answer quality.
"""

import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.cluster_labels import member_count, member_sources
from src.memory.models import ContextCluster, Conversation, EpisodicMemory
from src.memory.source import single_provenance
from src.workers.bg_client_factory import get_bg_model_name
from src.workers.clustering import _cluster_source_cast, run_cluster_assignment


def main():
    assert os.environ.get('ICE_TEST_DATABASE', '').startswith('ice_test_'), 'Use disposable_database.py.'
    assert get_bg_model_name() == 'gemma4:e4b', 'This check targets the current local pin.'
    texts = ['Mira plans the Cobalt deployment. Run staging tests before rollout.',
             'Chapter17: Cobalt deployment\n' + 'We are reviewing the staging rollout. ' * 35
             + 'Mira confirms: staging tests come first; cancel rollout if tests fail.']
    with SessionLocal() as db:
        cid = uuid.uuid4()
        db.add(Conversation(id=cid)); db.flush()
        ids = []
        for raw in texts:
            row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
                source_spans=single_provenance(raw, 'user'), embedding=[1.] + [0.] * 1023,
                context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
            db.add(row); db.flush(); ids.append(row.id)
        db.commit()
        started = time.perf_counter()
        result = run_cluster_assignment(db, [cid])
        cl = db.query(ContextCluster).filter_by(conversation_id=cid).one()
        cast = _cluster_source_cast(db, cl.id)
        receipt = dict(version='v3', background=get_bg_model_name(), ner=settings.background_ner_model,
            ner_max_chars=settings.cluster_entity_extraction_max_chars,
            elapsed_seconds=round(time.perf_counter()-started, 3), result=result,
            name=cl.name, description=cl.description, source_cast=sorted(cast),
            logical_members=member_count(db, cl.id))
        print(json.dumps(receipt, sort_keys=True), flush=True)
        assert result['created'] == 1 and result['assigned'] == 2
        assert result['labels']['accepted'] == result['labels']['attempted'] == 2
        assert cl.name != 'Unnamed Cluster' and 'DOMAIN:' in cl.description
        assert 'mira' in cast, 'The two distinct source mentions must reach clustering.'
        assert {row.id for row in member_sources(db, cl.id)} == set(ids)
        assert [db.get(EpisodicMemory, sid).raw_text for sid in ids] == texts
        raw = 'Mira is reviewing the Cobalt deployment staging rollout.'
        later = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
            source_spans=single_provenance(raw, 'user'),
            embedding=[.55, math.sqrt(1-.55**2)] + [0.] * 1022,
            context_reliance='Long_Term_Memory', idempotency_key=str(uuid.uuid4()))
        db.add(later); db.commit()
        assignment = run_cluster_assignment(db, [cid])
        print(json.dumps(dict(version='v3', low_cosine_assignment=assignment,
            logical_members=member_count(db, cl.id), base_cosine=.55), sort_keys=True), flush=True)
        assert assignment['assigned'] == 1 and assignment['created'] == 0
        assert assignment['labels']['attempted'] == 0
        assert member_count(db, cl.id) == 3


if __name__ == '__main__':
    main()
