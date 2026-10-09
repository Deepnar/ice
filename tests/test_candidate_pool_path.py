"""Actual pinned reranker through normal retrieve and unchanged final budgeting."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_reranker import exercise_candidate_pool_path
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__,
        'tests/smoke/test_reranker.py', 'src/retrieval/reranker.py', 'src/retrieval/orchestrator.py']],
        settings_keys=['retrieval_rerank_model', 'retrieval_rerank_revision', 'retrieval_rerank_candidates'],
        extra={'version':'v3','scope':'synthetic candidates; real retrieval/reranker/budget, no cloud answer'})
    result = []
    for relevant in (True, False):
        with pytest.MonkeyPatch.context() as patches:
            result.append(exercise_candidate_pool_path(patches, procedural_relevant=relevant, actual=True))
    print(json.dumps(dict(meta=meta,results=result),indent=2))
