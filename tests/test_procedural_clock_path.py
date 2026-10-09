"""Actual procedural source/retirement SQL, pinned reranking and final budgeting."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_procedural_clocks import exercise_retirement, exercise_source_clocks
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    meta = run_meta(script=__file__, args={}, inputs=[file_digest(p) for p in
        [__file__, 'tests/smoke/test_procedural_clocks.py', 'src/memory/procedural_support.py',
         'src/workers/procedural_decay.py', 'src/retrieval/orchestrator.py']],
        settings_keys=['procedural_stale_days', 'procedural_min_reinforcement',
                       'retrieval_rerank_model', 'retrieval_rerank_revision'], extra={'version': 'v3',
        'scope': 'synthetic clocks/SQL/retirement/final reader with actual reranker; source decisions controlled',
        'cloud_calls': 0})
    results = []
    for fn in (exercise_source_clocks, exercise_retirement):
        with pytest.MonkeyPatch.context() as patches:
            results.append(fn(patches, actual=True))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
