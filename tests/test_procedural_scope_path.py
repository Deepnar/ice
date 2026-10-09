"""Actual SQL/retrieve/final budgeting with bounded optional pinned reranking."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_procedural_candidate_scope import exercise_procedural_scope
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--before', action='store_true')
    args = parser.parse_args()
    meta = run_meta(script=__file__, args=vars(args), inputs=[file_digest(p) for p in
        [__file__, 'tests/smoke/test_procedural_candidate_scope.py', 'src/retrieval/orchestrator.py']],
        settings_keys=['retrieval_procedural_limit', 'procedural_min_conf', 'retrieval_rerank_model',
                       'retrieval_rerank_revision'], extra={'version': 'v3',
        'scope': 'synthetic SQL/retrieve/rerank/budget controls; other producers/classifier controlled',
        'cloud_calls': 0})
    results = []
    for case in ('project', 'allowed', 'excluded', 'topic', 'intent'):
        with pytest.MonkeyPatch.context() as patches:
            results.append(exercise_procedural_scope(patches, case,
                           expected=not args.before, actual=not args.before))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
