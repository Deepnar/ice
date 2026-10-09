"""Actual source-refinement/proof checks over the synthetic dated retrieval seam."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_temporal_axis_path import exercise_date_axis
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--warm-only', action='store_true')
    args = parser.parse_args()
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__,
        'tests/smoke/test_temporal_axis_path.py', 'src/retrieval/timescope.py',
        'src/retrieval/orchestrator.py', 'src/api/memory_preparation.py', 'src/api/source_proof.py']],
        settings_keys=['memory_source_gate_model', 'memory_source_gate_enabled'],
        extra={'version': 'v3', 'ranking_classifier_graph_extraction': 'controlled', 'cloud_calls': 0})
    results = []
    for cold in ((False,) if args.warm_only else (False, True)):
        with pytest.MonkeyPatch.context() as patches:
            results.append(exercise_date_axis(patches, cold=cold, actual_gate=True))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
