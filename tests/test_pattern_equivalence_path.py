"""Bounded real encoder/NLI through actual procedural writes in disposable SQL."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_pattern_support import exercise_pattern_writer
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__,
        'tests/smoke/test_pattern_support.py', 'src/memory/pattern_support.py',
        'src/workers/procedural_extractor.py']], settings_keys=['procedural_match_candidates',
        'embedding_model_name', 'source_support_model', 'source_support_revision',
        'source_support_threshold', 'source_support_max_tokens'],
        extra={'version': 'v3', 'scope': 'synthetic SQL/writer controls; real encoder/NLI, controlled BG proposal',
               'cloud_calls': 0})
    results = []
    for case in ('same', 'opposite', 'narrow', 'nonnearest'):
        with pytest.MonkeyPatch.context() as patches:
            results.append(exercise_pattern_writer(patches, case, actual=True))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
