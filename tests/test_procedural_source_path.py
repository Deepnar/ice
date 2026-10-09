"""Bounded actual E4B source checks through the procedural transaction."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_procedural_semantic_writer import exercise_source_writer
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__,
        'tests/smoke/test_procedural_semantic_writer.py', 'src/memory/procedural_support.py',
        'src/workers/procedural_extractor.py']], extra={'version': 'v3',
        'scope': 'actual E4B/source/writer; synthetic messages and controlled extraction proposal/encoder', 'cloud_calls': 0})
    results = []
    for case in ('supported', 'unsupported'):
        with pytest.MonkeyPatch.context() as patches:
            results.append(exercise_source_writer(patches, case, actual=True))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
