"""Bounded actual E4B decision through inline and periodic source-order paths."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest
from test_ordered_reconciliation import exercise_ordered_update
from scripts.z1.run_meta import file_digest, run_meta


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    results = []
    meta = run_meta(script=__file__, inputs=[file_digest(p) for p in [__file__, 'src/memory/source_order.py',
        'src/workers/codex_extractor.py', 'src/workers/maintenance_agent.py',
        'tests/smoke/test_ordered_reconciliation.py']],
        settings_keys=['codex_reconcile_input_tokens', 'background_model', 'background_model_mode'],
        extra={'scope': 'synthetic authority seam; controlled encoder/proposals/support',
               'version': 'v3', 'cloud_calls': 0})
    for decision, source in (
        ('expire_old', None),
        ('keep_both', 'Petra uses Nimbus in her test project. She still uses Atlas as her primary database.'),
    ):
        for periodic in (False, True):
            with pytest.MonkeyPatch.context() as patches:
                results.append(exercise_ordered_update(patches, periodic=periodic, actual=True,
                                                       decision=decision, new_source=source))
    print(json.dumps(dict(meta=meta, results=results), indent=2))
