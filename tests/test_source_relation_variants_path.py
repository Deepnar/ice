"""Real E4B relation selection through actual writer/readers in a disposable DB."""

import json
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest

from test_source_relation_variants import exercise_variant

if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-nli', action='store_true')
    args = parser.parse_args()
    results = []
    for outcome in ('raw', 'canonical'):
        with pytest.MonkeyPatch.context() as patches:
            results.append(exercise_variant(patches, outcome, actual=True, real_nli=args.real_nli))
    print(json.dumps(results, indent=2))
