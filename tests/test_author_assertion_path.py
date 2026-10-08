"""Real E4B through the graph writer; run only in a disposable database."""

import json
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "support"))
import pytest

from author_assertion_case import exercise_writer

if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipts', type=Path)
    args = parser.parse_args()

    def sink(receipts):
        if args.receipts:
            args.receipts.write_text(json.dumps(receipts, indent=2) + '\n')

    with pytest.MonkeyPatch.context() as patches:
        print(json.dumps(exercise_writer(patches, actual=True, receipt_sink=sink), indent=2))
