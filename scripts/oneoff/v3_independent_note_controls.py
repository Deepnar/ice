"""Bounded synthetic v3 controls, real NLI through the independent-note writer.

Run from repository root. No database writes, no personal corpus, no threshold fit.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.api.config import settings
from src.memory.source import single_provenance
from src.memory.support import verify_support
from src.workers.conversation_summary import _original_source, _source_note

CASES = [
    ('negation', 'user', 'Atlas keeps Redis disabled. Do not enable Redis.',
     'The user says Redis must remain disabled.', 'The user says Redis must be enabled.'),
    ('conditional', 'user', 'If Atlas gets approval, Atlas will deploy on Friday. Approval is still pending.',
     'The user says deployment on Friday depends on approval, which is still pending.',
     'The user says deployment on Friday is approved.'),
    ('speaker', 'assistant', 'I suggest using Redis, but the user has not agreed.',
     'The assistant suggests Redis; the user has not agreed.', 'The user decided to use Redis.'),
    ('correction', 'user', 'Atlas initially planned to use port 8391. Correction: Atlas now uses port 8392 instead.',
     'The user corrected the port from 8391 to 8392.', 'The user says Atlas now uses port 8391.'),
]


def run():
    rows = []
    for name, role, raw, positive, negative in CASES:
        turn = SimpleNamespace(id=name, timestamp=datetime(2026, 9, 20, tzinfo=timezone.utc),
            ts_provenance='original', raw_text=raw, source_spans=single_provenance(raw, role))
        source, known = _original_source(turn)
        for expected, candidate in [(True, positive), (False, negative)]:
            part = _source_note([turn], source, known,
                                lambda *a, **k: candidate, verify_support)
            admitted = part['mode'] == 'supported'
            rows.append(dict(case=name, expected_supported=expected, admitted=admitted,
                candidate=candidate, source=source, verification=part['verification'],
                original_preserved=admitted or part['text'] == source,
                passed=admitted == expected and (admitted or part['text'] == source)))
    result = dict(version='ICE v3', model=settings.source_support_model,
        revision=settings.source_support_revision, threshold=settings.source_support_threshold,
        scope='8 synthetic model-plus-writer decisions; not benchmark or general quality estimate',
        passed=sum(r['passed'] for r in rows), total=len(rows), rows=rows)
    target = Path('experiments/v3_repair/independent_notes/controls.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    run()
