"""Actual local question frames/source proof, followed by the final rescue path.

No answers generated; run through disposable_database.py. Source text is synthetic.
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'smoke'))
import pytest

from scripts.z1.run_meta import file_digest, run_meta
from src.api import source_proof
from src.api.source_need import judge_source_need
from test_temporal_axis_path import exercise_date_axis


def check_frames(*, only_temporal=False):
    source = '[source recorded: 2020-06-01T00:00:00Z] User: Petra worked at Atlas in 2010.'
    variants = ['Where did Petra work in 2010?', 'where did petra work in 2010',
                'ok so like in 2010 where was Petra working', 'where did petra wrk in 2010?']
    results = []
    captured = []
    original = source_proof.score_pairs
    def capture(pairs, **kwargs):
        probabilities = original(pairs, **kwargs)
        captured.append(dict(pairs=pairs, probabilities=probabilities))
        return probabilities
    with pytest.MonkeyPatch.context() as patches:
        patches.setattr(source_proof, 'score_pairs', capture)
        for question in ([] if only_temporal else variants):
            frame = source_proof.freeze_source_question(question)
            proof = source_proof.prove_source_fact(question, source, framed=frame)
            results.append(dict(case='workplace_variant', question=question, frame=asdict(frame), proof=asdict(proof)))
        frozen = source_proof.freeze_source_question(variants[0])
        temporal_cases = [
            ('undated_past_event', '[source recorded: 2010-06-01T00:00:00Z] User: Petra once worked at Atlas.'),
            ('unknown_import_clock', '[imported; source time unknown: 2010-06-01T00:00:00Z] User: Petra works at Atlas.')]
        opponents = temporal_cases if only_temporal else [
            ('wrong_year', 'User: Petra worked at Atlas in 2011.'),
            ('hypothetical', 'User: Petra might work at Atlas in 2010.'), *temporal_cases]
        for name, opposing in opponents:
            proof = source_proof.prove_source_fact(variants[0], opposing, framed=frozen)
            results.append(dict(case=name, frame=asdict(frozen), proof=asdict(proof)))
        if not only_temporal:
            city = 'Which city did Petra work in during 2010?'
            frozen = source_proof.freeze_source_question(city)
            for name, original_source in [('city_present', 'User: Petra worked for Atlas in Paris during 2010.'),
                                         ('city_absent', source)]:
                proof = source_proof.prove_source_fact(city, original_source, framed=frozen)
                results.append(dict(case=name, frame=asdict(frozen), proof=asdict(proof)))
            current = 'Petra worked at Atlas in 2010. Where did Petra work in 2010?'
            supplied = judge_source_need([dict(role='user', content=current)], current_only=True)
            results.append(dict(case='current_supplied', source_need=asdict(supplied)))
    return results, captured


if __name__ == '__main__':
    if not os.getenv('ICE_TEST_DATABASE', '').startswith('ice_test_'):
        raise SystemExit('Run through tests/support/disposable_database.py.')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--temporal-and-rescue-only', action='store_true',
                        help='diagnostic subset; not the full acceptance panel')
    args = parser.parse_args()
    meta = run_meta(script=__file__, args=vars(args), inputs=[file_digest(p) for p in
        [__file__, 'tests/smoke/test_temporal_axis_path.py', 'src/api/source_proof.py',
         'src/api/source_need.py', 'src/api/memory_preparation.py']],
        settings_keys=['memory_source_gate_enabled', 'memory_source_rescue_enabled',
                       'memory_source_gate_model', 'memory_source_proof_output_tokens',
                       'source_support_model', 'source_support_revision', 'source_support_threshold'],
        extra=dict(version='v3', cloud_calls=0,
                   scope='actual E4B/NLI/SQL/final preparation; classifier/ranking/graph producers controlled'))
    results, probabilities = check_frames(only_temporal=args.temporal_and_rescue_only)
    # Save diagnostic results before acceptance so failure is inspectable.
    print(json.dumps(dict(meta=meta, frame_controls=results, probabilities=probabilities), indent=2), flush=True)
    assert all(r['proof']['status'] == 'supported' for r in results if r['case'] in
               ('workplace_variant', 'city_present'))
    assert all(r['proof']['status'] != 'supported' for r in results if r['case'] in
               ('wrong_year', 'hypothetical', 'undated_past_event', 'unknown_import_clock', 'city_absent'))
    if not args.temporal_and_rescue_only:
        assert results[-1]['source_need']['decision'] == 'visible_evidence'
    with pytest.MonkeyPatch.context() as patches:
        result = exercise_date_axis(patches, actual_gate=True, base_retrieve=False)
    print(json.dumps(dict(final_rescue=result), indent=2), flush=True)
    assert result['source_action'] == 'rescue' and result['gold_in_final_prompt']
