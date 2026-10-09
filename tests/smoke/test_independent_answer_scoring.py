"""Independent factual grades cannot change with a comparator or a restart."""
import copy
import json
import sys
import tempfile
from pathlib import Path

import httpx
import openai
import pytest

from scripts.z1 import judge_answers, score_frozen_answers as score


def row(number=1, answer='7813'):
    return dict(probe_id=f'p{number}', conversation='synthetic', split_turn=80,
                question=f'Which port {number}?', expected_answer='7813',
                question_time='2026-01-01T00:00:00+00:00', question_time_provenance='original',
                gold_source_complete=True, gold_turn_text='User: use port 7813. END_SOURCE',
                label_review=dict(knowledge_scope='private_history', task_types=['episodic_lookup']),
                answer=answer, error=None, prompt_tokens=100)


def write_arm(root, tag, records):
    path = root / (tag + '.json')
    path.write_text(json.dumps(dict(version='v3', tag=tag, processing_complete=True, records=records)))
    return path


@pytest.fixture
def runtime(monkeypatch):
    with tempfile.TemporaryDirectory(prefix='v3-independent-score-', dir=Path(score.__file__).resolve().parents[2] / 'logs') as directory:
        root = Path(directory)
        monkeypatch.setenv('ICE_JUDGE_PROFILE', 'opencode-muse13')
        monkeypatch.setenv('PROBE_API_BASE_URL', 'https://test.invalid/v1')
        monkeypatch.setenv('PROBE_API_KEY', 'synthetic-key')
        monkeypatch.setattr(judge_answers, 'load_selected_env', lambda: None)
        monkeypatch.setattr(score.time, 'sleep', lambda _: None)
        calls, controls = [], {'response': None}

        def reply(request):
            calls.append(request)
            body = json.loads(request.content)
            assert request.url.path.endswith('/responses')
            assert body['model'] == 'muse-spark-1.3-contributor'
            assert request.headers['x-opencode-session'].startswith('ice-v3-independent-')
            assert body['temperature'] == 0 and body['max_output_tokens'] == 16000
            messages = body['input']
            evidence = json.loads(messages[1]['content'])
            assert 'END_SOURCE' in evidence['source']
            assert not {'arm', 'tag', 'model', 'absolute_grades', 'prompt_tokens'} & evidence.keys()
            if controls['response']:
                custom = controls['response'](messages, evidence)
                if custom is not None:
                    return custom
            if messages[0]['content'] == score.ABSOLUTE:
                assert not {'answer_A', 'answer_B'} & evidence.keys()
                value = dict(grade='correct' if evidence['answer'] == '7813' else 'incorrect',
                             reason='Matches or contradicts the source port', basis='source')
            else:
                assert messages[0]['content'] == score.PREFERENCE
                assert 'answer' not in evidence
                value = dict(verdict='A' if evidence['answer_A'] == '7813' else 'B',
                             reason='contradicts_source', note='One answer gives a different port')
            return completed(value)

        original = openai.OpenAI
        monkeypatch.setattr(openai, 'OpenAI', lambda **kwargs: original(**kwargs,
            http_client=httpx.Client(transport=httpx.MockTransport(reply))))

        def run(paths, *, mode='absolute', execute=True, resume=True):
            argv = ['score_frozen_answers.py', '--answers', *map(str, paths),
                    '--cache-dir', str(root / 'scores'), '--mode', mode]
            if execute: argv.append('--run')
            if resume: argv.append('--resume')
            monkeypatch.setattr(sys, 'argv', argv)
            return score.main()
        yield root, calls, controls, run


def completed(value):
    return httpx.Response(200, json=dict(id='synthetic-response', object='response', status='completed',
        output=[dict(type='message', role='assistant', content=[dict(type='output_text', text=json.dumps(value))])]))


def reports(root):
    return [json.loads(p.read_text()) for p in (root / 'scores/reports').glob('*.json')]


def test_actual_sdk_cache_reuses_unique_grades_and_keeps_preferences_separate(runtime):
    root, calls, _, run = runtime
    full = write_arm(root, 'full', [row()])
    same = write_arm(root, 'same', [row()])
    other = write_arm(root, 'other', [row(answer='9123')])
    original = {p: p.read_bytes() for p in (full, same, other)}
    assert run([full, other], execute=False) == 0
    assert not calls and not (root / 'scores').exists()
    assert run([full, other]) == 0
    assert len(calls) == 2
    assert run([full, same, other]) == 0
    assert len(calls) == 2, 'adding an identical answer to a contrast must reuse its absolute grade'
    absolute_bytes = {p: p.read_bytes() for p in (root / 'scores/cache').glob('*.json')}
    assert run([full, other], mode='preferences') == 0
    assert len(calls) == 4
    preference = next(r for r in reports(root) if r['mode'] == 'preferences')
    pair = preference['records'][0]
    assert pair['winner'] == 'A' and pair['order_consistent']
    assert [v['grade'] for v in pair['absolute_grades']] == ['correct', 'incorrect']
    assert all(p.read_bytes() == b for p, b in absolute_bytes.items())
    stamps = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (root / 'scores/reports').glob('*.json')}
    assert run([other, full]) == 0  # Absolute collection order cannot change a saved report.
    assert run([full, other], mode='preferences') == 0
    assert len(calls) == 4
    assert all(p.read_bytes() == b and p.stat().st_mtime_ns == stamp for p, (b, stamp) in stamps.items())
    assert {p: p.read_bytes() for p in original} == original


def test_interrupted_request_consumes_attempt_and_resume_is_bounded(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row()])
    controls['response'] = lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run([full])
    state = json.loads(next((root / 'scores/cache').glob('*.json')).read_text())
    assert len(state['attempts']) == 1 and state['attempts'][0]['raw']['reason'] == 'unconfirmed_call'
    controls['response'] = None
    assert run([full]) == 0 and len(calls) == 2
    assert run([full]) == 0 and len(calls) == 2


@pytest.mark.parametrize('failure', ['incomplete', 'malformed', 'uncertain'])
def test_provider_and_schema_failures_stay_ungraded_uncertain_is_valid(runtime, failure):
    root, calls, controls, run = runtime
    def response(*_):
        if failure == 'incomplete':
            return httpx.Response(200, json=dict(id='incomplete-id', object='response', status='incomplete',
                incomplete_details=dict(reason='max_output_tokens'), output=[]))
        if failure == 'malformed':
            return completed(dict(grade='correct', reason='Looks plausible', basis='source', A_grade='correct'))
        return completed(dict(grade='uncertain', reason='The date is ambiguous', basis='insufficient_evidence'))
    controls['response'] = response
    full = write_arm(root, 'full', [row()])
    assert run([full]) == 0
    result = reports(root)[0]
    assert result['processing_complete']
    assert len(calls) == (1 if failure == 'uncertain' else 2)
    if failure == 'uncertain':
        assert result['records'][0]['grade']['grade'] == 'uncertain' and result['complete']
    else:
        assert result['records'][0]['grade'] is None and not result['complete']
        state = json.loads(next((root / 'scores/cache').glob('*.json')).read_text())
        if failure == 'incomplete':
            assert state['attempts'][-1]['raw']['provider_completion']['incomplete_reason'] == 'max_output_tokens'
    assert run([full]) == 0


def test_saved_first_preference_survives_second_order_interruption(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row()])
    other = write_arm(root, 'other', [row(answer='9123')])
    assert run([full, other]) == 0
    def response(*_):
        if len(calls) == 4:
            raise KeyboardInterrupt()
    controls['response'] = response
    with pytest.raises(KeyboardInterrupt):
        run([full, other], mode='preferences')
    controls['response'] = None
    assert run([full, other], mode='preferences') == 0
    assert len(calls) == 5, 'first returned order or absolute grades were repeated'
    assert next(r for r in reports(root) if r['mode'] == 'preferences')['records'][0]['winner'] == 'A'


def test_order_disagreement_cannot_regrade_answers(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row()])
    other = write_arm(root, 'other', [row(answer='9123')])
    assert run([full, other]) == 0
    original = {p: p.read_bytes() for p in (root / 'scores/cache').glob('*.json')}
    controls['response'] = lambda *_: completed(dict(verdict='A', reason='more_complete', note='Synthetic position bias'))
    assert run([full, other], mode='preferences') == 0
    pair = next(r for r in reports(root) if r['mode'] == 'preferences')['records'][0]
    assert pair['winner'] == 'UNCERTAIN' and not pair['order_consistent']
    assert [v['grade'] for v in pair['absolute_grades']] == ['correct', 'incorrect']
    assert all(p.read_bytes() == b for p, b in original.items())


def test_fresh_outage_pauses_resume_skips_terminal_calls(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row(n) for n in range(1, 5)])
    controls['response'] = lambda *_: httpx.Response(500, json=dict(error=dict(message='synthetic outage', type='server')))
    assert run([full]) == 1 and len(calls) == 6
    controls['response'] = None
    assert run([full]) == 0 and len(calls) == 7
    result = reports(root)[0]
    assert result['processing_complete'] and not result['complete']
    assert result['task_counts'] == {'ungraded': 3, 'graded': 1}


def test_unavailable_answer_and_mismatch_make_no_preference_call(runtime):
    root, calls, _, run = runtime
    failed = row(answer='')
    failed.update(error='synthetic failure', error_type='CloudCompletionError',
                  cloud_attempts=2, fault_disposition='degraded_final')
    full = write_arm(root, 'full', [row()])
    other = write_arm(root, 'other', [failed])
    assert run([full, other]) == 0 and len(calls) == 1
    assert not reports(root)[0]['complete']
    assert run([full, other], mode='preferences') == 0 and len(calls) == 1
    pair = next(r for r in reports(root) if r['mode'] == 'preferences')['records'][0]
    assert pair['ungraded_reason'] == 'answer_unavailable' and pair['winner'] == 'ERROR'
    changed = row(answer='9123')
    changed['gold_turn_text'] = 'A different source'
    write_arm(root, 'other', [changed])
    with pytest.raises(ValueError, match='different occurrences or evidence'):
        run([full, other])
    assert len(calls) == 1


@pytest.mark.parametrize('change', ['question', 'expected_answer', 'gold_turn_text', 'answer', 'split_turn', 'question_time', 'profile'])
def test_grade_key_binds_all_material_inputs(change):
    original, identity = row(), dict(profile='fixture', rubric='actual')
    key, _ = score.task('absolute', score.case(original), {'answer': original['answer']}, identity)
    changed = copy.deepcopy(original)
    other_identity = copy.deepcopy(identity)
    if change == 'profile': other_identity['profile'] = 'other'
    elif change == 'split_turn': changed[change] += 1
    else: changed[change] += ' changed'
    new, _ = score.task('absolute', score.case(changed), {'answer': changed['answer']}, other_identity)
    assert new != key


def test_preference_schema_refuses_absolute_grade_fields():
    with pytest.raises(ValueError):
        score.validate(dict(verdict='A', reason='more_complete', note='reason', A_grade='correct'), 'preference')


def test_quota_pauses_immediately_without_hidden_retry(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row(), row(2)])
    controls['response'] = lambda *_: httpx.Response(429, json=dict(error=dict(type='quota', message='synthetic quota')))
    assert run([full]) == 1 and len(calls) == 1
    state = json.loads(next((root / 'scores/cache').glob('*.json')).read_text())
    assert state['attempts'][0]['raw']['operator_required']
    controls['response'] = None
    assert run([full]) == 0 and len(calls) == 3


def test_missing_status_and_corrupt_cache_cannot_become_grades(runtime):
    root, calls, controls, run = runtime
    full = write_arm(root, 'full', [row()])
    def no_status(*_):
        payload = json.loads(completed(dict(grade='correct', reason='fixture', basis='source')).content)
        payload.pop('status')
        return httpx.Response(200, json=payload)
    controls['response'] = no_status
    assert run([full]) == 0 and len(calls) == 2
    assert reports(root)[0]['records'][0]['grade'] is None
    path = next((root / 'scores/cache').glob('*.json'))
    saved = json.loads(path.read_text())
    saved['task']['messages'][1]['content'] += 'changed source'
    path.write_text(json.dumps(saved))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='cache evidence/identity differs'):
        run([full])
    assert len(calls) == 2 and path.read_bytes() == before
