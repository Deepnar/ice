"""Frozen inputs, blinded evidence and grouped independent review selection."""

import json
from copy import deepcopy

import pytest

from scripts.z1 import answer_review_packet as packet
from scripts.z1.judge_answers import probe_key


def fixture():
    records = [dict(probe_id=f'p{i:03d}', conversation='synthetic', split_turn=i + 10,
        question=f'Which color was chosen for crate {i}?', answer=f'Amber {i}.',
        expected_answer=f'Amber {i}.', gold_source_complete=True, gold_turn_text=f'User: Choose amber {i}.',
        question_time='2026-01-01', question_time_provenance='original',
        label_review=dict(knowledge_scope='private_history' if i % 2 else 'public_knowledge',
                          task_types=['episodic_lookup' if i % 3 else 'temporal'])) for i in range(80)]
    records[1]['development_repeat'] = dict(base_probe_id='p000', question_family_id='same-color')
    contrasts = [dict(complete=True, arm_a='full', results=[dict(r,
        arm_a_grade='correct' if j == 0 or i >= 36 else 'partial')
        for i, r in enumerate(records)]) for j in range(3)]
    return dict(version='v3', tag='full', records=records), contrasts


def test_blinded_complete_packet_and_frozen_family_partition():
    answers, contrasts = fixture()
    p, mapping, summary = packet.build_packet(answers, contrasts)
    assert summary['common_fully_graded'] == 80 and summary['disagreements'] == 36
    assert summary['selected'] == 60 and summary['qualification'] == 20 and summary['development'] == 40
    assert packet.build_packet(answers, contrasts) == (p, mapping, summary)
    first = [m for m in mapping if tuple(m['probe_key']) in
             {probe_key(answers['records'][0]), probe_key(answers['records'][1])}]
    assert len(first) == 2 and len({m['partition'] for m in first}) == 1
    assert all('answer_model' not in c and 'old_grades' not in c and 'partition' not in c for c in p['cases'])
    assert len({c['case_id'] for c in p['cases']}) == 60
    assert all(c['source'].startswith('User: Choose') for c in p['cases'])


@pytest.mark.parametrize('failure', ['duplicate', 'incomplete_source', 'missing_key', 'wrong_arm'])
def test_incomplete_or_mismatched_evidence_is_rejected(failure):
    answers, contrasts = fixture()
    if failure == 'duplicate':
        answers['records'].append(deepcopy(answers['records'][0]))
    elif failure == 'incomplete_source':
        answers['records'][0]['gold_source_complete'] = False
    elif failure == 'missing_key':
        answers['records'][0]['expected_answer'] = ''
    elif failure == 'wrong_arm':
        contrasts[0]['arm_a'] = 'vector_only'
    with pytest.raises(ValueError):
        packet.build_packet(answers, contrasts)


def test_resume_preserves_human_labels_and_rejects_different_cases(tmp_path):
    p, _, _ = packet.build_packet(*fixture())
    path = tmp_path / 'labels.json'
    edited = dict(packet_sha256='frozen', reviewer='Independent reviewer', cases=[
        dict(case_id=c['case_id'], grade='correct', reason='The source supplies the color.') for c in p['cases']])
    path.write_text(json.dumps(edited))
    before = path.read_bytes()
    packet.check_labels(path, 'frozen', p['cases'])
    assert path.read_bytes() == before
    with pytest.raises(ValueError):
        packet.check_labels(path, 'different', p['cases'])
    edited['cases'][0]['case_id'] = 'wrong'
    path.write_text(json.dumps(edited))
    with pytest.raises(ValueError):
        packet.check_labels(path, 'frozen', p['cases'])


def test_terminal_faults_do_not_disqualify_completed_successful_review_cohort():
    answers, contrasts = fixture()
    for c in contrasts:
        c.update(complete=False, processing_complete=True, operator_required=False)
    assert packet.build_packet(answers, contrasts)[2]['selected'] == 60
    contrasts[0]['operator_required'] = True
    with pytest.raises(ValueError):
        packet.build_packet(answers, contrasts)
