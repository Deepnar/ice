"""Blinded independent-answer review from immutable v3 saved outputs.

No DB/model/API calls. Labels are kept separately from immutable evidence.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.z1.answer_as_of import private_path
from scripts.z1.judge_answers import GRADES, probe_key
from scripts.z1.replay_checkpoint import atomic_json

POLICY = dict(version='v3', method='independent_answer_review_v1', seed=20261009,
              agreement_controls=24, qualification_rows=20, batch_size=10)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def indexed(records):
    result = {}
    for r in records:
        key = probe_key(r)
        if key in result:
            raise ValueError('duplicate question occurrence')
        result[key] = r
    return result


def stratum(row, grade):
    review = row.get('label_review') or {}
    size = len(row['answer'])
    return (grade, review.get('knowledge_scope', 'unknown'),
            tuple(sorted(review.get('task_types') or [])),
            'short' if size < 500 else 'medium' if size < 1500 else 'long')


def round_robin(candidates, strata):
    """Fixed-seed coverage; rare nonempty cells get an initial opportunity."""
    rng = random.Random(POLICY['seed'])
    buckets = defaultdict(list)
    for key in sorted(candidates):
        buckets[strata[key]].append(key)
    order = sorted(buckets, key=lambda k: (len(buckets[k]), k))
    for k in order:
        rng.shuffle(buckets[k])
    result = []
    while any(buckets.values()):
        for k in order:
            if buckets[k]:
                result.append(buckets[k].pop())
    return result


def build_packet(answers, contrasts):
    if answers.get('version') != 'v3' or len(contrasts) != 3:
        raise ValueError('requires v3 full answers and exactly three frozen contrasts')
    records = indexed(answers['records'])
    comparison_rows = []
    for c in contrasts:
        if (c.get('arm_a') != answers.get('tag') or c.get('operator_required')
                or not (c.get('processing_complete') or c.get('complete'))):
            raise ValueError('contrast is incomplete or does not use these full answers')
        comparison_rows.append(indexed(c['results']))
    disagreements, agreements, common = [], [], []
    strata = {}
    for key, r in records.items():
        grades = [c.get(key, {}).get('arm_a_grade') for c in comparison_rows]
        if (r.get('error') or not isinstance(r.get('answer'), str) or not r['answer'].strip()
                or any(g not in GRADES for g in grades)):
            continue
        if (r.get('gold_source_complete') is not True
                or not isinstance(r.get('gold_turn_text'), str) or not r['gold_turn_text'].strip()
                or not isinstance(r.get('expected_answer'), str) or not r['expected_answer'].strip()):
            raise ValueError('review occurrence lacks complete source/reviewed key')
        common.append(key)
        strata[key] = stratum(r, grades[0])
        (disagreements if len(set(grades)) > 1 else agreements).append(key)
    count = POLICY['agreement_controls']
    if len(agreements) < count:
        raise ValueError('not enough common agreement controls')
    selected = sorted(disagreements) + round_robin(agreements, strata)[:count]
    random.Random(POLICY['seed']).shuffle(selected)
    families = {}
    for row in records.values():
        repeat = row.get('development_repeat') or {}
        if repeat.get('base_probe_id') and repeat.get('question_family_id'):
            base = (row.get('conversation'), repeat['base_probe_id'])
            if base in families and families[base] != repeat['question_family_id']:
                raise ValueError('inconsistent repeated-question family')
            families[base] = repeat['question_family_id']
    cases, mapping, groups = [], [], defaultdict(list)
    for number, key in enumerate(selected, 1):
        row = records[key]
        case_id = f'case-{number:02d}'
        review = row.get('label_review') or {}
        case = dict(case_id=case_id, question=row['question'],
            question_time=row.get('question_time'), question_time_provenance=row.get('question_time_provenance'),
            expected_answer=row['expected_answer'], source=row['gold_turn_text'], answer=row['answer'],
            knowledge_scope=review.get('knowledge_scope'), task_types=review.get('task_types') or [])
        cases.append(case)
        repeat = row.get('development_repeat') or {}
        base = (row.get('conversation'), repeat.get('base_probe_id') or row.get('probe_id'))
        family = (row.get('conversation'), families.get(base) or base[1] or digest(row['question']))
        groups[family].append(key)
        mapping.append(dict(case_id=case_id, probe_key=key, family=family,
                            cohort='disagreement' if key in disagreements else 'agreement',
                            old_grades=[c[key]['arm_a_grade'] for c in comparison_rows]))
    # Grouped exact-size reservation before any human labels; ties are frozen.
    family_order = []
    by_key = {key: family for family, keys in groups.items() for key in keys}
    for key in round_robin(selected, strata):
        if by_key[key] not in family_order:
            family_order.append(by_key[key])
    target, choices = POLICY['qualification_rows'], {0: []}
    for family in family_order:
        weight = len(groups[family])
        for size, chosen in sorted(list(choices.items()), reverse=True):
            if size + weight <= target and size + weight not in choices:
                choices[size + weight] = chosen + [family]
    if target not in choices:
        raise ValueError('cannot reserve exact qualification size without splitting families')
    heldout = {key for family in choices[target] for key in groups[family]}
    for m in mapping:
        m['partition'] = 'qualification' if tuple(m['probe_key']) in heldout else 'development'
    packet = dict(version='v3', kind=POLICY['method'], cases=cases)
    summary = dict(common_fully_graded=len(common), disagreements=len(disagreements),
        agreement_controls=count, selected=len(cases), qualification=len(heldout),
        development=len(cases) - len(heldout), eligible_agreement_cells=len({strata[k] for k in agreements}),
        selected_agreement_cells=len({strata[k] for k in selected if k in agreements}),
        sample_basis='development answer occurrences; related families grouped, not population precision')
    return packet, mapping, summary


INSTRUCTIONS = """Grade each candidate answer independently as correct, partial, incorrect or uncertain.
Use the exact question and complete source. The expected answer is a reviewed
reference; flag an ambiguous question/key or insufficient evidence as uncertain.
Do not require omitted private history when the question permits public knowledge.
Historical assistant advice is evidence of what was said, not verified world truth.
Preserve role, date, polarity, every required part and stated uncertainty. An answer
can be correct without proving memory helped. Do not reward extra length by itself.
Give a short source-based reason. Enter grades/reasons by case ID in labels.json
or reply in chat. Leave uncertain cases uncertain. Existing model grades, arm names,
selection cohorts and qualification partition are intentionally hidden.
"""


def render_batch(cases):
    parts = ['# ICE v3 independent answer review\n\n' + INSTRUCTIONS]
    for c in cases:
        parts.append(f"\n## {c['case_id']}\n\nQuestion\n\n{c['question']}\n\n"
            f"Question cutoff: {c['question_time']} ({c['question_time_provenance']})\n\n"
            f"Reviewed reference answer\n\n{c['expected_answer']}\n\n"
            f"Complete source\n\n{c['source']}\n\nCandidate answer\n\n{c['answer']}\n")
    return '\n'.join(parts)


def check_labels(path, packet_hash, cases):
    if not path.exists():
        raise FileNotFoundError('review labels are missing')
    labels = json.loads(path.read_text())
    ids = [c['case_id'] for c in labels.get('cases', [])]
    if (labels.get('packet_sha256') != packet_hash or len(set(ids)) != len(ids)
            or set(ids) != {c['case_id'] for c in cases}):
        raise ValueError('label template has a different packet/case identity')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--contrast', type=Path, action='append', required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    inputs = [private_path(p) for p in [args.answers, *args.contrast]]
    raw = [p.read_bytes() for p in inputs]
    sha = [hashlib.sha256(value).hexdigest() for value in raw]
    answers, *contrasts = [json.loads(value) for value in raw]
    if any(c.get('answer_file_sha256', {}).get('a') != sha[0] for c in contrasts):
        raise ValueError('contrast hash does not bind the supplied full-answer bytes')
    packet, mapping, summary = build_packet(answers, contrasts)
    identity = dict(policy=POLICY, input_sha256=sha,
                    implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out = private_path(args.out_dir)
    payloads = {'packet.json': json.dumps(packet, indent=2, ensure_ascii=False) + '\n',
                'mapping-private.json': json.dumps(mapping, indent=2) + '\n'}
    for i in range(0, len(packet['cases']), POLICY['batch_size']):
        payloads[f'batch-{i // POLICY["batch_size"] + 1:02d}.md'] = render_batch(packet['cases'][i:i + POLICY['batch_size']])
    fingerprints = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in payloads.items()}
    manifest = dict(identity=identity, immutable_files=fingerprints, summary=summary)
    manifest_path = out / 'manifest.json'
    labels = out / 'labels.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('review identity changed; use a fresh directory')
        if any(not (out / name).exists() or hashlib.sha256((out / name).read_bytes()).hexdigest() != sha
               for name, sha in fingerprints.items()):
            raise ValueError('immutable review evidence changed or is missing')
        check_labels(labels, fingerprints['packet.json'], packet['cases'])
        print(json.dumps(dict(**summary, existing_verified=True, labels_retained=True)))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    # A interrupted construction can retry matching files without touching labels.
    for name, value in payloads.items():
        path = out / name
        if path.exists() and path.read_text() != value:
            raise ValueError('existing review artifact differs; use a fresh directory')
    for name, value in payloads.items():
        if not (out / name).exists():
            temporary = out / (name + '.tmp')
            temporary.write_text(value)
            temporary.replace(out / name)
    if labels.exists():
        check_labels(labels, fingerprints['packet.json'], packet['cases'])
    if not labels.exists():
        atomic_json(labels, dict(packet_sha256=fingerprints['packet.json'], reviewer=None,
            cases=[dict(case_id=c['case_id'], grade=None, reason='') for c in packet['cases']]))
    atomic_json(manifest_path, manifest)
    print(json.dumps(summary))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
