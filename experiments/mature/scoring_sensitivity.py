"""ICE v2 scoring sensitivity; private inputs, aggregate-only outputs.

Run from root: uv run python experiments/mature/scoring_sensitivity.py
Complete cases require explicit scores in both generalist arms. Failure-assigned
scores are excluded there, so this is selection sensitivity, not a replacement
estimand for operational reliability. Ordinal comparisons preserve paired probe
trajectories and do not assume equal distances between rubric levels.
"""
import json
import statistics
from collections import Counter
from pathlib import Path

from clustered_sensitivity import interval, mod

ICE = 'full_ice_generalist'
VEC = 'vector_rag_baseline_generalist'


def key(r):
    return tuple(r[k] for k in ('conversation_id', 'checkpoint_id', 'probe_id'))


def load(manual):
    root = Path(mod.RESULTS)
    master = json.loads((root/'master_results.json').read_text())['evaluation_run_results']
    evaluations = {key(r): r for r in json.loads((root/'evaluation_raw.json').read_text())}
    if manual:
        for r in json.loads((root/'manual_evaluation.json').read_text()):
            entry = evaluations.setdefault(key(r), {})
            if r.get('absolute_scores'):
                entry['absolute_scores'] = {c: {'score': s} for c, s in r['absolute_scores'].items() if s is not None}
    rows = []
    for r in master:
        absolute = evaluations.get(key(r), {}).get('absolute_scores', {}) or {}
        valid = {c: s['score'] for c in mod.CONDS
                 if isinstance((s := absolute.get(c, {})), dict) and 'score' in s}
        scores, origins = {}, {}
        for c in mod.CONDS:
            cond = r['conditions'].get(c)
            if c in valid:
                value, origin = valid[c], 'explicit'
            elif cond is not None and mod._is_failed_answer(cond.get('answer', '')):
                value, origin = 1, 'failed_answer'
            elif mod.PAIRED[c] in valid:
                value, origin = valid[mod.PAIRED[c]], 'sibling_routing'
            elif valid:
                value, origin = round(statistics.mean(valid.values())), 'rounded_record_mean'
            else:
                value, origin = 3, 'default_3'
            scores[c], origins[c] = value, origin
        rows.append(dict(conversation_id=r['conversation_id'], probe_id=r['probe_id'], scores=scores, origins=origins))
    return rows


def summarize(rows):
    groups, signs = {}, {}
    counts = Counter()
    for r in rows:
        k = (r['conversation_id'], r['probe_id'])
        d = r['scores'][ICE] - r['scores'][VEC]
        sign = int(d > 0) - int(d < 0)
        groups.setdefault(k, []).append(d)
        signs.setdefault(k, []).append(sign)
        counts['ice_higher' if d > 0 else 'vector_higher' if d < 0 else 'tie'] += 1
    def estimate(g):
        return interval([sum(v) for v in g.values()], [len(v) for v in g.values()])
    return dict(n=len(rows), clusters=len(groups), mean_difference=estimate(groups),
                ordinal_counts=dict(counts), net_ordinal_superiority=estimate(signs),
                score_origin={c: dict(Counter(r['origins'][c] for r in rows)) for c in mod.CONDS})


def main():
    merged, automated = load(True), load(False)
    assert [r['scores'] for r in merged] == [r['scores'] for r in mod.load_records()]
    out = dict(system='ICE v2', seed=20260911, resamples=20000,
               complete_case='explicit paired scores only; excludes failures without scores; selection-sensitive',
               ordinal='P(ICE score > vector score) minus P(ICE score < vector score); ties count zero; probe-cluster CI', views={})
    for name, rows in [('merged', merged), ('automated_only', automated)]:
        for complete in (False, True):
            selected = [r for r in rows if not complete or all(r['origins'][c]=='explicit' for c in (ICE,VEC))]
            for regime in ('all', 'ordinary', 'density'):
                rr = [r for r in selected if regime=='all' or (r['conversation_id']==mod.ICE_DEV)==(regime=='density')]
                out['views'][f'{name}_{"complete" if complete else "historical"}_{regime}'] = summarize(rr)
    Path('experiments/mature/results/scoring_sensitivity.json').write_text(json.dumps(out, indent=2)+'\n')
    for k,v in out['views'].items():
        print(k, v['n'], v['mean_difference'], v['ordinal_counts'], v['net_ordinal_superiority'])
    print('Merged origins:',out['views']['merged_historical_all']['score_origin'])
    print('Automated origins:',out['views']['automated_only_historical_all']['score_origin'])

if __name__ == '__main__':
    main()
