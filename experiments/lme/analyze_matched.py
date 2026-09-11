#!/usr/bin/env python3
"""ICE v2 matched LongMemEval analysis; reads local evidence, exports aggregates only.

Run from repository root:
  uv run python experiments/lme/analyze_matched.py
No model, database, or network calls. Questions are resampled jointly across arms
and phases. Intervals condition on these runs and this judge, not writer/judge
rerun variation. Cost fields retain their actual measurement boundaries.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARMS = ('full_ice', 'vector_rag')
PHASES = ('oracle', 'full')
SEED = 20260911
B = 20000


def bootstrap(values, seed=SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.concatenate([
        values[rng.integers(len(values), size=(min(1000, B-i), len(values)))].mean(axis=1)
        for i in range(0, B, 1000)
    ])
    return dict(n=len(values), difference_pp=float(100*values.mean()),
                ci95_pp=(100*np.quantile(samples, [.025, .975])).tolist())


def paired(x, y):
    x, y = np.asarray(x, int), np.asarray(y, int)
    discordant = x != y
    n = int(discordant.sum())
    smaller = min(int(((x==1)&(y==0)).sum()), int(((x==0)&(y==1)).sum()))
    exact_p = min(1.0, 2*sum(math.comb(n,k) for k in range(smaller+1))/2**n) if n else 1.0
    return dict(**bootstrap(x-y), exact_mcnemar_p=exact_p, both_correct=int(((x==1)&(y==1)).sum()),
                ice_only=int(((x==1)&(y==0)).sum()),
                vector_only=int(((x==0)&(y==1)).sum()),
                both_wrong=int(((x==0)&(y==0)).sum()))


def distribution(values):
    a = np.asarray([v for v in values if v is not None], float)
    if not len(a): return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()),
                p25=float(np.quantile(a,.25)), median=float(np.median(a)),
                p75=float(np.quantile(a,.75)), p95=float(np.quantile(a,.95)))


def analyze(root):
    records, verdicts, digest = {}, {}, hashlib.sha256()
    for phase in PHASES:
        records[phase], verdicts[phase] = {}, {}
        for path in sorted((root/phase/'answers').glob('*.json')):
            raw = path.read_bytes(); digest.update(raw)
            r = json.loads(raw); q = r['question_id']
            assert q not in records[phase]
            assert r['adapter_version'] == 'ice-v2-lme-sessions-v2'
            assert set(r['answers']) == set(ARMS)
            assert r['status'] == 'complete', (phase,q,r['status'])
            for a in ARMS:
                assert r['answers'][a]['model'] == 'gpt-5.6-luna'
            records[phase][q] = r
        for path in sorted((root/phase/'judgements').glob('*.json')):
            raw = path.read_bytes(); digest.update(raw)
            j = json.loads(raw); key = (j['question_id'],j['condition'])
            assert key not in verdicts[phase]
            assert j['judge_model'] == 'muse-spark-1.3-contributor'
            assert type(j['label']) is bool
            verdicts[phase][key] = int(j['label']) if j['spoke'] else None
        assert len(records[phase]) == 500
        assert len(verdicts[phase]) == 1000
    ids = sorted(records['oracle'])
    assert set(ids) == set(records['full'])
    def cat(p,q):
        r=records[p][q]
        return 'abstention' if r['is_abstention'] else r['question_type']
    assert all(cat('oracle',q)==cat('full',q) for q in ids)
    out = dict(system='ICE v2; v2-paper-eval', seed=SEED, resamples=B,
               source_sha256=digest.hexdigest(), phases={}, transitions={}, cost={})
    for phase in PHASES:
        out['phases'][phase] = {}
        for category in ['overall']+sorted({cat(phase,q) for q in ids}):
            qs=[q for q in ids if category=='overall' or cat(phase,q)==category]
            good=[q for q in qs if all(verdicts[phase][q,a] is not None for a in ARMS)]
            v = paired(*[[verdicts[phase][q,a] for q in good] for a in ARMS])
            v['arms']={}
            for a in ARMS:
                ys=[verdicts[phase][q,a] for q in qs if verdicts[phase][q,a] is not None]
                v['arms'][a]=dict(correct=sum(ys), n=len(ys), total=len(qs),
                    accuracy_pct=100*sum(ys)/len(ys),
                    all_case_bounds_pct=[100*sum(ys)/len(qs),100*(sum(ys)+len(qs)-len(ys))/len(qs)])
            out['phases'][phase][category]=v
        out['cost'][phase]={}
        for a in ARMS:
            out['cost'][phase][a]={}
            for category in ['overall']+sorted({cat(phase,q) for q in ids}):
                for outcome in ['all','correct','incorrect','missing']:
                    qs=[q for q in ids if (category=='overall' or cat(phase,q)==category)
                        and (outcome=='all' or verdicts[phase][q,a]=={'correct':1,'incorrect':0,'missing':None}.get(outcome))]
                    rows=[records[phase][q]['answers'][a] for q in qs]
                    out['cost'][phase][a][category+'/'+outcome]=dict(
                        n=len(rows),
                        fragments=distribution([r['fragments'] for r in rows]),
                        prompt_tokens_estimate=distribution([r['tokens_injected'] for r in rows]),
                        provider_input_tokens=distribution([r.get('provider_usage',{}).get('input_tokens') for r in rows]),
                        generation_seconds=distribution([r['seconds'] for r in rows]),
                        noncomplete=sum(r['status']!='complete' for r in rows),
                        empty_answers=sum(not r['answer'].strip() for r in rows))
    for a in ARMS:
        qs=[q for q in ids if all(verdicts[p][q,a] is not None for p in PHASES)]
        x=np.array([verdicts['oracle'][q,a] for q in qs]); y=np.array([verdicts['full'][q,a] for q in qs])
        out['transitions'][a]=dict(**bootstrap(x-y),correct_to_wrong=int(((x==1)&(y==0)).sum()),wrong_to_correct=int(((x==0)&(y==1)).sum()))
    complete=[q for q in ids if all(verdicts[p][q,a] is not None for p in PHASES for a in ARMS)]
    delta=[(verdicts['oracle'][q,'full_ice']-verdicts['full'][q,'full_ice'])-
           (verdicts['oracle'][q,'vector_rag']-verdicts['full'][q,'vector_rag']) for q in complete]
    out['degradation_difference']=bootstrap(delta)
    joint = Counter(''.join(str(verdicts[p][q,a]) for p in PHASES for a in ARMS) for q in complete)
    out['cross_phase_cells'] = dict(order=['oracle_ice','oracle_vector','full_ice','full_vector'],
                                   n=len(complete), counts=dict(sorted(joint.items())))
    # Independent arithmetic controls against the completed run report.
    assert [out['phases'][p]['overall']['arms'][a]['correct'] for p in PHASES for a in ARMS]==[254,364,215,347]
    assert [out['phases'][p]['overall']['n'] for p in PHASES]==[500,499]
    assert out['phases']['oracle']['overall']['ice_only']==26
    assert out['phases']['full']['overall']['vector_only']==156
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/lme/runs/v2-paper-eval-cloud-v1'))
    parser.add_argument('--output',type=Path,default=Path('experiments/lme/results/matched_cloud_analysis.json'))
    args=parser.parse_args(); out=analyze(args.root)
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    for p in PHASES:
        for c,v in out['phases'][p].items():
            print(p,c,v['n'],round(v['difference_pp'],2),[round(x,2) for x in v['ci95_pp']],
                  'paired cells', [v[k] for k in ['both_correct','ice_only','vector_only','both_wrong']])
        for a in ARMS: print(p,a,out['cost'][p][a]['overall/all'])
    print('transitions',out['transitions']); print('degradation difference',out['degradation_difference'])


if __name__=='__main__': main()
