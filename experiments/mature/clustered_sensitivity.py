"""ICE v2 LSREP sensitivity: cluster repeated observations by distinct probe.
Read-only on private source records; export aggregate statistics only.
Run from root: uv run python experiments/mature/clustered_sensitivity.py
"""
import importlib.util
import json
from pathlib import Path
import numpy as np

SPEC=importlib.util.spec_from_file_location('published',Path(__file__).with_name('exp2_bootstrap.py'))
mod=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(mod)


def interval(sums,counts,seed=20260911):
    rng=np.random.default_rng(seed); sums=np.asarray(sums); counts=np.asarray(counts)
    estimates=[]
    for _ in range(20):
        ix=rng.integers(len(sums),size=(1000,len(sums)))
        estimates.extend((sums[ix].sum(1)/counts[ix].sum(1)).tolist())
    return dict(delta=float(sums.sum()/counts.sum()),ci95=np.quantile(estimates,[.025,.975]).tolist())


def main():
    records=mod.load_records()
    master=json.loads(Path(mod.RESULTS,'master_results.json').read_text())['evaluation_run_results']
    assert len(records)==len(master)==1211
    for r,m in zip(records,master):
        r['probe_id']=m['probe_id'];r['checkpoint_id']=m['checkpoint_id']
    out={'system':'ICE v2','seed':20260911,'resamples':20000,
         'estimand':'observation-weighted mean ICE generalist minus vector generalist; pairs preserved; probe clusters resampled',
         'limitation':'conditions on four conversations from one author and recorded outcomes; not a user-population CI', 'views':{}}
    for view,rows in [('all',records),('ordinary',[r for r in records if r['conversation_id']!=mod.ICE_DEV]),('density',[r for r in records if r['conversation_id']==mod.ICE_DEV])]:
        groups={}
        for r in rows:
            g=groups.setdefault((r['conversation_id'],r['probe_id']),[])
            g.append(r['scores']['full_ice_generalist']-r['scores']['vector_rag_baseline_generalist'])
        out['views'][view]=dict(observations=len(rows),distinct_probes=len(groups),
                               **interval([sum(g) for g in groups.values()],[len(g) for g in groups.values()]))
    # Distinguish the three-dataset summary from inference over users.
    assert abs(out['views']['ordinary']['delta'])<.01
    assert .39<out['views']['all']['delta']<.40
    Path('experiments/mature/results/clustered_sensitivity.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
