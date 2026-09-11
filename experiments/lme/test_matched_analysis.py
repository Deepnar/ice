"""Numerical controls for paired resampling and aggregation (no services)."""
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('analysis',Path(__file__).with_name('analyze_matched.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def test_pairs():
    # Identical varying arms must have a point-mass zero difference. Independent
    # arm resampling would incorrectly manufacture uncertainty in this control.
    x=[0,1]*20
    assert a.paired(x,x)['ci95_pp']==[0.0,0.0]
    # Constant discordance pins the sign and all four count cells.
    r=a.paired([1]*8,[0]*8)
    assert r['difference_pp']==100 and r['ci95_pp']==[100,100]
    assert [r[k] for k in ['both_correct','ice_only','vector_only','both_wrong']]==[0,8,0,0]
    # Interchanging arm labels must reverse the paired interval.
    r=a.bootstrap([1,0,-1,1,0]);v=a.bootstrap([-1,0,1,-1,0])
    assert np.allclose(r['ci95_pp'],[-v['ci95_pp'][1],-v['ci95_pp'][0]])

def test_actual():
    r=a.analyze(Path('experiments/lme/runs/v2-paper-eval-cloud-v1'))
    assert r['phases']['full']['overall']['both_correct']==191
    assert r['phases']['full']['abstention']['exact_mcnemar_p']==0.0703125
    cells=r['cross_phase_cells']['counts']
    assert sum(cells.values())==499
    assert sum((int(k[0])-int(k[2])-int(k[1])+int(k[3]))*v for k,v in cells.items())==22
    assert r['phases']['full']['single-session-preference']['n']==29
    assert r['transitions']['full_ice']['correct_to_wrong']==68
    assert r['transitions']['vector_rag']['wrong_to_correct']==28
    assert np.isclose(r['degradation_difference']['difference_pp'],100*22/499)
    for p in a.PHASES:
        for arm in a.ARMS:
            rows=r['cost'][p][arm]
            assert sum(rows['overall/'+k]['n'] for k in ['correct','incorrect','missing'])==500
            assert rows['overall/all']['provider_input_tokens']['n']==500

if __name__=='__main__':
    test_pairs();test_actual();print('Paired controls and frozen-run arithmetic: passed')
