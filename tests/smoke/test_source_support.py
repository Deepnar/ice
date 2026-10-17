import pytest

from src.memory.support import verify_support


def scorer(entailment, contradiction=0.0):
    return lambda pairs: [dict(entailment=entailment, contradiction=contradiction,
                               neutral=1-entailment-contradiction)]


@pytest.mark.parametrize('score,contra,status', [(.1,.89,'unknown'),(.01,.98,'contradicted')])
def test_uncertainty_and_contradiction_are_not_support(score, contra, status):
    source, claim = 'Only if approved: use Redis. Approval was refused.', 'use Redis.'
    verdict = verify_support(source, claim, scorer=scorer(score,contra))
    assert verdict.status == status


@pytest.mark.parametrize('bad', [{}, {'entailment':float('nan'),'neutral':0,'contradiction':0},
                               {'entailment':.99,'neutral':.99,'contradiction':.99}])
def test_invalid_scores_are_unknown(bad):
    assert verify_support('Source','Claim',scorer=lambda pairs:[bad]).status == 'unknown'


def test_model_failure_is_unknown_and_yield_propagates():
    def broken(pairs):
        raise RuntimeError('unavailable')
    assert verify_support('Source','Claim',scorer=broken).status == 'unknown'
    from src.workers.runtime import JobYielded
    def yielding(pairs):
        raise JobYielded()
    with pytest.raises(JobYielded):
        verify_support('Source','Claim',scorer=yielding)
