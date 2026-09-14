from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.memory.claims import claim_representation
from src.memory.support import verify_support


def scorer(entailment, contradiction=0.0):
    return lambda pairs: [dict(entailment=entailment, contradiction=contradiction,
                               neutral=1-entailment-contradiction)]


def test_supported_compression_and_stale_hash_fallback():
    source = 'Mira chose PostgreSQL. It replaced SQLite.'
    claim = 'Mira chose PostgreSQL.'
    verdict = verify_support(source, claim, scorer=scorer(.99))
    row = SimpleNamespace(text=source, sentence=claim, verification=asdict(verdict))
    assert claim_representation(row) == claim
    row.text += ' But the decision was reversed.'
    assert claim_representation(row) == row.text


@pytest.mark.parametrize('score,contra,status', [(.1,.89,'unknown'),(.01,.98,'contradicted')])
def test_uncertainty_keeps_source(score, contra, status):
    source, claim = 'Only if approved: use Redis. Approval was refused.', 'use Redis.'
    verdict = verify_support(source, claim, scorer=scorer(score,contra))
    assert verdict.status == status
    row = SimpleNamespace(text=source,sentence=claim,verification=asdict(verdict))
    assert claim_representation(row) == source


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
