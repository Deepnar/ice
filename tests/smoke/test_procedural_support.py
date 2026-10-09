"""Source decision and receipt mechanics, independently of semantic accuracy."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.memory import procedural_support as support
from src.memory.source import digest


def sources():
    return [support.PatternSource(str(i), f'Review the plan before change {i}.', digest(f'raw {i}'))
            for i in (1, 2)]


def fake_review(pattern, originals, *, supports=None, kind='observed_request'):
    """Protocol fixture, never a habit-precision measurement."""
    def sender(body):
        cases = json.loads(body['messages'][-1]['content'])
        ds = [dict(id=c['id'], supports=supports(c['original']) if supports else True,
                   kind=kind if (supports(c['original']) if supports else True) else 'unsupported',
                   reason='controlled source decision') for c in cases]
        return dict(done=True, done_reason='stop', prompt_eval_count=512, eval_count=100,
                    message=dict(content=json.dumps(dict(decisions=ds))))
    return support.review_sources(pattern, originals, sender=sender)


def test_exact_sources_and_pattern_bind_admission():
    original = sources()
    proof = fake_review('The user requests plan review.', original)
    assert support.supported_ids(proof) == {'1', '2'}
    assert proof['status'] == 'supported' and proof['kind'] == 'observed_behavior'
    assert support.proof_current(proof, 'The user requests plan review.', original)
    assert not support.proof_current(proof, 'The user prefers changes first.', original)
    assert not support.proof_current(proof, 'The user requests plan review.',
                                     [replace(original[0], text='Different.'), original[1]])
    assert not support.proof_current(proof, 'The user requests plan review.', original[:1])


def test_extra_topic_citations_cannot_create_support():
    original = sources() + [support.PatternSource(str(i), 'Unrelated topic.', digest(str(i))) for i in range(3, 11)]
    proof = fake_review('The user requests plan review.', original, supports=lambda text: text.startswith('Review'))
    assert support.supported_ids(proof) == {'1', '2'} and proof['status'] == 'supported'
    proof = fake_review('The user requests plan review.', original, supports=lambda _: False)
    assert proof['status'] == 'unsupported' and support.supported_ids(proof) == set()


@pytest.mark.parametrize('bad', [
    {'decisions': []}, {'decisions': [dict(id=1, supports=True, kind='observed_request', reason='x')] * 2},
    {'decisions': [dict(id=i, supports='true', kind='observed_request', reason='x') for i in (1, 2)]},
    {'decisions': [dict(id=i, supports=True, kind='unsupported', reason='x') for i in (1, 2)]},
])
def test_bad_native_decisions_remain_unknown(bad):
    def sender(_):
        return dict(done=True, done_reason='stop', prompt_eval_count=512, message=dict(content=json.dumps(bad)))
    proof = support.review_sources('Requests reviews.', sources(), sender=sender)
    assert proof['status'] == 'unknown' and support.supported_ids(proof) == set()


@pytest.mark.parametrize('reason', ['length', None])
def test_truncated_native_positive_never_authorizes(reason):
    ds = [dict(id=i, supports=True, kind='observed_request', reason='x') for i in (1, 2)]
    proof = support.review_sources('Requests reviews.', sources(), sender=lambda _: dict(
        done=True, done_reason=reason, prompt_eval_count=512, message=dict(content=json.dumps(dict(decisions=ds)))))
    assert proof['status'] == 'unknown'


def test_unknown_and_yield_are_distinct():
    def broken(_):
        raise RuntimeError('unavailable')
    assert support.review_sources('Requests reviews.', sources(), sender=broken)['status'] == 'unknown'
    from src.workers.runtime import JobYielded
    def yielding(_):
        raise JobYielded()
    with pytest.raises(JobYielded):
        support.review_sources('Requests reviews.', sources(), sender=yielding)


def test_transport_outage_is_not_reclassified_as_content_failure():
    import httpx
    def disconnected(_):
        raise httpx.ConnectError('controlled disconnected provider')
    proof = support.review_sources('Requests reviews.', sources(), sender=disconnected)
    with pytest.raises(httpx.ConnectError):
        support.raise_unknown(proof)
    request = httpx.Request('POST', 'http://localhost/source-review')
    def failed(_):
        raise httpx.HTTPStatusError('controlled server error', request=request,
                                   response=httpx.Response(503, request=request))
    proof = support.review_sources('Requests reviews.', sources(), sender=failed)
    with pytest.raises(httpx.HTTPStatusError) as exc:
        support.raise_unknown(proof)
    assert exc.value.response.status_code == 503
    from scripts.z1.worker_recovery import model_failure, transport_failure
    assert model_failure(exc.value)
    assert transport_failure(dict(error_type=type(exc.value).__name__, error_status=exc.value.status_code))


def test_one_overlong_original_does_not_erase_two_other_supports(monkeypatch):
    monkeypatch.setattr(support, 'count_messages', lambda ms: 9000 if 'overlong' in ms[-1]['content'] else 100)
    original = sources() + [support.PatternSource('3', 'overlong', digest('raw 3'))]
    proof = fake_review('Requests reviews.', original)
    assert proof['status'] == 'supported' and support.supported_ids(proof) == {'1', '2'}


def test_explicit_instruction_type_is_retained():
    proof = fake_review('Prefers reviews first.', sources(), kind='explicit_instruction')
    assert proof['kind'] == 'explicit_instruction'


def test_source_calendar_is_distinct_from_ingest_and_conversation_order():
    old = datetime(2010, 1, 1, tzinfo=timezone.utc)
    imported = datetime(2026, 1, 1, tzinfo=timezone.utc)
    original = sources()
    original[0] = replace(original[0], recorded_at=old, ts_provenance='original',
        conversation_id='conversation-a', sequence={'sequence_sha256': 'a' * 64, 'position': 4})
    original[1] = replace(original[1], recorded_at=imported, ts_provenance='synthetic_raw_import',
        conversation_id='conversation-a', sequence={'sequence_sha256': 'a' * 64, 'position': 9})
    rendered = support.source_time_context(original)
    assert 'sources recorded: 2010-01-01T00:00:00Z' in rendered
    assert 'source calendar unknown: 1' in rendered and '2026' not in rendered
    assert 'verified order in conversation conversa: 4 to 9' in rendered
    # Display-only clock metadata cannot invalidate a source-text proof.
    proof = fake_review('Requests reviews.', original)
    assert support.proof_current(proof, 'Requests reviews.', original)


def test_source_orders_are_bounded_and_never_compared_across_conversations():
    original = [support.PatternSource(str(i), 'Request.', digest(str(i)),
        conversation_id=f'conv-{i}', sequence={'sequence_sha256': str(i) * 64, 'position': i * 10})
        for i in range(1, 5)]
    rendered = support.source_time_context(original)
    assert 'conv-1: 10' in rendered and 'conv-2: 20' in rendered
    assert '10 to 40' not in rendered and 'further source sequences: 2' in rendered
    assert 'source calendar unknown: 4' in rendered


@pytest.mark.parametrize('field,value', [('model', 'other-model'), ('prompt_sha256', 'x' * 64),
                                        ('usage', {}), ('status', 'unknown')])
def test_unconfirmed_or_stale_receipt_cannot_strengthen(field, value):
    proof = fake_review('Requests reviews.', sources())
    proof['observations'][0][field] = value
    assert not support.proof_current(proof, 'Requests reviews.', sources())


def test_observation_label_cannot_be_upgraded_without_explicit_evidence():
    proof = fake_review('Requests reviews.', sources())
    altered = deepcopy(proof)
    altered['kind'] = 'explicit_instruction'
    assert not support.proof_current(altered, 'Requests reviews.', sources())
