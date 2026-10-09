"""Fair bounded detection must change which core work reaches the real job."""

from types import SimpleNamespace as NS

import pytest

from src.api.config import settings
from src.workers import maintenance_agent as worker


def detector(name, calls, count=100):
    def detect(_db, cap):
        calls.append((name, cap))
        return [worker.WorkItem(name, 1) for _ in range(min(cap, count))]
    return detect


def test_busy_first_detector_cannot_starve_other_detector(monkeypatch):
    calls, receipts = [], []
    monkeypatch.setattr(settings, 'agent_max_scanned', 5)
    monkeypatch.setattr(worker, 'DETECTORS', {
        n: detector(n, calls) for n in ('reconciliation_leftover', 'pending_pileup', 'contradiction')})
    items = worker.detect_all(None, receipts=receipts)
    assert calls == [('reconciliation_leftover', 2), ('pending_pileup', 2), ('contradiction', 1)]
    assert len(items) == sum(r['candidate_scan_limit'] for r in receipts) == 5


def test_small_scan_budget_rotates_and_does_not_refill(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, 'agent_max_scanned', 1)
    monkeypatch.setattr(worker, '_scan_cursor', 0)
    monkeypatch.setattr(worker, 'DETECTORS', {n: detector(n, calls, count=0) for n in ('a', 'b', 'c')})
    for _ in range(3):
        assert worker.detect_all(None) == []
    assert calls == [('a', 1), ('b', 1), ('c', 1)]


def test_failed_detector_is_visible_without_consuming_other_quota(monkeypatch):
    def fail(_db, _cap):
        raise RuntimeError('Controlled outage')
    calls, receipts, rollbacks = [], [], []
    monkeypatch.setattr(settings, 'agent_max_scanned', 4)
    monkeypatch.setattr(worker, 'DETECTORS', {'a': fail, 'b': detector('b', calls)})
    assert len(worker.detect_all(NS(rollback=lambda: rollbacks.append(True)), receipts=receipts)) == 2
    assert rollbacks == [True] and receipts[0]['status'] == 'error'
    assert receipts[0]['error_type'] == 'RuntimeError' and calls == [('b', 2)]


def test_actual_run_reaches_later_detector_despite_blocked_backlog(monkeypatch):
    monkeypatch.setattr(settings, 'agent_max_scanned', 4)
    monkeypatch.setattr(worker, 'DETECTORS', {
        'reconciliation_leftover': detector('reconciliation_leftover', []),
        'pending_pileup': detector('pending_pileup', [], count=1)})
    monkeypatch.setattr(worker, '_decide_reconciliation', lambda _db, item, _decider:
        item.payload.update(_source_authority={'reason': 'source_order_unknown', 'eligible': False})
        or 'unsure')
    monkeypatch.setattr(worker, '_apply_reconciliation', lambda *_args: 'still_unsure')
    applied = []
    monkeypatch.setattr(worker, '_apply_pileup', lambda _db, item, _run:
        applied.append(item.item_type) or 'applied')
    result = worker.run_maintenance_agent(None, llm_decider=lambda *_a: None)
    assert applied == ['pending_pileup'] and result['applications'] == 1
    assert result['llm_decisions'] == 0
    assert result['source_authority_counts'] == {'source_order_unknown': 2}
    assert result['scanned'] == 3 and len(result['detector_receipts']) == 2


def test_job_counting_wrapper_preserves_provider_json_schema(monkeypatch):
    from src.workers import bg_client_factory

    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='{"decision":"keep_both"}'))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'controlled-background')
    monkeypatch.setattr(worker, 'DETECTORS', {'reconciliation_leftover':
        lambda _db, _cap: [worker.WorkItem('reconciliation_leftover', 1)]})
    monkeypatch.setattr(worker, '_decide_reconciliation', lambda _db, _item, decider:
        worker._ask_enum(decider, 'Controlled source comparison.', 'decision', ('keep_both', 'unsure')))
    monkeypatch.setattr(worker, '_apply_reconciliation', lambda _db, _item, decision, _run: decision)
    result = worker.run_maintenance_agent(None, llm_decider=worker.make_llm_decider())
    assert result['llm_decisions'] == len(requests) == 1
    schema = requests[0]['response_format']
    assert schema['type'] == 'json_schema'
    assert schema['json_schema']['schema']['properties']['decision']['enum'] == ['keep_both', 'unsure']


@pytest.mark.parametrize('finish', ['length', None, 'content_filter'])
def test_valid_looking_incomplete_verdict_is_not_maintenance_authority(monkeypatch, finish):
    from src.workers import bg_client_factory

    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        NS(chat=NS(completions=NS(create=lambda **_kw:
            NS(choices=[NS(finish_reason=finish, message=NS(content='{"decision":"expire_old"}'))])))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'controlled-background')
    assert worker.make_llm_decider()('Controlled source comparison.') is None
