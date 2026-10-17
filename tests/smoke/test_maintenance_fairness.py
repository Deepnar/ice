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
        n: detector(n, calls) for n in ('duplicate_entities', 'stale_slot', 'stale_work')})
    items = worker.detect_all(None, receipts=receipts)
    assert calls == [('duplicate_entities', 2), ('stale_slot', 2), ('stale_work', 1)]
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
    # CODEX_V4: the legacy edge detectors are gone; an undecidable duplicate
    # backlog must still leave room for the stale-slot detector.
    monkeypatch.setattr(settings, 'agent_max_scanned', 4)
    monkeypatch.setattr(worker, 'DETECTORS', {
        'duplicate_entities': detector('duplicate_entities', []),
        'stale_slot': detector('stale_slot', [], count=1)})
    monkeypatch.setattr(worker, '_decide_duplicate', lambda _db, item, _decider: 'unsure')
    reached = []
    monkeypatch.setattr(worker, '_decide_stale_slot', lambda _db, item, _decider:
        reached.append(item.item_type) or None)
    result = worker.run_maintenance_agent(None, llm_decider=lambda *_a: None)
    assert reached == ['stale_slot']
    assert result['outcomes'].get('duplicate_entities:unsure') == 2
    assert result['scanned'] == 3 and len(result['detector_receipts']) == 2


def test_job_counting_wrapper_preserves_provider_json_schema(monkeypatch):
    from src.workers import bg_client_factory

    requests = []
    def create(**kwargs):
        requests.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='{"verdict":"different"}'))])
    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'controlled-background')
    monkeypatch.setattr(worker, 'DETECTORS', {'duplicate_entities':
        lambda _db, _cap: [worker.WorkItem('duplicate_entities', 2)]})
    monkeypatch.setattr(worker, '_decide_duplicate', lambda _db, _item, decider:
        worker._ask_enum(decider, 'Controlled duplicate check.', 'verdict', ('same', 'different', 'unsure')))
    monkeypatch.setattr(worker, '_dismiss_pair', lambda _db, _item, verdict, _run: verdict)
    result = worker.run_maintenance_agent(None, llm_decider=worker.make_llm_decider())
    assert result['llm_decisions'] == len(requests) == 1
    schema = requests[0]['response_format']
    assert schema['type'] == 'json_schema'
    assert schema['json_schema']['schema']['properties']['verdict']['enum'] == ['same', 'different', 'unsure']


@pytest.mark.parametrize('finish', ['length', None, 'content_filter'])
def test_valid_looking_incomplete_verdict_is_not_maintenance_authority(monkeypatch, finish):
    from src.workers import bg_client_factory

    monkeypatch.setattr(bg_client_factory, 'get_bg_client', lambda:
        NS(chat=NS(completions=NS(create=lambda **_kw:
            NS(choices=[NS(finish_reason=finish, message=NS(content='{"decision":"expire_old"}'))])))))
    monkeypatch.setattr(bg_client_factory, 'get_bg_model_name', lambda: 'controlled-background')
    assert worker.make_llm_decider()('Controlled source comparison.') is None
