"""Provider residual duration is measured, never named as unobserved thinking."""

import pytest

from src.workers.bg_client_factory import native_call_timings


def test_complete_durations_preserve_unattributed_time():
    result = native_call_timings(dict(load_duration=300_000_000,
        prompt_eval_duration=40_000_000, eval_duration=500_000_000, total_duration=4_000_000_000))
    assert result['provider_total_seconds'] == 4.
    assert result['provider_load_seconds'] == .3
    assert result['provider_unattributed_seconds'] == 3.16
    assert not any('thinking' in key for key in result)


@pytest.mark.parametrize('invalid', [None, -1, True, float('nan'), float('inf')])
def test_missing_or_invalid_duration_cannot_become_zero_residual(invalid):
    result = native_call_timings(dict(load_duration=invalid,
        prompt_eval_duration=1, eval_duration=1, total_duration=3))
    assert result['provider_unattributed_seconds'] is None


def test_inconsistent_or_overlapping_durations_remain_unknown():
    result = native_call_timings(dict(load_duration=2, prompt_eval_duration=2,
                                    eval_duration=2, total_duration=3))
    assert result['provider_unattributed_seconds'] is None
