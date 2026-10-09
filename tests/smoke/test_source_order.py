"""Source order authority, independent of semantic replacement decisions."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from src.memory.source import single_provenance
from src.memory.source_order import (
    attach_sequence, compare_observations, sequence_receipt, verified_sequence,
)
from src.memory.time_format import source_stamp


def observation(position, *, calendar=False, sequence=True, **changes):
    result = dict(conversation_id='conversation', role='user', text=f'source {position}',
                  recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
                  + timedelta(days=position) if calendar else None,
                  sequence=sequence_receipt(f'source {position}', 'a' * 64, position)
                  if sequence is True else sequence if isinstance(sequence, dict) else None)
    result.update(changes)
    return result


def test_receipt_is_bound_to_complete_original_and_explicit_position():
    raw = 'Petra uses Atlas.'
    row = NS(raw_text=raw, source_spans=attach_sequence(single_provenance(raw, 'user'), raw,
             sequence_receipt(raw, 'a' * 64, 3)), timestamp=None, ts_provenance='unknown')
    assert verified_sequence(row)['position'] == 3
    assert source_stamp(row) == '[Verified conversation order: 3; calendar unknown] '
    row.raw_text += ' Updated.'
    assert verified_sequence(row) is None and source_stamp(row) == ''


@pytest.mark.parametrize('position', [0, -1, True, '1', None])
def test_position_must_be_explicit_positive_integer(position):
    with pytest.raises(ValueError):
        sequence_receipt('source', 'a' * 64, position)


def test_receipt_cannot_authorize_different_raw_bytes():
    raw = 'source'
    with pytest.raises(ValueError):
        attach_sequence(single_provenance(raw, 'user'), raw,
                        sequence_receipt('other', 'a' * 64, 1))


@pytest.mark.parametrize('calendar,sequence,basis', [
    (True, False, 'original_calendar'), (False, True, 'conversation_sequence'),
    (True, True, 'original_calendar'),
])
def test_latest_independent_old_source_selected(calendar, sequence, basis):
    old = [observation(i, calendar=calendar, sequence=sequence) for i in (2, 1)]
    new = [observation(3, calendar=calendar, sequence=sequence)]
    pair, reason = compare_observations(old, new)
    assert reason == 'eligible' and pair[0]['text'] == 'source 2'
    receipt = pair[1]['ordering']
    assert receipt['basis'] == basis
    if sequence:
        assert receipt['old_position'] == 2 and receipt['new_position'] == 3
    if not calendar:
        assert receipt['old_recorded_at'] is receipt['new_recorded_at'] is None


@pytest.mark.parametrize('new,reason', [
    (observation(1), 'new_source_not_later'),
    (observation(2), 'new_source_not_later'),
    (observation(3, conversation_id='other'), 'different_conversations'),
    (observation(3, role='assistant'), 'different_speakers'),
    (observation(3, sequence=None), 'source_order_unknown'),
    (observation(3, sequence=sequence_receipt('source 3', 'b' * 64, 3)), 'source_order_unknown'),
])
def test_incomparable_sources_preserve_both(new, reason):
    assert compare_observations([observation(2)], [new]) == (None, reason)


def test_clock_disagreement_defers_and_ambiguous_latest_cannot_be_chosen():
    old = observation(2, calendar=True)
    new = observation(3, calendar=True, recorded_at=old['recorded_at'] - timedelta(days=1))
    assert compare_observations([old], [new])[1] == 'calendar_sequence_disagreement'
    assert compare_observations([old, {**old, 'text': 'Other statement.'}],
                                [observation(3, calendar=True)])[1] == 'ambiguous_latest_source'


def test_mixed_known_unknown_dates_can_use_sequence_but_not_cross_role_corroboration():
    pair, reason = compare_observations([observation(1, calendar=True), observation(2)],
                                        [observation(3)])
    assert reason == 'eligible' and pair[1]['ordering']['basis'] == 'conversation_sequence'
    assert compare_observations([observation(1), observation(2, role='assistant')],
                                [observation(3)])[1] == 'different_speakers'
