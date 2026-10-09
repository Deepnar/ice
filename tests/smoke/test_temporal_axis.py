"""Resolved query dates do not establish an event/source clock equivalence."""

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.orchestrator import ContextFragment
from src.memory.tokens import count
from src.retrieval.timescope import CURRENT, TimeScope, detect_timescope, from_scope, to_scope_dict


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize('question', [
    'Where did Petra work in 2010?', 'where did petra work in 2010',
    'ok so like in 2010 where did Petra work', 'In 2010, where did Petra work?',
])
def test_date_expression_does_not_infer_semantic_axis(question):
    scope = detect_timescope(question, now=NOW, p_ltm=1., p_temporal=1.)
    assert scope.mode == 'range' and scope.axis == 'unknown'
    assert scope.t0.year == 2009 and scope.t1.year == 2011
    assert from_scope({'timescope': to_scope_dict(scope)}).axis == 'unknown'


def test_explicit_source_clock_and_legacy_scopes_keep_their_contract():
    scope = TimeScope(mode='range', t0=datetime(2010, 1, 1, tzinfo=timezone.utc),
                      t1=datetime(2011, 1, 1, tzinfo=timezone.utc))
    packed = to_scope_dict(scope)
    assert scope.source_clock and packed['axis'] == 'source_record'
    assert from_scope({'timescope': packed}) == scope
    packed.pop('axis')
    assert from_scope({'timescope': packed}) == scope
    packed['axis'] = 'invalid'
    assert from_scope({'timescope': packed}) == CURRENT


@pytest.mark.parametrize('axis', ['unknown', 'event'])
def test_no_source_date_predicate_recency_or_false_empty_window_note(axis):
    o = HybridRetrievalOrchestrator(None, None)
    scope = TimeScope(mode='as_of', t0=datetime(2010, 1, 1, tzinfo=timezone.utc),
                      t1=datetime(2011, 1, 1, tzinfo=timezone.utc), axis=axis)
    o._active_timescope = scope
    clause, _, parameters, floor = o._timescope_leg_filters()
    assert clause == '' and parameters == {} and floor == 0.
    assert o._recency_params(False)[1] == 0.
    assert o._append_empty_window_note([], None, None) == []
    o._active_timescope = replace(scope, axis='source_record')
    clause, _, parameters, floor = o._timescope_leg_filters()
    assert 'timestamp >= :ts_t0' in clause and parameters['ts_t0'].year == 2010
    assert o._recency_params(False)[1] > 0.


@pytest.mark.parametrize('available', [0, 39, 150])
def test_temporal_note_uses_only_remaining_budget(available):
    from types import SimpleNamespace as NS
    date = datetime(2010, 1, 1, tzinfo=timezone.utc)
    db = NS(execute=lambda *_a: NS(fetchall=lambda: [NS(timestamp=date)]))
    o = HybridRetrievalOrchestrator(db, None)
    o._active_timescope = TimeScope('range', date, date.replace(year=2011))
    evidence = ContextFragment('A supported user instruction.', 'procedural', 1.,
                               count('A supported user instruction.'))
    o.max_retrieval_tokens = evidence.token_count + available
    result = o._append_empty_window_note([evidence], [1.] + [0.] * 1023, None)
    assert evidence in result
    assert sum(f.token_count for f in result) <= o.max_retrieval_tokens
    assert all(f.token_count == count(f.text) for f in result)
    if available == 0:
        assert result == [evidence]
    else:
        assert len(result) == 2 and 'No episodic source records were selected' in result[1].text
        assert not result[1].origin_batch_ids
