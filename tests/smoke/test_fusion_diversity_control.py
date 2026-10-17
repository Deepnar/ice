"""v3 fallback diversity changes decisions without bypassing evidence guards."""

from types import SimpleNamespace

import pytest

from src.api.config import settings
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator


def candidates():
    return [ContextFragment('Highest scored source', 'episodic', .9, 50),
            ContextFragment('Next scored source', 'episodic', .8, 50),
            ContextFragment('Other family source', 'codex', .6, 50)]


@pytest.mark.parametrize('enabled', [True, False])
def test_opposing_switch_changes_fallback_selection(monkeypatch, enabled):
    monkeypatch.setattr(settings, 'retrieval_leg_guarantee_enabled', enabled)
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        candidates(), max_tokens=100)
    assert [f.text for f in selected] == ['Highest scored source',
        'Other family source' if enabled else 'Next scored source']
    assert sum(f.token_count for f in selected) == 100


@pytest.mark.parametrize('enabled', [True, False])
def test_reranker_order_is_not_replaced_by_old_fused_scores(monkeypatch, enabled):
    monkeypatch.setattr(settings, 'retrieval_leg_guarantee_enabled', enabled)
    items = candidates()
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        [items[2], items[0], items[1]], max_tokens=50, relevance_order=True)
    assert [f.text for f in selected] == ['Other family source']


@pytest.mark.parametrize('enabled', [True, False])
def test_oversize_source_does_not_block_late_evidence_or_duplicate_claim(monkeypatch, enabled):
    monkeypatch.setattr(settings, 'retrieval_leg_guarantee_enabled', enabled)
    monkeypatch.setattr(settings, 'retrieval_max_frags_per_turn', 1)
    monkeypatch.setattr(settings, 'retrieval_max_per_conversation', 1)
    items = [ContextFragment('Oversized parent', 'episodic', 10., 1000,
                             source_batch_id='source', conversation_id='history'),
             ContextFragment('The correction was applied.', 'episodic', 9., 8,
                             source_batch_id='source', conversation_id='history',
                             covers_entire_source=True),
             ContextFragment('Another candidate', 'episodic', 7., 2,
                             source_batch_id='other', conversation_id='history')]
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        items, max_tokens=12)
    assert selected == [items[1]]
    assert sum(f.token_count for f in selected) <= 12


@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('enabled', [True, False])
def test_real_retrieve_callers_reach_switch_and_report_mode(monkeypatch, wide, enabled):
    import src.retrieval.orchestrator as module

    monkeypatch.setattr(settings, 'retrieval_leg_guarantee_enabled', enabled)
    monkeypatch.setattr(settings, 'retrieval_rerank_enabled', False)
    monkeypatch.setattr(settings, 'retrieval_coverage_enabled', False)
    monkeypatch.setattr(settings, 'retrieval_wide_net_budget_floor', 100)
    monkeypatch.setattr(settings, 'retrieval_wide_net_budget_fraction', 1.)
    events, warnings = [], []
    monkeypatch.setattr(module, 'logger', SimpleNamespace(
        info=lambda event, **fields: events.append((event, fields)),
        debug=lambda *_a, **_kw: None,
        warning=lambda event, **fields: warnings.append((event, fields))))
    # Control only source-fetch/scorer boundaries; real fusion, deduplication,
    # main/wide callers and admission/budget logic participate. No SQL/model
    # semantics or source visibility are claimed by this control.
    db = SimpleNamespace(execute=lambda *_a, **_kw: SimpleNamespace(fetchall=lambda: []))
    reader = HybridRetrievalOrchestrator(db, None)
    reader.max_retrieval_tokens = 100
    items = candidates()
    monkeypatch.setattr(reader, '_facts_leg', lambda *_a, **_kw: items[2:])
    monkeypatch.setattr(reader, '_vector_episodic', lambda *_a, **_kw: items[:2])
    monkeypatch.setattr(reader, '_rows_to_fragments', lambda *_a, **_kw: items[:2])
    for name in ('_relevant_cluster_ids', '_bm25_episodic', '_procedural_lookup',
                 '_batch_summary_lookup', '_cold_lookup', '_vector_chunks'):
        monkeypatch.setattr(reader, name, lambda *_a, **_kw: [])
    monkeypatch.setattr(reader, '_apply_bonuses', lambda fragments, *_a: fragments)
    query = SimpleNamespace(prompt='Earlier evidence', context_reliance='Long_Term_Memory',
                            max_confidence=0. if wide else 1., intent_tags=[], topic_tags=[])
    selected = reader.retrieve(query, None, [1.], defer_exposure=True)
    assert [f.text for f in selected] == ['Highest scored source',
        'Other family source' if enabled else 'Next scored source']
    budgets = [fields for event, fields in events if event == 'leg_budget_share']
    assert len(budgets) == 1
    assert budgets[0]['selection_mode'] == ('fusion_family_diversity' if enabled
                                           else 'fusion_relevance')
    assert budgets[0]['total_tokens'] == 100
    assert not warnings
