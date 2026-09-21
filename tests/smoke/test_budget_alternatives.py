"""v3 evidence alternatives survive until an actual fitting representation wins."""

import pytest

from src.api.config import settings
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.retrieval.reranker import rerank


def fragment(text, tokens, source='turn', **kw):
    return ContextFragment(text, 'episodic', 1., tokens,
                           source_batch_id=source, conversation_id='other', **kw)


@pytest.mark.parametrize('ranked', [False, True])
def test_oversized_parents_do_not_consume_turn_or_conversation_allowance(monkeypatch, ranked):
    monkeypatch.setattr(settings, 'retrieval_max_frags_per_turn', 1)
    monkeypatch.setattr(settings, 'retrieval_max_per_conversation', 1)
    candidates = [fragment('Complete oversized source', 1000, covers_entire_source=True),
                  fragment('Another oversized option', 900), fragment('Late correction', 5)]
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        candidates, max_tokens=10, relevance_order=ranked)
    assert [f.text for f in selected] == ['Late correction']


@pytest.mark.parametrize('whole_first', [False, True])
def test_complete_source_and_its_excerpts_do_not_both_spend_budget(whole_first):
    whole = fragment('Complete source', 20, covers_entire_source=True)
    excerpt = fragment('Source excerpt', 5)
    candidates = [whole, excerpt] if whole_first else [excerpt, whole]
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        candidates, max_tokens=100, relevance_order=True)
    assert selected == candidates[:1]


def test_degraded_summary_cannot_claim_to_contain_every_source_detail():
    whole = fragment('Complete raw source', 1000, degrade_text='A summary.', covers_entire_source=True)
    excerpt = fragment('Additional exact detail', 5)
    selected = HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        [whole, excerpt], max_tokens=100, relevance_order=True)
    assert len(selected) == 2 and not selected[0].covers_entire_source


def test_reranker_skips_only_unscorable_pair_and_retains_complete_fallback():
    whole = fragment('An oversized original.', 1000, covers_entire_source=True)
    excerpt = fragment('An answer excerpt.', 5)
    ranked, success = rerank('Question', [whole, excerpt], scorer=lambda pairs: [None, 2.])
    assert success and ranked[0].text == excerpt.text
    assert ranked[1].text == whole.text and ranked[1].covers_entire_source


def test_all_unscored_keeps_original_candidates_and_order():
    candidates = [fragment('a', 1), fragment('b', 1, source='another')]
    assert rerank('Question', candidates, scorer=lambda pairs: [None, None]) == (candidates, False)


def test_reranked_summary_does_not_inherit_whole_source_marker():
    whole = fragment('Raw original.', 100, degrade_text='Compact note.', covers_entire_source=True)
    ranked, success = rerank('Question', [whole], scorer=lambda pairs: [0., 1.])
    assert success and ranked[0].text == 'Compact note.' and not ranked[0].covers_entire_source


def test_retrieval_failure_log_excludes_source_parameters(monkeypatch):
    from types import SimpleNamespace
    from sqlalchemy.exc import StatementError
    import src.retrieval.orchestrator as module

    records = []
    rollbacks = []
    monkeypatch.setattr(module, 'logger', SimpleNamespace(
        warning=lambda event, **fields: records.append((event, fields))))
    orch = module.HybridRetrievalOrchestrator.__new__(module.HybridRetrievalOrchestrator)
    orch.db = SimpleNamespace(rollback=lambda: rollbacks.append(True))
    failure = StatementError('private diagnosis', 'INSERT secret source',
                             {'raw_text': 'private memory evidence'}, ValueError('private detail'))
    orch._leg_degraded('cold.resurrect', failure)
    assert rollbacks == [True]
    assert records[0][1]['error_type'] == 'StatementError'
    assert 'private' not in str(records) and 'secret source' not in str(records)
