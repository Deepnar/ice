"""ICE v3 selected excerpts share context only with exact same-source coverage."""
from dataclasses import replace

import pytest

from src.api.config import settings
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator


def fragment(text, tokens, **kwargs):
    return ContextFragment(text, 'episodic', 1., tokens,
        source_batch_id='original', conversation_id='conversation',
        stored_artifact_ids=(('episodic_chunks', 'chunk-a'),), **kwargs)


def snippets():
    shorter = fragment('Requested exact value.', 4)
    longer = replace(shorter, text='Earlier context. Requested exact value. Later context.',
        token_count=9, stored_artifact_ids=shorter.stored_artifact_ids +
        (('episodic_chunks', 'chunk-b'),))
    return shorter, longer


def select(candidates, budget=100, ranked=True):
    return HybridRetrievalOrchestrator(None, None)._enforce_token_budget(
        candidates, max_tokens=budget, relevance_order=ranked,
        current_conversation_id='conversation')


@pytest.mark.parametrize('shorter_first', [False, True])
@pytest.mark.parametrize('ranked', [False, True])
def test_contained_excerpt_spends_budget_once(shorter_first, ranked):
    shorter, longer = snippets()
    candidates = [shorter, longer] if shorter_first else [longer, shorter]
    assert select(candidates, ranked=ranked) == [longer]


@pytest.mark.parametrize('changes', [
    {'source_batch_id': 'another-original'},
    {'source_batch_id': None},
    {'conversation_id': 'another-conversation'},
    {'conversation_id': None},
    {'text': 'A distinct value from the same source.'},
    {'text': 'Requested exact value. Unique surrounding context.'},
    {'stored_artifact_ids': ()},
    {'stored_artifact_ids': (('episodic_chunks', 'uncovered-chunk'),)},
    {'origin_batch_ids': ('uncovered-original',)},
    {'origin_edge_ids': ('uncovered-edge',)},
    {'source_type': 'procedural'},
])
def test_missing_or_distinct_evidence_is_not_covered(changes):
    shorter, longer = snippets()
    shorter = replace(shorter, **changes)
    assert not HybridRetrievalOrchestrator._same_source_excerpt_contains(longer, shorter)
    assert len(select([longer, shorter])) == 2


def test_whitespace_and_case_changes_do_not_invent_exact_coverage():
    shorter, longer = snippets()
    assert not HybridRetrievalOrchestrator._same_source_excerpt_contains(
        longer, replace(shorter, text='requested  exact value.'))


def test_empty_body_is_not_coverage():
    shorter, longer = snippets()
    assert not HybridRetrievalOrchestrator._same_source_excerpt_contains(
        longer, replace(shorter, text=''))


def test_complete_source_capability_cannot_be_removed_by_an_excerpt():
    shorter, longer = snippets()
    shorter = replace(shorter, covers_entire_source=True)
    assert not HybridRetrievalOrchestrator._same_source_excerpt_contains(longer, shorter)
    assert select([shorter, longer]) == [shorter]


def test_existing_complete_original_admission_is_preserved():
    shorter, longer = snippets()
    longer = replace(longer, covers_entire_source=True)
    assert select([shorter, longer], budget=9) == [shorter]


def test_oversized_cover_keeps_the_admitted_excerpt():
    shorter, longer = snippets()
    assert select([shorter, replace(longer, token_count=1000)], budget=9) == [shorter]


def test_degraded_cover_rechecks_actual_text_and_keeps_unique_detail(monkeypatch):
    monkeypatch.setattr('src.retrieval.orchestrator.count_tokens', lambda text: 2)
    shorter, longer = snippets()
    longer = replace(longer, token_count=1000, degrade_text='A different compact detail.')
    chosen = select([shorter, longer], budget=9)
    assert shorter in chosen and len(chosen) == 2
    assert chosen[1].text == longer.degrade_text


def test_existing_same_parent_cap_choice_is_preserved(monkeypatch):
    monkeypatch.setattr(settings, 'retrieval_max_frags_per_turn', 1)
    shorter, longer = snippets()
    assert select([shorter, longer], budget=9) == [shorter]


def test_disjoint_same_parent_still_obeys_the_original_cap(monkeypatch):
    monkeypatch.setattr(settings, 'retrieval_max_frags_per_turn', 1)
    shorter, longer = snippets()
    longer = replace(longer, text='A disjoint source window.')
    assert select([shorter, longer]) == [shorter]


def test_savings_do_not_backfill_and_displace_existing_selected_evidence():
    shorter, longer = snippets()
    later = replace(shorter, text='Independent later correction.', token_count=4,
        source_batch_id='later-original',
        stored_artifact_ids=(('episodic_chunks', 'later-chunk'),))
    assert select([longer, shorter, later], budget=13) == [longer]


def test_native_final_prompt_receives_only_the_retained_cover():
    from src.api.prompt_budget import assemble_budgeted_prompt

    shorter, longer = snippets()
    selected = select([longer, shorter])
    packed = assemble_budgeted_prompt(serving_window=20000, generation_reserve=256,
        memory_slots=[], retrieved_fragments=selected, user_message='Which exact value?',
        conversation_id='conversation')
    assert packed.visible_fragments == [longer]
    assert sum(message['content'].count(shorter.text) for message in packed.messages) == 1


def test_equal_selected_bodies_keep_one_representative():
    shorter, _ = snippets()
    same = replace(shorter, score=.5)
    assert HybridRetrievalOrchestrator._omit_covered_source_excerpts([shorter, same]) == [shorter]


def test_nested_selected_coverage_never_removes_all_representatives():
    shorter, longer = snippets()
    widest = replace(longer, text='Source beginning. ' + longer.text + ' Source ending.',
        token_count=14, stored_artifact_ids=longer.stored_artifact_ids +
        (('episodic_chunks', 'chunk-c'),))
    assert HybridRetrievalOrchestrator._omit_covered_source_excerpts([shorter, longer, widest]) == [widest]
