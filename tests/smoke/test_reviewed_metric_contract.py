"""Task intent and source age control metric admission, not catalog origins."""

import pytest

from scripts.z1.metric_contract import metric_contract, summarize_contracts


def review(*tasks, scope="private_history", recent=False):
    return dict(knowledge_scope=scope, task_types=list(tasks), recent_only_answerable=recent)


def test_precise_single_source_private_lookup_is_rank_eligible():
    result = metric_contract(review("episodic_lookup"), [17])
    assert result["lookup_rank_tuning_eligible"]
    assert result["rank_metric"] == "single_source_rank_diagnostic"
    assert result["semantic_support"] == "separate_qualified_review_required"


@pytest.mark.parametrize("tasks,gold,scope", [
    (["summary_synthesis"], [17], "private_history"),
    (["episodic_lookup", "summary_synthesis"], [17], "private_history"),
    (["temporal"], [17], "private_history"),
    (["knowledge_update"], [17, 40], "private_history"),
    (["multi_hop", "entity_relation"], [17, 40], "private_history"),
    (["procedural"], [17], "private_history"),
    (["negative"], [], "private_history"),
    (["abstention"], [], "private_history"),
    (["episodic_lookup"], [17, 40], "private_history"),
    (["episodic_lookup"], None, "private_history"),
    (["episodic_lookup"], [17], "public_knowledge"),
    (["episodic_lookup"], [17], "mixed"),
])
def test_non_lookup_contracts_never_enter_single_source_tuning(tasks, gold, scope):
    contract = metric_contract(review(*tasks, scope=scope), gold)
    assert not contract["lookup_rank_tuning_eligible"]
    assert contract["rank_metric"] != "single_source_rank_diagnostic"


@pytest.mark.parametrize("tasks", [[], ["ENTER_TYPE"], ["native_curated_codex"],
                                  ["episodic_lookup", "episodic_lookup"], [{}]])
def test_missing_or_catalog_types_cannot_replace_reviewed_intent(tasks):
    with pytest.raises(ValueError, match="reviewed"):
        metric_contract(review(*tasks), [17])


def test_before_and_unreviewed_age_never_count_as_long_term_lookup():
    recent = metric_contract(review("episodic_lookup"), [17], development_repeat=dict(phase="before"))
    unknown = metric_contract(review("episodic_lookup", recent=None), [17])
    assert recent["age_cohort"] == "recent_control" and not recent["lookup_rank_tuning_eligible"]
    assert unknown["age_cohort"] == "unreviewed_age" and not unknown["lookup_rank_tuning_eligible"]
    summary = summarize_contracts([recent, unknown])
    assert summary["occurrences"] == 2 and summary["lookup_rank_tuning_eligible"] == 0
    assert summary["independent_sample_count"] is None


@pytest.mark.parametrize("gold", [[True], [0], [1, 1], "1"])
def test_invalid_source_identity_is_not_a_precise_lookup(gold):
    with pytest.raises(ValueError, match="source turns"):
        metric_contract(review("episodic_lookup"), gold)


def test_task_group_counts_overlap_without_inflating_occurrences():
    contracts = [metric_contract(review("temporal", "knowledge_update"), [17, 40]),
                 metric_contract(review("episodic_lookup", scope="assistant_history"), [17])]
    summary = summarize_contracts(contracts)
    assert summary["occurrences"] == 2 and summary["lookup_rank_tuning_eligible"] == 1
    assert summary["task_types"]["temporal"] == summary["task_types"]["knowledge_update"] == 1
    assert summary["task_groups_overlap"]
