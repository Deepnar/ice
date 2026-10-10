"""Reviewed scorer-only task contracts; catalog provenance is not task intent."""

from collections import Counter

from scripts.z1.label_review import KNOWLEDGE_SCOPES, TASK_TYPES, question_family_counts

POLICY = "reviewed_metric_contract_v1"


def metric_contract(review, gold_turns, *, development_repeat=None):
    """Separate rank eligibility from coverage and independently judged answers."""
    if not isinstance(review, dict):
        raise ValueError("metric contract needs reviewed knowledge scope and task types")
    labels = review.get("task_types")
    scope = review.get("knowledge_scope")
    if (scope not in KNOWLEDGE_SCOPES or not isinstance(labels, list) or not labels
            or any(not isinstance(t, str) or t not in TASK_TYPES for t in labels)
            or len(labels) != len(set(labels))):
        raise ValueError("metric contract needs reviewed knowledge scope and task types")
    if gold_turns is not None and (not isinstance(gold_turns, list)
            or any(type(t) is not int or t < 1 for t in gold_turns)
            or len(set(gold_turns)) != len(gold_turns)):
        raise ValueError("metric contract source turns are malformed")
    repeat = {} if development_repeat is None else development_repeat
    if not isinstance(repeat, dict) or (repeat and repeat.get("phase") not in {"before", "after"}):
        raise ValueError("metric contract repeat phase is malformed")
    if repeat.get("phase") == "before":
        cohort = "recent_control"
    elif review.get("recent_only_answerable") is False:
        cohort = "older_source"
    else:
        cohort = "unreviewed_age"
    if cohort != "older_source":
        reason = cohort
    elif scope not in {"private_history", "assistant_history"}:
        reason = "public_or_mixed_knowledge"
    elif set(labels) != {"episodic_lookup"}:
        reason = "not_precise_lookup_only"
    elif not gold_turns or len(gold_turns) != 1:
        reason = "not_one_reviewed_source_turn"
    else:
        reason = "precise_single_source_lookup"
    eligible = reason == "precise_single_source_lookup"
    return dict(policy=POLICY, knowledge_scope=scope, task_types=sorted(labels),
                age_cohort=cohort, source_turn_count=len(gold_turns) if gold_turns is not None else None,
                rank_metric="single_source_rank_diagnostic" if eligible else
                            "annotated_source_coverage_diagnostic" if gold_turns else "unavailable",
                lookup_rank_tuning_eligible=eligible, rank_eligibility_reason=reason,
                semantic_support="separate_qualified_review_required")


def summarize_contracts(contracts):
    """Counts overlap across tasks, never imply independent questions or quality."""
    return dict(occurrences=len(contracts),
                lookup_rank_tuning_eligible=sum(c["lookup_rank_tuning_eligible"] for c in contracts),
                rank_eligibility_reasons=dict(sorted(Counter(c["rank_eligibility_reason"] for c in contracts).items())),
                knowledge_scopes=dict(sorted(Counter(c["knowledge_scope"] for c in contracts).items())),
                task_types={task: sum(task in c["task_types"] for c in contracts) for task in sorted(TASK_TYPES)},
                age_cohorts=dict(sorted(Counter(c["age_cohort"] for c in contracts).items())),
                task_groups_overlap=True, independent_sample_count=None)


def absolute_outcome_strata(records):
    """Own-arm grades with explicit unavailable/age/task denominators."""
    def summarize(rows):
        grades = Counter((r.get("grade") or {}).get("grade") for r in rows)
        return dict(occurrences=len(rows), statuses=dict(sorted(Counter(r["status"] for r in rows).items())),
                    grades={g: grades[g] for g in ("correct", "partial", "incorrect", "uncertain")},
                    without_grade=grades[None],
                    contracts=summarize_contracts([r["metric_contract"] for r in rows]),
                    question_families=question_family_counts([
                        dict(question_family_id=r.get("family")) for r in rows]))

    by_arm = {}
    for arm in sorted({r["arm"] for r in records}):
        rows = [r for r in records if r["arm"] == arm]
        older = [r for r in rows if r["metric_contract"]["age_cohort"] == "older_source"]
        by_arm[arm] = dict(
            all_occurrences=summarize(rows), older_source=summarize(older),
            recent_controls=summarize([r for r in rows if r["metric_contract"]["age_cohort"] == "recent_control"]),
            unreviewed_age=summarize([r for r in rows if r["metric_contract"]["age_cohort"] == "unreviewed_age"]),
            knowledge_scope={scope: summarize([r for r in older if r["knowledge_scope"] == scope])
                             for scope in sorted(KNOWLEDGE_SCOPES)},
            task_types={task: summarize([r for r in older if task in r["task_types"]])
                        for task in sorted(TASK_TYPES)})
    return dict(policy=POLICY, qualification="unqualified_diagnostic", task_groups_overlap=True,
                independent_sample_count=None, by_arm=by_arm)
