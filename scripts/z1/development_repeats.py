"""Small reviewed recent→old controls, separate from final semi-LSREP."""
from __future__ import annotations

import copy
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.z1.label_review import reviewed_expected_answer, validate_review

KINDS = {"development_repeat_recent", "development_repeat_delayed"}
POLICY = "reviewed_unchanged_fact_recent_to_old_v1"


def packet_hash(path: Path) -> str:
    path = path.resolve()
    if not path.is_relative_to(Path(__file__).resolve().parents[2] / "logs"):
        raise ValueError("development repeat review contains private data; keep it under logs/")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeat_probes(packet: dict, conversations: dict, scheduled: dict,
                  inputs: list, gap: int) -> list[dict]:
    """Require complete source review for both phases before adding any query."""
    if (packet.get("version") != "v3" or packet.get("kind") != "development_repeat_review"
            or packet.get("inputs") != inputs[:5] or packet.get("recent_window_turns") != gap):
        raise ValueError("development repeat review belongs to different inputs/window")
    families = packet.get("families")
    if not isinstance(families, list) or not 1 <= len(families) <= 4:
        raise ValueError("development repeat pass needs one to four reviewed families")
    catalog = {p["probe_id"]: p for group in scheduled.values() for p in group}
    checkpoints = set(scheduled)
    seen = set()
    result = []
    for family in families:
        identity = family.get("question_family_id")
        if not isinstance(identity, str) or not identity.strip() or identity in seen:
            raise ValueError("development repeat family is missing or duplicate")
        seen.add(identity)
        base = catalog.get(family.get("base_probe_id"))
        if not base or base.get("cutoff_kind") != "native_unlabeled_checkpoint":
            raise ValueError("development repeat must copy a captured native question")
        if family.get("expected_answer_unchanged") is not True:
            raise ValueError("development repeat requires reviewed unchanged answer")
        before, after = family.get("before", {}), family.get("after", {})
        if before.get("cutoff_turn") != base["split_turn"]:
            raise ValueError("development repeat before cutoff differs from native question")
        if (type(after.get("cutoff_turn")) is not int
                or after["cutoff_turn"] <= before["cutoff_turn"]
                or (base["conversation"], after["cutoff_turn"]) not in checkpoints):
            raise ValueError("development repeat after must be a later existing checkpoint")
        if (before.get("gold_turns") != after.get("gold_turns")
                or reviewed_expected_answer(before) != reviewed_expected_answer(after)):
            raise ValueError("development repeat answer/source changed between phases")
        for phase, review in (("before", before), ("after", after)):
            cutoff = review["cutoff_turn"]
            gold = review.get("gold_turns")
            if (not isinstance(gold, list) or not gold or any(type(t) is not int for t in gold)
                    or len(gold) != len(set(gold)) or min(gold) < 1
                    or max(gold) > cutoff or cutoff > len(conversations[base["conversation"]])):
                raise ValueError("development repeat has invalid source/cutoff bounds")
            recent = phase == "before"
            if (review.get("recent_only_answerable") is not recent
                    or (cutoff - max(gold) < gap) is not recent
                    or (recent and min(gold) <= cutoff - gap)
                    or (not recent and max(gold) > cutoff - gap)):
                raise ValueError("development repeat source/recent eligibility is inconsistent")
            if (not isinstance(review.get("reason"), str) or not review["reason"].strip()
                    or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()
                    or review.get("review_scope") != "complete_history_through_cutoff"):
                raise ValueError("development repeat needs complete source review and reviewer")
            # The recent phase is a control. All other admission facts still apply.
            validate_review({**review, "recent_only_answerable": False,
                             "question_family_id": identity}, cutoff)
            digest = hashlib.sha256(json.dumps([base["probe_id"], identity, phase, cutoff]).encode()).hexdigest()[:20]
            probe = copy.deepcopy(base)
            probe.update(probe_id="development-repeat:" + digest,
                         gold_turns=sorted(gold), source_split_turn=max(gold), split_turn=cutoff,
                         cutoff_kind="development_repeat_recent" if recent else "development_repeat_delayed",
                         probe_type="development_repeat_" + phase,
                         label_status="reviewed_development_repeat",
                         review={k: copy.deepcopy(v) for k, v in review.items()
                                 if k not in {"gold_turns", "expected_answer", "reviewed_expected_answer", "cutoff_turn"}},
                         development_repeat={"policy": POLICY, "question_family_id": identity,
                                             "phase": phase, "base_probe_id": base["probe_id"],
                                             "recent_window_turns": gap, "source_age_turns": cutoff - max(gold)},
                         catalog_expected_answer=base["expected_answer"],
                         expected_answer=reviewed_expected_answer(review))
            probe["review"]["question_family_id"] = identity
            result.append(probe)
    return result


def admit_repeats(trace_header: dict, frozen: list[dict], planned: list[dict], digest: str) -> list[dict]:
    extra = trace_header.get("meta", {}).get("extra", {})
    if (extra.get("development_repeat_review_sha256") != digest
            or extra.get("development_repeat_policy") != POLICY):
        raise ValueError("development repeat packet differs from pinned replay")
    actual = {p["probe_id"]: p for p in frozen if p.get("cutoff_kind") in KINDS}
    if set(actual) != {p["probe_id"] for p in planned}:
        raise ValueError("development repeat frozen schedule differs from review")
    chosen = []
    for plan in planned:
        probe = actual[plan["probe_id"]]
        if any(probe.get(k) != plan.get(k) for k in
               ("conversation", "question", "split_turn", "gold_turns", "cutoff_kind",
                "expected_answer", "development_repeat", "catalog_expected_answer")):
            raise ValueError("development repeat source/question/key differs from frozen input")
        chosen.append({**probe, "review": plan["review"]})
    return chosen


def retention_comparison(results: list[dict]) -> dict:
    """Compare the same reviewed question across time; never guess equivalence."""
    groups = defaultdict(dict)
    for row in results:
        meta = row.get("development_repeat")
        if not meta:
            continue
        family, phase = meta.get("question_family_id"), meta.get("phase")
        if meta.get("policy") != POLICY or not family or phase not in {"before", "after"}:
            raise ValueError("retention outcome lacks declared family/phase")
        if phase in groups[family]:
            raise ValueError("retention outcome has duplicate family phase")
        groups[family][phase] = row
    pairs = []
    ordinal = {"incorrect": 0, "partial": 1, "correct": 2}
    for family, phases in sorted(groups.items()):
        item = {"question_family_id": family, "status": "paired" if len(phases) == 2 else "missing_phase",
                "missing_phases": sorted({"before", "after"} - phases.keys()),
                "phases": {phase: {"probe_id": r["probe_id"], "cutoff_turn": r["split_turn"],
                                   "source_age_turns": r["development_repeat"]["source_age_turns"]}
                           for phase, r in phases.items()}, "arms": {}}
        for arm in ("arm_a", "arm_b"):
            early, late = phases.get("before", {}), phases.get("after", {})
            a, b = early.get(arm + "_grade"), late.get(arm + "_grade")
            resolved = (len(phases) == 2 and a in ordinal and b in ordinal
                        and early.get("winner") != "ERROR" and late.get("winner") != "ERROR")
            movement = "unresolved"
            if resolved:
                movement = "improved" if ordinal[b] > ordinal[a] else "worsened" if ordinal[b] < ordinal[a] else "unchanged"
            tokens = [r.get(arm + "_prompt_tokens_est") for r in (early, late)]
            evidence = [r.get(arm + "_gold_fragment_coverage", {}).get("selected_or_source_note") for r in (early, late)]
            item["arms"][arm] = {"before_grade": a, "after_grade": b, "movement": movement,
                                  "correct_to_not_correct": resolved and a == "correct" and b != "correct",
                                  "not_correct_to_correct": resolved and a != "correct" and b == "correct",
                                  "after_minus_before_prompt_tokens_est": tokens[1] - tokens[0]
                                  if all(type(t) in (int, float) for t in tokens) else None,
                                  "before_selected_source_count": evidence[0],
                                  "after_selected_source_count": evidence[1],
                                  "after_minus_before_selected_source_count": evidence[1] - evidence[0]
                                  if all(type(e) is int for e in evidence) else None}
        pairs.append(item)
    summary = {}
    for arm in ("arm_a", "arm_b"):
        observations = [p["arms"][arm] for p in pairs]
        costs = [r["after_minus_before_prompt_tokens_est"] for r in observations
                 if r["after_minus_before_prompt_tokens_est"] is not None]
        summary[arm] = {"grade_movements": dict(Counter(r["movement"] for r in observations)),
                        "correct_to_not_correct": sum(r["correct_to_not_correct"] for r in observations),
                        "not_correct_to_correct": sum(r["not_correct_to_correct"] for r in observations),
                        "median_after_minus_before_prompt_tokens_est": statistics.median(costs) if costs else None}
    return {"policy": POLICY, "development_only": True, "families": len(pairs),
            "paired_families": sum(p["status"] == "paired" for p in pairs),
            "independent_sample_count": None, "pairs": pairs, "summary": summary,
            "interpretation": "Dependent recent-to-old observations; time and intervening content confound causal attribution. Source presence is not semantic answer support."}
