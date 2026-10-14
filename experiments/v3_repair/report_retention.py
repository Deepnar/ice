"""Offline recent-to-old comparisons from sealed independent v3 grades.

Reads existing answer/plan/cache/report files; never calls or changes a judge.
The small unchanged-fact development policy is not final semi-LSREP scoring.
"""

import argparse
import fcntl
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import write_or_verify
from scripts.z1 import score_frozen_answers as scorer
from scripts.z1.answer_as_of import private_path
from scripts.z1.development_repeats import POLICY as REPEAT_POLICY

POLICY = "independent_development_retention_v1"
ORDINAL = {"incorrect": 0, "partial": 1, "correct": 2}


def paired_outcomes(arms, records):
    """Only explicit, compatible before/after labels establish a temporal pair."""
    grades = {(r["arm"], tuple(r["occurrence"])): r for r in records}
    if len(grades) != len(records):
        raise ValueError("duplicate independent outcome")
    groups, undeclared = defaultdict(dict), []
    for arm, rows in sorted(arms.items()):
        for occurrence, row in sorted(rows.items()):
            version, _, history, _ = occurrence
            if version != "v3":
                raise ValueError("retention needs a native v3 occurrence")
            meta = row["development_repeat"]
            if meta is None:
                continue
            family = meta.get("question_family_id")
            phase = meta.get("phase")
            if not isinstance(family, str) or not family.strip() or phase not in {"before", "after"}:
                undeclared.append(dict(arm=arm, occurrence=occurrence,
                                       reason="incomplete_repeat_metadata"))
                continue
            group = groups[arm, history, family]
            if phase in group:
                raise ValueError("duplicate repeat phase within history/family/arm")
            group[phase] = (occurrence, row, grades[arm, occurrence])

    pairs = []
    for (arm, history, family), phases in sorted(groups.items()):
        reasons = []
        missing = sorted({"before", "after"} - phases.keys())
        if missing:
            reasons.append("missing_phase")
        observations = {}
        for phase, (occurrence, row, result) in phases.items():
            _, _, _, cutoff = occurrence
            meta = row["development_repeat"]
            gold = row["gold_turns"]
            age, window = meta.get("source_age_turns"), meta.get("recent_window_turns")
            if (meta.get("policy") != REPEAT_POLICY or row["family"] != family
                    or not isinstance(meta.get("base_probe_id"), str) or not meta["base_probe_id"].strip()
                    or type(age) is not int or age < 0 or type(window) is not int or window < 1):
                reasons.append("invalid_repeat_contract")
            elif (not gold or age != cutoff - max(gold)
                    or min(gold) < 1 or max(gold) > cutoff
                    or (phase == "before" and min(gold) <= cutoff - window)
                    or (phase == "after" and age < window)):
                reasons.append("invalid_source_age_or_window")
            expected_cohort = "recent_control" if phase == "before" else "older_source"
            if row["metric_contract"]["age_cohort"] != expected_cohort:
                reasons.append("inconsistent_age_cohort")
            value = result["grade"]
            grade = value.get("grade") if isinstance(value, dict) else None
            if result["status"] != "graded" or grade not in ORDINAL:
                reasons.append("unresolved_grade")
            tokens = row["prompt_tokens"]
            if type(tokens) not in (int, float) or not math.isfinite(tokens) or tokens < 0:
                tokens = None
            observations[phase] = dict(occurrence=occurrence, cutoff_turn=cutoff,
                source_age_turns=age, recent_window_turns=window,
                question_time=row["evidence"].get("question_time"),
                question_time_provenance=row["evidence"].get("question_time_provenance"),
                status=result["status"], grade=grade, grade_key=result["grade_key"],
                prompt_tokens=tokens)
        if not missing:
            early_id, early, _ = phases["before"]
            late_id, late, _ = phases["after"]
            if early_id[3] >= late_id[3]:
                reasons.append("nonincreasing_cutoff")
            if (any(early["evidence"].get(k) != late["evidence"].get(k) for k in
                    ("question", "expected_answer", "source", "knowledge_scope", "task_types", "editorial_answer_contract"))
                    or early["gold_turns"] != late["gold_turns"]
                    or any(early["development_repeat"].get(k) != late["development_repeat"].get(k)
                           for k in ("base_probe_id", "recent_window_turns"))):
                reasons.append("changed_question_answer_or_source_contract")
        resolved = not reasons
        before, after = observations.get("before", {}), observations.get("after", {})
        a, b = before.get("grade"), after.get("grade")
        movement = ("improved" if ORDINAL[b] > ORDINAL[a] else
                    "worsened" if ORDINAL[b] < ORDINAL[a] else "unchanged") if resolved else "unresolved"
        costs = [r.get("prompt_tokens") for r in (before, after)]
        pairs.append(dict(arm=arm, history=history, question_family_id=family,
            status="resolved" if resolved else "unresolved", reasons=sorted(set(reasons)),
            missing_phases=missing, phases=observations, movement=movement,
            correct_to_not_correct=(a == "correct" and b != "correct") if resolved else None,
            not_correct_to_correct=(a != "correct" and b == "correct") if resolved else None,
            after_minus_before_prompt_tokens=(costs[1] - costs[0]) if all(t is not None for t in costs) else None))
    summary = {}
    for arm in sorted(arms):
        rows = [p for p in pairs if p["arm"] == arm]
        resolved = [p for p in rows if p["status"] == "resolved"]
        initially_correct = [p for p in resolved if p["phases"]["before"]["grade"] == "correct"]
        costs = [p["after_minus_before_prompt_tokens"] for p in resolved
                 if p["after_minus_before_prompt_tokens"] is not None]
        lost = sum(p["correct_to_not_correct"] for p in initially_correct)
        summary[arm] = dict(declared_families=len(rows), paired_families=sum(not p["missing_phases"] for p in rows),
            resolved_families=len(resolved), unresolved_families=len(rows) - len(resolved),
            incomplete_metadata_occurrences=sum(p["arm"] == arm for p in undeclared),
            movements=dict(Counter(p["movement"] for p in rows)),
            resolved_before_correct=len(initially_correct), correct_to_not_correct=lost,
            retained_correct=len(initially_correct) - lost,
            retained_correct_fraction=(len(initially_correct) - lost) / len(initially_correct) if initially_correct else None,
            paired_token_observations=len(costs), median_after_minus_before_prompt_tokens=statistics.median(costs) if costs else None)
    return dict(policy=POLICY, development_only=True, qualification="unqualified_diagnostic",
        independent_sample_count=None, pairs=pairs, by_arm=summary, undeclared_occurrences=undeclared,
        interpretation="Explicit unchanged-fact repeated questions only. Time and intervening content confound causality. "
                       "Uncertain/missing grades do not prove forgetting; source presence is not answer support. "
                       "Final correction and multiple-lag schedules require a separate reviewed contract.")


def derive(paths, report_path):
    """Rebuild the native scorer report against its bound answers and receipts."""
    bindings = {}

    def read(path):
        path = private_path(path)
        raw = path.read_bytes()
        bindings[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), mtime_ns=path.stat().st_mtime_ns)
        return json.loads(raw)

    report_path = private_path(report_path)
    report = read(report_path)
    if (report.get("version") != "v3" or report.get("policy") != scorer.POLICY
            or report.get("mode") != "absolute" or report.get("processing_complete") is not True):
        raise ValueError("requires a terminal independent absolute report")
    plan_key = report.get("plan_sha256")
    if not isinstance(plan_key, str) or len(plan_key) != 64 or any(c not in "0123456789abcdef" for c in plan_key):
        raise ValueError("invalid native scoring plan key")
    root = report_path.parent.parent
    if report_path.parent.name != "reports" or report_path.name != plan_key + ".json":
        raise ValueError("report must retain its native cache/plan directory layout")
    plan = read(root / "plans" / (plan_key + ".json"))
    if (scorer.digest(plan) != plan_key or plan.get("policy") != scorer.POLICY
            or plan.get("mode") != "absolute" or plan.get("pairs") != []):
        raise ValueError("scoring plan differs from report")
    identity = None
    for key in plan["task_keys"]:
        if not isinstance(key, str) or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid scoring task key")
        state = read(root / "cache" / (key + ".json"))
        current = state["task"]["judge"]
        if identity is not None and current != identity:
            raise ValueError("mixed judge identities in absolute plan")
        identity = current
    for path in paths:
        read(path)
    arms, inputs = scorer.collect(paths, identity or {})
    if inputs != plan.get("inputs"):
        raise ValueError("answer bytes differ from sealed scoring plan")
    tasks, pairs = scorer.work(arms, identity or {}, "absolute", root)
    if sorted(tasks) != plan["task_keys"]:
        raise ValueError("native absolute tasks differ from sealed plan")
    states = {key: scorer.cached(root, key, task) for key, task in tasks.items()}
    rebuilt = scorer.report(arms, tasks, pairs, states, plan_key, "absolute")
    if json.loads(json.dumps(rebuilt)) != report:
        raise ValueError("native report differs from its answers or grade receipts")
    result = paired_outcomes(arms, report["records"])
    for path in [Path(__file__), *[ROOT / p for p in scorer.implementation()],
                 ROOT / "experiments/v3_repair/prepare_inputs.py", ROOT / "scripts/z1/development_repeats.py"]:
        raw = path.read_bytes()
        bindings[str(path)] = dict(sha256=hashlib.sha256(raw).hexdigest(), mtime_ns=path.stat().st_mtime_ns)
    for path, binding in bindings.items():
        item = Path(path)
        if hashlib.sha256(item.read_bytes()).hexdigest() != binding["sha256"] or item.stat().st_mtime_ns != binding["mtime_ns"]:
            raise ValueError("retention input changed during reading")
    return dict(version="v3", kind="independent_retention_report", score_of_record=False,
        source_bindings=bindings, source_plan_sha256=plan_key, model_calls=0,
        original_changes=False, retention=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, nargs="+", required=True)
    parser.add_argument("--score-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.answers, args.score_report)
    out = private_path(args.out)
    if str(out) in result["source_bindings"] or str(out.with_suffix(out.suffix + ".lock")) in result["source_bindings"]:
        raise ValueError("output would overwrite a bound input")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(out.suffix + ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_or_verify(out, result)
    print(json.dumps(dict(version="v3", by_arm=result["retention"]["by_arm"], model_calls=0)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
