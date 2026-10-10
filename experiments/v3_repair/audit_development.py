"""Offline source/task diagnostics and pending content review for repaired v3."""

import argparse
import fcntl
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, encoded, write_or_verify
from scripts.z1.metric_contract import metric_contract, summarize_contracts
from scripts.z1.replay_validation import validate_complete_replay
from scripts.z1.report_v3_replay import summarize
from scripts.z1.run_meta import file_digest


def audit(root):
    config = json.loads((root / "development-run.json").read_text())
    if config["identity_sha256"] != digest(config["identity"]):
        raise ValueError("development plan identity changed")
    rows = [json.loads(line) for line in (root / "seed.jsonl").read_text().splitlines()]
    checked = validate_complete_replay(iter(rows), config["identity"]["expected_turns"])
    if rows[0]["meta"]["extra"].get("selected_replay_plan") != config["identity"]:
        raise ValueError("replay does not belong to this selected development plan")
    bound_inputs = rows[0]["meta"]["inputs"]
    for name in ("development-run.json", "reviewed-keys.json"):
        if file_digest(root / name) not in bound_inputs:
            raise ValueError("replay source-key/plan bytes changed")
    keys = json.loads((root / "reviewed-keys.json").read_text())["records"]
    expected = {(r["review"]["probe_id"], r["review"]["conversation"],
                 r["review"]["cutoff_turn"]): r for r in keys}
    if len(expected) != len(keys): raise ValueError("duplicate reviewed source keys")
    observed, cohorts, requirements = set(), defaultdict(Counter), []
    contracts = []
    for row in rows:
        if row["event"] != "as_of_probe": continue
        key = (row["probe_id"], row["conversation"], row["split_turn"])
        if key in observed or key not in expected: raise ValueError("duplicate or unreviewed development probe")
        observed.add(key)
        reviewed = expected[key]
        review = reviewed["review"]
        contract = metric_contract(review, reviewed["gold_turns"])
        if (row["question"] != review["question"] or row["gold_turns"] != reviewed["gold_turns"]
                or row["expected_answer"] != reviewed["reviewed_answer"]
                or row.get("label_review") != review or row.get("metric_contract") != contract):
            raise ValueError("probe source/question/task/answer differs from reviewed key")
        contracts.append(contract)
        stages = {"full": row["preflight"], **row["controls"]}
        if set(stages) != {"full", "no_codex", "vector_only", "recent_only"}:
            raise ValueError("development controls incomplete")
        for arm, stage in stages.items():
            coverage = stage["gold_fragment_coverage"]
            if coverage["gold_turns"] != len(row["gold_turns"]): raise ValueError("source coverage denominator differs")
            cohort = "single_source_lookup" if contract["lookup_rank_tuning_eligible"] else "other_annotated_tasks"
            counter = cohorts[arm + ":" + cohort]
            counter["occurrences"] += 1
            counter["gold_turns"] += coverage["gold_turns"]
            counter["source_turns_final_evidence"] += coverage["selected_or_source_note"]
            counter["total_prompt_tokens"] += stage["prompt_tokens"]
            if contract["lookup_rank_tuning_eligible"]:
                counter["source_turn_rank_at_5"] += coverage["source_turn_rank_at_5"]
                counter["source_turn_rank_at_10"] += coverage["source_turn_rank_at_10"]
        requirements.append(dict(probe_id=row["probe_id"], reviewed_through_turn=row["split_turn"],
            question=row["question"], reviewed_answer=row["expected_answer"],
            source_bodies=row["gold_sources"], metric_contract=contract,
            requirements=[], reason="", review_status="pending_required_units_and_context_support"))
    if observed != set(expected): raise ValueError("replay lacks a reviewed development occurrence")
    trace_sha = hashlib.sha256((root / "seed.jsonl").read_bytes()).hexdigest()
    packet = dict(kind="v3_required_evidence_review", trace_sha256=trace_sha,
                  records=requirements, qualification="pending", approved_units=0)
    report = dict(version="v3", score_of_record=False, trace_sha256=trace_sha,
        producer_identity=config["identity_sha256"], source_keys_sha256=digest(keys),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        complete_replay=checked["complete"], mechanism=summarize(root / "seed.jsonl"),
        metric_cohorts=summarize_contracts(contracts), source_diagnostics={k: dict(v) for k, v in cohorts.items()},
        semantic_review=dict(pending_occurrences=len(requirements), qualified_occurrences=0,
                             supported_answers="unmeasured", quality_per_token="unmeasured"),
        limits=["Source-ID/rank diagnostics are not semantic content support or memory-use attribution.",
                "Other reviewed tasks do not enter the precise lookup-rank cohort.",
                "Source keys and expected answers do not certify compressed or generated outputs.",
                "Model decisions and independent writer/lifecycle output review remain separate."])
    return report, packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root, out = args.run_dir.resolve(), args.out.resolve()
    if any(not p.is_relative_to((ROOT / "logs").resolve()) for p in (root, out)) or root == out:
        raise ValueError("distinct private replay and derived-output directories required")
    report, packet = audit(root)
    products = {"report.json": report, "content-review-template.json": packet}
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, value in products.items():
            if (out / name).exists() and (out / name).read_bytes() != encoded(value) + b"\n":
                raise ValueError("derived report inputs/instrument changed; preserve it and use a new directory")
        for name, value in products.items(): write_or_verify(out / name, value)
    print(json.dumps(dict(version="v3", reviewed_occurrences=report["metric_cohorts"]["occurrences"],
                         semantic_review=report["semantic_review"], score_of_record=False)))


if __name__ == "__main__": main()
