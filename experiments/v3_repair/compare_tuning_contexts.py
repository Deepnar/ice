"""Compare matched local v3 read coordinates, without answer-quality inference."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from experiments.v3_repair.run_tuning_contexts import source_probes, validate_unit
from scripts.z1.evidence_content import audit_probe

ARMS = ("full", "no_codex", "vector_only", "recent_only")
COMMON = ("producer", "producer_trace_sha256", "coordinate_seal", "code", "probes", "settings_sha256")


def load_contexts(directory, probes):
    header = json.loads((directory / "run.json").read_text())
    if digest(header["identity"]) != header["identity_sha256"]:
        raise ValueError("read collection identity changed")
    records = []
    if {p.name for p in (directory / "units").glob("*.json")} != {digest(key) + ".json" for key in probes}:
        raise ValueError("read collection occurrence roster differs")
    for key in sorted(probes):
        unit = validate_unit(directory / "units" / (digest(key) + ".json"), header, probes)
        current, original = unit["probe"], probes[key]
        if (set(current) != set(original) or any(current[k] != original[k] for k in original
                if k not in {"preflight", "controls"})
                or set(current["controls"]) != set(ARMS) - {"full"}):
            raise ValueError("coordinate question/source/task/cutoff or control roster changed")
        records.append(unit)
    expected = dict(**header, records=records); expected["complete"] = True
    if json.loads((directory / "contexts.json").read_text()) != expected:
        raise ValueError("coordinate aggregate differs from complete saved units")
    return header, {u["probe_id"]: u["probe"] for u in records}


def source_ids(stage):
    return sorted({key for fragment in stage["selected"]
                   for key in [fragment.get("source_row"), *(fragment.get("origin_batches") or [])] if key})


def reviewed_units(path, trace_sha, probes):
    if path is None: return {}
    packet = json.loads(path.read_text())
    if packet.get("kind") != "v3_required_evidence_review" or packet.get("trace_sha256") != trace_sha:
        raise ValueError("source-unit packet belongs to a different producer trace")
    reviews = {}
    for review in packet["records"]:
        key = review["probe_id"]
        if key not in probes or key in reviews:
            raise ValueError("unknown/duplicate required-unit occurrence")
        if any(requirement.get("semantic_review") for requirement in review["requirements"]):
            raise ValueError("source-unit packet cannot transfer content-bound semantic grades")
        # Validate source roles/hashes/offsets before interpreting any delta.
        audit_probe(probes[key], review)
        reviews[key] = review
    return reviews


def compare(root, baseline, candidate, requirements=None):
    producer, probes = source_probes(root)
    base_header, bases = load_contexts(baseline, probes)
    next_header, candidates = load_contexts(candidate, probes)
    first, second = base_header["identity"], next_header["identity"]
    if (first["coordinate"]["id"] != "baseline" or any(first[k] != second[k] for k in COMMON)
            or first.get("reader_adoption") != second.get("reader_adoption")
            or first["producer"] != producer["producer_identity"]
            or first["producer_trace_sha256"] != producer["trace_sha256"]
            or first["probes"] != sorted(probes)):
        raise ValueError("coordinate comparison is not the same producer/baseline/profile")
    reviews = reviewed_units(requirements, producer["trace_sha256"], probes)
    rows, totals = [], {arm: Counter() for arm in ARMS}
    for key in sorted(probes):
        before, after = bases[key], candidates[key]
        contract = probes[key]["metric_contract"]
        for arm in ARMS:
            a = before["preflight"] if arm == "full" else before["controls"][arm]
            b = after["preflight"] if arm == "full" else after["controls"][arm]
            original_ids, current_ids = set(source_ids(a)), set(source_ids(b))
            gate_fields = ("b2_retrieve", "final_retrieve", "source_action", "routed_model", "total_budget")
            changed = {field: dict(baseline=a["gate"].get(field), candidate=b["gate"].get(field))
                       for field in gate_fields if a["gate"].get(field) != b["gate"].get(field)}
            rank = {str(k): dict(baseline=a["gold_fragment_coverage"][k], candidate=b["gold_fragment_coverage"][k])
                    for k in ("source_turn_rank_at_5", "source_turn_rank_at_10")} if contract["lookup_rank_tuning_eligible"] else None
            literal = None
            if key in reviews:
                observations = {label: audit_probe(probe, reviews[key], arm=arm)["requirements"]
                                for label, probe in (("baseline", before), ("candidate", after))}
                literal = {label: {r["id"]: {stage: value["literal"] for stage, value in r["stages"].items()}
                                   for r in units} for label, units in observations.items()}
            row = dict(probe_id=key, arm=arm, metric_contract=contract,
                prompt_tokens=dict(baseline=a["prompt_tokens"], candidate=b["prompt_tokens"],
                                   delta=b["prompt_tokens"] - a["prompt_tokens"]),
                selected_tokens=dict(baseline=a["selected_tokens"], candidate=b["selected_tokens"],
                                     delta=b["selected_tokens"] - a["selected_tokens"]),
                rendered_input_changed=digest(a["prompt_messages"]) != digest(b["prompt_messages"]),
                gate_changes=changed, selected_sources_added=sorted(current_ids - original_ids),
                selected_sources_removed=sorted(original_ids - current_ids), source_rank_diagnostics=rank,
                source_coverage=dict(baseline=a["gold_fragment_coverage"], candidate=b["gold_fragment_coverage"]),
                required_literal_survival=literal, semantic_support="unmeasured", answer_quality="unmeasured")
            rows.append(row)
            counter = totals[arm]
            counter["occurrences"] += 1
            counter["rendered_inputs_changed"] += row["rendered_input_changed"]
            counter["gate_changes"] += bool(changed)
            counter["prompt_tokens_delta"] += row["prompt_tokens"]["delta"]
            counter["rank_eligible_occurrences"] += rank is not None
            counter["source_unit_reviewed_occurrences"] += literal is not None
    inputs = [root / "seed.jsonl", baseline / "contexts.json", candidate / "contexts.json"]
    if requirements is not None: inputs.append(requirements)
    return dict(version="v3", baseline_identity=base_header["identity_sha256"],
        candidate_identity=next_header["identity_sha256"], coordinate=second["coordinate"],
        input_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        evaluator_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ("comparison", Path(__file__)), ("source_content", ROOT / "scripts/z1/evidence_content.py"))},
        totals={arm: dict(value) for arm, value in totals.items()}, records=rows,
        score_of_record=False, promotion="not_performed", cloud_calls=False,
        limits=["Matched dependent occurrences are not independent question families.",
                "Only the reviewed lookup cohort enters precise rank diagnostics.",
                "Required-literal retention is not positive semantic support or memory-use evidence.",
                "Missing source-unit labels and other task kinds remain unmeasured.",
                "Native model/capacity freeze and independent answer grading remain separate gates."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-units", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = [args.development_run, args.baseline, args.candidate, args.out]
    if args.source_units is not None: paths.append(args.source_units)
    if any(not p.resolve().is_relative_to((ROOT / "logs").resolve()) for p in paths):
        raise ValueError("source/comparison artifacts must stay below private logs/")
    out = args.out.resolve()
    if any(out.is_relative_to(p.resolve()) for p in paths[:3]):
        raise ValueError("derived report must not alter its immutable input bundles")
    report = compare(args.development_run, args.baseline, args.candidate, args.source_units)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_or_verify(out, report)
    print(json.dumps(dict(version="v3", coordinate=report["coordinate"]["id"],
                         totals=report["totals"], score_of_record=False, promotion=report["promotion"])))


if __name__ == "__main__": main()
