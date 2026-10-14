"""Collect and compare full-ICE contexts for sealed v3 read coordinates.

Tuning selects ICE's own settings, so each coordinate needs only the `full`
arm. The completed four-arm baseline already holds the recent/vector/no-Codex
controls. This wrapper keeps `run_tuning_contexts.py` and
`compare_tuning_contexts.py` byte-identical, so the shared producer, code and
reader-profile identity still matches that baseline. It reuses their
validation, prefix restore and resumable units, and declares `arms=["full"]`
in its own identity.

`full` is the first arm and every probe runs inside a rolled-back read-only
transaction, so omitting the later arms cannot change the full context. The
repeated `baseline` coordinate measures the run-to-run floor that every other
coordinate's changed-input count is read against.

No writer replay, cloud call or answer grade. Local collection is launched
manually; completed units and coordinates resume unchanged.
"""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair import compare_tuning_contexts as matched
from experiments.v3_repair import run_tuning_contexts as runner
from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from scripts.z1.evidence_content import audit_probe

POLICY = "ice-v3-read-coordinate-full-arm-1"
ARMS = ("full",)


def setup(root, plan, identifier, out, *, owned=False, reader_adoption):
    """Extend the unchanged collector's identity; never recompute its fields."""
    with tempfile.TemporaryDirectory() as scratch:
        config, base, coordinate, probes = runner.setup(
            root, plan, identifier, Path(scratch), owned=owned, reader_adoption=reader_adoption)
    identity = dict(base["identity"], arms=list(ARMS), collector_policy=POLICY,
                    collector=fingerprint(Path(__file__).resolve()))
    header = dict(base, identity=identity, identity_sha256=digest(identity))
    out.mkdir(parents=True, exist_ok=True)
    write_or_verify(out / "run.json", header)
    for path in (out / "units").glob("*.json"):
        validate_unit(path, header, probes)
    return config, header, coordinate, probes


def validate_unit(path, header, probes):
    unit = runner.validate_unit(path, header, probes)
    probe = unit["probe"]
    if probe.get("controls") != {} or not isinstance(probe.get("preflight"), dict):
        raise ValueError("full-arm unit carries control arms or lacks its full context")
    return unit


def collect(root, out, header, coordinate, probes, config):
    from scripts.z1 import seed_v3
    with patch.object(seed_v3, "PROMPT_ARMS", ARMS):
        runner.collect(root, out, header, coordinate, probes, config)
    return load_contexts(out, probes)


def load_contexts(directory, probes):
    header = json.loads((directory / "run.json").read_text())
    identity = header["identity"]
    if (digest(identity) != header["identity_sha256"] or identity.get("arms") != list(ARMS)
            or identity.get("collector_policy") != POLICY):
        raise ValueError("full-arm collection identity changed")
    if {p.name for p in (directory / "units").glob("*.json")} != {digest(key) + ".json" for key in probes}:
        raise ValueError("full-arm collection occurrence roster differs")
    records = []
    for key in sorted(probes):
        unit = validate_unit(directory / "units" / (digest(key) + ".json"), header, probes)
        current, original = unit["probe"], probes[key]
        if (set(current) != set(original) | {"preflight", "controls"}
                or any(current[k] != original[k] for k in original if k not in {"preflight", "controls"})):
            raise ValueError("coordinate question/source/task/cutoff changed")
        records.append(unit)
    expected = dict(**header, records=records); expected["complete"] = True
    if json.loads((directory / "contexts.json").read_text()) != expected:
        raise ValueError("coordinate aggregate differs from complete saved units")
    return header, {u["probe_id"]: u["probe"] for u in records}


def compare(root, baseline, candidate, requirements=None):
    """Full-arm rows only; the shared identity must match the four-arm baseline."""
    producer, probes = runner.source_probes(root)
    base_header, bases = matched.load_contexts(baseline, probes)
    next_header, candidates = load_contexts(candidate, probes)
    first, second = base_header["identity"], next_header["identity"]
    if (first["coordinate"]["id"] != "baseline" or any(first[k] != second[k] for k in matched.COMMON)
            or first.get("reader_adoption") != second.get("reader_adoption")
            or first["producer"] != producer["producer_identity"]
            or first["producer_trace_sha256"] != producer["trace_sha256"]
            or first["probes"] != sorted(probes)):
        raise ValueError("coordinate comparison is not the same producer/baseline/profile")
    reviews = matched.reviewed_units(requirements, producer["trace_sha256"], probes)
    rows, totals = [], Counter()
    gate_fields = ("b2_retrieve", "final_retrieve", "source_action", "routed_model", "total_budget")
    for key in sorted(probes):
        a, b = bases[key]["preflight"], candidates[key]["preflight"]
        contract = probes[key]["metric_contract"]
        original_ids, current_ids = set(matched.source_ids(a)), set(matched.source_ids(b))
        changed = {field: dict(baseline=a["gate"].get(field), candidate=b["gate"].get(field))
                   for field in gate_fields if a["gate"].get(field) != b["gate"].get(field)}
        rank = {k: dict(baseline=a["gold_fragment_coverage"][k], candidate=b["gold_fragment_coverage"][k])
                for k in ("source_turn_rank_at_5", "source_turn_rank_at_10")} if contract["lookup_rank_tuning_eligible"] else None
        literal = None
        if key in reviews:
            literal = {label: {r["id"]: {stage: value["literal"] for stage, value in r["stages"].items()}
                               for r in audit_probe(probe, reviews[key])["requirements"]}
                       for label, probe in (("baseline", bases[key]), ("candidate", candidates[key]))}
        row = dict(probe_id=key, arm="full", metric_contract=contract,
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
        totals["occurrences"] += 1
        totals["rendered_inputs_changed"] += row["rendered_input_changed"]
        totals["selected_sources_changed"] += bool(current_ids ^ original_ids)
        totals["gate_changes"] += bool(changed)
        totals["prompt_tokens_delta"] += row["prompt_tokens"]["delta"]
        totals["rank_eligible_occurrences"] += rank is not None
        totals["source_unit_reviewed_occurrences"] += literal is not None
    inputs = [root / "seed.jsonl", baseline / "contexts.json", candidate / "contexts.json"]
    if requirements is not None: inputs.append(requirements)
    return dict(version="v3", arms=list(ARMS), baseline_identity=base_header["identity_sha256"],
        candidate_identity=next_header["identity_sha256"], coordinate=second["coordinate"],
        input_sha256={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        evaluator_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in (
            ("comparison", Path(__file__)), ("matched", ROOT / "experiments/v3_repair/compare_tuning_contexts.py"),
            ("source_content", ROOT / "scripts/z1/evidence_content.py"))},
        totals=dict(totals), records=rows, score_of_record=False, promotion="not_performed", cloud_calls=False,
        limits=["Full arm only; control arms come from the four-arm baseline.",
                "Read changed-input counts against the repeated-baseline floor.",
                "Matched dependent occurrences are not independent question families.",
                "Required-literal retention is not semantic support or answer quality."])


def plan_ids(plan, wanted):
    rows = json.loads((plan / "coordinates.json").read_text())["rows"]
    ids = [row["id"] for row in rows]
    if not wanted: return ids
    if len(set(wanted)) != len(wanted) or set(wanted) - set(ids):
        raise ValueError("unknown/duplicate coordinate requested")
    return [identifier for identifier in ids if identifier in wanted]


def execute(root, plan, out, ids, profile, *, owned):
    from src.workers.bg_client_factory import release_owned_models
    try:
        for number, identifier in enumerate(ids, 1):
            directory = out / identifier
            config, header, coordinate, probes = setup(root, plan, identifier, directory,
                                                       owned=owned, reader_adoption=profile)
            if (directory / "contexts.json").exists():
                load_contexts(directory, probes)
            else:
                collect(root, directory, header, coordinate, probes, config)
            print(f"v3 full-arm coordinates: {number}/{len(ids)} complete ({identifier})", flush=True)
    finally:
        release_owned_models()


def summarize(root, plan, out, baseline, reports, requirements):
    reports.mkdir(parents=True, exist_ok=True)
    _, probes = runner.source_probes(root)
    lines = []
    for identifier in plan_ids(plan, None):
        directory = out / identifier
        if not (directory / "contexts.json").exists():
            lines.append(dict(coordinate=identifier, status="not_collected")); continue
        report = compare(root, baseline, directory, requirements)
        write_or_verify(reports / f"{identifier}.json", report)
        lines.append(dict(coordinate=identifier, status="compared", **report["totals"]))
    summary = dict(version="v3", arms=list(ARMS), probes=len(probes), coordinates=lines,
                   score_of_record=False, promotion="not_performed", cloud_calls=False)
    (reports / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for line in lines:
        name = line["coordinate"] + (" (repeat floor)" if line["coordinate"] == "baseline" else "")
        if line["status"] != "compared":
            print(f"{name:<34} not collected"); continue
        print(f"{name:<34} inputs changed {line['rendered_inputs_changed']:>2}/{line['occurrences']}"
              f"  sources changed {line['selected_sources_changed']:>2}  gate {line['gate_changes']:>2}"
              f"  tokens {line['prompt_tokens_delta']:+d}")
    return summary


def private_paths(*paths):
    logs = (ROOT / "logs").resolve()
    if any(not p.is_relative_to(logs) for p in paths):
        raise ValueError("source, plan, profile and outputs must stay below private logs/")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--tuning-plan", required=True, type=Path)
    parser.add_argument("--reader-adoption", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Directory holding one subdirectory per coordinate.")
    parser.add_argument("--coordinate", action="append", default=[], help="Repeatable; default is every sealed coordinate.")
    parser.add_argument("--run", action="store_true", help="Collect locally; without it, only print the plan.")
    parser.add_argument("--compare-baseline", type=Path, help="Completed four-arm baseline collection to compare against.")
    parser.add_argument("--reports", type=Path, help="Comparison output directory (with --compare-baseline).")
    parser.add_argument("--source-units", type=Path)
    parser.add_argument("--execute-owned", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, plan, out, profile = (p.resolve() for p in (args.development_run, args.tuning_plan, args.out, args.reader_adoption))
    private_paths(root, plan, out, profile)
    if out.is_relative_to(root) or out.is_relative_to(plan):
        raise ValueError("private source/planner and collector output bundles must be separate")
    if args.compare_baseline is not None:
        baseline, reports = args.compare_baseline.resolve(), (args.reports or out / "comparisons").resolve()
        private_paths(baseline, reports)
        if reports.is_relative_to(baseline) or reports.is_relative_to(root):
            raise ValueError("derived reports must not alter their immutable input bundles")
        summarize(root, plan, out, baseline, reports, args.source_units)
        return 0
    ids = plan_ids(plan, args.coordinate)
    with (root / ".operator.lock").open("a") as lock:
        if not args.execute_owned: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.execute_owned:
            execute(root, plan, out, ids, profile, owned=True)
            return 0
        pending = [i for i in ids if not (out / i / "contexts.json").exists()]
        if not args.run:
            # Same producer/profile/coordinate validation as collection, no models.
            for identifier in ids:
                setup(root, plan, identifier, out / identifier, reader_adoption=profile)
            print(json.dumps(dict(version="v3", arms=list(ARMS), coordinates=len(ids), pending=pending,
                                  cloud_calls=False), indent=1))
            return 0
        config = json.loads((root / "development-run.json").read_text())
        from scripts.z1.run_v3_campaign import database_environment
        env = database_environment(config, root)
        command = [sys.executable, str(Path(__file__).resolve()), "--development-run", str(root),
                   "--tuning-plan", str(plan), "--reader-adoption", str(profile), "--out", str(out),
                   "--execute-owned", *[f"--coordinate={i}" for i in args.coordinate]]
        out.mkdir(parents=True, exist_ok=True)
        log = out / "collector.log"
        print(f"v3 full-arm collection of {len(pending)}/{len(ids)} coordinates; details: {log}", flush=True)
        with log.open("a") as sink:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT, check=False)
        print("v3 full-arm coordinates complete" if child.returncode == 0 else
              f"v3 full-arm collection paused; saved units retained. Inspect {log} and rerun identical arguments.")
        return child.returncode


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("v3 full-arm collection interrupted; saved units retained; rerun identical arguments.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v3 full-arm collection refused/paused: {exc}. Existing evidence retained.", file=sys.stderr)
        raise SystemExit(2)
