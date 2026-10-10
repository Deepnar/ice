"""Plan or manually answer a bounded v3 development coordinate shortlist.

Reuse the existing cloud SDK/answer cache and fault policy. No writer work,
qualified grade, validation execution or automatic tuning promotion.
"""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.compare_tuning_contexts import COMMON, load_contexts
from experiments.v3_repair.fault_receipts import FaultRecorder
from experiments.v3_repair.prepare_inputs import digest, encoded, fingerprint, write_or_verify
from experiments.v3_repair.run_tuning_contexts import source_probes

POLICY = "ice-v3-diagnostic-tuning-answer-bridge-1"
PROFILE = "opencode-luna6"
ARMS = ("full", "no_codex", "vector_only", "recent_only")


def immutable_lines(path, rows):
    content = b"".join(encoded(row) + b"\n" for row in rows)
    if path.exists():
        if path.read_bytes() != content: raise ValueError("answer projection changed; retain this bundle")
        return
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as sink:
        sink.write(content); sink.flush(); os.fsync(sink.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def plan(root, baseline, candidates, out):
    if len(candidates) > 2: raise ValueError("at most two surviving coordinates per stage")
    producer, probes = source_probes(root)
    collections = [load_contexts(path, probes) for path in [baseline, *candidates]]
    common = collections[0][0]["identity"]
    if (common["coordinate"]["id"] != "baseline"
            or common["producer"] != producer["producer_identity"]
            or common["producer_trace_sha256"] != producer["trace_sha256"]
            or common["probes"] != sorted(probes)):
        raise ValueError("answer baseline differs from complete repaired producer")
    identifiers, stages = set(), set()
    for header, _ in collections:
        identity = header["identity"]; coordinate = identity["coordinate"]
        identifier = coordinate["id"]
        if (identifier in identifiers or any(identity[key] != common[key] for key in COMMON)
                or coordinate.get("requires_writer_replay", False)):
            raise ValueError("answer shortlist differs in source/code/profile or repeats a coordinate")
        identifiers.add(identifier)
        if identifier != "baseline": stages.add(coordinate["stage"])
    if len(stages) > 1: raise ValueError("survivors must belong to one cheap read stage")
    # No provider credentials or SDK client are loaded during planning.
    from experiments.lme.cloud_provider import PROFILES
    profile = PROFILES[PROFILE]
    from scripts.z1 import answer_as_of
    code = [Path(__file__), ROOT / "experiments/v3_repair/fault_receipts.py", ROOT / "experiments/v3_repair/compare_tuning_contexts.py",
            ROOT / "experiments/v3_repair/run_tuning_contexts.py", ROOT / "experiments/v3_repair/audit_development.py",
            ROOT / "experiments/lme/cloud_provider.py", ROOT / "scripts/z1/answer_as_of.py",
            ROOT / "scripts/z1/cloud_recovery.py", ROOT / "scripts/z1/answer_resume.py",
            ROOT / "scripts/z1/answer_resume_repairs.json"]
    identity = dict(version="v3", policy=POLICY, producer=common["producer"],
        producer_trace_sha256=common["producer_trace_sha256"],
        collections=[header["identity_sha256"] for header, _ in collections],
        coordinate_ids=[header["identity"]["coordinate"]["id"] for header, _ in collections],
        stage=next(iter(stages), "baseline"), arms=list(ARMS), probes=sorted(probes),
        answer_profile=PROFILE, answer_model=profile.model, endpoint=profile.endpoint,
        decoding=dict(max_output_tokens=1500, temperature=0, temperature_sent=profile.supports_temperature),
        code=[fingerprint(path) for path in sorted(code)])
    manifest = dict(identity=identity, identity_sha256=digest(identity),
        expected_answers=len(probes) * len(collections) * len(ARMS),
        bounded_transient_attempts=2 * len(probes) * len(collections) * len(ARMS),
        source_labels="complete_selected_producer_reviewed_keys",
        content_writer_and_judge_qualification="pending_separate_gates",
        diagnostic_only=True, score_of_record=False, promotion="not_performed")
    out.mkdir(parents=True, exist_ok=True)
    with (root / "seed.jsonl").open() as source: source_run = json.loads(next(source))
    for header, current in collections:
        directory = out / header["identity_sha256"]; directory.mkdir(exist_ok=True)
        meta = copy.deepcopy(source_run)
        meta["meta"]["extra"]["diagnostic_coordinate_projection"] = dict(
            policy=POLICY, answer_plan=manifest["identity_sha256"], collection=header["identity_sha256"],
            old_parser_mode="allow_partial; complete_selected_producer_validated_by_adapter")
        rows = []
        for key in sorted(probes):
            row = copy.deepcopy(current[key])
            row["review"] = row["label_review"]
            rows.append(row)
        trace = directory / "projected.jsonl"
        immutable_lines(trace, [meta, *rows])
        # Qualify the real parser's input contract before any client construction.
        if len(answer_as_of.load_probes(trace, allow_partial=True)) != len(probes):
            raise ValueError("cloud parser dropped selected occurrences")
    write_or_verify(out / "answer-plan.json", manifest)
    return manifest


def source_text(probe):
    return "\n\n".join(
        f"Source turn {gold['turn']} at {gold['recorded_at']} "
        f"(timestamp provenance: {gold.get('ts_provenance', 'not recorded')})\n"
        f"User: {gold['prompt']}\nAssistant: {gold['response']}" for gold in probe["gold_sources"])


def validate_answers(output, probes, arm, *, profile_model):
    if not output.exists(): return
    result = json.loads(output.read_text())
    by_id = {p["probe_id"]: p for p in probes}
    seen = set()
    for row in result["records"]:
        key = row.get("probe_id")
        if key not in by_id or key in seen: raise ValueError("cached cloud occurrence differs")
        seen.add(key); probe = by_id[key]
        body = {k: v for k, v in row.items() if k != "answer_record_sha256"}
        if row.get("answer_record_sha256") != digest(body):
            raise ValueError("cached cloud answer body changed")
        expected = dict(conversation=probe["conversation"], split_turn=probe["split_turn"],
            question=probe["question"], expected_answer=probe["expected_answer"],
            catalog_expected_answer=probe.get("catalog_expected_answer", probe["expected_answer"]),
            gold_turns=probe["gold_turns"], gold_turn_text=source_text(probe),
            label_review=probe["review"], development_repeat=probe.get("development_repeat"),
            question_time=probe.get("question_time"), question_time_provenance=probe.get("question_time_provenance"),
            memory_processing=probe.get("memory_processing"), probe_type=probe["type"],
            answer_profile=PROFILE, answer_model=profile_model)
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("cached cloud question/source/reference/task metadata changed")
        stage = probe["preflight"] if arm == "full" else probe["controls"][arm]
        if row.get("answer_input_messages") != stage["prompt_messages"]:
            raise ValueError("cached cloud answering input changed")


def execute(out, manifest):
    from scripts.z1 import answer_as_of as native
    original_generator, original_atomic = native.TextGenerator, native.atomic_json
    for coordinate_id, identifier in zip(manifest["identity"]["coordinate_ids"], manifest["identity"]["collections"]):
        directory = out / identifier
        trace = directory / "projected.jsonl"
        probes = native.load_probes(trace, allow_partial=True)
        for arm in ARMS:
            output = directory / f"answers-{arm}.json"
            validate_answers(output, probes, arm, profile_model=manifest["identity"]["answer_model"])
            faults = FaultRecorder(directory / f"faults-{arm}.jsonl", manifest["identity_sha256"], private_root=native.LOGS)
            faults.verify()
            sessions = {f"ice-v3-z1-{arm}-{p['probe_id']}": p for p in probes}
            class NamespacedGenerator:
                def __init__(self, profile): self.inner = original_generator(profile)
                def generate(self, messages, **kwargs):
                    probe = sessions.get(kwargs.get("session_id"))
                    if probe is None: raise ValueError("cloud fault observer session is not a frozen occurrence")
                    saved = [r for r in json.loads(output.read_text())["records"] if r["probe_id"] == probe["probe_id"]]
                    stage = probe["preflight"] if arm == "full" else probe["controls"][arm]
                    if (len(saved) != 1 or saved[0].get("cloud_attempts") not in (1, 2)
                            or saved[0].get("error_type") != "UnconfirmedCloudCall" or saved[0].get("answer") != ""
                            or saved[0].get("answer_record_sha256") != digest({k: v for k, v in saved[0].items() if k != "answer_record_sha256"})
                            or saved[0].get("answer_input_sha256") != native.input_digest(messages)
                            or saved[0].get("answer_input_messages") != messages
                            or messages != stage["prompt_messages"]):
                        raise ValueError("cloud call lacks its exact durable attempt/input reservation")
                    context = dict(conversation=probe["conversation"], turn=probe["split_turn"], cutoff=probe["split_turn"],
                        probe_id=probe["probe_id"], source_time=probe.get("question_time"), job="answer", stage="cloud_answer",
                        attempt=saved[0]["cloud_attempts"], attempts=2, coordinate=coordinate_id, arm=arm,
                        model=manifest["identity"]["answer_model"], profile=PROFILE,
                        window=stage.get("context_ledger", {}).get("serving_window"),
                        artifact_id=saved[0]["answer_input_sha256"])
                    kwargs["session_id"] = f"ice-v3-tune-{identifier[:16]}-{kwargs['session_id']}"
                    return faults.call(lambda: self.inner.generate(messages, **kwargs), context=context)
            def bound_save(path, value):
                if path == output:
                    value["diagnostic_only"] = True; value["score_of_record"] = False
                    value["coordinate_answer_plan"] = manifest["identity_sha256"]
                    for row in value["records"]:
                        row["answer_record_sha256"] = digest({k: v for k, v in row.items() if k != "answer_record_sha256"})
                original_atomic(path, value)
            arguments = ["answer_as_of.py", "--trace", str(trace), "--out", str(output),
                "--arm", arm, "--profile", PROFILE, "--allow-partial", "--failure-policy", "continue"]
            if output.exists(): arguments.append("--resume")
            print(f"v3 diagnostic tuning answers: {identifier[:12]} {arm}; saved successes resume", flush=True)
            with patch.object(sys, "argv", arguments), patch.object(native, "TextGenerator", NamespacedGenerator), patch.object(native, "atomic_json", bound_save):
                code = native.main()
            if code: return code
            validate_answers(output, probes, arm, profile_model=manifest["identity"]["answer_model"])
            # The native cache keeps its original arm tag and immutable bytes.
            # Distinct derived names let the independent grader compare coordinates.
            graded_input = json.loads(output.read_text())
            graded_input["native_answer_tag"] = graded_input["tag"]
            graded_input["tag"] = f"{coordinate_id}:{arm}"
            graded_input["native_answer_cache_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            write_or_verify(directory / f"grading-{arm}.json", graded_input)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root, baseline, candidates, out = args.development_run.resolve(), args.baseline.resolve(), [p.resolve() for p in args.candidate], args.out.resolve()
    if (any(not p.is_relative_to((ROOT / "logs").resolve()) for p in [root, baseline, *candidates, out])
            or any(out.is_relative_to(p) or p.is_relative_to(out) for p in [root, baseline, *candidates])):
        raise ValueError("answer source/output bundles must be separate private logs/ directories")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".answer-bridge.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = plan(root, baseline, candidates, out)
        print(json.dumps(dict(version="v3", answer_model=manifest["identity"]["answer_model"],
            coordinates=len(manifest["identity"]["collections"]), expected_answers=manifest["expected_answers"],
            bounded_transient_attempts=manifest["bounded_transient_attempts"], diagnostic_only=True, score_of_record=False)))
        return execute(out, manifest) if args.run else 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("v3 tuning answers interrupted; cached successes/attempt receipts retained. Rerun identical arguments.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v3 tuning answers paused/refused: {exc}. Prior evidence retained.", file=sys.stderr)
        raise SystemExit(2)
