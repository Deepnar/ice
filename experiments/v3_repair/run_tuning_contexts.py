"""Plan or collect one v3 read coordinate from exact repaired as-of snapshots.

No writer replay or cloud calls. Completed local probe units resume unchanged.
The development database is restored to its final committed state afterward.
"""

import argparse
import copy
import fcntl
import json
import os
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.audit_development import audit
from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from experiments.v3_repair.tuning_plan import applied, coordinates

POLICY = "ice-v3-read-coordinate-contexts-1"


def current_code(identity):
    for record in identity["code"]:
        path = (ROOT / record["path"]).resolve()
        if not path.is_relative_to(ROOT) or fingerprint(path) != record:
            raise ValueError("producer/coordinate preparation code changed; preserve this state")


def source_probes(root):
    report, _ = audit(root)
    probes = {}
    with (root / "seed.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row.get("event") == "as_of_probe": probes[row["probe_id"]] = row
    return report, probes


def load_coordinate(plan, identifier):
    seal = json.loads((plan / "manifest.json").read_text())
    if digest(seal["identity"]) != seal["identity_sha256"]:
        raise ValueError("tuning seal identity changed")
    products = {}
    for name, expected in seal["artifacts"].items():
        value = json.loads((plan / name).read_text())
        if digest(value) != expected: raise ValueError("tuning plan product changed")
        products[name] = value
    matches = [r for r in products["coordinates.json"]["rows"] if r["id"] == identifier]
    if len(matches) != 1: raise ValueError("unknown/duplicate tuning coordinate")
    return seal, matches[0]


def resolved_baseline(config, owned=False):
    from sqlalchemy.engine import make_url
    from src.api.config import settings
    values = settings.model_dump(mode="json")
    url = make_url(settings.database_url)
    if owned:
        if url.database != config["database"] or os.environ.get("ICE_TEST_DATABASE") != config["database"]:
            raise ValueError("read collector lacks the producer's owned database")
        url = url.set(database=config["identity"]["source_database"])
    values["database_url"] = url.render_as_string(hide_password=False)
    if digest(values) != config["identity"]["settings_sha256"]:
        raise ValueError("baseline settings differ from the repaired writer")
    return values


def setup(root, plan, identifier, out, *, owned=False, reader_adoption=None):
    config = json.loads((root / "development-run.json").read_text())
    if digest(config["identity"]) != config["identity_sha256"]:
        raise ValueError("development plan identity changed")
    if config["identity"].get("partition") != "tuning":
        raise ValueError("validation outcomes remain reserved until qualified tuning/freeze")
    baseline = resolved_baseline(config, owned)
    seal, coordinate = load_coordinate(plan, identifier)
    adoption = None
    if reader_adoption is None:
        current_code(config["identity"])
        current_code(seal["identity"])
    else:
        from experiments.v3_repair.reader_adoption import validate
        adoption = validate(json.loads(reader_adoption.read_text()), root, config, seal)
    if (seal["identity"]["preparation"] != config["identity"]["preparation"]
            or seal["identity"]["settings_sha256"] != digest(baseline)):
        raise ValueError("coordinate plan belongs to a different preparation/baseline")
    from src.api.config import settings
    if coordinate not in coordinates(settings):
        raise ValueError("coordinate differs from current baseline policy")
    report, probes = source_probes(root)
    from scripts.z1.replay_checkpoint import code_files
    _, files = code_files()
    files += [Path(__file__), ROOT / "experiments/v3_repair/tuning_plan.py",
              ROOT / "experiments/v3_repair/audit_development.py", ROOT / "scripts/z1/metric_contract.py"]
    identity = dict(version="v3", policy=POLICY, producer=config["identity_sha256"],
        producer_trace_sha256=report["trace_sha256"], coordinate_seal=seal["identity_sha256"],
        coordinate=coordinate, code=[fingerprint(p) for p in sorted(set(files))],
        probes=sorted(probes), settings_sha256=digest(baseline))
    if adoption is not None:
        identity["reader_adoption"] = adoption
    header = dict(identity=identity, identity_sha256=digest(identity),
        score_of_record=False, cloud_calls=False, writer_replayed=False,
        semantic_quality="unqualified", complete=False)
    out.mkdir(parents=True, exist_ok=True)
    write_or_verify(out / "run.json", header)
    for path in (out / "units").glob("*.json"):
        validate_unit(path, header, probes)
    return config, header, coordinate, probes


def validate_unit(path, header, probes):
    unit = json.loads(path.read_text())
    identifier = unit.get("probe_id")
    body = {k: v for k, v in unit.items() if k != "unit_sha256"}
    if (identifier not in probes or unit.get("identity_sha256") != header["identity_sha256"]
            or unit.get("source_probe_sha256") != digest(probes[identifier])
            or unit.get("unit_sha256") != digest(body)
            or path.name != digest(identifier) + ".json"):
        raise ValueError("cached read unit differs from producer/coordinate identity or content")
    return unit


def pending_groups(out, probes):
    groups = defaultdict(list)
    for identifier, probe in probes.items():
        if not (out / "units" / (digest(identifier) + ".json")).exists():
            groups[(probe["conversation"], probe["split_turn"])].append(probe)
    return dict(sorted(groups.items()))


def checkpoint_receipt(root, directory, config, *, identity=None, producer_code=None):
    from scripts.z1 import snapshot
    from scripts.z1.replay_checkpoint import code_digest, prefix_digest
    receipt = json.loads((directory / "recovery.json").read_text())
    bound = receipt["identity"]
    with (root / "seed.jsonl").open() as source:
        meta = json.loads(next(source))["meta"]
    size = receipt["trace_bytes"]
    if (receipt["trace_path"] != str((root / "seed.jsonl").resolve())
            or type(size) is not int or not 0 < size <= (root / "seed.jsonl").stat().st_size
            or prefix_digest(root / "seed.jsonl", size) != receipt["trace_sha256"]
            or bound.get("selected_replay_plan") != config["identity"]
            or bound.get("database") != config["database"]
            or bound.get("code_sha256") != (producer_code or code_digest())
            or bound.get("tables") != snapshot.TABLES
            or bound.get("inputs") != meta["inputs"]
            or bound.get("models") != meta["extra"]["resolved_background_model"]
            or bound.get("writer_model_digests") != meta["extra"].get("writer_model_digests")
            or (identity is not None and bound != identity)):
        raise ValueError("as-of recovery differs from the complete producer execution/trace")
    return receipt


@contextmanager
def prefix_state(root, config, groups, *, adoption=None):
    """Validate before model load; restore final state even with a complete cache."""
    from scripts.z1 import snapshot
    producer_code = None
    if adoption is not None:
        from experiments.v3_repair.reader_adoption import validate_runtime
        validate_runtime(adoption, root, config)
        producer_code = adoption["identity"]["producer_native_code_sha256"]
    recovery = root / "seed.recovery"
    name = json.loads((recovery / "current.json").read_text())["generation"]
    if not name.startswith("generation-") or Path(name).name != name:
        raise ValueError("invalid producer recovery generation")
    final = recovery / name
    receipt = checkpoint_receipt(root, final, config, producer_code=producer_code)
    if receipt["generation"] != name or receipt["state"]["completed"] != config["identity"]["expected_turns"]:
        raise ValueError("producer has no complete final recovery state")
    prefixes = {}
    for slug, cutoff in groups:
        directory = recovery / "evaluation" / f"{slug}-{cutoff}"
        prefix = checkpoint_receipt(root, directory, config, identity=receipt["identity"],
                                    producer_code=producer_code)
        if prefix.get("evaluation_key") != f"{slug}-{cutoff}" or prefix["state"]["completed"].get(slug) != cutoff:
            raise ValueError("wrong as-of prefix checkpoint")
        prefixes[slug, cutoff] = (directory, prefix)
    try:
        yield snapshot, prefixes
    finally:
        with patch.object(snapshot, "SNAPDIR", final):
            if snapshot.restore("store"):
                raise RuntimeError("producer final restore failed; retain units/logs and inspect before reuse")


def collect(root, out, header, coordinate, probes, config):
    groups = pending_groups(out, probes)
    with prefix_state(root, config, groups, adoption=header["identity"].get("reader_adoption")) as (snapshot, prefixes):
        if groups:
            from src.api.config import settings
            from src.api.db import SessionLocal
            from src.classifier.classifier import PyTorchClassifier
            from src.memory.embedder import get_embedder
            from scripts.z1 import production_parity as pp, seed_v3
            embedder = get_embedder()
            classifier = PyTorchClassifier(model_path=settings.classifier_model_path, schema_path=settings.label_schema_path)
            units = out / "units"; units.mkdir(exist_ok=True)
            for (slug, cutoff), group in groups.items():
                directory, receipt = prefixes[slug, cutoff]
                with patch.object(snapshot, "SNAPDIR", directory):
                    if snapshot.restore("store"): raise RuntimeError("as-of restore failed")
                times = {key: datetime.fromisoformat(value) for key, value in receipt["state"]["seen_source_times"].items()}
                source_index = receipt["state"]["source_turn_by_id"]
                cid = receipt["state"]["conversation_ids"][slug]
                for probe in group:
                    now = datetime.fromisoformat(probe["question_time"])
                    with SessionLocal() as db, seed_v3.probe_observation(db), applied(settings, coordinate):
                        pre = pp.build(db, probe["question"], cid, classifier, embedder,
                            source_time=now, prompt_cap=coordinate.get("input_prompt_cap"))
                        originals = {gold["turn"]: set(gold["source_ids"]) for gold in probe["gold_sources"]}
                        controls, full = {}, None
                        for arm in seed_v3.PROMPT_ARMS:
                            current = copy.deepcopy(pre)
                            memory, generated, ranked, budgeted, notes = seed_v3.prepare_with_trace(
                                db, current, classifier, arm=arm, source_time=now)
                            stage = seed_v3.stage_record(current, memory, generated, ranked, budgeted, notes, include_prompt=True)
                            stage["lineage"] = seed_v3.lineage_at_cutoff(stage, times, now)
                            if stage["lineage"]["future_source_ids"]: raise ValueError("future evidence in coordinate input")
                            stage["gold_fragment_coverage"] = seed_v3.gold_fragment_coverage(stage, originals, source_index)
                            if arm == "full": full = stage
                            else: controls[arm] = stage
                        current_probe = {**probe, "preflight": full, "controls": controls}
                    unit = dict(probe_id=probe["probe_id"], identity_sha256=header["identity_sha256"],
                        source_probe_sha256=digest(probe), probe=current_probe, answer_quality="unmeasured")
                    unit["unit_sha256"] = digest(unit)
                    write_or_verify(units / (digest(probe["probe_id"]) + ".json"), unit)
                    done = len(list(units.glob("*.json")))
                    print(f"v3 coordinate {coordinate['id']} saved {done}/{len(probes)} local probes", flush=True)
    records = [validate_unit(out / "units" / (digest(identifier) + ".json"), header, probes)
               for identifier in sorted(probes)]
    result = dict(**header, records=records)
    result["complete"] = True
    write_or_verify(out / "contexts.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", required=True, type=Path)
    parser.add_argument("--tuning-plan", required=True, type=Path)
    parser.add_argument("--coordinate", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reader-adoption", type=Path,
                        help="Exact approved current-reader profile over frozen writer state; no replay.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--execute-owned", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root, plan, out = (p.resolve() for p in (args.development_run, args.tuning_plan, args.out))
    profile = args.reader_adoption.resolve() if args.reader_adoption is not None else None
    if (any(not p.is_relative_to((ROOT / "logs").resolve()) for p in (root, plan, out))
            or (profile is not None and not profile.is_relative_to((ROOT / "logs").resolve()))
            or out.is_relative_to(root) or out.is_relative_to(plan)):
        raise ValueError("private source/planner and collector output bundles must be separate")
    with (root / ".operator.lock").open("a") as lock:
        if not args.execute_owned: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, header, coordinate, probes = setup(root, plan, args.coordinate, out,
            owned=args.execute_owned, reader_adoption=profile)
        if not args.run and not args.execute_owned:
            print(json.dumps(dict(version="v3", coordinate=coordinate["id"],
                pending=sum(map(len, pending_groups(out, probes).values())), probes=len(probes), cloud_calls=False)))
            return 0
        if args.execute_owned:
            from src.workers.bg_client_factory import release_owned_models
            try: collect(root, out, header, coordinate, probes, config)
            finally: release_owned_models()
            return 0
        from scripts.z1.run_v3_campaign import database_environment
        env = database_environment(config, root)
        command = [sys.executable, str(Path(__file__).resolve()), "--development-run", str(root),
            "--tuning-plan", str(plan), "--coordinate", args.coordinate, "--out", str(out), "--execute-owned"]
        if profile is not None:
            command += ["--reader-adoption", str(profile)]
        log = out / "collector.log"
        print(f"v3 local coordinate collection; details: {log}", flush=True)
        with log.open("a") as sink:
            child = subprocess.run(command, cwd=ROOT, env=env, stdout=sink, stderr=subprocess.STDOUT, check=False)
        print("v3 coordinate complete" if child.returncode == 0 else
              f"v3 coordinate paused; saved units retained. Inspect {log} and rerun identical arguments.")
        return child.returncode


if __name__ == "__main__":
    try: raise SystemExit(main())
    except KeyboardInterrupt:
        print("v3 coordinate interrupted; saved units retained; rerun identical arguments.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v3 coordinate refused/paused: {exc}. Existing evidence retained.", file=sys.stderr)
        raise SystemExit(2)
