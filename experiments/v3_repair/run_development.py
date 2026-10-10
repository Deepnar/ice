"""Plan or manually replay one repaired ICE v3 development partition.

Default is offline. --run uses an owned isolated database and the existing
production-path replay/checkpoint engine. Rerun identical arguments to resume.
"""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import (
    DEVELOPMENT, digest, encoded, fingerprint, prepare, write_or_verify,
)
from scripts.z1.label_review import validate_review
from scripts.z1.metric_contract import metric_contract

POLICY = "ice-v3-repaired-development-replay-1"


def load_selected(inputs, partition):
    """Revalidate the complete offline seal; scorer keys never enter preparation."""
    if partition not in {"tuning", "validation"}:
        raise ValueError("unknown development partition")
    manifest = prepare(inputs)
    groups = json.loads((inputs / "development-originals.json").read_text())
    keys = json.loads((inputs / "development-keys.json").read_text())
    records = [r for r in keys["records"] if r["partition"] == partition]
    selected = {r["review"]["conversation"] for r in records}
    other = {r["review"]["conversation"] for r in keys["records"] if r["partition"] != partition}
    if not records or selected & other or not selected <= DEVELOPMENT.keys():
        raise ValueError("development partition is empty, overlapping or not a whole history")
    conversations, probes = {}, defaultdict(list)
    seen = set()
    for slug in sorted(selected):
        rows = groups[slug]
        if len(rows) != DEVELOPMENT[slug]:
            raise ValueError("development history count differs")
        conversations[slug] = []
        for number, row in enumerate(rows, 1):
            stamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            if (row["turn_number"] != number or stamp.tzinfo is None
                    or not row["prompt"].strip() or not row["response"].strip()):
                raise ValueError("invalid original replay pair")
            conversations[slug].append(dict(turn_number=number, timestamp=stamp,
                ts_provenance=row["timestamp_provenance"], prompt=row["prompt"], response=row["response"]))
    for row in records:
        review = row["review"]
        slug, cutoff, identifier = review["conversation"], review["cutoff_turn"], review["probe_id"]
        validate_review(review, cutoff)
        gold = row["gold_turns"]
        if (identifier in seen or type(cutoff) is not int or not 1 <= cutoff <= len(conversations[slug])
                or not gold or any(type(t) is not int or not 1 <= t <= cutoff for t in gold)
                or len(gold) != len(set(gold))
                or row["metric_contract"] != metric_contract(review, gold)):
            raise ValueError("invalid duplicate/future development source or metric contract")
        seen.add(identifier)
        probes[(slug, cutoff)].append(dict(
            probe_id=identifier, catalog_probe_id=review.get("catalog_probe_id"),
            probe_type=review.get("source", "reviewed_development"), question=review["question"],
            expected_answer=row["reviewed_answer"], catalog_expected_answer=review.get("expected_answer"),
            gold_turns=gold, source_split_turn=max(gold), cutoff_kind="reviewed_development_checkpoint",
            label_status="source_reviewed_valid", development_repeat=review.get("development_repeat"),
            metric_contract=row["metric_contract"], label_review=review))
    return manifest, conversations, probes, records


def planned_run(inputs, root, partition, every, *, owned=False):
    """Freeze before database access; identical plans resume without replacement."""
    if every < 1:
        raise ValueError("checkpoint interval must be positive")
    manifest, conversations, probes, records = load_selected(inputs, partition)
    from src.api.config import settings
    if (settings.background_model_mode != "shared" or not settings.background_model_name
            or not settings.codex_extraction_model or settings.codex_extraction_ner_tier != "background"):
        raise ValueError("development replay requires the declared shared writer/extractor/background-NER profile")
    from scripts.z1.replay_checkpoint import code_files
    from sqlalchemy.engine import make_url
    source_database = make_url(settings.database_url).database
    resolved = settings.model_dump(mode="json")
    resolved["database_url"] = make_url(settings.database_url).render_as_string(hide_password=False)
    if owned:
        original = json.loads((root / "development-run.json").read_text())
        if (source_database != original["database"]
                or os.environ.get("ICE_TEST_DATABASE") != original["database"]):
            raise ValueError("child is not attached to its declared owned database")
        source_database = original["identity"]["source_database"]
        resolved["database_url"] = make_url(settings.database_url).set(
            database=source_database).render_as_string(hide_password=False)
    _, replay_files = code_files()
    code = sorted(set(replay_files + list((ROOT / "scripts/z1").glob("*.py")) + [Path(__file__)]))
    identity = dict(policy=POLICY, version="v3", preparation=manifest["identity_sha256"],
        partition=partition, run_directory=str(root), checkpoint_every=every,
        originals_sha256=manifest["artifacts"]["development-originals.json"],
        keys_sha256=manifest["artifacts"]["development-keys.json"],
        code=[fingerprint(p) for p in code],
        settings_sha256=digest(resolved), source_database=source_database,
        expected_turns={s: len(r) for s, r in conversations.items()},
        probe_ids=[[p["probe_id"], s, n] for (s, n), ps in probes.items() for p in ps])
    identifier = digest(identity)
    config = dict(identity=identity, identity_sha256=identifier, campaign_id=identifier,
        database="ice_v3_campaign_" + identifier[:16], background_model=settings.background_model_name,
        codex_extraction_model=settings.codex_extraction_model, score_of_record=False,
        validation_execution="locked_until_qualified_tuning_freeze", answers_generated=False)
    products = {"development-run.json": config, "reviewed-keys.json": {"records": records}}
    for name, value in products.items():
        if (root / name).exists() and (root / name).read_bytes() != encoded(value) + b"\n":
            raise ValueError("development run identity changed; retain this bundle and use a new directory")
    for name, value in products.items():
        write_or_verify(root / name, value)
    return config, conversations, probes


def replay_args(root, every):
    trace = root / "seed.jsonl"
    pointer = root / "seed.recovery/current.json"
    if trace.exists() != pointer.exists():
        raise ValueError("trace/checkpoint pair is incomplete; preserve it for recovery review")
    return SimpleNamespace(out=str(trace), resume=trace.exists(), checkpoint_every=every,
        checkpoint_dir=str(root / "seed.recovery"), conversation=None, limit=0,
        no_maintenance=False, no_probe_controls=False, probe_panel="reviewed_development",
        worker_failure_policy="continue", development_repeat_review=None)


def execute_child(config, conversations, probes, root, inputs, every):
    from sqlalchemy.engine import make_url
    from src.api.config import settings
    if (make_url(settings.database_url).database != config["database"]
            or os.environ.get("ICE_TEST_DATABASE") != config["database"]):
        raise ValueError("development child lacks its owned isolated database")
    from scripts.z1.seed_v3 import run
    from src.workers.bg_client_factory import release_owned_models
    try:
        return run(replay_args(root, every), conversations, probes,
            expected_turns=config["identity"]["expected_turns"],
            input_files=[inputs / "manifest.json", inputs / "development-originals.json",
                         inputs / "development-keys.json", root / "development-run.json",
                         root / "reviewed-keys.json"], selected_plan=config["identity"])
    finally:
        release_owned_models()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--partition", choices=["tuning", "validation"], default="tuning")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--execute-owned", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    inputs, root = args.inputs.resolve(), args.run_dir.resolve()
    if any(not p.is_relative_to((ROOT / "logs").resolve()) for p in (inputs, root)) or inputs == root:
        raise ValueError("separate input/run bundles must stay under private logs/")
    root.mkdir(parents=True, exist_ok=True)
    # Parent holds the operator lock across the child. The worker uses the
    # existing independent checkpoint lock, never recursively this lock.
    with (root / ".operator.lock").open("a") as lock:
        if not args.execute_owned:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, conversations, probes = planned_run(inputs, root, args.partition,
                                                   args.checkpoint_every, owned=args.execute_owned)
        total, questions = sum(map(len, conversations.values())), sum(map(len, probes.values()))
        if not args.run and not args.execute_owned:
            print(json.dumps(dict(version="v3", partition=args.partition, turns=total,
                reviewed_questions=questions, database=config["database"], models_called=False,
                score_of_record=False, validation_execution=config["validation_execution"])))
            return 0
        if args.partition == "validation":
            raise ValueError("validation replay is reserved until qualified tuning/freeze; plan only")
        if args.execute_owned:
            return execute_child(config, conversations, probes, root, inputs, args.checkpoint_every)
        replay_args(root, args.checkpoint_every)  # Refuse malformed recovery before provisioning.
        from scripts.z1.run_v3_campaign import database_environment
        env = database_environment(config, root)
        from scripts.z1.campaign_progress import ArtifactProgress, TerminalProgress
        command = [sys.executable, str(Path(__file__).resolve()), "--inputs", str(inputs),
            "--run-dir", str(root), "--partition", args.partition, "--checkpoint-every",
            str(args.checkpoint_every), "--execute-owned"]
        log = root / "development.log"
        print(f"v3 repaired development replay: {total} turns/{questions} probes; details: {log}", flush=True)
        try:
            with log.open("a") as sink, TerminalProgress(
                    ArtifactProgress(root, "seed", turns=total, probes=questions), "v3 development"):
                subprocess.run(command, cwd=ROOT, env=env, stdout=sink,
                               stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Replay paused. Inspect {log}; rerun identical arguments to resume saved state.")
            return exc.returncode
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("v3 development interrupted; committed checkpoints retained. Rerun identical arguments.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"v3 development refused/paused: {exc}. Existing bundles retained.", file=sys.stderr)
        raise SystemExit(2)
