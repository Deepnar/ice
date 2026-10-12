"""Bind one approved reader repair to frozen v3 writer state; never replay it."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify

POLICY = "ice-v3-frozen-writer-current-reader-1"
BEFORE_REF = "80730ae"
TRANSITIONS = {
    "src/memory/summary_support.py": {
        "before": "be07047be33e3a2c6e0461264229104e51224ffd028c3a2dee2bb7ecd6ae1016",
        "after": "d391701b127e3e53a4978f983d3f7784e41e071daecbbbdfa95fa4e29a51405c",
        "reason": "Approved exact cited-span union and exact supported-v1-proof compatibility.",
    },
    "src/memory/assertion_support.py": {
        "before": "91feaf48c4672cd40de83d1841132f4a3c2466c8b5414d5edc6f09fe5d03d5b1",
        "after": "5521b2d8cae84ad72923b6c128bfa05fb40b74f1927095331d73814d7d856fe8",
        "reason": "Approved exact graph-argument consistency; frozen graph is not rewritten.",
    },
    "src/workers/codex_extractor.py": {
        "before": "ae4b57f6cbb31d76d889fe63349cb5afecbd7cef8f13f7fb742e13bb13ac5b3d",
        "after": "b21e8b287596d01d5f3526674ad3ba9f89ed2909f2436a981baaa3683bea08df",
        "reason": "Approved explicit graph tuple and native-envelope comment; no frozen writer replay.",
    },
    "src/workers/nuextract_transport.py": {
        "before": "4a0338a9719dc542838383c820a7732d36be429d6804310e95fc51acbd94139d",
        "after": "a11099a6a2b80029a8eb952e5f1933d7f536c2b0333b029ba1cf66d0e17addf8",
        "reason": "Approved matching nullable facts grammar; frozen extraction outcomes are retained.",
    },
    "src/workers/turn_density.py": {
        "before": "4bf4314d21994907393de946dd43c42d20324263044843df1ecb2ede074d6ecc",
        "after": "4aa0d135361bbd11878d9b7eb50af00859ca1335984522ae8770fab5fe82c879",
        "reason": "Approved complete-first summary preservation after incomplete optional retry; frozen writer outcomes are retained.",
    },
    "src/retrieval/orchestrator.py": {
        "before": "5027817122bad3a1eb9952132f056daebd3acaf574778da57826ac0a446d5b61",
        "after": "a79a4e93ebae97f4414d2c1cd6fb80fc5ca13de432f186a980ac0735bc0cc7e4",
        "reason": "Authorized tested native artifact IDs, preserved cold claim membership, source-covered graph packing and exact same-source selected excerpt omission; ranking/proofs and frozen writer outcomes remain unchanged.",
    },
    "src/retrieval/evolution.py": {
        "before": "a05d0f0923daf0a545d45ffe7a9412854e86504bec43726624b75942a9fd8051",
        "after": "5b9656eeda70f340d6a5c67cbd01d2f47ce4972bbb698d4a1708102d305d1cf5",
        "reason": "Approved rendered timeline contributors follow existing transition/token caps; source/clock and frozen writer outcomes are retained.",
    },
    "src/memory/conversation_notes.py": {
        "before": "a791b5b28674b2b0fff55501935c6df4eba59878f66efd6f8bd11476eab5a8b8",
        "after": "475d3f593e55f73a7d5abeb766c90ddc7aa05c1df4015f22252d7ef456a4f75d",
        "reason": "Approved exact selected note options; no new content or source policy.",
    },
    "src/api/prompt_assembler.py": {
        "before": "87ad681c5e9cb879334030de0b9de054f4c2ffb2ae404425c0c4231cf019f328",
        "after": "ad2b28e209adcf203d7d6d9274caa7e7f1d4e4f02c7b03bfc6070988ffded920",
        "reason": "Approved actual final recent/slot/bookmark/note block receipts; ordinary messages preserved.",
    },
    "src/api/prompt_budget.py": {
        "before": "e745aef5ce138de38e0b7020114ef4092b1b42ec45ff56b1937dd372ca373b32",
        "after": "30c1a6611937e8d385469caa8231441c07b276603dba6d32d6cc63e8c52a403e",
        "reason": "Approved fresh final observer and sharing of graph quotations only with complete covering originals retained after each native replacement and eviction.",
    },
    "src/api/memory_preparation.py": {
        "before": "9e398224c93a64de475f8299132ae8e6119eac2b8bec14b1d6c5f86cecb8bbee",
        "after": "79b966252ead94e36b8f43968fa94171e03c729860813b42daa6ab81c23781ba",
        "reason": "Approved explicit note/bookmark metadata forwarding through shared preparation.",
    },
    "scripts/z1/production_parity.py": {
        "before": "ed058e799864c1128c705b45d6a1098c26ab144ea922b35b941b6a59d1bdc5f5",
        "after": "8dd9a0e0401046ab4adbdee01fb807544c51bbd4d6954d975fb08d39d8c475ce",
        "reason": "Optional research bookmark receipts reach the approved shared observer without changed messages or writer outcomes.",
    },
    "scripts/z1/seed_v3.py": {
        "before": "053e893b754aceccb1989aac0aefe7a5feafc26875094d9239ff101e08fb5e53",
        "after": "9ec0814f29a82390b48d5e17424ee568e238c93c47216adfa68df125e3e8277f",
        "reason": "Exact research serialization of stored IDs, rendered versions and final block receipts; old captured writer/reader outcomes are retained.",
    },
    "experiments/v3_repair/tuning_plan.py": {
        "before": "c1480d90cfd749ebff1f690c5ce70bb85b7a5438dccccb1b4726b51602a9a070",
        "after": "ee728a131f54940bf82f7f04122ed0806b7d57a1abc460927fff63eca0c9b5e9",
        "reason": "Tested required content/noise/opposing-rejection dependency for the existing relevance floor; coordinates and frozen writer outcomes remain unchanged.",
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def before_bytes(relative):
    return subprocess.check_output(["git", "show", f"{BEFORE_REF}:{relative}"], cwd=ROOT)


def records(identities):
    declared = {}
    for identity in identities:
        seen = set()
        for record in identity["code"]:
            relative = record["path"]
            path = (ROOT / relative).resolve()
            if (not path.is_relative_to(ROOT) or path.relative_to(ROOT).as_posix() != relative
                    or relative in seen or set(record) != {"path", "sha256", "bytes"}):
                raise ValueError("invalid/duplicate producer code path")
            if relative in declared and declared[relative] != record:
                raise ValueError("producer and coordinate seals disagree on code")
            declared[relative] = record
            seen.add(relative)
    return declared


def code_transition(identities):
    """Unknown bytes cannot be approved by an operator-supplied allowlist."""
    from scripts.z1.replay_checkpoint import code_files

    declared = records(identities)
    current, changes, originals = [], {}, {}
    for relative, expected in sorted(declared.items()):
        path = ROOT / relative
        actual = fingerprint(path, root=ROOT)
        original = path.read_bytes()
        if actual != expected:
            approved = TRANSITIONS.get(relative)
            if (not approved or expected["sha256"] != approved["before"]
                    or actual["sha256"] != approved["after"]):
                raise ValueError("unregistered reader/producer code change: " + relative)
            original = before_bytes(relative)
            if (hashlib.sha256(original).hexdigest() != expected["sha256"]
                    or len(original) != expected["bytes"]):
                raise ValueError("original producer bytes do not match its seal")
            changes[relative] = dict(before=expected, after=actual, reason=approved["reason"])
        originals[relative] = original
        current.append(actual)
    if set(changes) != set(TRANSITIONS):
        raise ValueError("adoption requires the exact registered transition")
    root, native = code_files()
    if root != ROOT:
        raise ValueError("native code root differs")
    old_digest = hashlib.sha256()
    for path in native:
        relative = path.relative_to(ROOT).as_posix()
        if relative not in originals:
            raise ValueError("native reader path was not bound by producer: " + relative)
        old_digest.update(relative.encode())
        old_digest.update(originals[relative])
    for relative in ("experiments/v3_repair/reader_adoption.py",
                     "experiments/v3_repair/run_tuning_contexts.py",
                     "experiments/v3_repair/compare_tuning_contexts.py",
                     "experiments/v3_repair/audit_development.py",
                     "scripts/z1/metric_contract.py"):
        actual = fingerprint(ROOT / relative, root=ROOT)
        if not any(record["path"] == relative for record in current):
            current.append(actual)
    return sorted(current, key=lambda r: r["path"]), changes, old_digest.hexdigest()


def bundle_bindings(root, config, trace, native_digest):
    """Bind every original prefix before any model construction or restore."""
    from scripts.z1 import snapshot
    from scripts.z1.replay_checkpoint import prefix_digest

    raw = trace.read_bytes()
    meta = json.loads(raw.splitlines()[0])["meta"]
    footer = json.loads(raw.splitlines()[-1])
    if footer.get("event") != "complete":
        raise ValueError("adoption requires completed producer state")
    recovery = root / "seed.recovery"
    name = json.loads((recovery / "current.json").read_text())["generation"]
    if not name.startswith("generation-") or Path(name).name != name:
        raise ValueError("invalid final recovery pointer")
    final = recovery / name
    final_receipt = json.loads((final / "recovery.json").read_text())
    bound = final_receipt["identity"]
    if (bound.get("selected_replay_plan") != config["identity"]
            or bound.get("database") != config["database"]
            or bound.get("code_sha256") != native_digest
            or bound.get("tables") != snapshot.TABLES
            or bound.get("inputs") != meta["inputs"]
            or bound.get("models") != meta["extra"]["resolved_background_model"]
            or bound.get("writer_model_digests") != meta["extra"].get("writer_model_digests")
            or final_receipt["state"]["completed"] != config["identity"]["expected_turns"]):
        raise ValueError("frozen writer identity differs from original recovery")
    paths = [root / "development-run.json", trace, root / "schema.sql", root / "schema.json",
             recovery / "current.json"]
    evaluations = sorted(p for p in (recovery / "evaluation").iterdir() if p.is_dir())
    required = {f"{row['conversation']}-{row['split_turn']}" for row in
                (json.loads(line) for line in raw.splitlines()) if row.get("event") == "as_of_probe"}
    if not required <= {p.name for p in evaluations}:
        raise ValueError("missing producer evaluation checkpoint")
    for directory in [final, *evaluations]:
        receipt = json.loads((directory / "recovery.json").read_text())
        size = receipt["trace_bytes"]
        manifest = json.loads((directory / "store.manifest.json").read_text())
        if (receipt["identity"] != bound or receipt["trace_path"] != str(trace.resolve())
                or type(size) is not int or not 0 < size <= len(raw)
                or prefix_digest(trace, size) != receipt["trace_sha256"]
                or manifest["database"] != config["database"] or manifest["tables"] != snapshot.TABLES
                or sha(directory / "store.sql") != manifest["dump_sha256"]):
            raise ValueError("frozen prefix/dump/trace identity differs")
        if directory == final:
            if (manifest["counts"] != footer["table_counts"]
                    or manifest["sha256_by_table"] != footer["table_sha256"]):
                raise ValueError("final checkpoint differs from complete table identities")
        elif (receipt.get("evaluation_key") != directory.name
              or directory.name != next((f"{slug}-{cutoff}" for slug, cutoff in
                  receipt["state"]["completed"].items() if f"{slug}-{cutoff}" == directory.name), None)):
            raise ValueError("evaluation source cutoff differs")
        paths.extend(directory / filename for filename in ("recovery.json", "store.manifest.json", "store.sql"))
    return {path.relative_to(root).as_posix(): sha(path) for path in paths}


def build(root, config, seal):
    if (digest(config["identity"]) != config["identity_sha256"]
            or digest(seal["identity"]) != seal["identity_sha256"]
            or config["identity"].get("partition") != "tuning"
            or config["identity"]["preparation"] != seal["identity"]["preparation"]
            or config["identity"]["settings_sha256"] != seal["identity"]["settings_sha256"]):
        raise ValueError("reader profile requires the same sealed tuning producer")
    current, changes, native_digest = code_transition((config["identity"], seal["identity"]))
    bindings = bundle_bindings(root, config, root / "seed.jsonl", native_digest)
    identity = dict(policy=POLICY, producer=config["identity_sha256"],
        coordinate_seal=seal["identity_sha256"], coordinate_identity=seal["identity"],
        producer_directory=str(root.resolve()),
        settings_sha256=config["identity"]["settings_sha256"], source_bindings=bindings,
        producer_native_code_sha256=native_digest, reader_code=current, approved_changes=changes)
    return dict(version="v3", identity=identity, identity_sha256=digest(identity),
        writer_replayed=False, writer_result="frozen baseline, not new proof-writer outcomes",
        generated_note_policy="current reader may withhold old generated compositions",
        score_of_record=False, execution_ready=False, promotion_ready=False)


def validate(profile, root, config, seal):
    if profile != build(root, config, seal):
        raise ValueError("reader adoption profile/source/current bytes changed")
    return profile


def validate_runtime(profile, root, config):
    """Revalidate bound bytes before prefix restore, even with complete cache."""
    from experiments.v3_repair.run_tuning_contexts import resolved_baseline

    resolved_baseline(config, owned=True)
    identity = profile["identity"]
    seal = dict(identity=identity["coordinate_identity"], identity_sha256=identity["coordinate_seal"])
    validate(profile, root, config, seal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-run", type=Path, required=True)
    parser.add_argument("--tuning-plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, plan, out = (p.resolve() for p in (args.development_run, args.tuning_plan, args.out))
    if (any(not p.is_relative_to((ROOT / "logs").resolve()) for p in (root, plan, out))
            or out.is_relative_to(root) or out.is_relative_to(plan)):
        raise ValueError("reader profile must be separate from private frozen inputs")
    from experiments.v3_repair.run_tuning_contexts import load_coordinate, resolved_baseline, source_probes
    with (root / ".operator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = json.loads((root / "development-run.json").read_text())
        resolved_baseline(config)
        seal, _ = load_coordinate(plan, "baseline")
        source_probes(root)
        profile = build(root, config, seal)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_or_verify(out, profile)
    print(json.dumps(dict(version="v3", reader_adoption=profile["identity_sha256"],
        changed_paths=sorted(profile["identity"]["approved_changes"]),
        writer_replayed=False, model_calls=0, execution_ready=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
