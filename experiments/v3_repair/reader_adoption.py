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
# ⚠ CODEX_V4 (2026-10-16) deleted assertion_support.py, codex_extractor.py and
# nuextract_transport.py, so their transitions are retired: a frozen bundle that
# bound them now refuses at fingerprinting, which is correct — a pre-CODEX_V4
# writer's codex (edges/claims) is not readable by the facts reader. Adopt such
# a bundle at commit 8079eb2; Z2 builds fresh stores.
TRANSITIONS = {
    "src/memory/summary_support.py": {
        "before": "be07047be33e3a2c6e0461264229104e51224ffd028c3a2dee2bb7ecd6ae1016",
        "after": "5d68720a307e5bc9be95b5f5c57db94458c92940b2ef6a16e3fc9c3295c57e1e",
        "reason": "Approved exact cited-span union and exact supported-v1-proof compatibility. 2026-10-15 (SUMMARY_CLAIMS_V3): v3 quoted-claim method added behind summary_assertion_method; v2/v1 records verify under their own policy exactly as before; pieces and input-scaled caps are producer-side. 2026-10-16 (CODEX_V4): v4 claim method (modality + subject/relation/object, opening-context header, role-present speaker enum) behind summary_assertion_method, now the default; v2/v3 records still verify under their own policy.",
    },
    "src/workers/turn_density.py": {
        "before": "4bf4314d21994907393de946dd43c42d20324263044843df1ecb2ede074d6ecc",
        "after": "4aa0d135361bbd11878d9b7eb50af00859ca1335984522ae8770fab5fe82c879",
        "reason": "Approved complete-first summary preservation after incomplete optional retry; frozen writer outcomes are retained.",
    },
    "src/retrieval/orchestrator.py": {
        "before": "5027817122bad3a1eb9952132f056daebd3acaf574778da57826ac0a446d5b61",
        "after": "b76ca719fac794a6085c5eb0a545b5e7bda2ac706644573cab713b1de5436d3c",
        "reason": "Authorized tested native artifact IDs, preserved cold claim membership, source-covered graph packing and exact same-source selected excerpt omission; ranking/proofs and frozen writer outcomes remain unchanged. 2026-10-15 (audit A12): automatic top-k cluster scoping gated behind retrieval_auto_cluster_scope (default off); explicit cluster scope unchanged. 2026-10-16 (CODEX_V4): the facts leg replaced the graph, claims, enumeration and timeline legs; procedural scope resolver kept as _scope_batch_ids; inert claim/edge fragment fields removed. A pre-CODEX_V4 bundle's codex cannot be read by this reader. 2026-10-17: the wide-net path applies the D7 strength bonus after reranking, as the main path does.",
    },
    "src/retrieval/evolution.py": {
        "before": "a05d0f0923daf0a545d45ffe7a9412854e86504bec43726624b75942a9fd8051",
        "after": "7d2bcfa2c6e4a1fc3766f0e48dcb40426152cd63d13dcdf9e2743818eae117e3",
        "reason": "Approved rendered timeline contributors follow existing transition/token caps; source/clock and frozen writer outcomes are retained. 2026-10-16 (CODEX_V4): rewritten over facts (entity_diff + log_description_update only); the edge timeline is gone.",
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
        "after": "0ebe9573ea8ca689012154e2fb7ce429251a879dd62d8f72049a219b354fa029",
        "reason": "Approved fresh final observer and sharing of graph quotations only with complete covering originals retained after each native replacement and eviction. 2026-10-16 (CODEX_V4): graph-quote sharing removed; a fact is skipped when every source turn is already in the prompt.",
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
        "after": "408ee25028b9f2b19e000ff37b4136a13d60bdf3220f32dc5a8b2f3049ac5daf",
        "reason": "Exact research serialization of stored IDs, rendered versions and final block receipts; old captured writer/reader outcomes are retained. 2026-10-16 (CODEX_V4): fact tables, facts leg in the arms, fact metrics; legacy codex settings, tables and receipts removed.",
    },
    "experiments/v3_repair/tuning_plan.py": {
        "before": "c1480d90cfd749ebff1f690c5ce70bb85b7a5438dccccb1b4726b51602a9a070",
        "after": "52ba9734183961976bd500357ff714aedc767709c2e4783a1154cd652d48e1c6",
        "reason": "Tested required content/noise/opposing-rejection dependency for the existing relevance floor; coordinates and frozen writer outcomes remain unchanged. 2026-10-16 (CODEX_V4): candidate-limit knob codex_claim_candidate_limit → codex_fact_candidate_limit.",
    },
    "scripts/z1/evidence_content.py": {
        "before": "cb9caed2b5540eb4caaa04d507c00d0b2a0c8d93382b6689161ed7b5fbee9af7",
        "after": "4806b8373ff3a87e06d5ac07dae0849b394e1308bc3eba3f11e6048b36ba1c79",
        "reason": "Qualified research-only native attributed JSON observation; source-role/hash/offset and semantic gates preserved. Old measurements and frozen writer outputs are not rewritten.",
    },
    "scripts/z1/score_frozen_answers.py": {
        "before": "7d7b7025639f210b7c6f75948511058b8566329d9ed053d7d6fb5c8b2aa298a3",
        "after": "0edb0371d9b011bb1f42579600aeacb70b8b69a067dfc82469d6ce674c1b350c",
        "reason": "Qualified optional editorial required-content criteria in diagnostic judge inputs; answering messages, original references and saved answers/grades are unchanged. This reader profile performs no grading.",
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
