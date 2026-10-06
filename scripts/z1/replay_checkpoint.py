"""Durable, isolated replay recovery: store snapshot plus committed trace prefix."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts.z1 import snapshot


def background_model_digests(settings, background_model: str) -> dict[str, str]:
    """Ollama tags are mutable names; pin the local writer manifests too."""
    response = httpx.get(settings.ollama_base_url.rstrip("/") + "/api/tags", timeout=10)
    response.raise_for_status()
    tags = {row["name"]: row.get("digest") for row in response.json()["models"]}
    wanted = {background_model, settings.codex_extraction_model}
    result = {}
    for name in sorted(wanted):
        lookup = name if ":" in name else name + ":latest"
        digest = tags.get(lookup)
        if not isinstance(digest, str) or not digest:
            raise ValueError("required local writer model has no installed Ollama digest: " + name)
        result[name] = digest
    return result


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as sink:
        json.dump(value, sink, indent=2, default=str)
        sink.flush()
        os.fsync(sink.fileno())
    temp.replace(path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prefix_digest(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        left = size
        while left:
            chunk = source.read(min(left, 1 << 20))
            if not chunk:
                raise ValueError("trace is shorter than its recovery checkpoint")
            digest.update(chunk)
            left -= len(chunk)
    return digest.hexdigest()


INSTRUMENT_CHANGES = {"scripts/z1/worker_recovery.py", "scripts/z1/replay_checkpoint.py",
                      "scripts/z1/seed_v3.py"}
CONTINUATION_FORMAT = "ice-v3-instrument-continuation-1"
REPAIR_REGISTRY = Path(__file__).with_name("instrument_repairs.json")


def code_files():
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "src").rglob("*.py"))
    files += [root / "scripts/z1" / name for name in
              ("seed_v3.py", "production_parity.py", "historical_clock.py",
               "replay_checkpoint.py", "snapshot.py", "run_meta.py",
               "derive_retrieval_gt.py", "generate_probes.py", "development_repeats.py",
               "campaign_recovery.py", "worker_recovery.py")]
    return root, files


def code_digest(ref=None):
    root, files = code_files()
    digest = hashlib.sha256()
    for path in files:
        relative = str(path.relative_to(root))
        content = (subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=root)
                   if ref else path.read_bytes())
        digest.update(relative.encode())
        digest.update(content)
    return digest.hexdigest()


def registered_repair(old_code, new_code, changes):
    """The path allowlist alone never authorizes new tool bytes."""
    registered = json.loads(REPAIR_REGISTRY.read_text())
    if registered.get("format") == "ice-v3-reviewed-instrument-repairs-1":
        for record in registered.get("repairs", []):
            if (record.get("from_code") == old_code and record.get("to_code") == new_code
                    and record.get("code_changes") == changes
                    and changes and set(changes) <= INSTRUMENT_CHANGES):
                return record
    raise ValueError("code changes are not a registered reviewed instrument repair")


def instrument_changes(ref):
    """Refuse writer edits and require one exact registered tool transition."""
    root, files = code_files()
    changes = {}
    for path in files:
        relative = str(path.relative_to(root))
        before = subprocess.check_output(["git", "show", f"{ref}:{relative}"], cwd=root)
        after = path.read_bytes()
        if before != after:
            if relative not in INSTRUMENT_CHANGES:
                raise ValueError("continuation would change a writer or unapproved tool: " + relative)
            changes[relative] = {"before": hashlib.sha256(before).hexdigest(),
                                 "after": hashlib.sha256(after).hexdigest()}
    registered_repair(code_digest(ref), code_digest(), changes)
    return changes


def run_identity(meta: dict, args, settings) -> dict:
    """Pin execution bytes and settings without serializing credentials."""
    configuration = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    arguments = {k: v for k, v in vars(args).items()
                 if k not in {"resume", "check", "out", "checkpoint_dir"}}
    return {"format": "ice-v3-replay-recovery-1", "database": snapshot.DB,
            "code_sha256": code_digest(), "arguments": arguments,
            "settings_sha256": hashlib.sha256(configuration.encode()).hexdigest(),
            "inputs": meta["inputs"], "models": meta["extra"]["resolved_background_model"],
            "writer_model_digests": meta["extra"].get("writer_model_digests"),
            "plan": meta["extra"]["planned_probes"], "tables": snapshot.TABLES}


class Checkpoints:
    def __init__(self, root: Path, trace: Path, identity: dict):
        self.root, self.trace, self.identity = root, trace, identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "run.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise RuntimeError("this replay already has an active operator") from None
        self.pointer = root / "current.json"

    def close(self):
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()

    def publish_evaluation(self, directory: Path, key: str) -> None:
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in key):
            raise ValueError("invalid evaluation checkpoint key")
        root = self.root / "evaluation"
        root.mkdir(exist_ok=True)
        target = root / key
        if target.exists():
            if (target / "recovery.json").read_bytes() != (directory / "recovery.json").read_bytes():
                raise ValueError("evaluation checkpoint already has different trace/state")
            return
        temp = root / (".partial-" + uuid.uuid4().hex)
        temp.mkdir()
        for path in directory.iterdir():
            os.link(path, temp / path.name)
        temp.rename(target)
        fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def capture(self, sink, state: dict, *, evaluation_key=None) -> None:
        sink.flush()
        os.fsync(sink.fileno())
        size = self.trace.stat().st_size
        generation = "generation-" + uuid.uuid4().hex
        directory = self.root / generation
        directory.mkdir()
        with patch.object(snapshot, "SNAPDIR", directory):
            if snapshot.save("store"):
                raise RuntimeError("replay checkpoint snapshot failed")
        for path in directory.iterdir():
            with path.open("rb") as source:
                os.fsync(source.fileno())
        manifest = {"identity": self.identity, "trace_path": str(self.trace.resolve()),
                    "trace_bytes": size, "trace_sha256": prefix_digest(self.trace, size),
                    "generation": generation, "state": state,
                    "evaluation_key": evaluation_key}
        atomic_json(directory / "recovery.json", manifest)
        previous = json.loads(self.pointer.read_text())["generation"] if self.pointer.exists() else None
        atomic_json(self.pointer, {"generation": generation, "previous": previous})
        if evaluation_key:
            self.publish_evaluation(directory, evaluation_key)
        # Only generations owned by this recovery root; retain current + prior.
        for path in self.root.glob("generation-*"):
            if path.is_dir() and path.name not in {generation, previous}:
                shutil.rmtree(path)

    def validate(self) -> tuple[Path, dict]:
        if not self.pointer.exists():
            raise ValueError("no durable replay checkpoint; do not append to a partial trace")
        pointer = json.loads(self.pointer.read_text())
        name = pointer["generation"]
        if not name.startswith("generation-") or Path(name).name != name:
            raise ValueError("invalid recovery generation")
        directory = self.root / name
        manifest = json.loads((directory / "recovery.json").read_text())
        if manifest["trace_path"] != str(self.trace.resolve()):
            raise ValueError("replay inputs, code, settings, database or plan changed; refusing resume")
        if manifest["identity"] != self.identity:
            self.continuation(manifest)
        size = manifest["trace_bytes"]
        if prefix_digest(self.trace, size) != manifest["trace_sha256"]:
            raise ValueError("committed replay trace prefix changed; refusing resume")
        return directory, manifest

    def receipt_path(self, old, new):
        # Bind a receipt to both full identities; never overwrite the first
        # repair's receipt when preparing a later frozen prefix.
        key = hashlib.sha256(json.dumps([old, new], sort_keys=True).encode()).hexdigest()
        path = self.root / ("instrument-continuation-" + key + ".json")
        legacy = self.root / "instrument-continuation.json"
        if not path.exists() and legacy.exists():
            receipt = json.loads(legacy.read_text())
            if receipt.get("from_identity") == old and receipt.get("to_identity") == new:
                return legacy
        return path

    def continuation(self, manifest):
        path = self.receipt_path(manifest["identity"], self.identity)
        if not path.exists():
            raise ValueError("replay code changed without a verified instrument continuation; refusing resume")
        receipt = json.loads(path.read_text())
        old, new = receipt.get("from_identity"), receipt.get("to_identity")
        archive = receipt.get("archive", "")
        if (receipt.get("format") != CONTINUATION_FORMAT or old != manifest["identity"]
                or new != self.identity or not archive or Path(archive).name != archive
                or {k: v for k, v in old.items() if k != "code_sha256"}
                   != {k: v for k, v in new.items() if k != "code_sha256"}
                or receipt.get("trace_bytes") != manifest["trace_bytes"]
                or receipt.get("trace_sha256") != manifest["trace_sha256"]
                or receipt.get("completed") != manifest["state"]["completed"]):
            raise ValueError("instrument continuation does not match frozen checkpoint identity")
        baseline = receipt["baseline_commit"]
        if (code_digest(baseline) != old["code_sha256"] or code_digest() != new["code_sha256"]
                or instrument_changes(baseline) != receipt.get("code_changes")):
            raise ValueError("instrument continuation code is not the reviewed tool-only repair")
        saved = self.root / archive
        saved_manifest = (saved / "recovery.json").read_bytes()
        if (json.loads(saved_manifest) != manifest
                or hashlib.sha256(saved_manifest).hexdigest() != receipt.get("archive_manifest_sha256")
                or prefix_digest(saved / "trace-prefix.jsonl", manifest["trace_bytes"]) != manifest["trace_sha256"]):
            raise ValueError("instrument continuation original archive changed")
        return receipt

    def prepare_instrument_continuation(self, baseline):
        """Archive and authorize only the explicit tool-only code boundary.

        Never restore/change the database, trace, pointer or old manifests.
        The maintainer still runs the campaign; its normal restore verifies SQL.
        """
        directory, manifest = self.validate()
        old = manifest["identity"]
        changes = instrument_changes(baseline)
        if code_digest(baseline) != old["code_sha256"]:
            raise ValueError("Git baseline does not reproduce frozen checkpoint code")
        new = {**old, "code_sha256": code_digest()}
        receipt_path = self.receipt_path(old, new)
        if receipt_path.exists():
            previous_identity = self.identity
            self.identity = new
            try:
                return self.continuation(manifest)
            finally:
                self.identity = previous_identity
        archive = self.root / ("instrument-prefix-" + uuid.uuid4().hex)
        archive.mkdir()
        for path in directory.iterdir():
            os.link(path, archive / path.name)
        # The live trace will be truncated/appended. A hard link is unsafe here.
        with self.trace.open("rb") as source, (archive / "trace-prefix.jsonl").open("xb") as sink:
            left = manifest["trace_bytes"]
            while left:
                chunk = source.read(min(left, 1 << 20))
                if not chunk:
                    raise ValueError("committed trace truncated during continuation preparation")
                sink.write(chunk)
                left -= len(chunk)
            sink.flush()
            os.fsync(sink.fileno())
        if prefix_digest(archive / "trace-prefix.jsonl", manifest["trace_bytes"]) != manifest["trace_sha256"]:
            raise ValueError("continuation backup does not match original committed prefix")
        receipt = {"format": CONTINUATION_FORMAT, "baseline_commit": baseline,
                   "reason": registered_repair(old["code_sha256"], new["code_sha256"], changes)["reason"],
                   "from_identity": old, "to_identity": new, "code_changes": changes,
                   "trace_bytes": manifest["trace_bytes"], "trace_sha256": manifest["trace_sha256"],
                   "completed": manifest["state"]["completed"], "archive": archive.name,
                   "archive_manifest_sha256": hashlib.sha256((archive / "recovery.json").read_bytes()).hexdigest()}
        fd = os.open(archive, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        atomic_json(receipt_path, receipt)
        return receipt

    def recover(self) -> dict:
        directory, manifest = self.validate()
        continuation = self.continuation(manifest) if manifest["identity"] != self.identity else None
        size = manifest["trace_bytes"]
        with patch.object(snapshot, "SNAPDIR", directory):
            if snapshot.restore("store"):
                raise RuntimeError("recovery store could not be restored")
        if manifest.get("evaluation_key"):
            self.publish_evaluation(directory, manifest["evaluation_key"])
        # Preserve failed/in-flight observations separately; never score them.
        with self.trace.open("rb+") as trace:
            trace.seek(size)
            tail = trace.read()
            if tail:
                archive = self.root / ("unfinished-" + uuid.uuid4().hex + ".jsonl")
                with archive.open("xb") as sink:
                    sink.write(tail)
                    sink.flush()
                    os.fsync(sink.fileno())
            trace.truncate(size)
            trace.flush()
            os.fsync(trace.fileno())
        if continuation:
            with self.trace.open("a") as sink:
                sink.write(json.dumps({"event": "instrument_continuation", **continuation}) + "\n")
                sink.flush()
                os.fsync(sink.fileno())
        return manifest["state"]

    def complete_unchanged(self, expected_turns: dict, table_counts: dict) -> bool:
        """Finished resume must preserve bytes referenced by answer files."""
        self.validate()
        from scripts.z1.replay_validation import validate_complete_replay
        try:
            with self.trace.open() as source:
                last = validate_complete_replay((json.loads(line) for line in source),
                                                expected_turns)["complete"]
        except (ValueError, KeyError, TypeError):
            return False
        if table_counts != last["table_counts"] or snapshot.fingerprints() != last["table_sha256"]:
            raise ValueError("complete seed store changed; restore its final snapshot before continuing")
        return True
