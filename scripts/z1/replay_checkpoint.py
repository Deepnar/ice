"""Durable, isolated replay recovery: store snapshot plus committed trace prefix."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
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


def run_identity(meta: dict, args, settings) -> dict:
    """Pin execution bytes and settings without serializing credentials."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    files = sorted((root / "src").rglob("*.py"))
    files += [root / "scripts/z1" / name for name in
              ("seed_v3.py", "production_parity.py", "historical_clock.py",
               "replay_checkpoint.py", "snapshot.py", "run_meta.py",
               "derive_retrieval_gt.py", "generate_probes.py", "development_repeats.py",
               "campaign_recovery.py", "worker_recovery.py")]
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    configuration = json.dumps(settings.model_dump(mode="json"), sort_keys=True, default=str)
    arguments = {k: v for k, v in vars(args).items()
                 if k not in {"resume", "check", "out", "checkpoint_dir"}}
    return {"format": "ice-v3-replay-recovery-1", "database": snapshot.DB,
            "code_sha256": digest.hexdigest(), "arguments": arguments,
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
        if (manifest["identity"] != self.identity
                or manifest["trace_path"] != str(self.trace.resolve())):
            raise ValueError("replay inputs, code, settings, database or plan changed; refusing resume")
        size = manifest["trace_bytes"]
        if prefix_digest(self.trace, size) != manifest["trace_sha256"]:
            raise ValueError("committed replay trace prefix changed; refusing resume")
        return directory, manifest

    def recover(self) -> dict:
        directory, manifest = self.validate()
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
