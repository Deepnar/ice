#!/usr/bin/env python3
"""Restore false success-fault fields only against exact saved judge input hashes.

No cloud calls, regrading, identity edits or seed changes. Default is read-only;
--apply archives the original artifacts before restoring proven original bytes.
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.z1 import judge_answers
from scripts.z1.answer_resume import registered_runner_transition
from scripts.z1.replay_checkpoint import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def restored_input(raw, expected):
    if digest(raw) == expected:
        return raw, 0
    data = json.loads(raw)
    changed = 0
    for row in data["records"]:
        if (row.get("answer") and row.get("error") is None
                and row.get("error_type") == "UnconfirmedCloudCall"
                and row.get("cloud_attempts") == 2
                and row.get("fault_disposition") == "degraded_final"):
            row.pop("fault_disposition")
            changed += 1
    restored = json.dumps(data, indent=2, default=str).encode()
    if not changed or digest(restored) != expected:
        raise ValueError("false-success-only repair does not reproduce the judge's original input hash")
    return restored, changed


def repair(root, *, apply=False):
    root = root.resolve()
    if not root.is_relative_to(ROOT / "logs"):
        raise ValueError("private input recovery must stay under logs/")
    config = json.loads((root / "campaign.json").read_text())
    profile = config.get("judge_profile")
    if not profile:
        raise ValueError("input repair requires the saved explicit judge profile")
    paths = {"a": root / "answers-full.json", "b": root / "answers-no_codex.json"}
    judge_path = root / "judge-full-vs-no_codex.partial.json"
    with ExitStack() as locks:
        # Partial and final judge files share the final output's lock.
        lock_paths = [root / "campaign.lock", *(p.with_suffix(".lock") for p in paths.values()),
                      root / "judge-full-vs-no_codex.lock"]
        for lock in lock_paths:
            handle = locks.enter_context(lock.open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("an answer/judge operator is active; input repair refused") from None
        judge_bytes = judge_path.read_bytes()
        saved = json.loads(judge_bytes)
        prior_profile = os.environ.get("ICE_JUDGE_PROFILE")
        os.environ["ICE_JUDGE_PROFILE"] = profile
        try:
            identity = judge_answers.judge_provider_identity()
        finally:
            if prior_profile is None:
                os.environ.pop("ICE_JUDGE_PROFILE", None)
            else:
                os.environ["ICE_JUDGE_PROFILE"] = prior_profile
        identity.update(judge_implementation_sha256=digest((ROOT / "scripts/z1/judge_answers.py").read_bytes()),
                        judge_rubric_sha256=digest(judge_answers.SYSTEM_V3.encode()),
                        cloud_recovery_sha256=digest((ROOT / "scripts/z1/cloud_recovery.py").read_bytes()))
        if any(saved.get(k) != v for k, v in identity.items()):
            raise ValueError("judge code/rubric/provider identity changed; input repair refused")
        originals, restored, changes = {}, {}, {}
        runner = digest((ROOT / "scripts/z1/answer_as_of.py").read_bytes())
        for key, path in paths.items():
            originals[key] = path.read_bytes()
            data = json.loads(originals[key])
            if (data.get("version") != "v3" or not data.get("processing_complete")
                    or not registered_runner_transition(data["implementation_sha256"]["answer_runner"], runner)):
                raise ValueError("answer artifact is not from the registered completed runner")
            restored[key], changes[key] = restored_input(originals[key], saved["answer_file_sha256"][key])
        receipt = {"format": "ice-v3-false-success-input-recovery-1",
                   "judge_state_sha256": digest(judge_bytes), "judge_identity": identity,
                   "inputs": {key: {"file": path.name, "before_sha256": digest(originals[key]),
                                    "restored_sha256": digest(restored[key]), "false_fields_removed": changes[key]}
                              for key, path in paths.items()}, "applied": False}
        if not apply or not any(changes.values()):
            return receipt
        archive = Path(tempfile.mkdtemp(prefix="answer-input-recovery-", dir=root))
        for name, raw in [(judge_path.name, judge_bytes),
                          *((paths[key].name, raw) for key, raw in originals.items())]:
            with (archive / name).open("xb") as sink:
                sink.write(raw)
                sink.flush()
                os.fsync(sink.fileno())
        receipt["created_at"] = datetime.now(timezone.utc).isoformat()
        # This fsync also makes the archive directory entries durable.
        atomic_json(archive / "receipt.json", receipt)
        fd = os.open(root, os.O_DIRECTORY)
        try:
            os.fsync(fd)  # Publish the archive's parent entry before restoring inputs.
        finally:
            os.close(fd)
        for key, path in paths.items():
            if changes[key]:
                atomic_json(path, json.loads(restored[key]))
            if digest(path.read_bytes()) != saved["answer_file_sha256"][key]:
                raise RuntimeError("restored answer bytes differ from the original judge input")
        if judge_path.read_bytes() != judge_bytes:
            raise RuntimeError("judge state changed during input recovery")
        receipt.update(applied=True, archive=str(archive))
        atomic_json(archive / "receipt.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    receipt = repair(args.run_dir, apply=args.apply)
    print(json.dumps({"verified": True, "applied": receipt["applied"],
                      "inputs": receipt["inputs"], "archive": receipt.get("archive")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
