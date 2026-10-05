#!/usr/bin/env python3
"""Preserve a paused v3 checkpoint for the reviewed transport-guard correction.

No database restore, model call or campaign launch. Every memory writer and
other pinned file must match the run's Git baseline. No ignore-hash option.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.z1.replay_checkpoint import Checkpoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    logs = Path(__file__).resolve().parents[2] / "logs"
    if not root.is_relative_to(logs):
        raise ValueError("private continuation archive must stay under logs/")
    trace = root / "seed.jsonl"
    recovery = root / "seed.recovery"
    pointer = json.loads((recovery / "current.json").read_text())
    generation = pointer["generation"]
    if not generation.startswith("generation-") or Path(generation).name != generation:
        raise ValueError("invalid recovery generation")
    manifest = json.loads((recovery / generation / "recovery.json").read_text())
    with trace.open() as source:
        header = json.loads(source.readline())
    if header.get("event") != "run":
        raise ValueError("trace has no original provenance header")
    baseline = header["meta"]["git"]["commit"]
    point = Checkpoints(recovery, trace, manifest["identity"])
    try:
        receipt = point.prepare_instrument_continuation(baseline)
    finally:
        point.close()
    print("v3 instrument continuation prepared; original checkpoint/store/trace unchanged")
    print("durable turns retained:", sum(receipt["completed"].values()))
    print("Resume the same run-dir campaign command; no experiment launched here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
