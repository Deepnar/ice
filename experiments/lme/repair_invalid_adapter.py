#!/usr/bin/env python3
"""Reversibly retire flattened-session LME artifacts and preserve vector-RAG.

Adapter v1 flattened every LongMemEval history session into one ICE conversation
and passed its UUID object into session diversification. All ``full_ice`` answers
from that adapter are invalid. The direct SQL vector-RAG arm saw the same set of
stored turns and bypassed conversation diversification, so it can be retained.

Dry-run is the default. ``--apply`` copies every original answer into
``invalidated_adapter_v1/answers`` before replacing the active record with a
condition-level resume record containing only vector-RAG. Invalid full-ICE judge
files and old phase metadata/logs are archived alongside it.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from lme_run import ADAPTER_VERSION, DEFAULT_OUT, _atomic_write_json, _read_json


def repair_phase(phase: str, apply: bool) -> dict:
    phase_dir = DEFAULT_OUT / phase
    answers_dir = phase_dir / "answers"
    archive_dir = phase_dir / "invalidated_adapter_v1"
    archive_answers = archive_dir / "answers"
    archive_judgements = archive_dir / "judgements"

    stats = {
        "phase": phase,
        "answers_seen": 0,
        "answers_to_repair": 0,
        "vector_answers_preserved": 0,
        "full_ice_judgements_to_archive": 0,
        "metadata_to_archive": [],
        "apply": apply,
    }

    answer_paths = sorted(answers_dir.glob("*.json")) if answers_dir.exists() else []
    for path in answer_paths:
        stats["answers_seen"] += 1
        record = _read_json(path)
        if not record or record.get("adapter_version") == ADAPTER_VERSION:
            continue
        stats["answers_to_repair"] += 1
        vector_answer = (record.get("answers") or {}).get("vector_rag")
        if vector_answer is not None:
            stats["vector_answers_preserved"] += 1
        if not apply:
            continue

        archive_answers.mkdir(parents=True, exist_ok=True)
        backup_path = archive_answers / path.name
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

        old_conversation_id = record.pop("conversation_id", None)
        record.pop("total_seconds", None)
        record["adapter_version"] = ADAPTER_VERSION
        record["status"] = "adapter_v2_pending"
        record["answers"] = ({"vector_rag": vector_answer}
                             if vector_answer is not None else {})
        record["adapter_invalidation"] = {
            "invalid_adapter": "flattened-sessions-v1",
            "reason": (
                "history sessions were flattened and a UUID/string mismatch "
                "capped full_ice retrieval at three fragments"
            ),
            "legacy_conversation_id": old_conversation_id,
            "archived_original": str(backup_path),
            "repaired_utc": datetime.now(timezone.utc).isoformat(),
            "preserved_conditions": (["vector_rag"]
                                     if vector_answer is not None else []),
        }
        _atomic_write_json(path, record)

    judgement_dir = phase_dir / "judgements"
    full_ice_judgements = (sorted(judgement_dir.glob("*__full_ice.json"))
                           if judgement_dir.exists() else [])
    stats["full_ice_judgements_to_archive"] = len(full_ice_judgements)
    if apply and full_ice_judgements:
        archive_judgements.mkdir(parents=True, exist_ok=True)
        for path in full_ice_judgements:
            target = archive_judgements / path.name
            if not target.exists():
                shutil.copy2(path, target)
            path.unlink()

    for name in ("MANIFEST.json", "run.log", "failures.log"):
        path = phase_dir / name
        if path.exists():
            stats["metadata_to_archive"].append(name)
            if apply:
                archive_dir.mkdir(parents=True, exist_ok=True)
                target = archive_dir / name
                if not target.exists():
                    shutil.copy2(path, target)
                path.unlink()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", action="append", required=True,
                        choices=("oracle", "abstention", "stratified"))
    parser.add_argument("--apply", action="store_true",
                        help="perform the repair; default is a dry-run")
    args = parser.parse_args()

    summaries = [repair_phase(phase, args.apply) for phase in args.phase]
    print(json.dumps(summaries, indent=2))
    if not args.apply:
        print("dry-run only; pass --apply to archive and repair these artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
