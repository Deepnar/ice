"""Actual v3 preflight/write/post-flight recovery under injected worker faults.

Run with the disposable DB wrapper. Synthetic sources and short history test
fault recovery, not memory quality or complete corpus/maintenance coverage.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z1 import seed_v3
from scripts.z1.replay_validation import validate_complete_replay
from scripts.z1.worker_recovery import WorkerOutage
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import EpisodicMemory
from src.workers import post_flight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    slug = "synthetic"
    turns = [{"turn_number": n, "timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=n),
              "ts_provenance": "synthetic_raw_import", "prompt": f"I named synthetic test project number {n} Nimbus.",
              "response": "Understood."} for n in range(1, 5)]
    probe = {"probe_id": "synthetic-probe", "probe_type": "episodic", "question": "What did I name my project?",
             "expected_answer": "Nimbus", "gold_turns": [1], "source_split_turn": 1,
             "cutoff_kind": "delayed", "label_status": "fixture"}
    replay_args = SimpleNamespace(out=str(args.out), conversation=None, limit=0, probe_panel="existing",
        no_maintenance=False, no_probe_controls=False, resume=False, checkpoint_every=1,
        checkpoint_dir=None, development_repeat_review=None, worker_failure_policy="continue")
    calls = []
    original = post_flight.extract_codex
    def failed(**kwargs):
        calls.append(kwargs["batch_id"])
        if len(set(calls)) <= 3:
            raise ConnectionError("synthetic provider connection outage")
        return original(**kwargs)
    with patch.object(seed_v3, "EXPECTED", {slug: 4}), patch.object(settings, "maintenance_intervals", {}), \
         patch.object(post_flight, "extract_codex", failed):
        try:
            seed_v3.run(replay_args, {slug: turns}, {(slug, 2): [probe]})
        except WorkerOutage:
            pass
        else:
            raise AssertionError("persistent worker outage must pause")
        assert len(calls) == 6 and len(set(calls)) == 3
        recovery_root = args.out.with_suffix(".recovery")
        pointer = json.loads((recovery_root / "current.json").read_text())
        state = json.loads((recovery_root / pointer["generation"] / "recovery.json").read_text())["state"]
        assert state["completed"] == {slug: 3}
        assert len(state["worker_recovery"]["degraded"]) == 3
        with SessionLocal() as db:
            rows = db.query(EpisodicMemory).order_by(EpisodicMemory.timestamp).all()
            assert len(rows) == 3 and all(row.source_spans for row in rows)
            assert all(turn["prompt"] in row.raw_text for turn, row in zip(turns, rows))
        replay_args.resume = True
        assert seed_v3.run(replay_args, {slug: turns}, {(slug, 2): [probe]}) == 0
    rows = [json.loads(line) for line in args.out.open()]
    result = validate_complete_replay(rows, {slug: 4})["complete"]
    assert result["memory_processing"]["degraded_worker_calls"] == 3
    assert result["memory_processing"]["failed_attempts"] == 6
    assert [r["turn"] for r in rows if r["event"] == "written"] == [1, 2, 3, 4]
    captured = next(r for r in rows if r["event"] == "as_of_probe")
    assert captured["memory_processing"]["degraded_gold_turns"] == [1]
    assert captured["memory_processing"]["degraded_worker_calls"] == 2  # no future fault leakage
    assert "worker_degraded" not in json.dumps(captured["preflight"]["prompt_messages"])
    assert len(calls) == 7  # resume writes only the fourth turn, never the three completed sources
    assert list(recovery_root.glob("unfinished-*.jsonl"))
    with SessionLocal() as db:
        assert db.query(EpisodicMemory).count() == 4
    from src.workers.bg_client_factory import release_owned_models
    release_owned_models()
    print("v3 actual fault path passed: three originals/probe/degradation checkpointed before outage; resume processes only fourth turn; six failed attempts retained, no future fault/prompt leakage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
