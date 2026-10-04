"""Actual SQL output capture and an actual periodic writer; use a disposable DB."""
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z1 import seed_v3
from scripts.z1.worker_recovery import WorkerRecovery
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, ConversationSummary, ProceduralMemory
from src.workers.completion_text import IncompleteCompletion


def main():
    seed_v3.isolated_database()
    now = datetime.now(timezone.utc)
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        summary = ConversationSummary(conversation_id=conversation.id,
                                      summary_text="old fixture value", covers_turns=1)
        procedure = ProceduralMemory(pattern_name="disposable stale procedure",
                                     last_observed=now - timedelta(days=365),
                                     reinforcement_count=1, is_active=True)
        db.add_all([summary, procedure])
        db.commit()
        counts = seed_v3.memory_state(db)
        before = seed_v3.semantic_state(db, "conversation_summary")
        summary.summary_text = "corrected fixture value"
        db.commit()
        after = seed_v3.semantic_state(db, "conversation_summary")
        assert counts == seed_v3.memory_state(db)
        change, = seed_v3.semantic_changes(before, after)
        assert change["table"] == "conversation_summaries"
        assert change["key"] == [str(conversation.id)]
        assert change["before"]["summary_text"] == "old fixture value"
        assert change["after"]["summary_text"] == "corrected fixture value"
        assert "embedding" not in change["after"]

        sink = io.StringIO()
        with patch.object(settings, "maintenance_intervals", {"decay_procedural": 10}), \
                patch.object(settings, "procedural_stale_days", 30), \
                patch.object(settings, "procedural_min_reinforcement", 2):
            seed_v3.due_maintenance(db, sink, now, conversation.id, {})
        record, = [json.loads(line) for line in sink.getvalue().splitlines()]
        change, = record["semantic_changes"]
        assert change["table"] == "procedural_memory"
        assert change["before"]["is_active"] and not change["after"]["is_active"]
        assert record["semantic_observed_tables"] == ["procedural_memory"]
        assert record["job_elapsed_ms"] > 0 and record["observer_elapsed_ms"] > 0
        assert record["before"]["active_procedural"] == 1
        assert record["after"]["active_procedural"] == 0
        # A committed partial periodic job cannot be treated as an atomic
        # failure or immediately replayed. Capture its changed rows and retain
        # failed cadence separately from successful completion.
        import src.workers.conversation_summary as notes
        def partial_job(worker_db):
            target = worker_db.get(ConversationSummary, conversation.id)
            target.summary_text = "committed partial synthetic rewrite"
            worker_db.commit()
            raise IncompleteCompletion("synthetic incomplete completion")
        sink = io.StringIO()
        recovery = WorkerRecovery(sink, "continue")
        last_run = {}
        with patch.object(settings, "maintenance_intervals", {"conversation_summary": 10}), \
                patch.object(notes, "run_conversation_summaries", partial_job):
            seed_v3.due_maintenance(db, sink, now, conversation.id, last_run, recovery,
                                  {"conversation": "fixture", "turn": 1, "stage": "maintenance"})
            count = len(sink.getvalue())
            seed_v3.due_maintenance(db, sink, now + timedelta(seconds=5), conversation.id, last_run, recovery,
                                  {"conversation": "fixture", "turn": 2, "stage": "maintenance"})
            assert len(sink.getvalue()) == count
        rows = [json.loads(line) for line in sink.getvalue().splitlines()]
        assert [r["event"] for r in rows] == ["worker_attempt_failed", "worker_degraded", "maintenance"]
        assert rows[-1]["worker_status"] == "degraded" and rows[-1]["result"] is None
        assert rows[-1]["semantic_changes"][0]["after"]["summary_text"] == "committed partial synthetic rewrite"
        assert not last_run and recovery.state["last_attempt"] == {"conversation_summary": now.isoformat()}
    print("v3 semantic trace passed: unchanged-count rewrite, actual decay and committed partial failed-job changes/cadence captured; vectors omitted; disposable SQL only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
