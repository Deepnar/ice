"""Actual SQL output capture and an actual periodic writer; use a disposable DB."""
import io
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.z1 import seed_v3
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import Conversation, ConversationSummary, ProceduralMemory


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
    print("v3 semantic trace passed: unchanged-count text rewrite and actual procedural decay captured; vectors omitted; disposable SQL only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
