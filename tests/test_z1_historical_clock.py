"""Run with tests/support/disposable_database.py; never the working store."""
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from scripts.z1.historical_clock import historical_clock
from scripts.z1.seed_v3 import (isolated_database, memory_state, original_turn_count,
                               probe_observation, source_storage_at_cutoff)
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity, ColdStorage, Conversation, EpisodicMemory, MemorySlot
from src.memory.usage import record_graph_access
from src.workers import maintenance_agent


def main():
    isolated_database()
    source = datetime(2025, 1, 1, tzinfo=timezone.utc)
    db = SessionLocal()
    try:
        with historical_clock(source, db.get_bind()) as stats:
            conversation = Conversation()
            entity = CodexEntity(canonical_name="clock-control")
            slot = MemorySlot(slot_name=maintenance_agent.STALE_SLOT_NAME,
                              content="A pending invented task", updated_by="clock-control")
            db.add_all([conversation, entity, slot])
            db.commit()
            assert conversation.created_at == entity.last_updated == slot.last_updated == source
            assert db.execute(text("SELECT NOW()")).scalar() == source
            assert not maintenance_agent._detect_stale_slot(db, 1)
            assert stats["sql_now_statements"] == 1
        later = source + timedelta(days=settings.agent_stale_slot_days + 1)
        with historical_clock(later, db.get_bind()) as stats:
            detected = maintenance_agent._detect_stale_slot(db, 1)
            assert len(detected) == 1 and detected[0].item_type == "stale_slot"
            assert db.execute(text("SELECT now() - interval '1 day'")).scalar() == later - timedelta(days=1)
            assert stats["sql_now_statements"] == 1
        # The same source clock must govern physical warm-to-cold movement.
        with historical_clock(source, db.get_bind()):
            old = EpisodicMemory(conversation_id=conversation.id, batch_id=uuid.uuid4(),
                                 raw_text="An invented older fact", context_reliance="NONE",
                                 timestamp=source - timedelta(days=8), decay_score=0.001,
                                 idempotency_key="clock-old")
            fresh = EpisodicMemory(conversation_id=conversation.id, batch_id=uuid.uuid4(),
                                   raw_text="An invented current fact", context_reliance="NONE",
                                   timestamp=source, decay_score=1.0, idempotency_key="clock-fresh")
            db.add_all([old, fresh])
            db.commit()
            old_id = old.id
            old_batch_id = old.batch_id
            from src.workers.decay import apply_decay
            before = memory_state(db)
            apply_decay(cycles=1)
            db.expire_all()
            after = memory_state(db)
            cold = db.get(ColdStorage, old_id)
            assert cold and cold.archived_at == source and cold.batch_id == old_batch_id
            assert before["warm_turns"] == 2 and after["warm_turns"] == 1
            assert after["cold_turns"] == 1 and fresh.decay_score == 1.0
            assert original_turn_count(db, conversation.id) == 2
            storage = source_storage_at_cutoff(db, conversation.id)
            assert storage[str(old_id)]["tier"] == "cold"
            assert storage[str(fresh.id)]["tier"] == "warm"
        writes = settings.retrieval_strengthen_writes
        fresh_id = fresh.id
        with probe_observation(db):
            assert settings.retrieval_strengthen_writes is False
            assert db.execute(text("SHOW transaction_read_only")).scalar() == "on"
            try:
                db.execute(text("UPDATE episodic_memory SET decay_score=0.5 WHERE id=:id"),
                           {"id": fresh_id})
            except DBAPIError as exc:
                assert getattr(exc.orig, "sqlstate", None) == "25006"
            else:
                raise AssertionError("diagnostic probe transaction allowed a memory write")
        assert settings.retrieval_strengthen_writes == writes
        assert db.get(EpisodicMemory, fresh_id).decay_score == 1.0
        with historical_clock(source, db.get_bind()):
            target = CodexEntity(canonical_name="clock-target")
            db.add(target)
            db.flush()
            edge = CodexEdge(source_id=entity.id, target_id=target.id, relation="uses",
                             confidence="active", strength=1.0, source_batch=uuid.uuid4())
            db.add(edge)
            db.commit()
            fragment = SimpleNamespace(origin_edge_ids=[edge.id])
            record_graph_access(db, [fragment], stage="prompt_prepared")
            db.refresh(edge)
            assert edge.usage_count == 1 and edge.last_accessed_at == source
            strength = edge.strength
            with probe_observation(db):
                record_graph_access(db, [fragment], stage="prompt_prepared")
            db.refresh(edge)
            assert edge.usage_count == 1 and edge.strength == strength
            assert edge.confidence == "active"  # Exposure never creates corroboration.
        from src.memory import models
        assert models.datetime is datetime
        assert maintenance_agent.datetime is datetime
        actual = db.execute(text("SELECT NOW()")).scalar()
        assert abs((actual - datetime.now(timezone.utc)).total_seconds()) < 10
        print("v3 historical clock: ORM/default writes, SQL clocks, stale-slot detector, decay/cold movement, original counts, read-only probe isolation and restoration passed")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
