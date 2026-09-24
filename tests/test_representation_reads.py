"""v3 shared representation through three real database readers; no committed rows."""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.db import SessionLocal
from src.api.prompt_assembler import get_recent_turns
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services.retrieval_svc import recent_turns


def test_fact_source_time_is_separate_from_learning_time():
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        try:
            db.add(Conversation(id=cid))
            db.flush()
            db.add(
                EpisodicMemory(
                    conversation_id=cid,
                    batch_id=batch,
                    raw_text="Atlas uses Redis.",
                    timestamp=datetime(2021, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
                    ts_provenance="original",
                    context_reliance="Long_Term_Memory",
                    idempotency_key=f"source-clock-control:{batch}",
                )
            )
            db.flush()
            o = HybridRetrievalOrchestrator(db, None)
            edge = SimpleNamespace(
                source_batch=batch,
                relation="uses",
                negated=False,
                learned_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
                valid_from=None,
            )
            o._prime_edge_times([edge, edge])
            assert len(o._source_times) == 1
            rendered = o._fact_line(
                SimpleNamespace(canonical_name="Atlas"),
                edge,
                SimpleNamespace(canonical_name="Redis"),
            )
            assert "source recorded: 2021-01-02T03:04:05Z" in rendered
            assert "learned: 2026-09-13T00:00:00Z" in rendered
        finally:
            db.rollback()


def test_database_readers_preserve_uncompressed_evidence():
    cid, batch = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        try:
            db.add(Conversation(id=cid, kind="chat"))
            db.flush()
            row = EpisodicMemory(
                conversation_id=cid,
                batch_id=batch,
                raw_text="User: Keep persistence disabled.\n\nAssistant: "
                + "context " * 80
                + "The port is 8391.",
                summary_text="Enable persistence.",
                summary_coverage=None,
                abstract_text="Enable persistence.",
                inject_raw=False,
                is_private=False,
                is_document=False,
                context_reliance="Long_Term_Memory",
                idempotency_key=f"representation-control:{batch}",
            )
            db.add(row)
            db.flush()
            assert recent_turns(db, conversation_id=str(cid))[0]["text"].endswith(
                "8391."
            )
            rendered = " ".join(
                m["content"] for m in get_recent_turns(db, str(cid), max_tokens=4000)
            )
            assert "disabled" in rendered and "8391." in rendered
            assert "Enable persistence" not in rendered
            fragments = HybridRetrievalOrchestrator(db, None)._rows_to_fragments(
                [row], "episodic"
            )
            assert fragments and "8391." in fragments[0].text
            assert fragments[0].degrade_text is None
        finally:
            db.rollback()


if __name__ == "__main__":
    test_database_readers_preserve_uncompressed_evidence()
    test_fact_source_time_is_separate_from_learning_time()
    print("v3: three database reader paths passed; transaction rolled back")
