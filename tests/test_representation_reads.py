"""v3 shared representation through three real database readers; no committed rows."""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.db import SessionLocal
from src.api.prompt_assembler import get_recent_turns
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services.retrieval_svc import recent_turns


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
    print("v3: three database reader paths passed; transaction rolled back")
