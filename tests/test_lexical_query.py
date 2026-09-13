"""v3 lexical recall controls through PostgreSQL, with rolled-back fixtures."""

import uuid
from types import SimpleNamespace

import pytest

from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


@pytest.mark.parametrize("query,body", [
    ("8391", "The port is 8391."),
    ("東京", "東京"),
    ("καλημέρα", "καλημέρα"),
    ("привет", "привет"),
    ("unrelated " * 40 + "8391", "The port is 8391."),
    ("8391 ' | ! : & ( )", "The port is 8391."),
    ("src/auth.py", "Edit src/auth.py to configure authentication."),
])
def test_lexical_terms_survive(query, body, monkeypatch):
    cid, other = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        try:
            db.add_all([Conversation(id=cid), Conversation(id=other)])
            db.flush()
            rows = []
            for conv, content, private in [(cid, body, False), (cid, "unrelated control", False),
                                            (other, body, True)]:
                batch = uuid.uuid4()
                row = EpisodicMemory(conversation_id=conv, batch_id=batch,
                    raw_text=content, context_reliance="Long_Term_Memory",
                    is_private=private, decay_score=1.0,
                    idempotency_key=f"lexical-control:{batch}")
                db.add(row)
                rows.append(row)
            db.flush()
            o = HybridRetrievalOrchestrator(db, None)
            def fail(leg, err):
                pytest.fail(f"Unexpected fallback in {leg}: {type(err).__name__}")
            monkeypatch.setattr(o, "_leg_degraded", fail)
            classification = SimpleNamespace(prompt=query, intent_tags=[])
            hits = o._bm25_episodic(classification, None, str(cid))
            ids = {str(f.source_batch_id) for f in hits}
            assert str(rows[0].id) in ids
            assert str(rows[2].id) not in ids
            # Explicitly verify an unmatched query and stopword-only input.
            for empty in ("zyxneverpresent", "the and or"):
                assert not o._bm25_episodic(SimpleNamespace(prompt=empty, intent_tags=[]),
                                             None, str(cid))
            global_hits = o._bm25_episodic(classification, None)
            assert str(rows[2].id) not in {str(f.source_batch_id) for f in global_hits}
        finally:
            db.rollback()
