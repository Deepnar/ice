"""v3 source metadata survives database migration, rendering and cold storage.

Run only through tests/support/disposable_database.py: the decay worker is global.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal, engine
from src.api.prompt_assembler import get_recent_turns
from src.memory.models import ColdStorage, Conversation, EpisodicMemory
from src.memory.source import chat_provenance, source_units
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.workers.decay import apply_decay

assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
        and make_url(settings.database_url).database == os.environ["ICE_TEST_DATABASE"]), \
    "Run via tests/support/disposable_database.py; this suite calls global workers."


def test_migration_roundtrip():
    # The disposable runner created the current ORM schema. Downgrade only the
    # new additive revision, then apply it again; do not run old data backfills.
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.stamp(config, "c8f0d21a6e39")
    command.downgrade(config, "a1c4e7b90d22")
    assert "source_spans" not in {c["name"] for c in inspect(engine).get_columns("episodic_memory")}
    command.upgrade(config, "c8f0d21a6e39")
    assert {"source_spans", "ts_provenance"} <= {
        c["name"] for c in inspect(engine).get_columns("cold_storage")}


def test_all_ingestion_writers_record_roles(monkeypatch):
    from src.api import main
    from src.ingestion.importer import _store_turn
    from src.ingestion.documents.ingest import _store_section
    from src.services import bookmarks

    user, assistant = "My quoted Assistant: example.", "An assistant suggestion."
    now = datetime.now(timezone.utc)
    embedder = SimpleNamespace(encode=lambda *args, **kwargs: [0.0] * 1024)
    monkeypatch.setattr(main, "classifier", SimpleNamespace(embedder=embedder))
    monkeypatch.setattr(main, "core", None)
    monkeypatch.setattr(bookmarks, "_enqueue_extraction", lambda *args: None)
    cids = [uuid.uuid4() for _ in range(3)]
    with SessionLocal() as db:
        db.add_all([Conversation(id=cid) for cid in cids])
        db.commit()
    stream = "data: " + json.dumps({"choices": [{"delta": {"content": assistant}}]}) + "\n\n"
    asyncio.run(main.store_turn_async(str(uuid.uuid4()), user, cids[0], [], [],
                                     "Long_Term_Memory", [[stream]]))
    with SessionLocal() as db:
        _store_turn(db, cids[1], user, assistant, now, str(uuid.uuid4()),
                    "preserve", now, "original", None, embedder)
        _store_section(db, SimpleNamespace(id=uuid.uuid4()), cids[2], user, now, 0,
                       None, embedder)
        note = bookmarks.remember_note(db, user, embedder)
        for cid in cids[:2]:
            turn = db.query(EpisodicMemory).filter_by(conversation_id=cid).one()
            assert [(u.role, u.text) for u in source_units(turn)] == [("user", user), ("assistant", assistant)]
        doc = db.query(EpisodicMemory).filter_by(conversation_id=cids[2]).one()
        assert [(u.role, u.text) for u in source_units(doc)] == [("document", user)]
        assert source_units(db.get(EpisodicMemory, uuid.UUID(note["id"])))[0].role == "user"


def test_source_survives_render_archive_restore(monkeypatch):
    user, assistant = "Quoted example:\n\nAssistant: use Redis. 🌟", "I suggest PostgreSQL."
    raw = f"User: {user}\n\nAssistant: {assistant}"
    spans = chat_provenance(user, assistant)
    cid, rid = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid))
        db.flush()
        db.add(EpisodicMemory(id=rid, conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=raw, source_spans=spans, context_reliance="Long_Term_Memory",
            ts_provenance="synthetic_raw_import", decay_score=0.0001,
            is_archived=False, timestamp=datetime.now(timezone.utc) - timedelta(days=90),
            inject_raw=True, idempotency_key=str(uuid.uuid4())))
        db.commit()
        messages = get_recent_turns(db, str(cid), max_tokens=4000)
        assert user in messages[0]["content"] and messages[1]["content"] == assistant
    apply_decay()
    with SessionLocal() as db:
        cold = db.get(ColdStorage, rid)
        assert cold is not None and db.get(EpisodicMemory, rid) is None
        assert cold.source_spans == spans and cold.ts_provenance == "synthetic_raw_import"
        embedder = SimpleNamespace(encode=lambda *args, **kwargs: [0.0] * 1024)
        o = HybridRetrievalOrchestrator(db, embedder)
        o._cold_hits = {str(rid): cold}
        monkeypatch.setattr(settings, "retrieval_strengthen_writes", True)
        o._resurrect_cold_hits([ContextFragment(raw, "episodic", 1.0, 20,
                                               source_batch_id=str(rid))])
        db.expire_all()
        turn = db.get(EpisodicMemory, rid)
        assert turn is not None and db.get(ColdStorage, rid) is None
        assert turn.source_spans == spans and turn.ts_provenance == "synthetic_raw_import"
        assert [(u.role, u.text) for u in source_units(turn)] == [("user", user), ("assistant", assistant)]
