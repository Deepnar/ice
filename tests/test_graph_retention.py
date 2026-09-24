"""v3 graph retention, independent support and non-destructive aging."""

import os
import uuid

from alembic import command
from alembic.config import Config
from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity
from src.memory.usage import evidence_after_eviction, record_graph_access
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator
from src.workers.codex_decay import decay_codex_edges
from src.workers.codex_extractor import _observe_edge

assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
        and make_url(settings.database_url).database == os.environ["ICE_TEST_DATABASE"]), \
    "Run through tests/support/disposable_database.py; the decay worker is global."


def test_usage_migration_roundtrip():
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.stamp(config, "d9a1f4b72c60")
    command.downgrade(config, "c8f0d21a6e39")
    command.upgrade(config, "d9a1f4b72c60")


def test_quiet_and_used_facts_survive_without_false_promotion(monkeypatch):
    monkeypatch.setattr(settings, "retrieval_strengthen_writes", True)
    monkeypatch.setattr(settings, "codex_decay_daily", 0.01)
    with SessionLocal() as db:
        a, b = [CodexEntity(canonical_name=f"retention_{uuid.uuid4().hex}") for _ in range(2)]
        db.add_all([a, b])
        db.flush()
        quiet = CodexEdge(source_id=a.id, target_id=b.id, relation="uses",
            source_batch=uuid.uuid4(), strength=0.001, confidence="active", extraction_confidence=0.9)
        used = CodexEdge(source_id=a.id, target_id=b.id, relation="knows",
            source_batch=uuid.uuid4(), strength=1.0, confidence="pending", extraction_confidence=0.9)
        weak = CodexEdge(source_id=a.id, target_id=b.id, relation="likes",
            source_batch=uuid.uuid4(), strength=10.0, confidence="pending", extraction_confidence=0.35)
        db.add_all([quiet, used, weak])
        db.commit()
        ids = [e.id for e in (quiet, used, weak)]
    decay_codex_edges(cycles=96)
    with SessionLocal() as db:
        quiet, used, weak = [db.get(CodexEdge, e) for e in ids]
        assert all(e.valid_until is None and e.unlearned_at is None for e in (quiet, used, weak))
        assert quiet.confidence == "active" and used.confidence == "pending"
        o = HybridRetrievalOrchestrator(db, None)
        assert o._edge_trust(quiet) >= settings.codex_direct_trust_floor
        weak.strength = settings.codex_retention_cap
        assert o._edge_trust(weak) < settings.codex_direct_trust_floor
        frag = ContextFragment("actual fact", "codex", 1.0, 3, origin_edge_ids=(str(used.id),))
        record_graph_access(db, evidence_after_eviction([frag], ["evidence"]), stage="prompt_prepared")
        db.refresh(used)
        assert used.usage_count == 0
        for _ in range(20):
            record_graph_access(db, [frag, frag], stage="context_returned")
        db.refresh(used)
        assert used.usage_count == 20 and used.last_accessed_at is not None
        assert used.strength > settings.codex_retention_floor
        assert used.confidence == "pending" and used.extraction_confidence == 0.9
        assert not _observe_edge(used, used.source_batch, 0.9)
        assert used.confidence == "pending"
        assert _observe_edge(used, uuid.uuid4(), 0.95)
        assert used.confidence == "active" and len(used.observed_batches) == 2
        monkeypatch.setattr(settings, "retrieval_strengthen_writes", False)
        record_graph_access(db, [frag], stage="context_returned")
        assert used.usage_count == 20
        db.rollback()
