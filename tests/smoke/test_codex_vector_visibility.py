"""G29: nearest-vector graph anchors obey the same project filter as exact match."""

import os
import uuid

import pytest

from src.api.db import SessionLocal
from src.memory.models import CodexEntity
from src.retrieval.orchestrator import HybridRetrievalOrchestrator


class _Embedder:
    def encode(self, values, **_kwargs):
        return [[1.0] + [0.0] * 1023 for _ in values]


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_nearest_vector_anchor_filters_before_limit():
    db = SessionLocal()
    suffix = uuid.uuid4().hex
    project_a, project_b = uuid.uuid4(), uuid.uuid4()
    derived_a = CodexEntity(
        canonical_name=f"vector-scope-a-{suffix}", source="derived",
        project_id=project_a, embedding=[1.0] + [0.0] * 1023,
    )
    derived_b = CodexEntity(
        canonical_name=f"vector-scope-b-{suffix}", source="derived",
        project_id=project_b, embedding=[0.98, 0.199] + [0.0] * 1022,
    )
    conversation = CodexEntity(
        canonical_name=f"vector-scope-chat-{suffix}", source="conversation",
        embedding=[0.9, 0.4359] + [0.0] * 1022,
    )
    try:
        db.add_all([derived_a, derived_b, conversation])
        db.flush()
        orch = HybridRetrievalOrchestrator(db, _Embedder())

        orch._scope_project_id = None
        assert [row.id for row in orch._match_entities_by_similarity(["probe"])] == [
            conversation.id], "unscoped query admitted a project-derived vector"

        orch._scope_project_id = project_a
        assert [row.id for row in orch._match_entities_by_similarity(["probe"])] == [
            derived_a.id], "project A did not resolve its nearest visible vector"

        orch._scope_project_id = project_b
        assert [row.id for row in orch._match_entities_by_similarity(["probe"])] == [
            derived_b.id], "hidden rank-1 project A vector crowded out project B"
    finally:
        db.rollback()
        db.close()
