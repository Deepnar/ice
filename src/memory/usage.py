"""Graph retention from attributed context exposure, never truth promotion."""

import uuid

import structlog
from sqlalchemy import text

from src.api.config import settings

logger = structlog.get_logger("ice.memory.usage")


def evidence_after_eviction(fragments, plan):
    return [] if "evidence" in plan else fragments


def record_graph_access(db, fragments, *, stage):
    if stage not in {"prompt_prepared", "context_returned"}:
        raise ValueError("Unknown memory exposure stage")
    if not settings.retrieval_strengthen_writes:
        return
    ids = {uuid.UUID(str(e)) for f in fragments
           for e in getattr(f, "origin_edge_ids", ())}
    if not ids:
        return
    try:
        result = db.execute(text("""
            UPDATE codex_edges
            SET strength = LEAST(:cap, COALESCE(strength, 0) + :increment),
                usage_count = usage_count + 1,
                last_accessed_at = NOW()
            WHERE id = ANY(:ids) AND valid_until IS NULL
              AND source = 'conversation'
        """), {"ids": list(ids), "cap": settings.codex_retention_cap,
               "increment": settings.codex_retention_increment})
        db.commit()
        logger.info("graph_memory_access", stage=stage, edges=result.rowcount,
                    edge_ids=sorted(str(e) for e in ids))
    except Exception as exc:
        db.rollback()
        logger.warning("graph_memory_access_failed", stage=stage,
                       error_type=type(exc).__name__)
