"""Codex fact usage from attributed context exposure, never truth promotion."""

import uuid

import structlog
from sqlalchemy import text

from src.api.config import settings

logger = structlog.get_logger("ice.memory.usage")


def evidence_after_eviction(fragments, plan):
    return [] if "evidence" in plan else fragments


def record_graph_access(db, fragments, *, stage):
    """Count a codex fact's exposure in the final prompt (or MCP context)."""
    if stage not in {"prompt_prepared", "context_returned"}:
        raise ValueError("Unknown memory exposure stage")
    if not settings.retrieval_strengthen_writes:
        return
    _record_fact_usage(db, fragments, stage=stage)


def _record_fact_usage(db, fragments, *, stage):
    """CODEX_V4 D7: a fact that reached the final prompt (or the MCP context)
    gains one usage point; the daily decay job ages it. Usage ranks, never admits."""
    ids = {uuid.UUID(str(f_id)) for f in fragments
           for f_id in getattr(f, "origin_fact_ids", ())}
    if not ids:
        return
    try:
        result = db.execute(text("""
            UPDATE codex_facts SET usage = usage + 1, last_used_at = NOW()
            WHERE id = ANY(:ids) AND valid_until IS NULL
        """), {"ids": list(ids)})
        db.commit()
        logger.info("fact_memory_access", stage=stage, facts=result.rowcount)
    except Exception as exc:
        db.rollback()
        logger.warning("fact_memory_access_failed", stage=stage, error_type=type(exc).__name__)
