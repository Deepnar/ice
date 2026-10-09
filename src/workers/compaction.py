"""Compaction Engine Subsystem – Ledger Compression Plane."""

from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import CodexEdge, CodexEntity, CodexEvent, CodexSnapshot

logger = structlog.get_logger("ice.workers.compaction")


def _edge_record(edge):
    """Stored projection metadata, not a fresh factual support decision."""
    result = {}
    for column in CodexEdge.__table__.columns:
        value = getattr(edge, column.name)
        if isinstance(value, datetime):
            value = value.isoformat()
        elif column.name in {'id', 'source_id', 'target_id', 'source_batch'}:
            value = str(value) if value is not None else None
        elif column.name == 'observed_batches':
            value = [str(v) for v in value] if value is not None else None
        result[column.name] = value
    return result


def compact_entities():
    """Compresses append-only transaction logs into fast entity state
    snapshots. Scheduled by the maintenance runtime at 24h (G10 — beat never
    scheduled it); lossless per the Track-T constraint (events are marked
    compacted, never deleted — the journal stays readable history)."""
    db = SessionLocal()
    stats = dict(method='graph_projection_checkpoint_v2', snapshots=0, events_compacted=0)
    try:
        # A concurrent writer must not make entity, edges and consumed events
        # describe different database views. This session alone owns the level.
        db.connection(execution_options={'isolation_level': 'REPEATABLE READ'})
        subq = (
            select(CodexEvent.entity_id)
            .where(CodexEvent.compacted == False)
            .group_by(CodexEvent.entity_id)
            .having(func.count(CodexEvent.id) >= settings.compaction_event_threshold)
            .subquery()
        )
        target_entities = db.query(subq.c.entity_id).all()
        for (entity_id,) in target_entities:
            entity = db.get(CodexEntity, entity_id)
            if not entity:
                continue

            events = db.query(CodexEvent).filter(
                CodexEvent.entity_id == entity_id,
                CodexEvent.compacted == False
            ).order_by(CodexEvent.timestamp.asc(), CodexEvent.id.asc()).all()
            edges = db.query(CodexEdge).filter_by(source_id=entity_id).order_by(CodexEdge.id.asc()).all()
            context_payload = entity.context_payload or ""
            properties = entity.properties or {}
            aliases = list(entity.aliases) if entity.aliases else []

            db.add(CodexSnapshot(
                entity_id=entity_id,
                snapshot_ts=datetime.now(timezone.utc),
                last_event_id=events[-1].id,
                full_state={
                    "version": 2,
                    "basis": "current_graph_projection",
                    "edges": [_edge_record(edge) for edge in edges],
                    # Unexpired IDs are a storage-state index, not validity at
                    # every world date or an eligibility/trust decision.
                    "active_edges": [str(edge.id) for edge in edges
                                     if edge.valid_until is None and edge.unlearned_at is None],
                    "context_payload": context_payload,
                    "properties": properties,
                    "aliases": aliases
                }
            ))

            for event in events:
                event.compacted = True
            stats['snapshots'] += 1
            stats['events_compacted'] += len(events)
        db.commit()
        logger.info('graph_checkpoint_pass_complete', **stats)
        return stats

    except Exception as exc:
        db.rollback()
        logger.error("compaction_pass_failed", error=str(exc))
        raise
    finally:
        db.close()
