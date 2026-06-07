"""Compaction Engine Subsystem – Ledger Compression Plane."""

import uuid
from datetime import datetime, timezone
import structlog
from sqlalchemy import func, select

from src.api.db import SessionLocal
from src.memory.models import CodexEntity, CodexEvent, CodexSnapshot
from src.workers.gpu_check import is_gpu_busy
from src.workers.celery_app import app

logger = structlog.get_logger("ice.workers.compaction")
EVENT_THRESHOLD = 100


@app.task(bind=True, max_retries=3, default_retry_delay=60)
def compact_entities(self):
    """Compresses append-only transaction logs into fast entity state snapshots."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        subq = (
            select(CodexEvent.entity_id)
            .where(CodexEvent.compacted == False)
            .group_by(CodexEvent.entity_id)
            .having(func.count(CodexEvent.id) >= EVENT_THRESHOLD)
            .subquery()
        )
        
        target_entities = db.query(subq.c.entity_id).all()
        
        for (entity_id,) in target_entities:
            entity = db.query(CodexEntity).get(entity_id)
            if not entity:
                continue

            events = db.query(CodexEvent).filter(
                CodexEvent.entity_id == entity_id,
                CodexEvent.compacted == False
            ).order_by(CodexEvent.timestamp.asc()).all()

            # Reconstruction map using composited signature strings to protect overlapping paths
            active_edges = set() 
            context_payload = entity.context_payload or ""
            properties = entity.properties or {}
            aliases = list(entity.aliases) if entity.aliases else []
            last_event_id = None

            for event in events:
                payload = event.payload or {}
                rel = payload.get("relation")
                tgt = payload.get("target_id")
                edge_sig = f"{rel}:{tgt}" if (rel and tgt) else None

                if event.event_type == "edge_added" and edge_sig:
                    active_edges.add(edge_sig)
                elif event.event_type == "edge_expired" and edge_sig:
                    active_edges.discard(edge_sig)
                    
                last_event_id = event.id

            db.add(CodexSnapshot(
                entity_id=entity_id,
                snapshot_ts=datetime.now(timezone.utc),
                last_event_id=last_event_id,
                full_state={
                    "active_edges": list(active_edges),
                    "context_payload": context_payload,
                    "properties": properties,
                    "aliases": aliases
                }
            ))

            for event in events:
                event.compacted = True

            db.commit()
            logger.info("entity_historical_ledger_compacted", entity_id=str(entity_id))

    except Exception as exc:
        db.rollback()
        logger.error("compaction_pass_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()