"""Sentinel Monitor – evaluates declarative rules and fires actions."""

import json

import structlog
from datetime import datetime, timezone, timedelta
from sqlalchemy import text

from src.api.db import SessionLocal
from src.memory.models import SentinelRule, SentinelEvent
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy

logger = structlog.get_logger("ice.workers.sentinel")


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def monitor_sentinels(self):
    """Periodic task: evaluate all active sentinel rules."""
    if is_gpu_busy():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        rules = db.query(SentinelRule).filter_by(is_active=True).all()
        for rule in rules:
            if rule.last_fired_at and (now - rule.last_fired_at).total_seconds() < rule.cooldown_seconds:
                continue
            if _evaluate_rule(rule, db):
                event = SentinelEvent(
                    rule_id=rule.id,
                    fired_at=now,
                    trigger_state=rule.trigger_conditions,
                    action_taken=rule.action_type
                )
                db.add(event)
                rule.last_fired_at = now
                logger.info("sentinel_fired", rule_name=rule.name, action=rule.action_type)

                # Perform action
                _execute_action(rule, db)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("sentinel_monitor_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _evaluate_rule(rule, db) -> bool:
    cond = rule.trigger_conditions or {}
    if isinstance(cond, str):
        import json
        cond = json.loads(cond)
    ttype = rule.trigger_type

    if ttype == "threshold":
        # Generic threshold check: if cond has keys like min_pending_edges, evaluate
        if "consecutive_zero_retrieval" in cond:
            # check recent turns for zero retrieval
            limit = cond["consecutive_zero_retrieval"]
            # Query the most recent episodic turns where context_reliance = 'Long_Term_Memory'
            # and check if any retrieval happened (we don't log retrieval per turn, so approximate)
            # For now, placeholder – would need retrieval logging.
            return False  # Not implemented fully; would need to track retrieval success
        if "min_pending_edges" in cond:
            min_pend = cond["min_pending_edges"]
            min_active_overlap = cond.get("min_active_overlap", 1)
            # Find entities with many pending edges and overlapping active edges
            rows = db.execute(text("""
                SELECT e.id FROM codex_entities e
                JOIN codex_edges pe ON pe.source_id = e.id AND pe.confidence = 'pending'
                JOIN codex_edges ae ON ae.source_id = e.id AND ae.target_id = pe.target_id
                    AND ae.confidence = 'active' AND ae.relation = pe.relation AND ae.valid_until IS NULL
                GROUP BY e.id
                HAVING COUNT(DISTINCT pe.id) > :min_pen
                   AND COUNT(DISTINCT ae.id) > :min_act
                LIMIT 1
            """), {"min_pen": min_pend, "min_act": min_active_overlap}).fetchall()
            return len(rows) > 0

    elif ttype == "frequency":
        # Not yet implemented
        pass

    elif ttype == "absence":
        if "table" in cond and cond["table"] == "memory_slots":
            # Check if pending_items content hasn't been modified in max_age_days
            max_age = cond.get("max_age_days", 14)
            row = db.execute(text("""
                SELECT 1 FROM memory_slots
                WHERE slot_name = 'pending_items'
                  AND content IS NOT NULL AND content != ''
                  AND last_updated < :cutoff
                LIMIT 1
            """), {"cutoff": datetime.now(timezone.utc) - timedelta(days=max_age)}).first()
            return row is not None

    return False


def _execute_action(rule, db):
    action = rule.action_type
    payload = rule.action_payload or {}

    if action == "log_event":
        pass  # already logged
    elif action == "notify":
        # Write to a notifications table or just log (for now, log)
        logger.info("sentinel_notify", rule=rule.name, message=payload.get("message", ""))
    elif action == "schedule_worker":
        worker = payload.get("worker")
        if worker:
            # Dynamically call the Celery task (lazy import)
            import importlib
            module_name, task_name = worker.rsplit(".", 1)
            mod = importlib.import_module(module_name)
            task = getattr(mod, task_name)
            task.delay()
    elif action == "create_review_item":
        db.execute(
            text("INSERT INTO review_queue (item_type, item_content) VALUES ('sentinel_review', :payload)"),
            {"payload": json.dumps({"rule": rule.name})}
        )