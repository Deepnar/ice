"""Sentinel Monitor – evaluates declarative rules and fires actions."""

import structlog
from datetime import datetime, timezone
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

            # Evaluate trigger_conditions against current database state
            if _evaluate_rule(rule, db):
                # Fire the action – currently only log_event is implemented
                event = SentinelEvent(
                    rule_id=rule.id,
                    fired_at=now,
                    trigger_state={},
                    action_taken=rule.action_type
                )
                db.add(event)
                rule.last_fired_at = now
                logger.info("sentinel_fired", rule_name=rule.name, action=rule.action_type)

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("sentinel_monitor_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


def _evaluate_rule(rule, db) -> bool:
    """Placeholder for rule evaluation. Real implementation would parse trigger_conditions."""
    return False  # no rules fire by default until conditions are populated