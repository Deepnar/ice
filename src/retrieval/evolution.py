"""T4: evolution surfacing — git log & diff for ideas, over the CODEX_V4 facts.

Every replacement is recorded on the fact itself (valid_until + superseded_by,
never rewritten) and journaled in codex_events, so `entity_diff` is `diff` and
the entity view's replaced facts are `log`. `log_description_update` (D13)
journals the one mutable-in-place entity field so note bodies never lose
history either. Expiry by deletion (C10) is not evolution, and a fact whose
sources were all deleted carries deleted text: neither function shows it.

DB-reading only, no LLM, free of orchestrator state (db + entity in, data
out) — E0 wraps these as services. Do NOT add REST endpoints here (E0's job).
The v2/v3 edge timeline this module used to build is described in
docs/outdated/ICE_Architecture_section4_codex_v2_v3.md.
"""

import uuid

import structlog
from sqlalchemy.orm import Session

from src.memory.facts import has_source
from src.memory.models import CodexEvent, CodexFact, CodexFactEntity

logger = structlog.get_logger("ice.retrieval.evolution")


def entity_diff(db: Session, entity, t0, t1) -> dict:
    """What changed for this entity inside [t0, t1): facts first stated in the
    window → "added"; facts that stopped being current in the window →
    "expired" (with the replacing fact's sentence when one replaced it).
    "retracted" stays in the shape for callers; facts carry negation in their
    sentence, so it is empty. Items carry raw datetimes."""
    rows = (db.query(CodexFact)
            .join(CodexFactEntity, CodexFactEntity.fact_id == CodexFact.id)
            .filter(CodexFactEntity.entity_id == entity.id, has_source()).distinct().all())
    added, expired = [], []
    for fact in rows:
        if fact.mentioned_at is not None and t0 <= fact.mentioned_at < t1:
            added.append({"fact": fact.text, "speaker": fact.speaker, "date": fact.mentioned_at})
        if fact.valid_until is not None and t0 <= fact.valid_until < t1:
            successor = db.get(CodexFact, fact.superseded_by) if fact.superseded_by else None
            expired.append({"fact": fact.text, "date": fact.valid_until,
                            "replaced_by": successor.text if successor else None})
    return {"added": added, "expired": expired, "retracted": []}


def log_description_update(db: Session, entity, old, new, source,
                           batch_source=None) -> None:
    """D13 never-overwrite: every `entity.description` write site calls this
    so the one mutable-in-place field joins the journal. Truncated snippets
    keep the event small; the caller owns the commit. *batch_source* lets
    agent writes stamp their agent_run_id (D1/D2 D4); default stays a fresh
    batch for standalone edits."""
    db.add(CodexEvent(
        entity_id=entity.id,
        event_type="description_updated",
        payload={"old": (old or "")[:300], "new": (new or "")[:300],
                 "source": source},
        batch_source=batch_source or uuid.uuid4(),
    ))
    logger.info("description_updated", entity=entity.canonical_name, source=source)
