"""T4: evolution surfacing — git log & diff for ideas.

The storage layer already records every supersession (bi-temporal codex_edges
+ the codex_events journal); this module is the porcelain that reads them.
T3's valid_at(T) filtering is `checkout`; `build_entity_timeline` is `log`,
`entity_diff` is `diff`, and `log_description_update` (D13) journals the one
mutable-in-place field so note bodies never lose history either.

The D6 discriminator lives here and nowhere else: an expired edge counts as
*history* only when its expiry is event-backed (a real supersession — A6
reconciliation, A8 negation, property update) and the reason is not
`source_deleted` (C10: deletion is not evolution). An eventless expiry is
codex_decay forgetting — a faded idea is not a revised idea, so it never
enters a timeline.

DB-reading only, no LLM, free of orchestrator state (db + entity in, data
out) — E0 wraps these as services and D1's agent can drive them unmodified.
Do NOT add REST endpoints here (E0's job).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from src.api.config import settings
from src.memory.models import CodexEdge, CodexEntity, CodexEvent

logger = structlog.get_logger("ice.retrieval.evolution")

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _expiry_events(db: Session, edge_ids) -> dict:
    """edge_id (str) → (event_timestamp, reason) for event-backed expiries,
    one query for all candidates. Applies the D6/C10 discriminator: eventless
    expiries simply don't appear, and `source_deleted` expiries are dropped
    so every consumer (timeline, diff, gate) shares the same rule."""
    if not edge_ids:
        return {}
    rows = db.execute(text("""
        SELECT payload->>'edge_id' AS edge_id,
               timestamp,
               payload->>'reason'  AS reason
        FROM codex_events
        WHERE event_type = 'edge_expired'
          AND payload->>'edge_id' = ANY(:ids)
    """), {"ids": [str(i) for i in edge_ids]}).fetchall()
    return {r.edge_id: (r.timestamp, r.reason or "updated")
            for r in rows if r.reason != "source_deleted"}


def history_exists(db: Session, entity_id) -> bool:
    """Cheap gate for the timeline wiring: does this entity carry any real
    supersession history — an expired edge whose expiry is event-backed and
    not a deletion?"""
    row = db.execute(text("""
        SELECT 1
        FROM codex_edges e
        JOIN codex_events ev
          ON ev.event_type = 'edge_expired'
         AND ev.payload->>'edge_id' = e.id::text
        WHERE (e.source_id = :eid OR e.target_id = :eid)
          AND e.valid_until IS NOT NULL
          AND COALESCE(ev.payload->>'reason', '') != 'source_deleted'
        LIMIT 1
    """), {"eid": str(entity_id)}).fetchone()
    return row is not None


def build_entity_timeline(db: Session, entity, allowed_batch_ids=None,
                          t0=None, t1=None, max_transitions=8) -> Optional[str]:
    """Render an entity's supersession history as a compact dated block:

        [Timeline: saga 2 ending]
        2025-11 – 2026-02: saga 2 ending --planned_as--> multiverse destruction  (superseded: antonym_superseded)
        2026-02 – now: saga 2 ending --planned_as--> pact mercy killing

    Live edges always belong; expired edges only when event-backed (D6).
    Returns None when there is no history to tell (no event-backed expired
    member) — a timeline of purely-current facts would just repeat the note.
    Month-precision dates (day precision implies false exactness); negated
    edges render NOT <relation>; the expiry reason comes from the event
    payload (fallback "updated"). A window bounds the story: candidates must
    be established by t1 and still alive at t0 (span overlap). Output keeps
    the most recent `max_transitions` lines and is word-trimmed to
    `settings.timeline_max_tokens`; truncation is announced, never silent.
    """
    until = t1 or datetime.now(timezone.utc)
    q = db.query(CodexEdge).filter(
        or_(CodexEdge.source_id == entity.id, CodexEdge.target_id == entity.id),
        or_(CodexEdge.valid_from.is_(None), CodexEdge.valid_from <= until),
    )
    if allowed_batch_ids is not None:
        q = q.filter(CodexEdge.source_batch.in_(allowed_batch_ids))
    if t0 is not None:
        q = q.filter(or_(CodexEdge.valid_until.is_(None), CodexEdge.valid_until >= t0))
    candidates = q.all()
    if not candidates:
        return None

    events = _expiry_events(db, [e.id for e in candidates if e.valid_until is not None])
    members = [e for e in candidates
               if e.valid_until is None or str(e.id) in events]
    # No event-backed expired member = no supersession history (D-U2: real
    # history only) — the anchor's live edges are already in its codex note.
    if not any(e.valid_until is not None for e in members):
        return None

    ent_ids = {e.source_id for e in members} | {e.target_id for e in members}
    names = {en.id: en.canonical_name for en in
             db.query(CodexEntity).filter(CodexEntity.id.in_(ent_ids)).all()}

    # Group by (relation, direction): consecutive members of a group form a
    # supersession chain. Groups ordered by their newest member, lines within
    # a group oldest→newest, so the block reads chronologically while chains
    # stay adjacent.
    groups: dict = {}
    for e in members:
        direction = "out" if e.source_id == entity.id else "in"
        groups.setdefault((e.relation, direction), []).append(e)

    lines = []
    for key in sorted(groups, key=lambda k: max((e.valid_from or _EPOCH)
                                                for e in groups[k])):
        for e in sorted(groups[key], key=lambda e: e.valid_from or _EPOCH):
            start = e.valid_from.strftime("%Y-%m") if e.valid_from else "?"
            if e.valid_until is not None:
                end = e.valid_until.strftime("%Y-%m")
                suffix = f"  (superseded: {events[str(e.id)][1]})"
            else:
                end = "now"
                suffix = ""
            rel = f"NOT {e.relation}" if e.negated else e.relation
            src = names.get(e.source_id, "?")
            tgt = names.get(e.target_id, "?")
            lines.append(f"{start} – {end}: {src} --{rel}--> {tgt}{suffix}")

    omitted = False
    if len(lines) > max_transitions:
        lines = lines[-max_transitions:]
        omitted = True
    header = f"[Timeline: {entity.canonical_name}]"

    def _render():
        parts = [header] + (["(earlier history omitted)"] if omitted else []) + lines
        return "\n".join(parts)

    # Token cap (D7: a life story must not eat the window) — drop the oldest
    # remaining lines before ever cutting mid-line.
    while len(lines) > 1 and len(_render().split()) * 1.33 > settings.timeline_max_tokens:
        lines.pop(0)
        omitted = True
    return _render()


def entity_diff(db: Session, entity, t0, t1) -> dict:
    """What changed for this entity inside [t0, t1): edges established in the
    window → "added" ("retracted" when negated — an A8 negation is the
    retraction of a fact, not a new link); event-backed expiries whose event
    falls in the window → "expired". Items carry raw datetimes (F3's UI
    formats itself) and the other endpoint's canonical name. Not wired to any
    endpoint here — the E0 service candidate and F3's timeline-overlay
    backend; exercised by tests until then."""
    edges = db.query(CodexEdge).filter(
        or_(CodexEdge.source_id == entity.id, CodexEdge.target_id == entity.id)
    ).all()
    ent_ids = {e.source_id for e in edges} | {e.target_id for e in edges}
    names = {en.id: en.canonical_name for en in
             db.query(CodexEntity).filter(CodexEntity.id.in_(ent_ids)).all()} if ent_ids else {}

    def _other(e):
        oid = e.target_id if e.source_id == entity.id else e.source_id
        return names.get(oid, "?")

    added, retracted, expired = [], [], []
    for e in edges:
        if e.valid_from is not None and t0 <= e.valid_from < t1:
            item = {"relation": e.relation, "other": _other(e), "date": e.valid_from}
            (retracted if e.negated else added).append(item)

    events = _expiry_events(db, [e.id for e in edges if e.valid_until is not None])
    for e in edges:
        info = events.get(str(e.id))
        if info and t0 <= info[0] < t1:
            expired.append({"relation": e.relation, "other": _other(e),
                            "date": info[0], "reason": info[1]})
    return {"added": added, "expired": expired, "retracted": retracted}


def log_description_update(db: Session, entity, old, new, source) -> None:
    """D13 never-overwrite: every `entity.description` write site calls this
    so the one mutable-in-place field joins the journal. Truncated snippets
    keep the event small; the caller owns the commit."""
    db.add(CodexEvent(
        entity_id=entity.id,
        event_type="description_updated",
        payload={"old": (old or "")[:300], "new": (new or "")[:300],
                 "source": source},
        batch_source=uuid.uuid4(),
    ))
    logger.info("description_updated", entity=entity.canonical_name, source=source)
