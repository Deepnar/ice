"""Codex surgery callables (the D1/D2 seam; future graph surgery lives here —
codex_extractor is big enough).

`merge_entities` is D5's real merge. It is dispatched by review-queue approval
(`src/services/review.py`, Tier-2 `entity_merge` proposals) and called directly
by the maintenance agent's Tier-0 path. Nothing is hard-deleted (T-track:
history must survive): the absorbed row stays as an expired "husk" whose
canonical_name is renamed out of the resolution space — `get_or_create_entity`
resolves canonical_name BEFORE aliases, so an unrenamed husk would keep
capturing future extractions — and whose original name lives on as an alias of
the keeper. Every mutation is journaled (`source: "maintenance_agent"`, G17;
`batch_source = agent_run_id`, D4).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, Tuple

import structlog
from sqlalchemy import or_
from sqlalchemy.orm.attributes import flag_modified

logger = structlog.get_logger("ice.workers.codex_ops")

SOURCE = "maintenance_agent"


def _run_batch(agent_run_id) -> Tuple[uuid.UUID, Optional[str]]:
    """batch_source is a UUID column; the run id normally is one, but tolerate
    arbitrary strings (they still land in event payloads as agent_run_id)."""
    if agent_run_id is None:
        return uuid.uuid4(), None
    try:
        return uuid.UUID(str(agent_run_id)), str(agent_run_id)
    except ValueError:
        return uuid.uuid4(), str(agent_run_id)


def merge_entities(db, keep_id, absorb_id, agent_run_id=None) -> dict:
    """Fold *absorb_id* into *keep_id* (spec D1_D2 D5). Owns its transaction
    (commits on success); returns a summary dict. Skips loudly (log + status)
    instead of raising when either entity is missing/already merged — the
    caller may be the review-approve endpoint acting on a stale proposal."""
    # Lazy imports: codex_extractor loads a SentenceTransformer at import time.
    from src.memory.models import CodexEdge, CodexEntity, CodexEvent
    from src.retrieval.evolution import log_description_update
    from src.workers.codex_extractor import _regenerate_context_payload

    batch, run_ref = _run_batch(agent_run_id)
    keep = db.query(CodexEntity).get(keep_id)
    absorb = db.query(CodexEntity).get(absorb_id)

    def _skip(reason: str) -> dict:
        logger.warning("agent_action", action="entity_merge", outcome="skipped",
                       reason=reason, keep_id=str(keep_id),
                       absorb_id=str(absorb_id), agent_run_id=run_ref)
        return {"status": "skipped", "reason": reason}

    if keep is None or absorb is None:
        return _skip("entity_missing")
    if keep.id == absorb.id:
        return _skip("same_entity")
    # Re-check both are still live inside this transaction (spec §4: a
    # concurrent run or an earlier approval may have merged one already).
    if (keep.properties or {}).get("merged_into"):
        return _skip("keep_already_merged")
    if (absorb.properties or {}).get("merged_into"):
        return _skip("absorb_already_merged")

    now = datetime.now(timezone.utc)
    absorbed_name = absorb.canonical_name

    # ── 1. Re-point absorb's edges (live AND expired — the keeper owns the
    # history; the events move with them below). A live edge that duplicates
    # an existing live keep-edge is expired (journaled), never deleted; the
    # survivor keeps max strength/confidence and the earlier valid_from (fact
    # lines render "(since …)" from it — the fact held since the earlier
    # assertion). Edges BETWEEN the pair would become self-loops: expired
    # in place, endpoints untouched (honest history beats a keep→keep loop).
    keep_live = {}
    for e in db.query(CodexEdge).filter(
            CodexEdge.valid_until == None,  # noqa: E711
            or_(CodexEdge.source_id == keep.id, CodexEdge.target_id == keep.id)).all():
        direction = "out" if e.source_id == keep.id else "in"
        other = e.target_id if direction == "out" else e.source_id
        keep_live[(e.relation, other, direction, bool(e.negated))] = e

    pair_ids = {keep.id, absorb.id}
    repointed = deduped = loops = 0
    absorb_edges = db.query(CodexEdge).filter(
        or_(CodexEdge.source_id == absorb.id,
            CodexEdge.target_id == absorb.id)).all()
    for e in absorb_edges:
        if e.source_id in pair_ids and e.target_id in pair_ids:
            if e.valid_until is None:
                e.valid_until = now
                db.add(CodexEvent(
                    entity_id=keep.id, event_type="edge_expired",
                    payload={"edge_id": str(e.id), "reason": "entity_merge_self_loop",
                             "source": SOURCE, "agent_run_id": run_ref},
                    timestamp=now, batch_source=batch))
                loops += 1
            continue
        if e.source_id == absorb.id:
            e.source_id = keep.id
            direction, other = "out", e.target_id
        else:
            e.target_id = keep.id
            direction, other = "in", e.source_id
        if e.valid_until is not None:
            repointed += 1
            continue
        key = (e.relation, other, direction, bool(e.negated))
        survivor = keep_live.get(key)
        if survivor is not None and survivor.id != e.id:
            survivor.strength = max(survivor.strength or 0.0, e.strength or 0.0)
            survivor.extraction_confidence = max(
                survivor.extraction_confidence or 0.0,
                e.extraction_confidence or 0.0)
            if e.confidence == "active":
                survivor.confidence = "active"
            if e.valid_from and survivor.valid_from:
                survivor.valid_from = min(survivor.valid_from, e.valid_from)
            e.valid_until = now
            db.add(CodexEvent(
                entity_id=keep.id, event_type="edge_expired",
                payload={"edge_id": str(e.id), "reason": "duplicate_merged",
                         "kept_edge_id": str(survivor.id),
                         "source": SOURCE, "agent_run_id": run_ref},
                timestamp=now, batch_source=batch))
            deduped += 1
        else:
            keep_live[key] = e
            repointed += 1

    # ── 2. Union aliases/tags/properties. absorb's canonical becomes an alias
    # of keep (already casefolded at creation — get_or_create's alias match is
    # exact on the lowercased name, so this is the form that must be present).
    keep.aliases = list(dict.fromkeys(
        list(keep.aliases or []) + list(absorb.aliases or []) + [absorbed_name]))
    keep.tags = list(dict.fromkeys(list(keep.tags or []) + list(absorb.tags or [])))
    if absorb.properties:
        merged_props = dict(keep.properties or {})
        for k, v in absorb.properties.items():
            merged_props.setdefault(k, v)
        keep.properties = merged_props
        flag_modified(keep, "properties")

    # ── 3. Keep the longer description (journaled — D13 never-overwrite).
    if len(absorb.description or "") > len(keep.description or ""):
        old_desc = keep.description
        keep.description = absorb.description
        log_description_update(db, keep, old_desc, absorb.description,
                               source=SOURCE, batch_source=batch)

    # ── 4. Move absorb's journal to keep (autoflush lands the events added
    # above first, so merge-time expiries move too — they are keep's history).
    events_moved = db.query(CodexEvent).filter(
        CodexEvent.entity_id == absorb.id).update(
        {CodexEvent.entity_id: keep.id}, synchronize_session=False)

    # ── 5. Expire the absorbed entity: rename it out of the resolution space
    # (marker suffix keeps the original readable), empty its aliases (they
    # live on the keeper now), stamp merged_into/merged_at. Row, embedding and
    # description are kept — no hard delete (T-track).
    absorb.canonical_name = f"{absorbed_name} [merged:{str(absorb.id)[:8]}]"
    absorb.aliases = []
    husk_props = dict(absorb.properties or {})
    husk_props["merged_into"] = str(keep.id)
    husk_props["merged_at"] = now.isoformat()
    absorb.properties = husk_props
    flag_modified(absorb, "properties")
    absorb.context_payload = f"[merged into {keep.canonical_name}]"
    absorb.last_updated = now

    # ── 6. Rebuild the keeper's note + one summarizing journal entry.
    _regenerate_context_payload(keep, db)
    keep.last_updated = now
    db.add(CodexEvent(
        entity_id=keep.id, event_type="entity_merged",
        payload={"absorbed_id": str(absorb.id), "absorbed_name": absorbed_name,
                 "edges_repointed": repointed, "edges_deduped": deduped,
                 "self_loops_expired": loops, "events_moved": int(events_moved or 0),
                 "source": SOURCE, "agent_run_id": run_ref},
        timestamp=now, batch_source=batch))

    db.commit()
    result = {"status": "merged", "kept_id": str(keep.id),
              "absorbed_id": str(absorb.id), "edges_repointed": repointed,
              "edges_deduped": deduped, "self_loops_expired": loops,
              "events_moved": int(events_moved or 0)}
    logger.info("agent_action", action="entity_merge", outcome="applied",
                kept=keep.canonical_name, absorbed=absorbed_name,
                agent_run_id=run_ref, **{k: v for k, v in result.items()
                                         if k not in ("status",)})
    return result
