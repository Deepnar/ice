"""Conversation-deletion service (C10) — manifest-first cascade + /forget.

`delete_conversation` is the ONE deletion path (REST DELETE, `ice_control
delete_conversation`, and C11's `/delete-conversation` all call it). It is
manifest-first: phase A is pure reads that build the per-store manifest and
the exact work lists; phase B applies them in FK-safe order inside the one
open transaction. `dry_run=True` runs phase A only — the confirmation dialog
and the C11 confirm step render the identical manifest a real run would.

Cascade semantics (spec D2, CODEX_V4 D10): codex facts corroborated by OTHER
conversations survive with their support recounted; a fact left with no source
is expired-with-journal (`fact_expired`, reason `source_deleted`), never
hard-deleted, and the deleted turns' quotes are removed. Everything
conversation-owned (turns, chunks, links, cold rows, summaries, replays,
curated labels, conversation-tier slots) is deleted; procedural patterns are pruned; live decisions extracted from the
conversation expire bi-temporally; pending review items referencing its
batches go `stale`. Logs are NOT touched — G25 owns redaction, and the
manifest's caveat string says so (honesty over pretense).

`propose_forget`/`apply_forget` are the /forget pair: fuzzy + destructive ⇒
review-queue proposal (`forget_request`), applied only on approval.
"""
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.memory import facts as fact_store
from src.memory.models import (
    BatchSummary,
    CodexEntity,
    CodexEvent,
    CodexFact,
    CodexFactEntity,
    CodexFactSource,
    ColdStorage,
    ContextCluster,
    Conversation,
    ConversationNote,
    ConversationSummary,
    CuratedLabel,
    Decision,
    EpisodicClusterLink,
    EpisodicMemory,
    MemorySlot,
    ProceduralMemory,
    ReviewQueue,
    SessionReplay,
    SessionSummary,
)
from src.services.errors import ConflictError, NotFoundError, ValidationError

logger = structlog.get_logger("ice.services.conversations")

# G25 restated in every manifest so the user-facing claim never overpromises.
LOGS_CAVEAT = (
    "Logs are NOT touched: proxy/worker log files may retain excerpts of this "
    "conversation until rotation or G25 log redaction. Deletion covers the "
    "memory stores listed in this manifest."
)


def _user_authored_entities(db: Session, entity_ids: list) -> set:
    """Entity ids (str) whose description was ever edited by a human surface
    (mcp_edit / manual_edit journal events) — deletion never eats a note body
    the user wrote themselves."""
    if not entity_ids:
        return set()
    rows = db.execute(text("""
        SELECT DISTINCT entity_id::text AS eid
        FROM codex_events
        WHERE event_type = 'description_updated'
          AND entity_id = ANY(:ids)
          AND payload->>'source' IN ('mcp_edit', 'manual_edit')
    """), {"ids": [str(i) for i in entity_ids]}).fetchall()
    return {r.eid for r in rows}


def _expire_entity(db: Session, entity: CodexEntity, batch_id) -> None:
    """Husk an entity the deleted conversation solely supported (no liveness
    column exists — mirror of D1/D2's merge-husk): unmatchable canonical name,
    emptied aliases, NULLed embedding, journaled. Row kept — events reference
    it and the audit trail must survive."""
    old_name = entity.canonical_name
    marker = f" [deleted:{str(entity.id)[:8]}]"
    props = dict(entity.properties or {})
    props["deleted_reason"] = "source_deleted"
    props["deleted_at"] = datetime.now(timezone.utc).isoformat()
    entity.canonical_name = old_name + marker
    entity.aliases = []
    entity.embedding = None
    entity.context_payload = "[deleted]"
    entity.properties = props
    db.add(CodexEvent(
        entity_id=entity.id,
        event_type="entity_expired",
        payload={"reason": "source_deleted", "name": old_name,
                 "source": "user_deletion"},
        timestamp=datetime.now(timezone.utc),
        batch_source=batch_id,
    ))


def delete_conversation(db: Session, conv_id: str, dry_run: bool = False) -> dict:
    """Delete a conversation with correct cascade semantics. Returns the
    manifest (identical for dry_run and real runs); one transaction, one
    commit at the end. Refused mid-generation (the live stream's post-flight
    would write into the void)."""
    from src.workers.runtime import get_runtime

    conv_uuid = uuid.UUID(str(conv_id))
    conv = db.query(Conversation).filter_by(id=conv_uuid).first()
    if not conv:
        raise NotFoundError("Conversation not found")
    runtime = get_runtime()
    if runtime is not None and runtime.generation_in_flight:
        raise ConflictError(
            "a chat generation is in progress — deletion refused; retry "
            "after the stream completes")

    # ── Phase A: pure reads — the manifest and the exact work lists ──
    turn_rows = db.query(EpisodicMemory.id, EpisodicMemory.batch_id).filter_by(
        conversation_id=conv_uuid).all()
    turn_ids = [r.id for r in turn_rows]
    batches = {r.batch_id for r in turn_rows}
    batches.update(r.batch_id for r in db.query(ColdStorage.batch_id).filter_by(
        conversation_id=conv_uuid).all() if r.batch_id is not None)

    # CODEX_V4 D10: sole support expires (journaled), corroborated facts keep
    # going on their other sources; secondary observations are pruned too.
    sole_facts, surviving_facts = fact_store.source_plan(db, batches)
    n_fact_sources = db.query(CodexFactSource).filter(
        CodexFactSource.source_batch.in_(batches)).count() if batches else 0

    # Entities the sole-support expiries may orphan: conversation-source, no
    # current fact left after ours expire, no user-authored description.
    touched_entity_ids = {r.entity_id for r in db.query(CodexFactEntity.entity_id).filter(
        CodexFactEntity.fact_id.in_(sole_facts)).all()}
    entities_to_expire: list = []
    if touched_entity_ids:
        user_authored = _user_authored_entities(db, list(touched_entity_ids))
        for ent in db.query(CodexEntity).filter(
                CodexEntity.id.in_(touched_entity_ids)).all():
            if (ent.source == "conversation"
                    and fact_store.live_fact_links(db, ent.id, excluding=sole_facts) == 0
                    and str(ent.id) not in user_authored):
                entities_to_expire.append(ent)

    # Episodic-side counts.
    n_chunks = 0
    n_links = 0
    if turn_ids:
        n_chunks = db.execute(text(
            "SELECT count(*) FROM episodic_chunks WHERE turn_id = ANY(:ids)"
        ), {"ids": [str(i) for i in turn_ids]}).scalar() or 0
        n_links = db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).count()
    n_cold = db.query(ColdStorage).filter_by(conversation_id=conv_uuid).count()

    # Clusters: empty = no member links and no legacy cluster_id refs outside
    # this conversation (both FK paths); surviving born-here clusters get
    # their conversation anchor NULLed so the row delete doesn't FK-fail.
    cand_cluster_ids = set()
    if turn_ids:
        cand_cluster_ids |= {r.cluster_id for r in db.query(
            EpisodicClusterLink.cluster_id).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).distinct().all()}
        cand_cluster_ids |= {r.cluster_id for r in db.query(
            EpisodicMemory.cluster_id).filter(
            EpisodicMemory.conversation_id == conv_uuid,
            EpisodicMemory.cluster_id.isnot(None)).distinct().all()}
    born_here = {r.id for r in db.query(ContextCluster.id).filter_by(
        conversation_id=conv_uuid).all()}
    cand_cluster_ids |= born_here
    empty_clusters: list = []
    detach_clusters: list = []
    for cid in cand_cluster_ids:
        link_q = db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.cluster_id == cid)
        if turn_ids:
            link_q = link_q.filter(
                ~EpisodicClusterLink.episodic_id.in_(turn_ids))
        other_links = link_q.count()
        other_refs = db.query(EpisodicMemory).filter(
            EpisodicMemory.cluster_id == cid,
            EpisodicMemory.conversation_id != conv_uuid).count()
        if other_links == 0 and other_refs == 0:
            empty_clusters.append(cid)
        elif cid in born_here:
            detach_clusters.append(cid)

    # Procedural: prune this conversation's batches out of the evidence sets.
    # (models.py uses the generic sqlalchemy ARRAY, which has no .overlap();
    # the && operator via raw SQL is the reliable form.)
    proc_rows = []
    if batches:
        proc_id_rows = db.execute(text("""
            SELECT id FROM procedural_memory
            WHERE source_batch_ids && CAST(:b AS uuid[])
        """), {"b": [str(x) for x in batches]}).fetchall()
        if proc_id_rows:
            proc_rows = db.query(ProceduralMemory).filter(
                ProceduralMemory.id.in_([r.id for r in proc_id_rows])).all()
    proc_deactivate = [p for p in proc_rows
                       if not [b for b in (p.source_batch_ids or [])
                               if b not in batches]]

    # Summaries / replays / training hygiene / decisions.
    n_batch_summaries = db.query(BatchSummary).filter_by(
        conversation_id=conv_uuid).count()
    n_conv_summaries = db.query(ConversationSummary).filter_by(
        conversation_id=conv_uuid).count()
    n_conv_notes = db.query(ConversationNote).filter_by(
        conversation_id=conv_uuid).count()
    n_replays = db.query(SessionReplay).filter_by(
        conversation_id=conv_uuid).count()
    n_session_summaries = db.query(SessionSummary).filter_by(
        conversation_id=conv_uuid).count()
    n_conv_slots = db.query(MemorySlot).filter_by(
        conversation_id=conv_uuid).count()
    n_curated = 0
    decisions_to_expire: list = []
    if batches:
        n_curated = db.query(CuratedLabel).filter(
            CuratedLabel.batch_id.in_(batches)).count()
        decisions_to_expire = db.query(Decision).filter(
            Decision.source_batch.in_(batches),
            Decision.valid_until.is_(None)).all()

    # Review queue: pending items referencing this conversation's content
    # (spec rev 6 — only three item types can).
    stale_items: list = []
    turn_id_strs = {str(t) for t in turn_ids}
    for item in db.query(ReviewQueue).filter(
            ReviewQueue.status == "pending",
            ReviewQueue.item_type.in_(
                ["decision_supersession", "forget_request"])).all():
        content = item.item_content or {}
        if item.item_type == "decision_supersession" and batches:
            ref_ids = [content.get("new_id"), content.get("old_id")]
            refs = [uuid.UUID(r) for r in ref_ids if r]
            if refs and db.query(Decision).filter(
                    Decision.id.in_(refs),
                    Decision.source_batch.in_(batches)).count():
                stale_items.append(item)
        elif item.item_type == "forget_request":
            listed = {t.get("id") for t in content.get("turns", [])}
            if (content.get("conversation_id") == str(conv_uuid)
                    or (listed & turn_id_strs)):
                stale_items.append(item)

    manifest = {
        "conversation_id": str(conv_uuid),
        "dry_run": dry_run,
        "batches": len(batches),
        "deleted": {
            "episodic_turns": len(turn_ids),
            "episodic_chunks": int(n_chunks),
            "cluster_links": n_links,
            "empty_clusters": len(empty_clusters),
            "clusters_detached": len(detach_clusters),
            "cold_storage_rows": n_cold,
            "batch_summaries": n_batch_summaries,
            "conversation_summaries": n_conv_summaries,
            "conversation_notes": n_conv_notes,
            "session_replays": n_replays,
            "session_summaries": n_session_summaries,
            "conversation_slots": n_conv_slots,
            "curated_labels": n_curated,
        },
        "codex": {
            "entities_expired": len(entities_to_expire),
            "facts_expired": len(sole_facts),
            "facts_support_pruned": len(surviving_facts),
            "fact_sources_deleted": int(n_fact_sources),
        },
        "decisions_expired": len(decisions_to_expire),
        "procedural_pruned": len(proc_rows),
        "procedural_deactivated": len(proc_deactivate),
        "review_items_staled": len(stale_items),
        "logs_caveat": LOGS_CAVEAT,
    }

    if dry_run:
        logger.info("conversation_delete_dry_run",
                    conversation_id=str(conv_uuid),
                    turns=len(turn_ids), facts_expired=len(sole_facts))
        return manifest

    # ── Phase B: apply, FK-safe order, one commit ──
    deletion_batch = uuid.uuid4()
    fact_store.apply_source_plan(db, sole_facts, surviving_facts, batches, deletion_batch)
    for ent in entities_to_expire:
        _expire_entity(db, ent, deletion_batch)
    db.flush()

    if turn_ids:
        db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).delete(
            synchronize_session=False)
        # Chunks CASCADE from the parent turn at the DB level (C2 migration).
        db.query(EpisodicMemory).filter(
            EpisodicMemory.conversation_id == conv_uuid).delete(
            synchronize_session=False)
    db.query(ColdStorage).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)

    for p in proc_rows:
        p.source_batch_ids = [b for b in (p.source_batch_ids or [])
                              if b not in batches]
        if not p.source_batch_ids:
            p.is_active = False

    db.query(BatchSummary).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)
    # Counted above, deleted explicitly — the DB CASCADE would fire on the
    # row delete anyway, but the manifest must never read a count it can't see.
    db.query(ConversationSummary).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)
    db.query(SessionReplay).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)
    db.query(SessionSummary).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)
    if batches:
        db.query(CuratedLabel).filter(
            CuratedLabel.batch_id.in_(batches)).delete(
            synchronize_session=False)
    for d in decisions_to_expire:
        d.valid_until = datetime.now(timezone.utc)
    for item in stale_items:
        item.status = "stale"
    db.query(MemorySlot).filter_by(conversation_id=conv_uuid).delete(
        synchronize_session=False)

    for cid in empty_clusters:
        db.query(ContextCluster).filter_by(id=cid).delete(
            synchronize_session=False)
    for cid in detach_clusters:
        db.query(ContextCluster).filter_by(id=cid).update(
            {"conversation_id": None}, synchronize_session=False)

    db.query(Conversation).filter_by(id=conv_uuid).delete(
        synchronize_session=False)
    db.commit()
    logger.info("conversation_deleted", conversation_id=str(conv_uuid),
                turns=len(turn_ids), facts_expired=len(sole_facts),
                facts_kept=len(surviving_facts),
                entities_expired=len(entities_to_expire))
    return manifest


# ── /forget: fuzzy + destructive ⇒ review-queue proposal (D4) ──────────────

def _forget_embedder():
    from src.api.core import get_core
    core = get_core()
    if core is None:
        raise RuntimeError(
            "ICE core not started — create_core() must run before /forget")
    return core.classifier.embedder


def propose_forget(db: Session, conv_id: str, query: str,
                   embedder=None, proposed_by: str = "chat_command",
                   limit: int = 5) -> dict:
    """Build a `forget_request` review-queue proposal: the top matching turns
    and current facts (embedding, visibility-guarded) listed for approval. NEVER applies anything — the day a fuzzy match eats
    the wrong memory is the day trust dies. Zero matches ⇒ nothing queued."""
    query = (query or "").strip()
    if not query:
        raise ValidationError("/forget needs the text to forget")
    if embedder is None:
        embedder = _forget_embedder()
    vec = embedder.encode(query, convert_to_tensor=False)
    emb = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    from pgvector.sqlalchemy import Vector as PgVector
    from sqlalchemy import bindparam
    # C9 lesson: the PgVector bindparam is load-bearing — a plain list binds
    # as double precision[] and the search dies silently.
    turn_rows = db.execute(text("""
        SELECT id, conversation_id, timestamp, raw_text, summary_text,
               1 - (embedding <=> :emb) AS sim
        FROM episodic_memory
        WHERE embedding IS NOT NULL
          AND (is_private = FALSE OR conversation_id = :conv)
        ORDER BY embedding <=> :emb
        LIMIT :lim
    """).bindparams(bindparam("emb", type_=PgVector)),
        {"emb": emb, "conv": str(conv_id), "lim": limit}).fetchall()
    turns = [
        {"id": str(r.id),
         "date": r.timestamp.strftime("%Y-%m-%d") if r.timestamp else None,
         "excerpt": (r.summary_text or r.raw_text or "")[:140],
         "similarity": round(float(r.sim), 3)}
        for r in turn_rows
    ]

    fact_rows = db.execute(text("""
        SELECT f.id, f.text, 1 - (f.embedding <=> :emb) AS sim
        FROM codex_facts f
        WHERE f.embedding IS NOT NULL AND f.valid_until IS NULL
          AND EXISTS (SELECT 1 FROM codex_fact_sources s
                      JOIN episodic_memory e ON e.id = s.episodic_id
                      WHERE s.fact_id = f.id
                        AND (e.is_private = FALSE OR e.conversation_id = :conv))
        ORDER BY f.embedding <=> :emb
        LIMIT :lim
    """).bindparams(bindparam("emb", type_=PgVector)),
        {"emb": emb, "conv": str(conv_id), "lim": limit}).fetchall()
    facts = [{"id": str(r.id), "description": r.text[:200],
              "similarity": round(float(r.sim), 3)} for r in fact_rows]

    if not turns and not facts:
        logger.info("forget_no_matches", query_words=len(query.split()))
        return {"queued": False, "matches": 0, "turns": [], "facts": []}

    item = ReviewQueue(item_type="forget_request", item_content={
        "query": query,
        "conversation_id": str(conv_id),
        "turns": turns,
        "facts": facts,
        "proposed_by": proposed_by,
        "suggested_action": (
            f"forget: delete {len(turns)} turn(s) and expire "
            f"{len(facts)} fact(s) matching the request"),
    })
    db.add(item)
    db.flush()
    out = {"queued": True, "item_id": str(item.id),
           "matches": len(turns) + len(facts), "turns": turns, "facts": facts}
    db.commit()
    logger.info("forget_proposed", item_id=out["item_id"],
                turns=len(turns), facts=len(facts), proposed_by=proposed_by)
    return out


def apply_forget(db: Session, item_content: dict) -> dict:
    """The review-approve arm for `forget_request`: delete the listed turns
    (links first, chunks CASCADE, curated labels + cold rows by provenance)
    and expire the listed facts (journaled, reason `user_forget`). Tolerates
    rows that vanished since the proposal (bulk deletes no-op on missing ids).
    The caller (review service) owns the commit."""
    turn_ids = [uuid.UUID(t["id"]) for t in item_content.get("turns", [])]
    fact_ids = [uuid.UUID(f["id"]) for f in item_content.get("facts", [])]
    source = item_content.get("proposed_by", "chat_command")

    deleted_turns = 0
    batches = set()
    sole_facts, surviving_facts = [], []
    if turn_ids:
        batch_rows = db.query(EpisodicMemory.batch_id).filter(
            EpisodicMemory.id.in_(turn_ids)).all()
        batches = {r.batch_id for r in batch_rows}
        batches.update(r.batch_id for r in db.query(ColdStorage.batch_id).filter(
            ColdStorage.id.in_(turn_ids)).all() if r.batch_id is not None)
        sole_facts, surviving_facts = fact_store.source_plan(db, batches)
        db.query(EpisodicClusterLink).filter(
            EpisodicClusterLink.episodic_id.in_(turn_ids)).delete(
            synchronize_session=False)
        deleted_turns = db.query(EpisodicMemory).filter(
            EpisodicMemory.id.in_(turn_ids)).delete(
            synchronize_session=False)
        if batches:
            db.query(CuratedLabel).filter(
                CuratedLabel.batch_id.in_(batches)).delete(
                synchronize_session=False)
        deleted_turns += db.query(ColdStorage).filter(
            ColdStorage.id.in_(turn_ids)).delete(synchronize_session=False)

    forget_batch = uuid.uuid4()
    fact_store.apply_source_plan(db, sole_facts, surviving_facts, batches,
                                 forget_batch, reason="user_forget")
    facts_expired = len(sole_facts)
    for fid in fact_ids:
        fact = db.get(CodexFact, fid)
        if fact is not None and fact.valid_until is None:
            fact_store.expire_fact(db, fact, forget_batch, "user_forget", source=source)
            facts_expired += 1

    logger.info("forget_applied", turns_deleted=deleted_turns,
                facts_expired=facts_expired, source=source)
    return {"turns_deleted": deleted_turns, "facts_expired": facts_expired}
