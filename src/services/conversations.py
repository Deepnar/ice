"""Conversation-deletion service (C10) — manifest-first cascade + /forget.

`delete_conversation` is the ONE deletion path (REST DELETE, `ice_control
delete_conversation`, and C11's `/delete-conversation` all call it). It is
manifest-first: phase A is pure reads that build the per-store manifest and
the exact work lists; phase B applies them in FK-safe order inside the one
open transaction. `dry_run=True` runs phase A only — the confirmation dialog
and the C11 confirm step render the identical manifest a real run would.

Cascade semantics (spec D2): codex facts corroborated by OTHER conversations
survive; sole-support edges are expired-with-journal (`edge_expired`, reason
`source_deleted` — excluded from T4 timelines: deletion is not evolution),
never hard-deleted. Everything conversation-owned (turns, chunks, links, cold
rows, summaries, replays, curated labels, conversation-tier slots, and the
G32/a1 relation-gap rows — which hold raw subject/object text, so a cascade
that skipped them left content behind) is
deleted; procedural patterns are pruned; live decisions extracted from the
conversation expire bi-temporally; pending review items referencing its
batches go `stale`. Logs are NOT touched — G25 owns redaction, and the
manifest's caveat string says so (honesty over pretense).

`propose_forget`/`apply_forget` are the /forget pair: fuzzy + destructive ⇒
review-queue proposal (`forget_request`), applied only on approval.
"""
import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import or_, text
from sqlalchemy.orm import Session

from src.memory.models import (
    BatchSummary,
    CodexEdge,
    CodexEntity,
    CodexEvent,
    CodexRelationGap,
    ColdStorage,
    ContextCluster,
    Conversation,
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


def _external_evidence(db: Session, edge_ids: list, batches: set) -> set:
    """Edge ids (str) corroborated by at least one `edge_added` event whose
    batch_source lies OUTSIDE the deleted conversation's batches — the fact
    stands on other conversations (spec rev 2: raw event counts would let a
    double-extraction inside the deleted conversation immunize its own edge).
    """
    if not edge_ids:
        return set()
    rows = db.execute(text("""
        SELECT DISTINCT payload->>'edge_id' AS eid
        FROM codex_events
        WHERE event_type = 'edge_added'
          AND payload->>'edge_id' = ANY(:ids)
          AND NOT (batch_source = ANY(:batches))
    """), {"ids": [str(i) for i in edge_ids],
           "batches": [str(b) for b in batches]}).fetchall()
    return {r.eid for r in rows}


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
    from src.workers.codex_extractor import (  # lazy: embedder at import
        _expire_edge,
        _regenerate_context_payload,
    )
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

    # Codex: live conversation-source edges born from this conversation.
    edges = []
    if batches:
        edges = db.query(CodexEdge).filter(
            CodexEdge.source_batch.in_(batches),
            CodexEdge.valid_until.is_(None),
            CodexEdge.source == "conversation",
        ).all()
    corroborated = _external_evidence(db, [e.id for e in edges], batches)
    sole_edges = [e for e in edges if str(e.id) not in corroborated]
    kept_edges = len(edges) - len(sole_edges)
    sole_ids = {e.id for e in sole_edges}

    # Entities the sole-support expiries may orphan: conversation-source,
    # zero live edges left after ours expire, no user-authored description.
    touched_entity_ids = ({e.source_id for e in sole_edges}
                          | {e.target_id for e in sole_edges})
    entities_to_expire: list = []
    surviving_touched: list = []
    if touched_entity_ids:
        user_authored = _user_authored_entities(db, list(touched_entity_ids))
        for ent in db.query(CodexEntity).filter(
                CodexEntity.id.in_(touched_entity_ids)).all():
            remaining = db.query(CodexEdge).filter(
                or_(CodexEdge.source_id == ent.id,
                    CodexEdge.target_id == ent.id),
                CodexEdge.valid_until.is_(None),
                ~CodexEdge.id.in_(sole_ids),
            ).count()
            if (remaining == 0 and ent.source == "conversation"
                    and str(ent.id) not in user_authored):
                entities_to_expire.append(ent)
            else:
                surviving_touched.append(ent)

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
    n_replays = db.query(SessionReplay).filter_by(
        conversation_id=conv_uuid).count()
    n_session_summaries = db.query(SessionSummary).filter_by(
        conversation_id=conv_uuid).count()
    n_conv_slots = db.query(MemorySlot).filter_by(
        conversation_id=conv_uuid).count()
    # G32/a1's gap ledger holds the SUBJECT and OBJECT of facts the vocabulary
    # could not express — real content from this conversation's turns. The table
    # arrived (2026-08-04) after this cascade was written (2026-07-19) and has
    # no FK, so deletion silently left it behind: "forget this conversation"
    # kept the model's wording plus both entity names. Matched on either
    # provenance column, because batch_id is nullable on rows written by paths
    # that had no batch.
    gap_filter = or_(CodexRelationGap.conversation_id == conv_uuid,
                     CodexRelationGap.batch_id.in_(batches)) if batches \
        else (CodexRelationGap.conversation_id == conv_uuid)
    n_relation_gaps = db.query(CodexRelationGap).filter(gap_filter).count()
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
                ["codex_reconciliation", "decision_supersession",
                 "forget_request"])).all():
        content = item.item_content or {}
        if item.item_type == "codex_reconciliation" and batches:
            ref = content.get("old_edge_id")
            edge = db.query(CodexEdge).filter_by(
                id=uuid.UUID(ref)).first() if ref else None
            if edge is not None and edge.source_batch in batches:
                stale_items.append(item)
        elif item.item_type == "decision_supersession" and batches:
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
            "session_replays": n_replays,
            "session_summaries": n_session_summaries,
            "conversation_slots": n_conv_slots,
            "curated_labels": n_curated,
        },
        "codex": {
            "edges_expired": len(sole_edges),
            "edges_kept_corroborated": kept_edges,
            "entities_expired": len(entities_to_expire),
            "relation_gaps_deleted": n_relation_gaps,
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
                    turns=len(turn_ids), edges_expired=len(sole_edges))
        return manifest

    # ── Phase B: apply, FK-safe order, one commit ──
    deletion_batch = uuid.uuid4()
    for e in sole_edges:
        _expire_edge(db, e.id, deletion_batch, "source_deleted",
                     source="user_deletion")
    for ent in entities_to_expire:
        _expire_entity(db, ent, deletion_batch)
    for ent in surviving_touched:
        _regenerate_context_payload(ent, db)
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
    # Deleted, not expired: a gap row is not a fact the graph holds, so there is
    # no bi-temporal history to preserve — it is raw extracted content from a
    # turn the user asked to be forgotten. Same `gap_filter` phase A counted, so
    # the manifest and the delete cannot disagree.
    db.query(CodexRelationGap).filter(gap_filter).delete(
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
                turns=len(turn_ids), edges_expired=len(sole_edges),
                edges_kept=kept_edges,
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
    (embedding, visibility-guarded) and live edges (entity-name containment)
    listed for approval. NEVER applies anything — the day a fuzzy match eats
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

    ent_rows = db.execute(text("""
        SELECT id, canonical_name FROM codex_entities
        WHERE length(canonical_name) >= 3
          AND source = 'conversation'
          AND :q ILIKE '%' || canonical_name || '%'
        LIMIT 10
    """), {"q": query}).fetchall()
    edges: list = []
    if ent_rows:
        ent_ids = [r.id for r in ent_rows]
        names = {r.id: r.canonical_name for r in ent_rows}
        for e in db.query(CodexEdge).filter(
                or_(CodexEdge.source_id.in_(ent_ids),
                    CodexEdge.target_id.in_(ent_ids)),
                CodexEdge.valid_until.is_(None),
                CodexEdge.source == "conversation").limit(limit).all():
            src = names.get(e.source_id) or _entity_name(db, e.source_id)
            tgt = names.get(e.target_id) or _entity_name(db, e.target_id)
            rel = f"NOT {e.relation}" if e.negated else e.relation
            edges.append({"id": str(e.id),
                          "description": f"{src} --{rel}--> {tgt}"})

    if not turns and not edges:
        logger.info("forget_no_matches", query_words=len(query.split()))
        return {"queued": False, "matches": 0, "turns": [], "edges": []}

    item = ReviewQueue(item_type="forget_request", item_content={
        "query": query,
        "conversation_id": str(conv_id),
        "turns": turns,
        "edges": edges,
        "proposed_by": proposed_by,
        "suggested_action": (
            f"forget: delete {len(turns)} turn(s) and expire "
            f"{len(edges)} edge(s) matching the request"),
    })
    db.add(item)
    db.flush()
    out = {"queued": True, "item_id": str(item.id),
           "matches": len(turns) + len(edges), "turns": turns, "edges": edges}
    db.commit()
    logger.info("forget_proposed", item_id=out["item_id"],
                turns=len(turns), edges=len(edges), proposed_by=proposed_by)
    return out


def _entity_name(db: Session, entity_id) -> str:
    ent = db.query(CodexEntity).filter_by(id=entity_id).first()
    return ent.canonical_name if ent else "?"


def apply_forget(db: Session, item_content: dict) -> dict:
    """The review-approve arm for `forget_request`: delete the listed turns
    (links first, chunks CASCADE, curated labels + cold rows by provenance)
    and expire the listed edges (journaled, reason `user_forget`). Tolerates
    rows that vanished since the proposal (bulk deletes no-op on missing ids).
    The caller (review service) owns the commit."""
    from src.workers.codex_extractor import (  # lazy: embedder at import
        _expire_edge,
        _regenerate_context_payload,
    )
    turn_ids = [uuid.UUID(t["id"]) for t in item_content.get("turns", [])]
    edge_ids = [uuid.UUID(e["id"]) for e in item_content.get("edges", [])]
    source = item_content.get("proposed_by", "chat_command")

    deleted_turns = 0
    if turn_ids:
        batch_rows = db.query(EpisodicMemory.batch_id).filter(
            EpisodicMemory.id.in_(turn_ids)).all()
        batches = {r.batch_id for r in batch_rows}
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
        db.query(ColdStorage).filter(ColdStorage.id.in_(turn_ids)).delete(
            synchronize_session=False)

    forget_batch = uuid.uuid4()
    expired = 0
    touched: set = set()
    for eid in edge_ids:
        edge = db.query(CodexEdge).filter_by(id=eid).first()
        if edge is not None and edge.valid_until is None:
            _expire_edge(db, eid, forget_batch, "user_forget", source=source)
            expired += 1
            touched |= {edge.source_id, edge.target_id}
    if touched:
        for ent in db.query(CodexEntity).filter(
                CodexEntity.id.in_(touched)).all():
            _regenerate_context_payload(ent, db)

    logger.info("forget_applied", turns_deleted=deleted_turns,
                edges_expired=expired, source=source)
    return {"turns_deleted": deleted_turns, "edges_expired": expired}
