"""Memory Maintenance Agent (D1) — deterministic worklist + bounded LLM decisions.

NOT a free-roaming tool loop (spec D1_D2 D1): cheap SQL detectors produce
typed work items; the background model decides per item from a FIXED enum
(JSON mode, temperature 0, unparseable ⇒ unsure ⇒ no write); execution goes
through named callables (A6's ``reconcile_conflict``, D5's ``merge_entities``).
A small model gets reliability from constrained choices, not open planning.

Three risk tiers govern write authority (D2):

  Tier 0 — deterministic, auto-applied: normalization-equal duplicate merges.
  Tier 1 — auto-applied + journaled: reconciliation leftovers re-decided with
           fuller context (LLM), pending-edge dedupe and contradiction
           resolution (deterministic, via the A6 machinery).
  Tier 2 — LLM-proposed, review-queued (≤5/run): near-duplicate entity merges,
           stale-slot rewrites. The queue stays the safety net.

Every run gets an ``agent_run_id``; every graph write journals CodexEvents
with ``source: "maintenance_agent"`` (G17) and ``batch_source = agent_run_id``
(D4). Runs are idempotent: detectors re-derive state from the DB, caps make
runs incremental (D8). Scheduled by the maintenance runtime (gpu lane,
overdue 12h + session-end burst).

E1 look-ahead: detectors are a pluggable registry keyed by item type — the
coding-side decisions table reuses this exact pattern.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.api.config import settings
import structlog
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified

logger = structlog.get_logger("ice.workers.maintenance_agent")

SOURCE = "maintenance_agent"

# Detector 5 — the sentinel absence rule, ported. A slot NAME, not a knob:
# it identifies the slot the detector watches, so it stays a constant.
STALE_SLOT_NAME = "pending_items"



@dataclass
class WorkItem:
    item_type: str      # detector registry key
    tier: int           # 0 | 1 | 2
    payload: dict = field(default_factory=dict)


# ═══ Detectors (D3) — deterministic SQL, run first, capped ═══════════════════

def _detect_reconciliation_leftovers(db, cap: int) -> list:
    """D3.1: pending codex_reconciliation items the in-line reconciler refused
    to guess at. The agent re-decides WITH context the in-line pass lacked
    (both edges' source turns). ≥settings.agent_max_attempts prior attempts ⇒ never
    retried (check 6)."""
    rows = db.execute(text("""
        SELECT id, item_content FROM review_queue
        WHERE item_type = 'codex_reconciliation' AND status = 'pending'
          AND COALESCE((item_content->>'agent_attempts')::int, 0) < :max_att
        ORDER BY created_at ASC
        LIMIT :cap
    """), {"max_att": settings.agent_max_attempts, "cap": cap}).fetchall()
    return [WorkItem("reconciliation_leftover", 1,
                     {"review_id": str(r.id), "content": dict(r.item_content or {})})
            for r in rows]


def _detect_pending_pileup(db, cap: int) -> list:
    """D3.2: the sentinel's one real query (JOIN/HAVING shape ported; LIMIT 1
    existence check replaced by the capped entity list, and pending edges
    filtered to live ones so already-fixed entities stop re-detecting)."""
    rows = db.execute(text("""
        SELECT e.id FROM codex_entities e
        JOIN codex_edges pe ON pe.source_id = e.id AND pe.confidence = 'pending'
            AND pe.valid_until IS NULL
        JOIN codex_edges ae ON ae.source_id = e.id AND ae.target_id = pe.target_id
            AND ae.confidence = 'active' AND ae.relation = pe.relation
            AND ae.valid_until IS NULL
        GROUP BY e.id
        HAVING COUNT(DISTINCT pe.id) > :min_pen
           AND COUNT(DISTINCT ae.id) > :min_act
        LIMIT :cap
    """), {"min_pen": settings.agent_pileup_min_pending, "min_act": settings.agent_pileup_min_active_overlap,
           "cap": cap}).fetchall()
    return [WorkItem("pending_pileup", 1, {"entity_id": str(r.id)}) for r in rows]


def _norm(name: str) -> str:
    return (name or "").strip().casefold()


def _pair_key(id_a, id_b) -> str:
    return "|".join(sorted([str(id_a), str(id_b)]))


def _detect_duplicate_entities(db, cap: int) -> list:
    """D3.3: candidate pairs = casefold name/alias intersection ∪ embedding
    cosine ≥ threshold ∧ same entity_type; ranked name-overlap first then by
    cosine; top settings.agent_dup_pairs_per_run. Normalization-equal ⇒ Tier 0 auto-merge;
    cosine-only pairs ALWAYS need the LLM verdict (Tier 2 — spec §4 guard).
    Pairs the user rejected, and pairs already proposed/decided, are skipped
    forever (pair_key stamped in item_content)."""
    cap = min(cap, settings.agent_dup_pairs_per_run)
    if cap <= 0:
        return []

    seen_keys = {r[0] for r in db.execute(text(
        "SELECT item_content->>'pair_key' FROM review_queue "
        "WHERE item_type = 'entity_merge' "
        "  AND item_content->>'pair_key' IS NOT NULL")).fetchall()}

    # Name-overlap channel (Tier 0 candidates). Canonical names are stored
    # casefolded, but aliases keep original case — normalize in Python; the
    # entity table is personal-scale (thousands, not millions).
    # E1b: derived entities (code graph / project facts) are regenerable —
    # merging them is wrong and the next re-parse would undo it anyway.
    ents = db.execute(text("""
        SELECT id, canonical_name, aliases, entity_type
        FROM codex_entities
        WHERE (properties->>'merged_into') IS NULL
          AND source = 'conversation'
    """)).fetchall()
    by_name: dict = {}
    for e in ents:
        for n in {_norm(e.canonical_name), *(_norm(a) for a in (e.aliases or []))}:
            if n:
                by_name.setdefault(n, []).append(e)
    name_pairs = {}
    for _, group in by_name.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.id != b.id:
                    name_pairs[_pair_key(a.id, b.id)] = (a.id, b.id)

    # Cosine channel (Tier 2 candidates); pgvector <=> is cosine distance.
    cos_rows = db.execute(text("""
        SELECT a.id AS a_id, b.id AS b_id,
               1 - (a.embedding <=> b.embedding) AS cos
        FROM codex_entities a
        JOIN codex_entities b ON a.id < b.id
         AND COALESCE(a.entity_type, 'entity') = COALESCE(b.entity_type, 'entity')
        WHERE a.embedding IS NOT NULL AND b.embedding IS NOT NULL
          AND a.source = 'conversation' AND b.source = 'conversation'
          AND (a.properties->>'merged_into') IS NULL
          AND (b.properties->>'merged_into') IS NULL
          AND 1 - (a.embedding <=> b.embedding) >= :thresh
        ORDER BY cos DESC
        LIMIT :cap
    """), {"thresh": settings.agent_dup_cosine_threshold, "cap": cap}).fetchall()

    ranked = [(key, ids, None) for key, ids in name_pairs.items()]
    ranked += [(_pair_key(r.a_id, r.b_id), (r.a_id, r.b_id), float(r.cos))
               for r in cos_rows]

    items, used = [], set()
    for key, (id_a, id_b), cos in ranked:
        if len(items) >= cap:
            break
        if key in seen_keys or key in used:
            continue
        used.add(key)
        keep_id, absorb_id = _merge_order(db, id_a, id_b)
        items.append(WorkItem(
            "duplicate_entities", 0 if cos is None else 2,
            {"keep_id": str(keep_id), "absorb_id": str(absorb_id),
             "pair_key": key, "cosine": cos}))
    return items


def _merge_order(db, id_a, id_b):
    """Keep = higher mention count (COUNT of codex_events — A7.3's
    definition); tie-break the older first event; then id for determinism."""
    rows = db.execute(text("""
        SELECT entity_id, COUNT(*) AS n, MIN(timestamp) AS first_ev
        FROM codex_events WHERE entity_id IN (:a, :b) GROUP BY entity_id
    """), {"a": id_a, "b": id_b}).fetchall()
    stats = {str(r.entity_id): (r.n, r.first_ev) for r in rows}
    far_future = datetime.max.replace(tzinfo=timezone.utc)

    def rank(eid):
        n, first = stats.get(str(eid), (0, None))
        return (-n, first or far_future, str(eid))

    keep, absorb = sorted([id_a, id_b], key=rank)
    return keep, absorb


def _detect_contradictions(db, cap: int) -> list:
    """D3.4: (a) a live positive edge coexisting with a live negated edge for
    the same (source, relation, target) — A8 residue the in-line retraction
    missed cross-batch; (b) two live antonym edges for one entity pair — A6
    cross-batch misses. Resolution is A6's deterministic newer-supersedes."""
    items = []
    rows = db.execute(text("""
        SELECT p.id AS pos_id, n.id AS neg_id, p.source_id, p.target_id,
               p.relation, p.valid_from AS pos_from, n.valid_from AS neg_from
        FROM codex_edges p
        JOIN codex_edges n ON n.source_id = p.source_id
         AND n.target_id = p.target_id AND n.relation = p.relation
         AND n.negated = TRUE AND n.valid_until IS NULL
        WHERE p.negated = FALSE AND p.valid_until IS NULL
        LIMIT :cap
    """), {"cap": cap}).fetchall()
    for r in rows:
        # Newer assertion wins; the negation retracts the positive on a tie
        # (that is what the in-line A8 path would have done).
        if r.pos_from and r.neg_from and r.pos_from > r.neg_from:
            old_id, old_rel, new_id, new_rel = r.neg_id, r.relation, r.pos_id, r.relation
        else:
            old_id, old_rel, new_id, new_rel = r.pos_id, r.relation, r.neg_id, r.relation
        items.append(WorkItem("contradiction", 1, {
            "kind": "polarity", "old_edge_id": str(old_id),
            "old_relation": old_rel, "new_edge_id": str(new_id),
            "new_relation": new_rel, "source_id": str(r.source_id),
            "target_id": str(r.target_id)}))

    if len(items) < cap:
        # Lazy: codex_extractor loads an embedder at import.
        from src.workers.codex_extractor import _ANTONYM_PAIRS
        for rel_a, rel_b in _ANTONYM_PAIRS:
            if len(items) >= cap:
                break
            rows = db.execute(text("""
                SELECT a.id AS a_id, b.id AS b_id, a.source_id, a.target_id,
                       a.valid_from AS a_from, b.valid_from AS b_from
                FROM codex_edges a
                JOIN codex_edges b ON a.source_id = b.source_id
                 AND a.target_id = b.target_id AND b.relation = :rel_b
                 AND b.valid_until IS NULL AND b.negated = FALSE
                WHERE a.relation = :rel_a AND a.valid_until IS NULL
                  AND a.negated = FALSE
                LIMIT :cap
            """), {"rel_a": rel_a, "rel_b": rel_b,
                   "cap": cap - len(items)}).fetchall()
            for r in rows:
                if r.a_from and r.b_from and r.a_from > r.b_from:
                    old_id, old_rel, new_id, new_rel = r.b_id, rel_b, r.a_id, rel_a
                else:
                    old_id, old_rel, new_id, new_rel = r.a_id, rel_a, r.b_id, rel_b
                items.append(WorkItem("contradiction", 1, {
                    "kind": "antonym", "old_edge_id": str(old_id),
                    "old_relation": old_rel, "new_edge_id": str(new_id),
                    "new_relation": new_rel, "source_id": str(r.source_id),
                    "target_id": str(r.target_id)}))
    return items


def _detect_stale_slot(db, cap: int) -> list:
    """D3.5: the sentinel absence rule, ported — the pending_items slot has
    content but hasn't been touched in settings.agent_stale_slot_days. Skipped while a
    proposal for the slot is already pending, or the user rejected one since
    the slot last changed (their "no" stands until the slot moves)."""
    if cap <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.agent_stale_slot_days)
    row = db.execute(text("""
        SELECT slot_name, last_updated FROM memory_slots
        WHERE slot_name = :slot
          AND content IS NOT NULL AND content != ''
          AND last_updated < :cutoff
        LIMIT 1
    """), {"slot": STALE_SLOT_NAME, "cutoff": cutoff}).first()
    if row is None:
        return []
    blocker = db.execute(text("""
        SELECT 1 FROM review_queue
        WHERE item_type = 'memory_slot_update'
          AND item_content->>'slot_name' = :slot
          AND (status = 'pending'
               OR (status = 'rejected' AND created_at > :since))
        LIMIT 1
    """), {"slot": STALE_SLOT_NAME, "since": row.last_updated}).first()
    if blocker is not None:
        return []
    return [WorkItem("stale_slot", 2, {
        "slot_name": row.slot_name,
        "last_updated": row.last_updated.isoformat() if row.last_updated else None})]




def _detect_stale_work(db, cap: int) -> list:
    """E3 (coding-core D5 step 4): project tasks untouched > settings.agent_stale_task_days
    become Tier-2 proposals — detection is deterministic (no LLM), the review
    queue is the notifier (no parallel notification system). Skipped while a
    proposal for the task is pending, or rejected since the task last moved
    (the user's "no" stands until the task moves). Payload carries branch +
    goal so drift is visible to the reviewer."""
    if cap <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.agent_stale_task_days)
    try:
        rows = db.execute(text("""
            SELECT t.id, t.title, t.status, t.updated_at, t.project_id,
                   p.slug, ps.current_branch, ps.goal
            FROM tasks t
            JOIN projects p ON p.id = t.project_id
            LEFT JOIN project_state ps ON ps.project_id = t.project_id
            WHERE t.status IN ('pending', 'active')
              AND t.updated_at < :cutoff
              AND NOT EXISTS (
                  SELECT 1 FROM review_queue rq
                  WHERE rq.item_type = 'stale_work'
                    AND rq.item_content->>'task_id' = t.id::text
                    AND (rq.status = 'pending'
                         OR (rq.status = 'rejected' AND rq.created_at > t.updated_at))
              )
            ORDER BY t.updated_at ASC
            LIMIT :cap
        """), {"cutoff": cutoff, "cap": cap}).fetchall()
    except Exception:
        db.rollback()
        return []
    now = datetime.now(timezone.utc)
    return [WorkItem("stale_work", 2, {
        "task_id": str(r.id), "title": r.title, "status": r.status,
        "project": r.slug, "project_id": str(r.project_id),
        "days_stale": int((now - r.updated_at).total_seconds() // 86400)
        if r.updated_at else None,
        "current_branch": r.current_branch, "goal": r.goal,
        "suggested_action": f'review stale task "{r.title}" '
                            f"({r.slug}) — finish, re-scope, or drop it",
    }) for r in rows]


# Registry (E1 look-ahead: item types are pluggable; spec order = priority).
DETECTORS: dict = {
    "reconciliation_leftover": _detect_reconciliation_leftovers,
    "pending_pileup": _detect_pending_pileup,
    "duplicate_entities": _detect_duplicate_entities,
    "contradiction": _detect_contradictions,
    "stale_slot": _detect_stale_slot,
    # E3: stale project work (coding core) — deterministic, Tier-2.
    "stale_work": _detect_stale_work,
}


def detect_all(db) -> list:
    items = []
    for name, fn in DETECTORS.items():
        remaining = settings.agent_max_scanned - len(items)
        if remaining <= 0:
            break
        try:
            found = fn(db, remaining)
        except Exception as exc:
            db.rollback()
            logger.error("agent_detector_failed", detector=name, error=str(exc))
            continue
        if found:
            logger.info("agent_detector", detector=name, items=len(found))
            items.extend(found)
    return items[:settings.agent_max_scanned]


# ═══ Decide (fixed enums; bg model in JSON mode; unparseable ⇒ unsure) ═══════

_USE_DEFAULT = object()


def make_llm_decider():
    """One bounded JSON completion on the background model: temperature 0,
    JSON response format, short output. Returns a dict or None (None and any
    error read as 'unsure' — the agent never acts on garbage)."""
    from src.workers.bg_client_factory import (
        bg_timeout,
        get_bg_client,
        get_bg_model_name,
        json_schema,
    )
    client = get_bg_client()

    def _decide(prompt: str, max_tokens: int = 200,
                schema: Optional[dict] = None) -> Optional[dict]:
        try:
            resp = client.chat.completions.create(
                model=get_bg_model_name(),
                messages=[{"role": "system",
                           "content": "You output exactly one JSON object and nothing else."},
                          {"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=max_tokens,
                # `{"type": "json_object"}` used to be here and it did NOTHING:
                # measured 2026-08-04, it scored 0/8 for conformance, exactly as
                # if no constraint had been sent, while `json_schema` scored 8/8
                # through the same endpoint. Both return HTTP 200, which is why
                # it went unnoticed. When the caller knows the answer's shape it
                # passes one; otherwise we send no constraint rather than a
                # decorative one.
                response_format=json_schema("agent_decision", schema) if schema else None,
                timeout=bg_timeout(max_tokens))
            raw = (resp.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except Exception as exc:
            logger.warning("agent_llm_failed", error=str(exc))
            return None

    return _decide


def _turn_for_batch(db, batch_id) -> str:
    if batch_id is None:
        return ""
    row = db.execute(text(
        "SELECT raw_text, summary_text FROM episodic_memory "
        "WHERE batch_id = :b LIMIT 1"), {"b": batch_id}).first()
    if row is None:
        return ""
    return (row.raw_text or row.summary_text or "")[:600]


def _ask_enum(llm_decider, prompt: str, key: str, allowed: tuple) -> str:
    """One decider call, answer forced into the enum; anything else ⇒ unsure.

    An enum is the right tool HERE, unlike the codex relation vocabulary: this
    is a closed decision set that already contains "unsure", so a model with no
    good answer has a legal way to say so. The relation vocabulary has no such
    escape and forcing a choice there produced ~78% wrong facts (PROVENANCE,
    2026-08-04) — the difference is whether declining is expressible.
    """
    options = tuple(allowed) + (() if "unsure" in allowed else ("unsure",))
    schema = {"type": "object",
              "properties": {key: {"type": "string", "enum": list(options)}},
              "required": [key]}
    try:
        parsed = llm_decider(prompt, schema=schema)
    except TypeError:
        # a caller-supplied stub decider (tests) may not accept the kwarg
        parsed = llm_decider(prompt)
    if not isinstance(parsed, dict):
        return "unsure"
    value = str(parsed.get(key, "")).strip().lower().replace(" ", "_").replace("-", "_")
    return value if value in allowed else "unsure"


def _decide_reconciliation(db, item: WorkItem, llm_decider) -> str:
    """D2 Tier 1: re-run the A6 supersession decision WITH the source turns of
    both edges (the in-line reconciler only saw the new turn)."""
    from src.memory.models import CodexEdge  # lazy for symmetry with callables

    content = item.payload["content"]
    old_edge = db.query(CodexEdge).get(uuid.UUID(content["old_edge_id"]))
    if old_edge is None or old_edge.valid_until is not None:
        return "already_resolved"          # someone else settled it
    new = content.get("new") or {}
    new_edge = _find_edge_by_names(db, new.get("subject"), new.get("relation"),
                                   new.get("object"))
    old_turn = _turn_for_batch(db, old_edge.source_batch)
    new_turn = _turn_for_batch(db, new_edge.source_batch) if new_edge is not None else ""
    prompt = (
        "Two stored facts about the same subject may conflict. Decide how to "
        "reconcile them using ONLY the conversation excerpts.\n"
        f"OLD fact: {new.get('subject', '?')} {content.get('old_relation', '?')} "
        f"{content.get('old_object', '?')}\n"
        f"  from conversation: {old_turn or content.get('turn_excerpt', '')}\n"
        f"NEW fact: {new.get('subject', '?')} {new.get('relation', '?')} {new.get('object', '?')}\n"
        f"  from conversation: {new_turn or content.get('turn_excerpt', '')}\n\n"
        'Reply with one JSON object: {"decision": "<value>"} where <value> is one of:\n'
        "expire_old — the new fact replaced/superseded the old one\n"
        "keep_both — both facts are true at the same time\n"
        "reject_new — the new fact is wrong or was never actually asserted\n"
        "unsure — the excerpts do not settle it"
    )
    return _ask_enum(llm_decider, prompt, "decision",
                     ("expire_old", "keep_both", "reject_new", "unsure"))


def _find_edge_by_names(db, subject, relation, obj):
    """Resolve the reconciliation item's stored triplet names back to the live
    edge (names are canonical — reconcile_conflict wrote them from the
    entities). Lookup only; never creates entities."""
    from src.memory.models import CodexEdge, CodexEntity
    if not (subject and relation and obj):
        return None
    subj_e = db.query(CodexEntity).filter_by(canonical_name=_norm(subject)).first()
    obj_e = db.query(CodexEntity).filter_by(canonical_name=_norm(obj)).first()
    if subj_e is None or obj_e is None:
        return None
    return db.query(CodexEdge).filter(
        CodexEdge.source_id == subj_e.id, CodexEdge.target_id == obj_e.id,
        CodexEdge.relation == relation,
        CodexEdge.valid_until == None,  # noqa: E711
    ).first()


def _decide_duplicate(db, item: WorkItem, llm_decider) -> str:
    """D3.3 Tier 2: one-word verdict; only 'same' proposes."""
    from src.memory.models import CodexEntity
    keep = db.query(CodexEntity).get(uuid.UUID(item.payload["keep_id"]))
    absorb = db.query(CodexEntity).get(uuid.UUID(item.payload["absorb_id"]))
    if keep is None or absorb is None:
        return "unsure"

    def describe(e):
        parts = [f'"{e.canonical_name}" (type: {e.entity_type or "entity"}']
        if e.aliases:
            parts.append(f'aliases: {", ".join(e.aliases[:5])}')
        parts = ["; ".join(parts) + ")"]
        if e.description:
            parts.append(f"note: {e.description[:200]}")
        return " — ".join(parts)

    cos = item.payload.get("cosine")
    prompt = (
        "Do these two knowledge-graph entries refer to the SAME thing "
        "(one is a duplicate/alternate name of the other), or to two "
        "different things?\n"
        f"A: {describe(keep)}\n"
        f"B: {describe(absorb)}\n"
        + (f"Name-embedding similarity: {cos:.2f}\n" if cos is not None else "")
        + '\nReply with one JSON object: {"verdict": "same" | "different" | "unsure"}.'
    )
    return _ask_enum(llm_decider, prompt, "verdict", ("same", "different", "unsure"))


def _decide_stale_slot(db, item: WorkItem, llm_decider) -> Optional[str]:
    """D3.5 Tier 2: a content suggestion (not an enum) from recent turns;
    empty/None ⇒ nothing to propose."""
    slot = db.execute(text(
        "SELECT content FROM memory_slots WHERE slot_name = :s LIMIT 1"),
        {"s": item.payload["slot_name"]}).first()
    if slot is None:
        return None
    recents = db.execute(text("""
        SELECT timestamp, COALESCE(summary_text, raw_text) AS txt
        FROM episodic_memory
        WHERE is_private = FALSE
        ORDER BY timestamp DESC LIMIT 5
    """)).fetchall()
    recent_lines = "\n".join(
        f"- [{r.timestamp:%Y-%m-%d}] {(r.txt or '')[:200]}" for r in recents)
    prompt = (
        f'The user\'s "{item.payload["slot_name"]}" memory slot has not changed in '
        f"{settings.agent_stale_slot_days}+ days. Suggest an updated version: keep items that "
        "still look open, drop ones the recent conversations show as done or "
        "abandoned. Do not invent new items.\n"
        f"Current content:\n{(slot.content or '')[:600]}\n\n"
        f"Recent conversation summaries:\n{recent_lines}\n\n"
        'Reply with one JSON object: {"suggestion": "<the updated slot content>"} '
        '— or {"suggestion": ""} if the current content still looks right.'
    )
    parsed = llm_decider(prompt, 400) if llm_decider else None
    if not isinstance(parsed, dict):
        return None
    suggestion = str(parsed.get("suggestion", "")).strip()
    return suggestion or None


# ═══ Apply (tier rules; callables; Tier-2 → review queue) ═══════════════════

def _apply_merge(db, item: WorkItem, run_id) -> str:
    from src.workers.codex_ops import merge_entities  # lazy: embedder at import
    result = merge_entities(db, uuid.UUID(item.payload["keep_id"]),
                            uuid.UUID(item.payload["absorb_id"]),
                            agent_run_id=str(run_id))
    return "applied" if result.get("status") == "merged" else "skipped"


def _apply_pileup(db, item: WorkItem, run_id) -> str:
    """Expire live pending edges that exactly duplicate a live active edge
    (same source/target/relation); the active edge keeps the max
    extraction_confidence seen. Deterministic, journaled via _expire_edge."""
    from src.memory.models import CodexEdge
    from src.workers.codex_extractor import _expire_edge
    entity_id = uuid.UUID(item.payload["entity_id"])
    rows = db.execute(text("""
        SELECT pe.id AS pending_id, ae.id AS active_id
        FROM codex_edges pe
        JOIN codex_edges ae ON ae.source_id = pe.source_id
         AND ae.target_id = pe.target_id AND ae.relation = pe.relation
         AND ae.confidence = 'active' AND ae.valid_until IS NULL
        WHERE pe.source_id = :eid AND pe.confidence = 'pending'
          AND pe.valid_until IS NULL
    """), {"eid": entity_id}).fetchall()
    if not rows:
        return "noop"
    for r in rows:
        pending = db.query(CodexEdge).get(r.pending_id)
        active = db.query(CodexEdge).get(r.active_id)
        if pending is None or active is None or pending.valid_until is not None:
            continue
        active.extraction_confidence = max(
            active.extraction_confidence or 0.0,
            pending.extraction_confidence or 0.0)
        _expire_edge(db, pending.id, run_id, "pending_duplicate", source=SOURCE)
    db.commit()
    logger.info("agent_action", action="pending_pileup", outcome="applied",
                entity_id=str(entity_id), expired=len(rows),
                agent_run_id=str(run_id))
    return "applied"


def _apply_contradiction(db, item: WorkItem, run_id) -> str:
    """Newer-supersedes through A6's reconcile_conflict (antonym arm — the
    deterministic path; a polarity clash IS the antonym of its positive)."""
    from src.memory.models import CodexEdge, CodexEntity
    from src.workers.codex_extractor import reconcile_conflict
    old_edge = db.query(CodexEdge).get(uuid.UUID(item.payload["old_edge_id"]))
    new_edge = db.query(CodexEdge).get(uuid.UUID(item.payload["new_edge_id"]))
    if old_edge is None or new_edge is None or old_edge.valid_until is not None:
        return "noop"
    subj = db.query(CodexEntity).get(new_edge.source_id)
    obj = db.query(CodexEntity).get(new_edge.target_id)
    conflict = {"type": "antonym", "old_edge_id": old_edge.id,
                "old_relation": old_edge.relation,
                "old_target_id": old_edge.target_id}
    reconcile_conflict(db, conflict, subj, item.payload["new_relation"], obj,
                       run_id, None, None, source=SOURCE)
    db.commit()
    logger.info("agent_action", action="contradiction", outcome="applied",
                kind=item.payload.get("kind"),
                old_edge_id=item.payload["old_edge_id"],
                agent_run_id=str(run_id))
    return "applied"


def _apply_reconciliation(db, item: WorkItem, decision: str, run_id) -> str:
    from src.memory.models import ReviewQueue
    from src.workers.codex_extractor import _expire_edge
    review = db.query(ReviewQueue).get(uuid.UUID(item.payload["review_id"]))
    if review is None:
        return "noop"
    content = dict(review.item_content or {})

    if decision == "unsure":
        content["agent_attempts"] = int(content.get("agent_attempts", 0)) + 1
        review.item_content = content
        flag_modified(review, "item_content")
        db.commit()
        logger.info("agent_action", action="reconciliation_leftover",
                    outcome="still_unsure", review_id=item.payload["review_id"],
                    attempts=content["agent_attempts"], agent_run_id=str(run_id))
        return "still_unsure"

    if decision == "expire_old":
        _expire_edge(db, uuid.UUID(content["old_edge_id"]), run_id,
                     "supersession", source=SOURCE)
    elif decision == "reject_new":
        new = content.get("new") or {}
        new_edge = _find_edge_by_names(db, new.get("subject"),
                                       new.get("relation"), new.get("object"))
        if new_edge is not None:
            _expire_edge(db, new_edge.id, run_id, "reconciliation_reject",
                         source=SOURCE)
    # keep_both / already_resolved: no graph write.
    content["decision"] = decision
    content["agent_run_id"] = str(run_id)
    review.item_content = content
    flag_modified(review, "item_content")
    review.status = "resolved"       # not "approved" — the agent decided, not the user
    db.commit()
    logger.info("agent_action", action="reconciliation_leftover",
                outcome=decision, review_id=item.payload["review_id"],
                agent_run_id=str(run_id))
    return "applied" if decision in ("expire_old", "reject_new") else decision


def _propose_merge(db, item: WorkItem, run_id) -> str:
    from src.memory.models import CodexEntity, ReviewQueue
    keep = db.query(CodexEntity).get(uuid.UUID(item.payload["keep_id"]))
    absorb = db.query(CodexEntity).get(uuid.UUID(item.payload["absorb_id"]))
    cos = item.payload.get("cosine")
    # Self-describing for F2 (renders from item_content alone).
    db.add(ReviewQueue(item_type="entity_merge", item_content={
        "keep_id": item.payload["keep_id"], "absorb_id": item.payload["absorb_id"],
        "keep_name": keep.canonical_name if keep else "?",
        "absorb_name": absorb.canonical_name if absorb else "?",
        "pair_key": item.payload["pair_key"],
        "evidence": {"cosine": cos, "llm_verdict": "same"},
        "suggested_action": f'merge "{absorb.canonical_name if absorb else "?"}" '
                            f'into "{keep.canonical_name if keep else "?"}"',
        "agent_run_id": str(run_id)}))
    db.commit()
    logger.info("agent_action", action="duplicate_entities", outcome="proposed",
                pair_key=item.payload["pair_key"], agent_run_id=str(run_id))
    return "proposed"


def _dismiss_pair(db, item: WorkItem, verdict: str, run_id) -> str:
    """LLM says 'different': record it (status resolved, never re-proposed —
    the pair_key skip covers every non-fresh status) without user noise."""
    from src.memory.models import ReviewQueue
    db.add(ReviewQueue(item_type="entity_merge", status="resolved", item_content={
        "keep_id": item.payload["keep_id"], "absorb_id": item.payload["absorb_id"],
        "pair_key": item.payload["pair_key"], "decision": verdict,
        "evidence": {"cosine": item.payload.get("cosine")},
        "agent_run_id": str(run_id)}))
    db.commit()
    logger.info("agent_action", action="duplicate_entities", outcome=verdict,
                pair_key=item.payload["pair_key"], agent_run_id=str(run_id))
    return verdict


def _propose_slot(db, item: WorkItem, suggestion: str, run_id) -> str:
    from src.memory.models import ReviewQueue
    db.add(ReviewQueue(item_type="memory_slot_update", item_content={
        "slot_name": item.payload["slot_name"],
        "proposed_content": suggestion,
        "proposed_by": "agent",       # C9 D7: recorded as updated_by on apply
        "reason": f"slot untouched > {settings.agent_stale_slot_days}d (maintenance agent)",
        "agent_run_id": str(run_id)}))
    db.commit()
    logger.info("agent_action", action="stale_slot", outcome="proposed",
                slot=item.payload["slot_name"], agent_run_id=str(run_id))
    return "proposed"


# ═══ The run (spec §2) ═══════════════════════════════════════════════════════

def run_maintenance_agent(db, llm_decider=_USE_DEFAULT) -> dict:
    """One capped, idempotent maintenance pass. *llm_decider* defaults to the
    background model (built lazily); pass None to skip every LLM-gated
    decision (Tier 0 and the deterministic Tier-1 items still run — graceful
    degradation, spec §4) or a stub callable in tests."""
    if llm_decider is _USE_DEFAULT:
        llm_decider = make_llm_decider()

    run_id = uuid.uuid4()
    items = detect_all(db)
    counters = {"llm_decisions": 0, "applications": 0, "proposals": 0}
    # Count actual decider invocations (an item that resolves without the LLM
    # must not consume budget); the cap pre-check in _process stays.
    if llm_decider is not None:
        inner = llm_decider

        def llm_decider(prompt, max_tokens=200):
            counters["llm_decisions"] += 1
            return inner(prompt, max_tokens)

    outcomes: dict = {}

    def record(item_type: str, outcome: str):
        outcomes[f"{item_type}:{outcome}"] = outcomes.get(f"{item_type}:{outcome}", 0) + 1

    for item in items:
        try:
            outcome = _process(db, item, llm_decider, run_id, counters)
        except Exception as exc:
            db.rollback()
            logger.error("agent_action_failed", item_type=item.item_type,
                         error=str(exc), agent_run_id=str(run_id))
            outcome = "error"
        record(item.item_type, outcome)

    logger.info("agent_run_summary", agent_run_id=str(run_id),
                scanned=len(items), outcomes=outcomes, **counters)
    return {"agent_run_id": str(run_id), "scanned": len(items),
            "outcomes": outcomes, **counters}


def _process(db, item: WorkItem, llm_decider, run_id, counters) -> str:
    needs_llm = (item.item_type in ("reconciliation_leftover", "stale_slot")
                 or (item.item_type == "duplicate_entities" and item.tier == 2))
    if needs_llm:
        if llm_decider is None:
            return "skipped_no_llm"
        if counters["llm_decisions"] >= settings.agent_max_llm_decisions:
            return "skipped_cap"

    if item.item_type == "duplicate_entities" and item.tier == 0:
        if counters["applications"] >= settings.agent_max_applications:
            return "skipped_cap"
        outcome = _apply_merge(db, item, run_id)
        if outcome == "applied":
            counters["applications"] += 1
        return outcome

    if item.item_type == "pending_pileup":
        if counters["applications"] >= settings.agent_max_applications:
            return "skipped_cap"
        outcome = _apply_pileup(db, item, run_id)
        if outcome == "applied":
            counters["applications"] += 1
        return outcome

    if item.item_type == "contradiction":
        if counters["applications"] >= settings.agent_max_applications:
            return "skipped_cap"
        outcome = _apply_contradiction(db, item, run_id)
        if outcome == "applied":
            counters["applications"] += 1
        return outcome

    if item.item_type == "reconciliation_leftover":
        decision = _decide_reconciliation(db, item, llm_decider)
        if decision in ("expire_old", "reject_new") and \
                counters["applications"] >= settings.agent_max_applications:
            return "skipped_cap"
        outcome = _apply_reconciliation(db, item, decision, run_id)
        if outcome == "applied":
            counters["applications"] += 1
        return outcome

    if item.item_type == "duplicate_entities":       # tier 2
        verdict = _decide_duplicate(db, item, llm_decider)
        if verdict == "same":
            if counters["proposals"] >= settings.agent_max_tier2_proposals:
                return "skipped_cap"
            counters["proposals"] += 1
            return _propose_merge(db, item, run_id)
        if verdict == "different":
            return _dismiss_pair(db, item, verdict, run_id)
        return "unsure"

    if item.item_type == "stale_slot":
        suggestion = _decide_stale_slot(db, item, llm_decider)
        if not suggestion:
            return "no_suggestion"
        if counters["proposals"] >= settings.agent_max_tier2_proposals:
            return "skipped_cap"
        counters["proposals"] += 1
        return _propose_slot(db, item, suggestion, run_id)

    if item.item_type == "stale_work":
        # E3: deterministic Tier-2 — the item payload IS the proposal
        # (self-describing for F2); no LLM decision needed.
        if counters["proposals"] >= settings.agent_max_tier2_proposals:
            return "skipped_cap"
        from src.memory.models import ReviewQueue
        db.add(ReviewQueue(item_type="stale_work", item_content={
            **item.payload, "agent_run_id": str(run_id)}))
        db.commit()
        counters["proposals"] += 1
        logger.info("agent_action", action="stale_work", outcome="proposed",
                    task_id=item.payload.get("task_id"),
                    agent_run_id=str(run_id))
        return "proposed"

    logger.warning("agent_unknown_item_type", item_type=item.item_type)
    return "unknown"
