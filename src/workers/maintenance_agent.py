"""Memory Maintenance Agent (D1) — deterministic worklist + bounded LLM decisions.

NOT a free-roaming tool loop (spec D1_D2 D1): cheap SQL detectors produce
typed work items; the background model decides per item from a FIXED enum
(JSON mode, temperature 0, unparseable ⇒ unsure ⇒ no write); execution goes
through named callables (A6's ``reconcile_conflict``, D5's ``merge_entities``).
A small model gets reliability from constrained choices, not open planning.

Three risk tiers govern write authority (D2):

  Tier 0 — deterministic, auto-applied: normalization-equal duplicate merges.
  Tier 1 — auto-applied + journaled: reconciliation leftovers re-decided with
           fuller context (LLM), pending-edge dedupe.
  Tier 2 — LLM-decided: near-duplicate entity merges (CODEX_V4 D12) and
           stale-slot removals (G77) apply themselves with an `applied` audit
           row that `review.undo` reverses; a stale-slot rewrite that adds
           text is dismissed. Stale-work notices are written `applied` (G77).

Every run gets an ``agent_run_id``; every graph write journals CodexEvents
with ``source: "maintenance_agent"`` (G17) and ``batch_source = agent_run_id``
(D4). Runs are idempotent: detectors re-derive state from the DB, caps make
runs incremental (D8). Scheduled by the maintenance runtime (gpu lane,
overdue 12h + session-end burst).

E1 look-ahead: detectors are a pluggable registry keyed by item type — the
coding-side decisions table reuses this exact pattern.
"""
import inspect
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.api.config import settings
import structlog

from src.workers.llm_json import parse_object
from sqlalchemy import text

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


def _norm(name: str) -> str:
    return (name or "").strip().casefold()


# Separator-like punctuation only. Everything else — emoji, `~`, `*`, `+`, `%`,
# currency — SURVIVES, because it distinguishes entities: `choose 🇦` is not
# `choose 🇪`, and stripping non-ASCII collapsed three real options into one.
_MERGE_SEPS = re.compile(r"[\s\-_:;,()\[\]{}\"'/\\.]+")
_DIGIT_ALPHA = re.compile(r"(\d)([a-z])")
_ALPHA_DIGIT = re.compile(r"([a-z])(\d)")


def merge_key(name: str) -> str:
    """G50: a deterministic identity key — two names with the same key are the
    same entity, with no model and no review.

    Casefold, split digit/letter runs (`18gb` → `18 gb`), collapse separator
    punctuation. **Token ORDER is preserved on purpose.** Sorting the tokens
    finds 11 more groups on the real store and two of them are wrong in the way
    that cannot be undone: `see you`/`you see`, and `brahma's creation without
    shiva's dissolution`/`shiva's dissolution without brahma's creation` — a
    converse, the entity-side twin of TRAPS #26. A wrong auto-merge writes a
    false identity permanently, so permutation matches belong in Tier 2 where a
    model and a human look at them.

    Measured on 7,949 real entities: 20 groups merge, and every safety probe
    holds — `~15 gb`≠`~17 gb`, `12th boards`≠`10th boards`,
    `1-2 percent`≠`1-4 percent`, `devstral-small`≠`devstral-small-2`.
    """
    s = (name or "").strip().casefold()
    s = _DIGIT_ALPHA.sub(r"\1 \2", s)
    s = _ALPHA_DIGIT.sub(r"\1 \2", s)
    return " ".join(t for t in _MERGE_SEPS.sub(" ", s).split() if t)


_WORDNUM = frozenset(
    "one two three four five six seven eight nine ten eleven twelve "
    "first second third fourth fifth sixth seventh eighth ninth tenth "
    "half quarter double triple".split())
_GENDER = frozenset(
    "his her he she him hers himself herself mr mrs ms sir madam".split())
_TENSE = frozenset(
    "is was are were has had have will would did does do being been "
    "am shall should can could".split())
_TOKENS = re.compile(r"[\s\-_:;,()\[\]{}\"'/\\.]+")
_NUMERALS = re.compile(r"\d+")


def difference_kind(a: str, b: str) -> str:
    """G50: what KIND of difference separates two names — 'merge', 'reject:<why>'
    or 'defer'. Deterministic: no model, no embedding, no vector.

    **Why this exists rather than a cosine threshold.** Measured 2026-08-17 over
    all 2,247 pairs above cosine 0.90 on the arm-1 store: only **42 (1.9%)** are
    `merge_key`-equal and safe, while **583 (26%)** differ by a token that
    changes what the name refers to. Cosine does not sort by safety and the top
    of the band is the worst part — 0.98+ holds `two sagas`/`four sagas`,
    `8 gb`/`4 gb`, `qwen3-32b`/`qwen3-30b` and `gemma-4-e4b`/`gemma-4-e4b-q4`
    (base versus quantised) sitting beside genuinely safe pairs. So the
    candidate queue is 98% pending REJECTIONS, not pending merges, and the
    cheapest correct thing is to type the difference and drop the rejects at
    detection — no model, no review queue, no human.

    Ordered most-specific first: a pair differing by BOTH a digit and a tense
    word is a digit rejection, because the digit is the stronger signal.
    """
    # ⚑ A MISSING NAME MUST NEVER PRODUCE A MERGE. Both keys are "" when a
    # lookup misses, and `"" == ""` made every pair read as `merge` — so a
    # silently empty name map would have auto-merged the entire candidate set,
    # which is the one direction this function must not fail in. Caught
    # 2026-08-17 while wiring the detector, before it reached the queue.
    if not (a or "").strip() or not (b or "").strip():
        return "defer"
    if merge_key(a) == merge_key(b):
        return "merge"
    if sorted(_NUMERALS.findall(a or "")) != sorted(_NUMERALS.findall(b or "")):
        return "reject:digits"
    ta = {t for t in _TOKENS.split((a or "").casefold()) if t}
    tb = {t for t in _TOKENS.split((b or "").casefold()) if t}
    diff = ta ^ tb
    if diff & _WORDNUM:
        return "reject:wordnum"
    if diff & _GENDER:
        return "reject:gender"
    if diff & _TENSE:
        return "reject:tense"
    # ⚑ SAME TOKENS, DIFFERENT ORDER — stays a REJECTION even though it looks
    # like the safest case of all. `shiva without brahma`/`brahma without shiva`
    # is live in the store at cosine 0.9639: a converse, the entity-side twin of
    # TRAPS #26. Merging it writes a false identity that cannot be undone.
    if ta == tb:
        return "reject:permutation"
    return "defer"


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
    # G50: a pair already typed as a rejection is settled — `8 gb`/`4 gb` will
    # not stop differing by a digit. Without this the detector re-derives the
    # same verdict and writes a duplicate memo on every run, and the rejection
    # work would grow linearly with run count instead of draining.
    seen_keys |= {r[0] for r in db.execute(text(
        "SELECT payload->>'pair_key' FROM codex_events "
        "WHERE event_type = 'merge_rejected' "
        "  AND payload->>'pair_key' IS NOT NULL")).fetchall()}

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
    # G50: keyed by merge_key, not by casefold. The casefold channel could
    # never fire — `canonical_name` is UNIQUE, so two rows cannot share one, and
    # aliases mirror it; measured 0 collisions across 7,949 entities. The Tier 0
    # channel, the one designed to need no judgement, had therefore never merged
    # anything. merge_key finds 20 groups on the same store with the same
    # safety, e.g. `gemma 4 e4b`/`gemma-4-e4b`/`gemma4:e4b`, `qwen3-8b`/
    # `qwen3:8b`, `~18 gb`/`~18gb`, `manhattan 5 lb book`/`manhattan 5lb book`.
    by_name: dict = {}
    for e in ents:
        for n in {merge_key(e.canonical_name), *(merge_key(a) for a in (e.aliases or []))}:
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

    # Lazy, like the other imports in this module's decision paths.
    from src.memory.models import CodexEvent

    # G50: names for the typed rejection below. One query rather than a lookup
    # per pair — the cosine channel alone can return `cap` rows and each would
    # otherwise cost a round trip.
    pair_ids = {i for _k, ids, _c in ranked for i in ids}
    names = {}
    if pair_ids:
        names = {r.id: r.canonical_name for r in db.execute(text(
            "SELECT id, canonical_name FROM codex_entities WHERE id = ANY(:ids)"),
            {"ids": list(pair_ids)}).fetchall()}

    items, used = [], set()
    rejected = Counter()
    reject_batch = uuid.uuid4()
    for key, (id_a, id_b), cos in ranked:
        if len(items) >= cap:
            break
        if key in seen_keys or key in used:
            continue
        used.add(key)
        # ⚑ G50: TYPED REJECTION BEFORE THE QUEUE. Measured 2026-08-17 over all
        # 2,247 pairs above cosine 0.90: **583 (26%)** differ by a digit, a
        # spelled-out number, a gender word, a tense word or a token
        # permutation — every one of which changes what the name refers to.
        # Those never needed a model or a human, and queueing them is what made
        # consolidation look like a scaling problem: the backlog is mostly
        # pending REJECTIONS, not pending merges. Dropped here, journalled so a
        # later run does not re-derive the same verdict.
        kind = difference_kind(names.get(id_a, ""), names.get(id_b, ""))
        if kind.startswith("reject:"):
            rejected[kind] += 1
            db.add(CodexEvent(
                entity_id=id_a, event_type="merge_rejected",
                # batch_source is NOT NULL — a detection pass has no graph batch
                # of its own, so it gets one per run, the same shape
                # codex_ops._run_batch() uses when there is no agent_run_id.
                batch_source=reject_batch,
                payload={"other_id": str(id_b), "kind": kind,
                         "pair_key": key, "cosine": cos}))
            continue
        keep_id, absorb_id = _merge_order(db, id_a, id_b)
        items.append(WorkItem(
            "duplicate_entities", 0 if cos is None else 2,
            {"keep_id": str(keep_id), "absorb_id": str(absorb_id),
             "pair_key": key, "cosine": cos, "difference_kind": kind}))
    if rejected:
        db.commit()
        # WARNING, and the RATE is the finding: if this fires on nearly every
        # candidate, the cosine threshold is manufacturing the queue rather
        # than discovering it — which is exactly what it was measured doing.
        logger.warning("codex_merge_candidates_rejected",
                       total=sum(rejected.values()), by_kind=dict(rejected),
                       queued=len(items))
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



def _detect_stale_slot(db, cap: int) -> list:
    """D3.5: the sentinel absence rule, ported — a pending_items slot (any tier)
    has content but hasn't been touched in settings.agent_stale_slot_days.
    Keyed by the slot ROW: since C9 the name alone matches one row per tier and
    anchor. Skipped while a proposal for the slot is pending, or a rejected /
    dismissed one is newer than the slot's last change (the "no" stands until
    the slot moves). Python's clock, not SQL NOW() — replays patch it."""
    if cap <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.agent_stale_slot_days)
    rows = db.execute(text("""
        SELECT s.id, s.slot_name, s.scope_tier, s.project_id, s.conversation_id,
               s.last_updated
        FROM memory_slots s
        WHERE s.slot_name = :slot AND s.is_active
          AND s.content IS NOT NULL AND s.content != ''
          AND s.last_updated < :cutoff
          AND NOT EXISTS (
              SELECT 1 FROM review_queue rq
              WHERE rq.item_type = 'memory_slot_update'
                AND (rq.item_content->>'slot_id' = s.id::text
                     OR (rq.item_content->>'slot_id' IS NULL
                         AND s.scope_tier = 'global'
                         AND rq.item_content->>'slot_name' = s.slot_name))
                AND (rq.status = 'pending'
                     OR (rq.status IN ('rejected', 'dismissed')
                         AND rq.created_at > s.last_updated)))
        ORDER BY s.last_updated ASC
        LIMIT :cap
    """), {"slot": STALE_SLOT_NAME, "cutoff": cutoff, "cap": cap}).fetchall()
    return [WorkItem("stale_slot", 2, {
        "slot_id": str(r.id), "slot_name": r.slot_name, "scope_tier": r.scope_tier,
        "project_id": str(r.project_id) if r.project_id else None,
        "conversation_id": str(r.conversation_id) if r.conversation_id else None,
        "last_updated": r.last_updated.isoformat() if r.last_updated else None})
        for r in rows]




def _detect_stale_work(db, cap: int) -> list:
    """E3 (coding-core D5 step 4): project tasks untouched > settings.agent_stale_task_days
    become notices — detection is deterministic (no LLM), the review queue is
    the notifier (no parallel notification system). G77: a notice changes no
    state, so it is written `applied` (nothing waits on a person). Skipped
    while a notice for the task is newer than the task's last move (or a legacy
    pending / rejected one is). Payload carries branch + goal so drift is
    visible."""
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
                         OR (rq.status IN ('rejected', 'applied')
                             AND rq.created_at > t.updated_at))
              )
            ORDER BY t.updated_at ASC
            LIMIT :cap
        """), {"cutoff": cutoff, "cap": cap}).fetchall()
    except Exception as exc:                                       # noqa: BLE001
        # A schema error must not read as "nothing stale" (TRAPS 11).
        db.rollback()
        logger.warning("stale_work_detector_failed", error=f"{type(exc).__name__}: {exc}"[:200],
                       fallback="no_items")
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
    "duplicate_entities": _detect_duplicate_entities,
    "stale_slot": _detect_stale_slot,
    # E3: stale project work (coding core) — deterministic, Tier-2.
    "stale_work": _detect_stale_work,
}


_scan_cursor = 0


def detect_all(db, *, receipts=None) -> list:
    """Reserve bounded opportunity per detector; never refill a busy backlog."""
    global _scan_cursor

    items = []
    detectors = list(DETECTORS.items())
    if not detectors:
        return items
    total = max(0, settings.agent_max_scanned)
    quota, remainder = divmod(total, len(detectors))
    start = _scan_cursor % len(detectors) if total < len(detectors) else 0
    if 0 < total < len(detectors):
        _scan_cursor = (start + total) % len(detectors)
    recipients = {(start + i) % len(detectors) for i in range(remainder)}
    for i, (name, fn) in enumerate(detectors):
        cap = quota + int(i in recipients)
        receipt = dict(detector=name, candidate_scan_limit=cap, returned=0,
                       status="not_scheduled" if cap == 0 else "complete")
        if receipts is not None:
            receipts.append(receipt)
        if cap == 0:
            continue
        try:
            found = fn(db, cap) or []
        except Exception as exc:
            db.rollback()
            logger.error("agent_detector_failed", detector=name, error=str(exc))
            receipt["status"] = "error"
            receipt["error_type"] = type(exc).__name__
            continue
        if len(found) > cap:
            logger.warning("agent_detector_exceeded_quota", detector=name,
                           returned=len(found), quota=cap)
            receipt["status"] = "exceeded_quota"
            found = found[:cap]
        receipt["returned"] = len(found)
        if found:
            items.extend(found)
        logger.info("agent_detector", **receipt)
    return items


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
            choice = resp.choices[0]
            if getattr(choice, 'finish_reason', None) != 'stop':
                logger.warning('agent_llm_incomplete', finish_reason=getattr(choice, 'finish_reason', None))
                return None
            raw = (choice.message.content or "").strip()
            # G29: shared salvage — this fence-strip was byte-identical to the
            # (since deleted) codex_extractor's; it lives in llm_json now.
            return parse_object(raw, where="maintenance_agent")
        except Exception as exc:
            logger.warning("agent_llm_failed", error=str(exc))
            return None

    return _decide



def _ask_enum(llm_decider, prompt: str, key: str, allowed: tuple) -> str:
    """One decider call, answer forced into the enum; anything else ⇒ unsure.

    An enum is the right tool HERE, unlike the old codex relation vocabulary
    (removed with CODEX_V4): this is a closed decision set that already contains
    "unsure", so a model with no good answer has a legal way to say so. The
    vocabulary had no such escape and forcing a choice there produced ~78% wrong
    facts (PROVENANCE, 2026-08-04) — the difference is whether declining is
    expressible.
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
    empty/None ⇒ nothing to propose. A conversation-tier slot is judged
    against its own conversation's turns; other tiers against any."""
    slot = db.execute(text(
        "SELECT content FROM memory_slots WHERE id = :id"),
        {"id": item.payload["slot_id"]}).first()
    if slot is None:
        return None
    conv_id = item.payload.get("conversation_id")
    recents = db.execute(text(f"""
        SELECT timestamp, COALESCE(summary_text, raw_text) AS txt
        FROM episodic_memory
        WHERE is_private = FALSE{" AND conversation_id = :cid" if conv_id else ""}
        ORDER BY timestamp DESC LIMIT 5
    """), {"cid": conv_id} if conv_id else {}).fetchall()
    recent_lines = "\n".join(
        f"- [{r.timestamp:%Y-%m-%d}] {(r.txt or '')[:200]}" for r in recents)
    prompt = (
        f'The user\'s "{item.payload["slot_name"]}" memory slot has not changed in '
        f"{settings.agent_stale_slot_days}+ days. Suggest an updated version: copy each "
        "line that still looks open exactly as written, and leave out lines the recent "
        "conversations show as done or abandoned. Do not add, merge or reword lines.\n"
        f"Current content:\n{slot.content or ''}\n\n"
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



def _apply_merge_audited(db, item: WorkItem, run_id) -> str:
    """Tier 2 'same': merge now, then record it in review_queue as applied
    (F2 can list and undo it; nothing waits for approval)."""
    from src.memory.models import CodexEntity, ReviewQueue
    from src.workers.codex_ops import merge_entities
    keep = db.query(CodexEntity).get(uuid.UUID(item.payload["keep_id"]))
    absorb = db.query(CodexEntity).get(uuid.UUID(item.payload["absorb_id"]))
    keep_name = keep.canonical_name if keep else "?"
    absorb_name = absorb.canonical_name if absorb else "?"
    result = merge_entities(db, uuid.UUID(item.payload["keep_id"]),
                            uuid.UUID(item.payload["absorb_id"]), agent_run_id=str(run_id))
    if result.get("status") != "merged":
        return "skipped"
    db.add(ReviewQueue(item_type="entity_merge", status="applied", item_content={
        "keep_id": item.payload["keep_id"], "absorb_id": item.payload["absorb_id"],
        "keep_name": keep_name, "absorb_name": absorb_name,
        "pair_key": item.payload["pair_key"],
        "evidence": {"cosine": item.payload.get("cosine"), "llm_verdict": "same"},
        "merge_event_id": result.get("merge_event_id"),
        "undo": "codex_ops.unmerge_entities(merge_event_id)",
        "agent_run_id": str(run_id)}))
    db.commit()
    logger.info("agent_action", action="duplicate_entities", outcome="applied",
                pair_key=item.payload["pair_key"], agent_run_id=str(run_id))
    return "applied"


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


def _slot_lines(content: str) -> list:
    """Lines compared by their words: whitespace and a leading bullet mark do not
    make a kept line 'new'."""
    out = []
    for line in (content or "").splitlines():
        words = " ".join(line.split()).lstrip("-*•· ").strip()
        if words:
            out.append(words)
    return out


def _apply_slot_removal(db, item: WorkItem, suggestion: str, run_id) -> str:
    """G77 (maintainer, 2026-10-17): a stale-slot suggestion applies itself only
    when it is a pure removal — every suggested line is already a line of the
    slot ("drop what is done" is checkable; anything added is not). Applied
    through slots.write_journaled (undo restores the old text); a suggestion
    that adds or rewords text is dismissed, which also blocks re-asking until
    the slot changes."""
    from src.memory.models import MemorySlot, ReviewQueue
    from src.services import slots as slots_svc
    slot = db.get(MemorySlot, uuid.UUID(item.payload["slot_id"]))
    if slot is None:
        return "skipped"
    current = (slot.content or "").splitlines()
    keys = [_slot_lines(line) for line in current]
    current_set = {k[0] for k in keys if k}
    kept = set(_slot_lines(suggestion))
    if not kept <= current_set:
        db.add(ReviewQueue(item_type="memory_slot_update", status="dismissed", item_content={
            "slot_id": str(slot.id), "slot_name": slot.slot_name,
            "proposed_content": suggestion, "proposed_by": "agent",
            "reason": "adds_or_rewords_text (G77: only pure removals apply)",
            "added_lines": sorted(kept - current_set)[:10],
            "agent_run_id": str(run_id)}))
        db.commit()
        logger.info("agent_action", action="stale_slot", outcome="dismissed_adds_text",
                    slot=slot.slot_name, agent_run_id=str(run_id))
        return "dismissed"
    if kept == current_set:
        return "no_change"
    new_lines = [line for line, k in zip(current, keys) if not k or k[0] in kept]
    removed = [k[0] for k in keys if k and k[0] not in kept]
    slots_svc.write_journaled(
        db, slot.slot_name, "\n".join(new_lines).strip(), updated_by="agent",
        scope_tier=slot.scope_tier or "global", project_id=slot.project_id,
        conversation_id=slot.conversation_id,
        audit={"reason": f"slot untouched > {settings.agent_stale_slot_days}d; "
                         "lines shown done or abandoned (maintenance agent)",
               "removed_lines": removed, "agent_run_id": str(run_id)})
    logger.info("agent_action", action="stale_slot", outcome="applied",
                slot=slot.slot_name, removed=len(removed), agent_run_id=str(run_id))
    return "applied"


# ═══ The run (spec §2) ═══════════════════════════════════════════════════════

def run_maintenance_agent(db, llm_decider=_USE_DEFAULT) -> dict:
    """One capped, idempotent maintenance pass. *llm_decider* defaults to the
    background model (built lazily); pass None to skip every LLM-gated
    decision (Tier 0 and the deterministic Tier-1 items still run — graceful
    degradation, spec §4) or a stub callable in tests."""
    if llm_decider is _USE_DEFAULT:
        llm_decider = make_llm_decider()

    run_id = uuid.uuid4()
    detector_receipts = []
    items = detect_all(db, receipts=detector_receipts)
    counters = {"llm_decisions": 0, "applications": 0, "proposals": 0}
    # Count actual decider invocations (an item that resolves without the LLM
    # must not consume budget); the cap pre-check in _process stays.
    if llm_decider is not None:
        inner = llm_decider
        parameters = inspect.signature(inner).parameters
        accepts_schema = ('schema' in parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))

        def llm_decider(prompt, max_tokens=200, **kwargs):
            # Schema-limited test callbacks do not use a provider. The real
            # decider must receive its constraint through this counting seam.
            if not accepts_schema:
                kwargs.pop('schema', None)
            counters["llm_decisions"] += 1
            return inner(prompt, max_tokens, **kwargs)

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
                scanned=len(items), outcomes=outcomes, detector_receipts=detector_receipts,
                **counters)
    return {"agent_run_id": str(run_id), "scanned": len(items),
            "outcomes": outcomes, "detector_receipts": detector_receipts, **counters}


def _process(db, item: WorkItem, llm_decider, run_id, counters) -> str:
    needs_llm = (item.item_type == "stale_slot"
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

    if item.item_type == "duplicate_entities":       # tier 2
        verdict = _decide_duplicate(db, item, llm_decider)
        if verdict == "same":
            # CODEX_V4 D12: no human wall — the model-confirmed merge applies
            # itself (undoable via codex_ops.unmerge_entities) and the review
            # queue keeps an audit row, not a pending gate.
            if counters["applications"] >= settings.agent_max_applications:
                return "skipped_cap"
            outcome = _apply_merge_audited(db, item, run_id)
            if outcome == "applied":
                counters["applications"] += 1
            return outcome
        if verdict == "different":
            return _dismiss_pair(db, item, verdict, run_id)
        return "unsure"

    if item.item_type == "stale_slot":
        suggestion = _decide_stale_slot(db, item, llm_decider)
        if not suggestion:
            return "no_suggestion"
        if counters["applications"] >= settings.agent_max_applications:
            return "skipped_cap"
        outcome = _apply_slot_removal(db, item, suggestion, run_id)
        if outcome == "applied":
            counters["applications"] += 1
        return outcome

    if item.item_type == "stale_work":
        # E3: deterministic — the item payload IS the notice (self-describing
        # for F2); no LLM decision needed. G77: it changes no state, so it is
        # written `applied` — a logged notice, not a pending gate.
        if counters["proposals"] >= settings.agent_max_tier2_proposals:
            return "skipped_cap"
        from src.memory.models import ReviewQueue
        db.add(ReviewQueue(item_type="stale_work", status="applied", item_content={
            **item.payload, "agent_run_id": str(run_id)}))
        db.commit()
        counters["proposals"] += 1
        logger.info("agent_action", action="stale_work", outcome="notified",
                    task_id=item.payload.get("task_id"),
                    agent_run_id=str(run_id))
        return "notified"

    logger.warning("agent_unknown_item_type", item_type=item.item_type)
    return "unknown"
