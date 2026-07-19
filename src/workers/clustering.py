"""Clustering Worker – v5 (roadmap C5): fixes the *structural* causes of
Exp2's "2–3 mega-clusters + dozens of singletons" — which were creation- and
assignment-side, not threshold-side.

WHAT v5 CHANGES AND WHY
  1. EXCLUSIVE ASSIGNMENT (kills the mega-cluster convergence machine).
     v4 assigned a turn to EVERY cluster above threshold; each of those
     clusters then recomputed its mean centroid *including that turn*, so
     clusters sharing members were dragged toward each other every run until
     the merge pass glued them — multi-assignment + mean-centroids is a
     positive feedback loop, and it (not the merge thresholds) is the root
     cause of the observed mega-clusters. A turn now joins only its single
     best-scoring cluster.

  2. WAIT-FOR-A-FRIEND CREATION (kills the singleton spray).
     v4 turned any turn that missed every cluster into an instant one-turn
     cluster, and nothing ever revisited it. Now an unmatched turn *waits*:
     each run, waiting turns of a conversation are pairwise-scored (embedding
     + entity overlap + same-session bonus) and grouped by union-find; a
     cluster is created only from a group of ≥2 mutually-similar turns — so
     new clusters are born with two turns of evidence (better centroid,
     better name). A turn that stays alone longer than SINGLETON_AGE_HOURS
     becomes a singleton cluster after all (one-off topics are real).
     Waiting is safe for retrieval: the cluster-scope filter explicitly
     passes turns that have no cluster links yet.

  3. SESSION AFFINITY (the C6 payoff).
     Turns from the same sitting (episodic_memory.session_id, 30-min-gap
     sessions) are likely same-topic, so sharing a session with a cluster's
     existing members now adds SESSION_AFFINITY_BONUS. This replaces v4's
     temporal-proximity bonus, which fired only when the turn was the
     cluster's *immediate* predecessor and cost two DB queries per
     turn×cluster comparison to detect — the session set is precomputed once
     per run with one grouped query per conversation.

  4. NOISE BONUSES REMOVED.
     v4's name-overlap bonus intersected the WHOLE turn text (stopwords
     included) with the cluster name+description — for any long turn it sat
     at its 0.10 cap against every cluster, a constant threshold shift that
     discriminated nothing (the same flaw the v4 docstring called out for
     tags). Removed. The tag bonus is kept but capped (within one
     conversation most clusters share topic tags, so uncapped it was also
     mostly a constant offset).

  5. NER-GROUNDED NAMING (name/description quality, needed for C6-F UX).
     The name/description prompts now receive the entities that actually
     recur across ≥2 member turns (from the shared MicroNER), so the
     RECURRING_ENTITIES line is grounded in detected names instead of asking
     the model to spot them from excerpts alone — the same
     ground-then-generate pattern A2 uses for Codex extraction.

  6. DETERMINISTIC QUEUE. The unassigned-turns query had LIMIT without
     ORDER BY (nondeterministic which 25 turns a run processed). Oldest
     first now — which is also what makes the waiting-turn age-out work.

  7. REPAIR PATH FOR EXISTING PATHOLOGY. merge_similar_clusters (which was
     never beat-scheduled — bug G10/G21 — it is now) gained a singleton
     RE-ABSORPTION pass: a one-turn cluster whose member scores above the
     assignment threshold against a sibling cluster is folded into it and
     deleted. That repairs the dozens of singletons already in the DB, not
     just future ones. The pairwise merge keeps v4's conservative gates
     (raw-sim floor 0.82, adjusted 0.90, tight entity cap) — with the
     convergence loop gone (#1), merge pressure is honest again.

  Plain callables (run_cluster_assignment / run_cluster_merge) driven by the
  C7 maintenance runtime (cadence + session-gap freshening) and, later,
  Track D's maintenance agent.

CARRIED OVER FROM v4 (still correct):
  - Entity overlap as the continuity signal bridging low bulk-text cosine
    between scenes of one story (see v4 notes: threshold tuning can't fix a
    missing signal).
  - Centroid renormalization after every mean (an average of unit vectors
    is not unit length; unnormalized centroids deflate dot-product scores
    as clusters grow).
  - Name/description regeneration only every NAME_REGEN_INTERVAL members.
  - Real MicroNER (shared with the orchestrator) over a regex proxy.
"""

import structlog
import re
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from typing import Optional
from src.memory.models import (
    EpisodicMemory,
    ContextCluster,
    EpisodicClusterLink,
)
from src.workers.bg_client_factory import bg_timeout, get_bg_client, get_bg_model_name
from src.memory.embedder import get_embedder

logger = structlog.get_logger("ice.workers.clustering")
bg_client = get_bg_client()

# The process-shared native-width embedder (G13/G23).
cluster_embedder = get_embedder()

SIMILARITY_THRESHOLD = 0.6              # assignment bar (adjusted score)
MERGE_SIMILARITY_THRESHOLD = 0.90       # adjusted bar for merging two clusters
MERGE_MIN_RAW_SIM = 0.82                # raw-centroid floor to even consider merging
MAX_TURNS_PER_RUN = 25

ENTITY_OVERLAP_BONUS_PER_SHARED = 0.08
ENTITY_OVERLAP_BONUS_CAP = 0.30         # assignment/pairing cap
MERGE_ENTITY_BONUS_CAP = 0.10           # merging is permanent — be conservative

TAG_OVERLAP_BONUS = 0.05
TAG_OVERLAP_BONUS_CAP = 0.10            # v5: capped — within one conversation tags
                                        # are near-uniform, so uncapped this was a
                                        # constant offset, not a signal
SESSION_AFFINITY_BONUS = 0.10           # v5: same sitting ⇒ likely same topic (C6)
SINGLETON_AGE_HOURS = 24.0              # how long a lone turn waits for a friend

NAME_REGEN_INTERVAL = 5

from src.retrieval.ner_utils import extract_entities

ENTITY_EXTRACTION_MAX_CHARS = 2000


def _extract_turn_entities(text_: str) -> set:
    """Entity strings from a turn's raw_text via the shared MicroNER
    (truncated: the recurring cast almost always appears in a scene's
    opening portion, and per-turn NER cost compounds across the batch)."""
    return set(extract_entities(
        text_ or "", cluster_embedder, max_chars=ENTITY_EXTRACTION_MAX_CHARS
    ))


def _recurring_member_entities(turns, min_occurrences: int = 2, top_n: int = 15) -> list:
    """v5: the NER-grounding for naming — entities that appear in at least
    *min_occurrences* member turns, most frequent first. For 1-turn clusters
    the single turn's entities are used as-is (nothing to recur across)."""
    counts = {}
    for t in turns[:10]:
        for e in _extract_turn_entities(t.raw_text or ""):
            counts[e] = counts.get(e, 0) + 1
    if len(turns) == 1:
        recurring = list(counts.keys())
    else:
        recurring = [e for e, c in counts.items() if c >= min_occurrences]
    recurring.sort(key=lambda e: counts[e], reverse=True)
    return recurring[:top_n]


def _member_snippets(turns) -> str:
    snippets = []
    for t in turns[:10]:
        raw = t.raw_text or ""
        if len(raw.split()) < 20:
            continue
        cleaned = re.sub(r'(?i)^.*\bchapter\s*\d+.*$', '', raw, flags=re.MULTILINE)
        snippets.append(" ".join(cleaned.split()[:200]))
    return "\n\n".join(snippets)


def _generate_cluster_description(turns, recurring_entities=None):
    combined = _member_snippets(turns)
    # v5: ground the RECURRING_ENTITIES line in NER-detected names (the same
    # ground-then-generate pattern as Codex A2) instead of hoping the model
    # spots recurrence from excerpts.
    ner_hint = ""
    if recurring_entities:
        ner_hint = (
            "\n\nNames detected as recurring across these turns by entity "
            f"recognition (ground your RECURRING_ENTITIES line in these): "
            f"{', '.join(recurring_entities)}"
        )
    prompt = (
        "Below are excerpts from several conversation turns that all belong to the same "
        "cluster. Write a precise description of the RECURRING THEME or DOMAIN this "
        "cluster represents — NOT a summary of any single turn or exchange.\n\n"
        "Structure your answer as exactly these labeled lines:\n"
        "DOMAIN: <the general subject area, project, or topic this belongs to>\n"
        "CONTENT_TYPE: <facts | planning | brainstorming | narrative prose | technical "
        "decisions | personal reflection — pick the best fit(s)>\n"
        "RECURRING_ENTITIES: <comma‑separated list of the named people, characters, "
        "tools, projects, libraries, or other proper nouns that appear repeatedly across "
        "these excerpts — this is the most important line, list every recurring name you "
        "can identify>\n"
        "SETTING_OR_CONTEXT: <the recurring context, location, or environment; if none "
        "is clear, write 'n/a'>\n\n"
        "Do not add any other text, headers, or commentary."
    )
    try:
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[
                {"role": "system", "content": "You are a precise topic and cast categoriser."},
                {"role": "user", "content": f"{prompt}\n\n{combined}{ner_hint}"}
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=bg_timeout(200),
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error("cluster_description_generation_failed", error=str(e))
        return ""


def _generate_cluster_name(turns, recurring_entities=None):
    combined = _member_snippets(turns)
    ner_hint = ""
    if recurring_entities:
        ner_hint = (
            "\n\nRecurring names detected across these turns: "
            f"{', '.join(recurring_entities)}"
        )
    prompt = (
        "Below are excerpts from several conversation turns that belong to the same "
        "cluster. Give this cluster a short, distinct name (3‑5 words) that captures "
        "the RECURRING TOPIC or THEME these turns share.\n"
        "NEVER be too general; be specific with the naming. \n\n"
        "RULES:\n"
        "- NEVER use a turn number, chapter number, or date in the name.\n"
        "- NEVER write a name that describes a single event or moment.\n"
        "- The name must describe a CATEGORY or RECURRING SUBJECT — not a specific "
        "action or exchange.\n"
        "- The name must be generic enough that a similar future turn about the SAME "
        "topic would also be assigned to this cluster.\n\n"
        "Output ONLY the name, nothing else."
    )
    try:
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[
                {"role": "system", "content": "You are a naming assistant."},
                {"role": "user", "content": f"{prompt}\n\n{combined}{ner_hint}"}
            ],
            temperature=0.0,
            max_tokens=20,
            timeout=15.0,
        )
        name = completion.choices[0].message.content.strip()
        name = name.strip('"\'').strip()
        return name if name else "Unnamed Cluster"
    except Exception as e:
        logger.error("cluster_name_generation_failed", error=str(e))
        return "Unnamed Cluster"


def _embed_text(text_):
    return cluster_embedder.encode(text_, convert_to_tensor=False).tolist()


def _parse_embedding(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [float(x) for x in value.strip('[]').split(',')]
    return list(value)


def _normalize(vec):
    """An average of unit vectors is NOT unit length; unnormalized centroids
    silently deflate dot-product scores as clusters grow (verified in v4)."""
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        return [x / norm for x in vec]
    return vec


def _recompute_centroid_from_members(db, cluster_id) -> Optional[list]:
    member_embs = db.execute(
        text("""
            SELECT e.embedding
            FROM episodic_memory e
            JOIN episodic_cluster_links l ON l.episodic_id = e.id
            WHERE l.cluster_id = :cid AND e.embedding IS NOT NULL
        """),
        {"cid": cluster_id}
    ).fetchall()

    parsed = []
    for (emb,) in member_embs:
        emb_list = _parse_embedding(emb)
        if emb_list:
            parsed.append(emb_list)
    if not parsed:
        return None

    dim = len(parsed[0])
    avg = [0.0] * dim
    for emb in parsed:
        for i in range(dim):
            avg[i] += emb[i]
    for i in range(dim):
        avg[i] /= len(parsed)

    return _normalize(avg)


def _extract_recurring_cast_from_description(description: str) -> set:
    m = re.search(r'RECURRING_ENTITIES:\s*(.+)', description or "", re.IGNORECASE)
    if not m:
        return set()
    names = [n.strip() for n in m.group(1).split(',')]
    return {n for n in names if n and n.lower() != 'n/a'}


def _cluster_session_sets(db, conversation_id) -> dict:
    """cluster_id → set of session_ids present among its members. One grouped
    query per conversation per run (v5 — replaces v4's two-queries-per-
    turn×cluster temporal check)."""
    rows = db.execute(text("""
        SELECT l.cluster_id AS cid, e.session_id AS sid
        FROM episodic_cluster_links l
        JOIN episodic_memory e ON e.id = l.episodic_id
        WHERE e.conversation_id = :conv AND e.session_id IS NOT NULL
        GROUP BY l.cluster_id, e.session_id
    """), {"conv": conversation_id}).fetchall()
    out = {}
    for r in rows:
        out.setdefault(r.cid, set()).add(r.sid)
    return out


def _score_against_cluster(turn_emb, turn_tags, turn_entities, turn_session, state):
    """Adjusted assignment score of one turn against one cluster state."""
    cl = state["cluster"]
    base_sim = sum(a * b for a, b in zip(turn_emb, state["embedding"]))
    tag_bonus = min(TAG_OVERLAP_BONUS_CAP,
                    TAG_OVERLAP_BONUS * len(set(turn_tags) & set(cl.tags or [])))
    entity_bonus = min(ENTITY_OVERLAP_BONUS_CAP,
                       ENTITY_OVERLAP_BONUS_PER_SHARED * len(turn_entities & state["cast"]))
    session_bonus = SESSION_AFFINITY_BONUS if (
        turn_session is not None and turn_session in state["sessions"]
    ) else 0.0
    return base_sim + tag_bonus + entity_bonus + session_bonus


def _pair_score(a, b):
    """Adjusted similarity between two *waiting* turns (dicts with emb /
    entities / session_id) — the wait-for-a-friend grouping signal."""
    base = sum(x * y for x, y in zip(a["emb"], b["emb"]))
    entity_bonus = min(ENTITY_OVERLAP_BONUS_CAP,
                       ENTITY_OVERLAP_BONUS_PER_SHARED * len(a["entities"] & b["entities"]))
    session_bonus = SESSION_AFFINITY_BONUS if (
        a["session_id"] is not None and a["session_id"] == b["session_id"]
    ) else 0.0
    return base + entity_bonus + session_bonus


def _create_cluster_from_members(db, conversation_id, members, cluster_state):
    """Create a cluster from ≥1 waiting turns; centroid = normalized mean;
    name/description NER-grounded from the members. Appends to cluster_state
    so later turns in the same run see it."""
    turns = [m["turn"] for m in members]
    dim = len(members[0]["emb"])
    avg = [0.0] * dim
    for m in members:
        for i in range(dim):
            avg[i] += m["emb"][i]
    centroid = _normalize([x / len(members) for x in avg])

    tags = sorted({t for m in members for t in (m["tags"] or [])})
    recurring = _recurring_member_entities(turns)
    name = _generate_cluster_name(turns, recurring)
    desc = _generate_cluster_description(turns, recurring)

    cluster = ContextCluster(
        name=name if name else "Unnamed Cluster",
        description=desc if desc else "",
        conversation_id=conversation_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        embedding=centroid,
        tags=tags,
    )
    db.add(cluster)
    db.flush()
    for m in members:
        db.add(EpisodicClusterLink(episodic_id=m["id"], cluster_id=cluster.id))
    db.flush()

    cluster_state.append({
        "cluster": cluster,
        "embedding": centroid,
        "cast": _extract_recurring_cast_from_description(desc) | set(recurring),
        "sessions": {m["session_id"] for m in members if m["session_id"] is not None},
    })
    logger.info("cluster_created", name=cluster.name, members=len(members),
                conversation_id=str(conversation_id))
    return cluster


def _assign_turn_to_cluster(db, turn_info, state):
    """Link a turn to a cluster; refresh tags, centroid, session set, and the
    name/description every NAME_REGEN_INTERVAL members."""
    cl = state["cluster"]
    if not db.query(EpisodicClusterLink).filter_by(
            episodic_id=turn_info["id"], cluster_id=cl.id).first():
        db.add(EpisodicClusterLink(episodic_id=turn_info["id"], cluster_id=cl.id))

    existing_tags = set(cl.tags or [])
    new_tags = set(turn_info["tags"] or [])
    if new_tags - existing_tags:
        cl.tags = list(existing_tags | new_tags)
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(cl, "tags")

    db.flush()
    member_count = db.query(EpisodicClusterLink).filter_by(cluster_id=cl.id).count()

    if member_count > 1 and member_count % NAME_REGEN_INTERVAL == 0:
        linked_turns = db.query(EpisodicMemory).filter(
            EpisodicMemory.id.in_(
                db.query(EpisodicClusterLink.episodic_id).filter(
                    EpisodicClusterLink.cluster_id == cl.id
                )
            )
        ).all()
        recurring = _recurring_member_entities(linked_turns)
        new_name = _generate_cluster_name(linked_turns, recurring)
        new_desc = _generate_cluster_description(linked_turns, recurring)
        if new_name:
            cl.name = new_name
        if new_desc:
            cl.description = new_desc
        state["cast"] = _extract_recurring_cast_from_description(cl.description or "") | set(recurring)

    new_centroid = _recompute_centroid_from_members(db, cl.id)
    if new_centroid is not None:
        cl.embedding = new_centroid
        cl.updated_at = datetime.now(timezone.utc)
        state["embedding"] = new_centroid
    if turn_info["session_id"] is not None:
        state["sessions"].add(turn_info["session_id"])


def run_cluster_assignment(db, conversation_ids=None) -> dict:
    """Plain callable (C7/D1-composable): assign unassigned turns. Returns
    stats for logging/telemetry. *conversation_ids* optionally restricts the
    run to specific conversations — the hook C7's session-triggered
    "cluster this sitting now" uses."""
    stats = {"processed": 0, "assigned": 0, "created": 0, "waiting": 0, "singletons": 0}
    now = datetime.now(timezone.utc)

    conv_filter = "AND e.conversation_id = ANY(:conv_ids)" if conversation_ids else ""
    params = {"limit": MAX_TURNS_PER_RUN}
    if conversation_ids:
        params["conv_ids"] = list(conversation_ids)

    # v5: deterministic queue — oldest first (v4 had LIMIT without ORDER BY),
    # which is also what lets lone turns age out into singletons.
    unassigned_rows = db.execute(text(f"""
        SELECT e.id, e.topic_tags, e.embedding, e.conversation_id,
               e.session_id, e.timestamp
        FROM episodic_memory e
        WHERE e.is_private = FALSE
          {conv_filter}
          AND NOT EXISTS (
            SELECT 1 FROM episodic_cluster_links l WHERE l.episodic_id = e.id
        )
        ORDER BY e.timestamp ASC
        LIMIT :limit
    """), params).fetchall()

    if not unassigned_rows:
        return stats

    conv_turns = {}
    for row in unassigned_rows:
        conv_turns.setdefault(str(row.conversation_id), []).append(row)

    for cid, rows in conv_turns.items():
        existing_clusters = db.query(ContextCluster).filter(
            ContextCluster.embedding != None,
            ContextCluster.conversation_id == cid
        ).all()
        session_sets = _cluster_session_sets(db, cid)
        cluster_state = []
        for cl in existing_clusters:
            cluster_state.append({
                "cluster": cl,
                "embedding": _parse_embedding(cl.embedding),
                "cast": _extract_recurring_cast_from_description(cl.description or ""),
                "sessions": session_sets.get(cl.id, set()),
            })

        waiting = []
        for row in rows:
            turn = db.query(EpisodicMemory).get(row.id)
            if not turn:
                continue
            stats["processed"] += 1
            emb = _parse_embedding(row.embedding) or _embed_text(turn.raw_text)
            info = {
                "id": row.id,
                "turn": turn,
                "emb": emb,
                "tags": row.topic_tags or [],
                "entities": _extract_turn_entities(turn.raw_text or ""),
                "session_id": row.session_id,
                "timestamp": row.timestamp,
            }

            # v5: EXCLUSIVE assignment — single best cluster above threshold
            # (multi-assignment was the mega-cluster convergence machine).
            best, best_score = None, SIMILARITY_THRESHOLD
            for state in cluster_state:
                if state["embedding"] is None:
                    continue
                score = _score_against_cluster(
                    emb, info["tags"], info["entities"], info["session_id"], state
                )
                if score >= best_score:
                    best, best_score = state, score
            if best is not None:
                _assign_turn_to_cluster(db, info, best)
                stats["assigned"] += 1
            else:
                waiting.append(info)

        if not waiting:
            continue

        # v5: wait-for-a-friend — group mutually-similar waiting turns via
        # union-find; only groups of ≥2 become clusters now.
        parent = list(range(len(waiting)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(waiting)):
            for j in range(i + 1, len(waiting)):
                if _pair_score(waiting[i], waiting[j]) >= SIMILARITY_THRESHOLD:
                    parent[find(i)] = find(j)

        groups = {}
        for i in range(len(waiting)):
            groups.setdefault(find(i), []).append(waiting[i])

        for members in groups.values():
            if len(members) >= 2:
                _create_cluster_from_members(db, cid, members, cluster_state)
                stats["created"] += 1
                stats["assigned"] += len(members)
            else:
                lone = members[0]
                age = now - lone["timestamp"]
                if age > timedelta(hours=SINGLETON_AGE_HOURS):
                    # One-off topics are real — after waiting long enough,
                    # a lone turn earns its own cluster.
                    _create_cluster_from_members(db, cid, [lone], cluster_state)
                    stats["created"] += 1
                    stats["singletons"] += 1
                else:
                    stats["waiting"] += 1

    db.commit()
    logger.info("cluster_assignment_complete", **stats)
    return stats


def run_cluster_merge(db, conversation_ids=None) -> dict:
    """Plain callable (C7/D1-composable): singleton re-absorption + pairwise
    merge of converged centroids. *conversation_ids* optionally restricts the
    run (C7 session-triggered maintenance)."""
    stats = {"absorbed": 0, "merged": 0}

    conv_q = db.query(ContextCluster.conversation_id).filter(
        ContextCluster.conversation_id != None,
        ContextCluster.embedding != None
    )
    if conversation_ids:
        conv_q = conv_q.filter(ContextCluster.conversation_id.in_(list(conversation_ids)))
    conv_ids = conv_q.distinct().all()

    for (cid,) in conv_ids:
        clusters = db.query(ContextCluster).filter(
            ContextCluster.conversation_id == cid,
            ContextCluster.embedding != None
        ).order_by(ContextCluster.created_at.asc()).all()
        if len(clusters) < 2:
            continue

        session_sets = _cluster_session_sets(db, cid)
        states = []
        member_counts = {}
        for cl in clusters:
            states.append({
                "cluster": cl,
                "embedding": _parse_embedding(cl.embedding),
                "cast": _extract_recurring_cast_from_description(cl.description or ""),
                "sessions": session_sets.get(cl.id, set()),
            })
            member_counts[cl.id] = db.query(EpisodicClusterLink).filter_by(
                cluster_id=cl.id).count()

        # ── v5 repair pass: singleton re-absorption ──
        # A one-turn cluster whose member fits a sibling cluster (by the same
        # assignment scoring) is folded in and deleted — this is what heals
        # the "dozens of singletons" already accumulated in the DB.
        removed_ids = set()
        for state in states:
            cl = state["cluster"]
            if member_counts.get(cl.id, 0) != 1:
                continue
            link = db.query(EpisodicClusterLink).filter_by(cluster_id=cl.id).first()
            if not link:
                continue
            turn = db.query(EpisodicMemory).get(link.episodic_id)
            if turn is None or turn.embedding is None:
                continue
            emb = _parse_embedding(turn.embedding)
            entities = _extract_turn_entities(turn.raw_text or "")
            best, best_score = None, SIMILARITY_THRESHOLD
            for other in states:
                if other["cluster"].id == cl.id or other["cluster"].id in removed_ids:
                    continue
                if other["embedding"] is None:
                    continue
                score = _score_against_cluster(
                    emb, turn.topic_tags or [], entities, turn.session_id, other
                )
                if score >= best_score:
                    best, best_score = other, score
            if best is None:
                continue
            target = best["cluster"]
            logger.info("singleton_absorbed", singleton=cl.name, into=target.name,
                        score=round(best_score, 3))
            if not db.query(EpisodicClusterLink).filter_by(
                    episodic_id=link.episodic_id, cluster_id=target.id).first():
                db.add(EpisodicClusterLink(episodic_id=link.episodic_id,
                                           cluster_id=target.id))
            db.delete(link)
            db.flush()
            new_centroid = _recompute_centroid_from_members(db, target.id)
            if new_centroid is not None:
                target.embedding = new_centroid
                target.updated_at = datetime.now(timezone.utc)
                best["embedding"] = new_centroid
            if turn.session_id is not None:
                best["sessions"].add(turn.session_id)
            db.delete(cl)
            removed_ids.add(cl.id)
            stats["absorbed"] += 1
            db.flush()

        # ── Pairwise merge of converged centroids (v4 gates kept) ──
        live = [s for s in states if s["cluster"].id not in removed_ids]
        merged_ids = set()
        for i in range(len(live)):
            older = live[i]["cluster"]
            if older.id in merged_ids:
                continue
            older_cast = live[i]["cast"]
            for j in range(i + 1, len(live)):
                newer = live[j]["cluster"]
                if newer.id in merged_ids:
                    continue
                emb_older = _parse_embedding(older.embedding)
                emb_newer = live[j]["embedding"]
                if emb_older is None or emb_newer is None:
                    continue
                sim = sum(a * b for a, b in zip(emb_older, emb_newer))
                if sim < MERGE_MIN_RAW_SIM:
                    continue
                shared = older_cast & live[j]["cast"]
                entity_bonus = min(MERGE_ENTITY_BONUS_CAP,
                                   ENTITY_OVERLAP_BONUS_PER_SHARED * len(shared))
                if sim + entity_bonus < MERGE_SIMILARITY_THRESHOLD:
                    continue

                logger.info("merging_clusters", older=older.name, newer=newer.name,
                            similarity=round(sim, 3), entity_bonus=round(entity_bonus, 3))
                newer_links = db.query(EpisodicClusterLink).filter_by(cluster_id=newer.id).all()
                for link in newer_links:
                    if not db.query(EpisodicClusterLink).filter_by(
                            episodic_id=link.episodic_id, cluster_id=older.id).first():
                        db.add(EpisodicClusterLink(episodic_id=link.episodic_id,
                                                   cluster_id=older.id))
                    db.delete(link)

                older_tags = set(older.tags or [])
                newer_tags = set(newer.tags or [])
                if newer_tags - older_tags:
                    older.tags = list(older_tags | newer_tags)
                    from sqlalchemy.orm.attributes import flag_modified
                    flag_modified(older, "tags")

                db.flush()
                new_centroid = _recompute_centroid_from_members(db, older.id)
                if new_centroid is not None:
                    older.embedding = new_centroid
                    older.updated_at = datetime.now(timezone.utc)
                db.delete(newer)
                merged_ids.add(newer.id)
                stats["merged"] += 1
                db.flush()

    db.commit()
    logger.info("cluster_merge_complete", **stats)
    return stats


