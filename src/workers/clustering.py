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
     better name). A turn that stays alone longer than settings.cluster_singleton_age_hours
     becomes a singleton cluster after all (one-off topics are real).
     Waiting is safe for retrieval: the cluster-scope filter explicitly
     passes turns that have no cluster links yet.

  3. SESSION AFFINITY (the C6 payoff).
     Turns from the same sitting (episodic_memory.session_id, 30-min-gap
     sessions) are likely same-topic, so sharing a session with a cluster's
     existing members now adds settings.cluster_session_affinity_bonus. This replaces v4's
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
     recur across ≥2 member turns (from background NuNER), so the
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
  - Name/description regeneration only every settings.cluster_name_regen_interval members.
  - Existing background NuNER over complete source windows by default;
    generated label text never supplies overlap evidence.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

import structlog
from sqlalchemy import text

from src.api.config import settings
from src.memory.cluster_labels import (cached_entities, generate_label, label_receipts,
    member_count, member_input, member_sessions, member_sources, member_vectors,
    rebase_cold_members, recurring_entities)
from src.memory.embedder import get_embedder
from src.memory.models import ContextCluster, EpisodicClusterLink, EpisodicMemory
from src.memory.source import digest
from src.retrieval.ner_utils import extract_entities

logger = structlog.get_logger("ice.workers.clustering")
cluster_embedder = get_embedder()


def _extract_turn_entities(text_: str) -> set:
    """Existing background NuNER covers all source windows by default."""
    raw = text_ or ""
    cap = settings.cluster_entity_extraction_max_chars
    if cap is not None and len(raw) > cap:
        logger.warning("cluster_entities_prefix_only", original_chars=len(raw), supplied_chars=cap,
                       reason="explicit character cap omits source tail")
    key = (digest(raw), settings.background_ner_model, cap)
    return cached_entities(key, lambda: {
        ' '.join(entity.split()).casefold() for entity in extract_entities(
            raw, cluster_embedder, max_chars=cap, tier="background")})


def _recurring_member_entities(turns, min_occurrences: int = 2, top_n: int = 15) -> list:
    return recurring_entities(turns, _extract_turn_entities,
                              min_occurrences=min_occurrences, top_n=top_n)


def _cluster_source_cast(db, cluster_id):
    return set(_recurring_member_entities(member_sources(db, cluster_id)))


def _member_snippets(turns) -> str:
    """Compatibility name: complete attributed member originals, never prefixes."""
    return member_input(turns)


def _generate_cluster_description(turns, recurring_entities=None):
    return generate_label(turns, recurring_entities, kind='description') or ""


def _generate_cluster_name(turns, recurring_entities=None):
    return generate_label(turns, recurring_entities, kind='name') or ""


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
    member_embs = member_vectors(db, cluster_id)

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


def _cluster_session_sets(db, conversation_id) -> dict:
    """cluster_id → set of session_ids present among its members. One grouped
    query per conversation per run (v5 — replaces v4's two-queries-per-
    turn×cluster temporal check)."""
    rows = member_sessions(db, conversation_id)
    out = {}
    for r in rows:
        out.setdefault(r.cid, set()).add(r.sid)
    return out


def _score_against_cluster(turn_emb, turn_tags, turn_entities, turn_session, state):
    """Adjusted assignment score of one turn against one cluster state."""
    cl = state["cluster"]
    base_sim = sum(a * b for a, b in zip(turn_emb, state["embedding"]))
    tag_bonus = min(settings.cluster_tag_overlap_bonus_cap,
                    settings.cluster_tag_overlap_bonus * len(set(turn_tags) & set(cl.tags or [])))
    entity_bonus = min(settings.cluster_entity_overlap_bonus_cap,
                       settings.cluster_entity_overlap_bonus_per_shared * len(turn_entities & state["cast"]))
    session_bonus = settings.cluster_session_affinity_bonus if (
        turn_session is not None and turn_session in state["sessions"]
    ) else 0.0
    if base_sim < settings.cluster_min_raw_similarity:
        return base_sim  # audit A12: bonuses separate similar turns, never rescue unrelated ones
    return base_sim + tag_bonus + entity_bonus + session_bonus


def _pair_score(a, b):
    """Adjusted similarity between two *waiting* turns (dicts with emb /
    entities / session_id) — the wait-for-a-friend grouping signal."""
    base = sum(x * y for x, y in zip(a["emb"], b["emb"]))
    entity_bonus = min(settings.cluster_entity_overlap_bonus_cap,
                       settings.cluster_entity_overlap_bonus_per_shared * len(a["entities"] & b["entities"]))
    session_bonus = settings.cluster_session_affinity_bonus if (
        a["session_id"] is not None and a["session_id"] == b["session_id"]
    ) else 0.0
    if base < settings.cluster_min_raw_similarity:
        return base  # audit A12: see _score_against_cluster
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
        "cast": set(recurring),
        "sessions": {m["session_id"] for m in members if m["session_id"] is not None},
    })
    logger.info("cluster_created", name=cluster.name, members=len(members),
                conversation_id=str(conversation_id))
    return cluster


def _assign_turn_to_cluster(db, turn_info, state):
    """Link a turn to a cluster; refresh tags, centroid, session set, and the
    name/description every settings.cluster_name_regen_interval members."""
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
    count = member_count(db, cl.id)
    state["cast"] = _cluster_source_cast(db, cl.id)

    if count > 1 and count % settings.cluster_name_regen_interval == 0:
        linked_turns = member_sources(db, cl.id)
        recurring = _recurring_member_entities(linked_turns)
        new_name = _generate_cluster_name(linked_turns, recurring)
        new_desc = _generate_cluster_description(linked_turns, recurring)
        if new_name:
            cl.name = new_name
        if new_desc:
            cl.description = new_desc

    new_centroid = _recompute_centroid_from_members(db, cl.id)
    if new_centroid is not None:
        cl.embedding = new_centroid
        cl.updated_at = datetime.now(timezone.utc)
        state["embedding"] = new_centroid
    if turn_info["session_id"] is not None:
        state["sessions"].add(turn_info["session_id"])


def run_cluster_assignment(db, conversation_ids=None) -> dict:
    with label_receipts() as receipt:
        result = _run_cluster_assignment(db, conversation_ids)
        result['labels'] = receipt
        return result


def _run_cluster_assignment(db, conversation_ids=None) -> dict:
    """Plain callable (C7/D1-composable): assign unassigned turns. Returns
    stats for logging/telemetry. *conversation_ids* optionally restricts the
    run to specific conversations — the hook C7's session-triggered
    "cluster this sitting now" uses."""
    stats = {"processed": 0, "assigned": 0, "created": 0, "waiting": 0, "singletons": 0}
    now = datetime.now(timezone.utc)

    conv_filter = "AND e.conversation_id = ANY(:conv_ids)" if conversation_ids else ""
    params = {"limit": settings.cluster_max_turns_per_run}
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
                "cast": _cluster_source_cast(db, cl.id),
                "sessions": session_sets.get(cl.id, set()),
            })

        waiting = []
        for row in rows:
            turn = db.get(EpisodicMemory, row.id)
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
            best, best_score = None, settings.cluster_similarity_threshold
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
                if _pair_score(waiting[i], waiting[j]) >= settings.cluster_similarity_threshold:
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
                if age > timedelta(hours=settings.cluster_singleton_age_hours):
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
    with label_receipts() as receipt:
        result = _run_cluster_merge(db, conversation_ids)
        result['labels'] = receipt
        return result


def _run_cluster_merge(db, conversation_ids=None) -> dict:
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
                "cast": _cluster_source_cast(db, cl.id),
                "sessions": session_sets.get(cl.id, set()),
            })
            member_counts[cl.id] = member_count(db, cl.id)

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
            turn = db.get(EpisodicMemory, link.episodic_id)
            if turn is None or turn.embedding is None:
                continue
            emb = _parse_embedding(turn.embedding)
            entities = _extract_turn_entities(turn.raw_text or "")
            best, best_score = None, settings.cluster_similarity_threshold
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
            rebase_cold_members(db, cl.id, target.id)
            new_centroid = _recompute_centroid_from_members(db, target.id)
            if new_centroid is not None:
                target.embedding = new_centroid
                target.updated_at = datetime.now(timezone.utc)
                best["embedding"] = new_centroid
            if turn.session_id is not None:
                best["sessions"].add(turn.session_id)
            best["cast"] = _cluster_source_cast(db, target.id)
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
                if sim < settings.cluster_merge_min_raw_sim:
                    continue
                shared = older_cast & live[j]["cast"]
                entity_bonus = min(settings.cluster_merge_entity_bonus_cap,
                                   settings.cluster_entity_overlap_bonus_per_shared * len(shared))
                if sim + entity_bonus < settings.cluster_merge_similarity_threshold:
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
                rebase_cold_members(db, newer.id, older.id)
                new_centroid = _recompute_centroid_from_members(db, older.id)
                if new_centroid is not None:
                    older.embedding = new_centroid
                    older.updated_at = datetime.now(timezone.utc)
                live[i]["cast"] = _cluster_source_cast(db, older.id)
                older_cast = live[i]["cast"]
                db.delete(newer)
                merged_ids.add(newer.id)
                stats["merged"] += 1
                db.flush()

    db.commit()
    logger.info("cluster_merge_complete", **stats)
    return stats
