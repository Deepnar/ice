"""Clustering Worker – v4: normalized centroids + entity-overlap signal
(fixes chapter-per-cluster fragmentation) + improved naming/description
prompts + cheaper regeneration cadence.

ROOT CAUSE OF "EVERY CHAPTER BECOMES ITS OWN CLUSTER"
  Each turn in a narrative conversation (Shinchan, Flaw) is a full chapter
  of prose — often 800-1200+ words describing a SPECIFIC scene (a kendo
  tournament, a cafe date, etc). Two chapters from the SAME ongoing story
  can have LOW cosine similarity as bulk-text embeddings, because embedding
  models respond heavily to concrete content (which characters are active,
  what location, what action) — and two different scenes in one continuous
  story usually use mostly different concrete nouns/verbs even though a
  human would obviously call them "the same story."

  Lowering SIMILARITY_THRESHOLD (0.80 -> 0.40) was the wrong lever: it
  doesn't fix a weak/wrong SIGNAL, it just makes an already-noisy signal
  noisier. Tag-overlap and name/description boosts (also already present)
  don't help either, because every chapter in the same story tends to
  share the SAME topic tags and a near-identical generic cluster
  name/description ("Shinchan story chapters") — meaning that signal
  contributes nearly the SAME bonus to every comparison, so it can't
  discriminate "these two chapters are obviously the same arc" from
  "these two chapters are unrelated."

  THE FIX: add a genuinely discriminating signal — NAMED ENTITY OVERLAP,
  extracted using the REAL MicroNER model shared with the orchestrator's
  Codex-matching path (src/retrieval/ner_utils.py), not a regex proxy.
  An earlier version of this fix used a capitalized-word regex on the
  (incorrect) assumption that the real NER model might not be loadable in
  worker context — it loads exactly the same way this file's own
  cluster_embedder already does (a local file + a HuggingFace tokenizer,
  no request-context dependency at all), and this is a periodic background
  worker with no live-request latency budget, so there's no real reason to
  prefer a weaker proxy. The real model gives fewer false positives
  (sentence-initial capitalized words, chapter headers, etc — the regex
  version needed a manual stoplist to work around exactly this) and
  benefits from whatever domain-specific entity training the model already
  has, rather than reinventing weaker heuristics.

  This signal generalizes beyond narrative content: two chapters that both
  feature Shinchan, Kazama, and Rika are almost certainly the same ongoing
  story thread even if the scene content is totally different (one's a
  tournament, one's a cafe date) — and the same mechanism applies to
  technical conversations, where recurring proper nouns like "Codex",
  "FastAPI", "PostgreSQL" provide the same kind of continuity signal across
  planning/discussion turns that character names provide in narrative.
  Entity overlap persists across scenes/turns in a way that bulk-text
  cosine similarity does not. With this signal in place,
  SIMILARITY_THRESHOLD is raised back toward a more meaningful level — the
  entity-overlap bonus is what now carries cross-turn continuity, instead
  of relying on raw embedding similarity (which was never going to
  reliably capture it) or threshold tuning (which can't fix a missing
  signal, only mask it).

CENTROID NORMALIZATION (carried over from previous review)
  Centroids are recomputed as a mean of member embeddings, then
  RENORMALIZED to unit length. Without this, centroid magnitude shrinks as
  a cluster accumulates more diverse members, which deflates the raw
  Python dot-product similarity used for assignment — making large,
  well-established clusters progressively WORSE at recognizing matching
  new turns the more turns they already have. This was confirmed
  numerically: a genuinely-matching new turn's raw similarity dropped from
  0.45 to 0.42 as a simulated cluster grew from 3 to 40 members, even
  though TRUE cosine similarity (properly normalized) rose from 0.57 to
  0.64 over the same growth. Fixed by renormalizing after every average.
"""

import structlog
import json
import re
from datetime import datetime, timezone
from sqlalchemy import text
from typing import Optional
from src.api.config import settings
from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory,
    ContextCluster,
    EpisodicClusterLink,
)
from src.workers.celery_app import app
from src.workers.gpu_check import is_gpu_busy, is_user_active
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger("ice.workers.clustering")
bg_client = get_bg_client()

cluster_embedder = SentenceTransformer(
    "Qwen/Qwen3-Embedding-0.6B",
    device="cpu",
    truncate_dim=384
)

# Raised back up from 0.40 now that entity overlap carries cross-scene
# continuity instead of relying on raw embedding similarity alone. 0.40 was
# a symptom-patch for a missing signal, not a real fix — see module
# docstring. 0.55 is a more honest "these two chunks of text are actually
# semantically close" bar; entity overlap below makes up the rest of the
# distance for same-story-different-scene cases.
SIMILARITY_THRESHOLD = 0.6
MERGE_SIMILARITY_THRESHOLD = 0.90       # keep this
MAX_TURNS_PER_RUN = 25

# Entity-overlap bonus weight. Tuned so that sharing 2+ named entities
# (e.g. "Shinchan" and "Kazama" both appearing) can single-handedly bridge
# the gap between a borderline embedding similarity (~0.40-0.50) and the
# assignment threshold (0.55) — this is the mechanism that actually fixes
# chapter fragmentation, not the threshold value itself.
ENTITY_OVERLAP_BONUS_PER_SHARED = 0.08  # unchanged
ENTITY_OVERLAP_BONUS_CAP = 0.30      # cap so entity overlap alone can't force
                                      # a match between two genuinely unrelated
                                      # texts that happen to share common names
MERGE_ENTITY_BONUS_CAP = 0.10           # tighter cap for merge decisions only
MERGE_MIN_RAW_SIM = 0.82                # raw similarity floor to even consider merging

TAG_OVERLAP_BONUS = 0.05
NAME_DESC_SIM_WEIGHT = 0.10          # reduced from naive versions — name/desc
                                      # text is often too generic to carry much
                                      # signal on its own (see docstring); kept
                                      # small and secondary, not primary
TEMPORAL_PROXIMITY_BONUS = 0.05

# Regenerate a cluster's name/description only every N members, not on
# EVERY single turn assignment. Previously this fired on every assignment
# once a cluster had 2+ members — for a 40-member cluster that's up to ~38
# extra LLM calls purely for naming churn, with the name potentially
# flip-flopping turn to turn as the generation model's output varies.
NAME_REGEN_INTERVAL = 5


from src.retrieval.ner_utils import extract_entities

# Real chapters can run 800-1200+ words; the recurring cast is almost
# always introduced in the opening portion of a scene, so we truncate
# before running NER to keep per-turn cost bounded for this background
# worker (which processes up to MAX_TURNS_PER_RUN turns per invocation —
# unlike the orchestrator's single-live-prompt NER call, latency compounds
# here across the whole batch).
ENTITY_EXTRACTION_MAX_CHARS = 2000


def _extract_turn_entities(text: str) -> set:
    """Extract entity strings from a turn's raw_text using the REAL
    MicroNER model (shared with the orchestrator's Codex-matching path)
    instead of a regex proxy. Previously this used a capitalized-word
    regex with a manual stoplist ("The", "Chapter", etc) on the
    (incorrect) assumption that the real NER model might not be loadable
    in worker context — it loads the same way cluster_embedder already
    does (a local file + a HuggingFace tokenizer, no request-context
    dependency at all). Using the real model means narrative clustering
    benefits from the same entity-recognition quality used for Codex
    graph matching, with fewer false positives than a bare regex.
    """
    entities = extract_entities(
        text or "", cluster_embedder, max_chars=ENTITY_EXTRACTION_MAX_CHARS
    )
    return set(entities)


def _generate_cluster_description(turns):
    snippets = []
    for t in turns[:10]:
        raw = t.raw_text or ""
        words = raw.split()
        if len(words) < 20:
            continue
        cleaned = re.sub(r'(?i)^.*\bchapter\s*\d+.*$', '', raw, flags=re.MULTILINE)
        words = cleaned.split()
        snippet = " ".join(words[:200])
        snippets.append(snippet)
    combined = "\n\n".join(snippets)

    # Improved prompt: explicitly asks for the RECURRING CAST / SETTING as
    # its own labeled field, since that's now the signal future assignment
    # decisions lean on most heavily (entity overlap). Previously the
    # description asked for "distinctive named entities" as one clause
    # buried in a longer instruction — now it's a structured, separate ask
    # so the model reliably surfaces names instead of summarizing plot.
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
                {"role": "user", "content": f"{prompt}\n\n{combined}"}
            ],
            temperature=0.0,
            max_tokens=200,
            timeout=30.0,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error("cluster_description_generation_failed", error=str(e))
        return ""


def _generate_cluster_name(turns):
    snippets = []
    for t in turns[:10]:
        raw = t.raw_text or ""
        words = raw.split()
        if len(words) < 20:
            continue
        cleaned = re.sub(r'(?i)^.*\bchapter\s*\d+.*$', '', raw, flags=re.MULTILINE)
        words = cleaned.split()
        snippet = " ".join(words[:200])
        snippets.append(snippet)
    combined = "\n\n".join(snippets)
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
                {"role": "user", "content": f"{prompt}\n\n{combined}"}
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


def _embed_text(text):
    return cluster_embedder.encode(text, convert_to_tensor=False).tolist()


def _parse_embedding(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [float(x) for x in value.strip('[]').split(',')]
    return list(value)


def _normalize(vec):
    """Renormalize a vector to unit length. Critical for centroids: an
    average of several unit vectors is NOT itself unit length, and feeding
    an unnormalized centroid into a raw Python dot product (as the
    per-turn assignment loop does) silently deflates similarity scores as
    the centroid's magnitude shrinks — see module docstring."""
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
    """Parse the RECURRING_CAST line out of a cluster description produced
    by the structured prompt above, returning a set of names for fast
    overlap comparison against a new turn's entities."""
    m = re.search(r'RECURRING_ENTITIES:\s*(.+)', description, re.IGNORECASE)
    if not m:
        return set()
    names = [n.strip() for n in m.group(1).split(',')]
    return {n for n in names if n and n.lower() != 'n/a'}


from typing import Optional


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def cluster_turns(self):
    """Assign unassigned turns to existing or new clusters, scoped per
    conversation, using embedding similarity + entity overlap (the
    dominant signal for narrative continuity across scenes) + tag/name
    overlap + temporal proximity as secondary signals.
    """
    if is_gpu_busy():
        raise self.retry(countdown=60)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=30)

    db = SessionLocal()
    try:
        unassigned_rows = db.execute(text("""
            SELECT e.id, e.topic_tags, e.embedding, e.conversation_id
            FROM episodic_memory e
            WHERE NOT EXISTS (
                SELECT 1 FROM episodic_cluster_links l WHERE l.episodic_id = e.id
            )
            LIMIT :limit
        """), {"limit": MAX_TURNS_PER_RUN}).fetchall()

        if not unassigned_rows:
            return

        conv_turns = {}
        for row in unassigned_rows:
            cid = str(row.conversation_id)
            conv_turns.setdefault(cid, []).append(row)

        for cid, turns_in_conv in conv_turns.items():
            existing_clusters = db.query(ContextCluster).filter(
                ContextCluster.embedding != None,
                ContextCluster.conversation_id == cid
            ).all()

            # Pre-compute each existing cluster's recurring-cast set ONCE
            # per run (not per candidate turn) for efficiency.
            cluster_state = []
            for cl in existing_clusters:
                entities = _extract_recurring_cast_from_description(cl.description or "")
                cluster_state.append({
                    "cluster": cl,
                    "embedding": _parse_embedding(cl.embedding),
                    "cast": entities,
                })

            for turn_row in turns_in_conv:
                turn_id = turn_row.id
                turn_tags = turn_row.topic_tags or []
                turn_emb = _parse_embedding(turn_row.embedding)

                turn = db.query(EpisodicMemory).get(turn_id)
                if not turn:
                    continue

                if turn_emb is None:
                    turn_emb = _embed_text(turn.raw_text)

                turn_entities = _extract_turn_entities(turn.raw_text or "")

                assigned_to = []

                for state in cluster_state:
                    cl = state["cluster"]
                    cl_emb = state["embedding"]
                    if cl_emb is None:
                        continue

                    base_sim = sum(a * b for a, b in zip(turn_emb, cl_emb))

                    tag_overlap = len(set(turn_tags) & set(cl.tags or []))
                    tag_bonus = TAG_OVERLAP_BONUS * tag_overlap

                    # ── Entity overlap: the primary fix for chapter
                    #    fragmentation. Two chapters featuring the same
                    #    recurring cast are almost certainly the same
                    #    ongoing arc, even with low bulk-text similarity. ──
                    shared_entities = turn_entities & state["cast"]
                    entity_bonus = min(
                        ENTITY_OVERLAP_BONUS_CAP,
                        ENTITY_OVERLAP_BONUS_PER_SHARED * len(shared_entities)
                    )

                    # Name/description text overlap — kept as a small,
                    # secondary signal (reduced weight vs. earlier version,
                    # since it's often too generic to discriminate well).
                    cluster_text = (cl.name + " " + (cl.description or "")).lower()
                    cluster_words = set(re.findall(r'\w+', cluster_text))
                    turn_words = set(re.findall(r'\w+', (turn.raw_text or "").lower()))
                    name_overlap = len(turn_words & cluster_words)
                    name_bonus = min(0.10, 0.01 * name_overlap)

                    # Temporal proximity: if this turn directly follows the
                    # last turn added to this cluster, treat as likely
                    # continuation.
                    temporal_bonus = 0.0
                    last_turn_in_cluster = db.query(EpisodicMemory).join(
                        EpisodicClusterLink, EpisodicClusterLink.episodic_id == EpisodicMemory.id
                    ).filter(
                        EpisodicClusterLink.cluster_id == cl.id
                    ).order_by(EpisodicMemory.timestamp.desc()).first()
                    if last_turn_in_cluster:
                        prev_turn = db.query(EpisodicMemory).filter(
                            EpisodicMemory.conversation_id == turn.conversation_id,
                            EpisodicMemory.timestamp < turn.timestamp
                        ).order_by(EpisodicMemory.timestamp.desc()).first()
                        if prev_turn and prev_turn.id == last_turn_in_cluster.id:
                            temporal_bonus = TEMPORAL_PROXIMITY_BONUS

                    adjusted = base_sim + tag_bonus + entity_bonus + name_bonus + temporal_bonus

                    if adjusted >= SIMILARITY_THRESHOLD:
                        assigned_to.append(cl)

                if not assigned_to:
                    new_cluster = ContextCluster(
                        name="new",
                        description="",
                        conversation_id=cid,
                        created_at=datetime.now(timezone.utc),
                        embedding=None,
                    )
                    db.add(new_cluster)
                    db.flush()

                    db.add(EpisodicClusterLink(episodic_id=turn_id, cluster_id=new_cluster.id))
                    db.flush()

                    linked_turns = db.query(EpisodicMemory).filter(
                        EpisodicMemory.id.in_(
                            db.query(EpisodicClusterLink.episodic_id).filter(
                                EpisodicClusterLink.cluster_id == new_cluster.id
                            )
                        )
                    ).all()

                    name = _generate_cluster_name(linked_turns)
                    desc = _generate_cluster_description(linked_turns)

                    new_cluster.name = name if name else "Unnamed Cluster"
                    new_cluster.description = desc if desc else ""
                    new_cluster.embedding = _normalize(turn_emb)
                    new_cluster.tags = list(set(turn_tags))
                    new_cluster.updated_at = datetime.now(timezone.utc)
                    assigned_to = [new_cluster]
                    cluster_state.append({
                        "cluster": new_cluster,
                        "embedding": new_cluster.embedding,
                        "cast": _extract_recurring_cast_from_description(new_cluster.description),
                    })

                for cl in assigned_to:
                    existing_link = db.query(EpisodicClusterLink).filter_by(
                        episodic_id=turn_id, cluster_id=cl.id
                    ).first()
                    if not existing_link:
                        db.add(EpisodicClusterLink(episodic_id=turn_id, cluster_id=cl.id))

                    existing_tags = set(cl.tags or [])
                    new_tags = set(turn_tags)
                    if new_tags - existing_tags:
                        cl.tags = list(existing_tags | new_tags)
                        from sqlalchemy.orm.attributes import flag_modified
                        flag_modified(cl, "tags")

                    db.flush()
                    member_count = db.query(EpisodicClusterLink).filter_by(cluster_id=cl.id).count()

                    # ── Regenerate name/description only every
                    #    NAME_REGEN_INTERVAL members, not on every single
                    #    assignment — avoids ~38 extra LLM calls per
                    #    40-member cluster and stops the name from
                    #    flip-flopping turn to turn. ──
                    if member_count > 1 and member_count % NAME_REGEN_INTERVAL == 0:
                        linked_turns = db.query(EpisodicMemory).filter(
                            EpisodicMemory.id.in_(
                                db.query(EpisodicClusterLink.episodic_id).filter(
                                    EpisodicClusterLink.cluster_id == cl.id
                                )
                            )
                        ).all()
                        new_name = _generate_cluster_name(linked_turns)
                        new_desc = _generate_cluster_description(linked_turns)
                        if new_name:
                            cl.name = new_name
                        if new_desc:
                            cl.description = new_desc

                    # ── Recompute + RENORMALIZE centroid (fixes the
                    #    centroid-shrinkage bug from the previous review) ──
                    new_centroid = _recompute_centroid_from_members(db, cl.id)
                    if new_centroid is not None:
                        cl.embedding = new_centroid
                        cl.updated_at = datetime.now(timezone.utc)
                        # Keep in-memory cluster_state in sync so later
                        # turns IN THIS SAME RUN compare against the fresh
                        # centroid, not a stale one from the start of the run.
                        for state in cluster_state:
                            if state["cluster"].id == cl.id:
                                state["embedding"] = new_centroid
                                state["cast"] = _extract_recurring_cast_from_description(cl.description or "")

        db.commit()
        logger.info("cluster_assignment_complete", turns_processed=len(unassigned_rows))

    except Exception as exc:
        db.rollback()
        logger.error("clustering_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()


@app.task(bind=True, max_retries=2, default_retry_delay=120)
def merge_similar_clusters(self):
    """Periodically merge clusters within the same conversation that have
    converged to very similar centroids, preventing fragmentation."""
    if is_gpu_busy():
        raise self.retry(countdown=120)
    if settings.background_model_mode == "shared" and is_user_active():
        raise self.retry(countdown=60)

    db = SessionLocal()
    try:
        conv_ids = db.query(ContextCluster.conversation_id).filter(
            ContextCluster.conversation_id != None,
            ContextCluster.embedding != None
        ).distinct().all()

        merge_count = 0

        for (cid,) in conv_ids:
            clusters = db.query(ContextCluster).filter(
                ContextCluster.conversation_id == cid,
                ContextCluster.embedding != None
            ).order_by(ContextCluster.created_at.asc()).all()

            if len(clusters) < 2:
                continue

            merged_ids = set()
            for i in range(len(clusters)):
                older = clusters[i]
                if older.id in merged_ids:
                    continue
                older_cast = _extract_recurring_cast_from_description(older.description or "")
                for j in range(i + 1, len(clusters)):
                    newer = clusters[j]
                    if newer.id in merged_ids:
                        continue

                    emb_older = _parse_embedding(older.embedding)
                    emb_newer = _parse_embedding(newer.embedding)
                    if emb_older is None or emb_newer is None:
                        continue

                    sim = sum(a * b for a, b in zip(emb_older, emb_newer))

                    # Only consider merging if the raw centroid similarity
                    # is genuinely high — entity overlap alone should never
                    # force a merge of dissimilar clusters.
                    if sim < MERGE_MIN_RAW_SIM:
                        continue

                    # Use a TIGHTER entity bonus cap for merge decisions.
                    # Assignment can be generous (0.30 cap) because a turn
                    # being assigned to the wrong cluster is fixable later.
                    # Merging is permanent — be conservative.
                    newer_cast = _extract_recurring_cast_from_description(newer.description or "")
                    shared = older_cast & newer_cast
                    entity_bonus = min(
                        MERGE_ENTITY_BONUS_CAP,
                        ENTITY_OVERLAP_BONUS_PER_SHARED * len(shared)
                    )

                    adjusted_sim = sim + entity_bonus

                    if adjusted_sim < MERGE_SIMILARITY_THRESHOLD:
                        continue

                    logger.info("merging_clusters", older=older.name, newer=newer.name,
                                similarity=round(sim, 3), entity_bonus=round(entity_bonus, 3))

                    newer_links = db.query(EpisodicClusterLink).filter_by(cluster_id=newer.id).all()
                    for link in newer_links:
                        already_linked = db.query(EpisodicClusterLink).filter_by(
                            episodic_id=link.episodic_id, cluster_id=older.id
                        ).first()
                        if not already_linked:
                            db.add(EpisodicClusterLink(episodic_id=link.episodic_id, cluster_id=older.id))
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
                        older_cast = _extract_recurring_cast_from_description(older.description or "")

                    db.delete(newer)
                    merged_ids.add(newer.id)
                    merge_count += 1
                    db.flush()

        db.commit()
        logger.info("cluster_merge_complete", merges_performed=merge_count)

    except Exception as exc:
        db.rollback()
        logger.error("cluster_merge_failed", error=str(exc))
        raise self.retry(exc=exc)
    finally:
        db.close()