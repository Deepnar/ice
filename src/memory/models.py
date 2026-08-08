import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    # C12: 'chat' (default) | 'document' | 'transcript'. An ingested document is
    # its OWN conversation whose turns are its sections — that is what makes C6
    # sharing, C4's conversation summary, C2/C3 chunk retrieval and C5
    # clustering apply to it with no new machinery. The kind is also the
    # visibility rule: non-'chat' conversations are OPT-IN everywhere (a doc is
    # readable only where it is currently enabled), resolved in
    # services/scoping.py and enforced by C6's existing exclusion filter.
    kind = Column(Text, nullable=False, default="chat", server_default="chat")
    # none | auto | project | manual — the vocabulary is enforced by
    # services.scoping.VALID_SCOPE_TYPES; an unknown string used to be stored
    # verbatim and degrade silently to 'auto' (C6).
    memory_scope_type = Column(Text, nullable=False, default="auto")
    cluster_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    # C6: the three id sets scope resolution reads. `included_*` is the
    # cross-chat set a 'manual' conversation reads (own id implicit); the two
    # `excluded_*` sets are honored in EVERY mode (user decision 2026-07-28) —
    # "keep this memory, stop retrieving it", the missing middle between C6's
    # add-to-scope and C10's delete-forever.
    included_conversation_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    excluded_conversation_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    excluded_cluster_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    # C7 D9 (G8): session-stickiness state, formerly the in-memory
    # SESSION_STATE dict in main.py. The topic-shift overlap check reads the
    # previous turn's tags straight off the latest episodic row — no columns
    # needed for those.
    sticky_model = Column(Text, nullable=True)
    consecutive_shifts = Column(Integer, nullable=False, default=0, server_default="0")
    # E1 (D11): attaching a conversation to a project gives it coding scope —
    # retrieval resolves to the project's conversations' batch-set ∪ its
    # code-graph allowance. Incognito ('none') conversations refuse attachment.
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)

    episodic_turns = relationship("EpisodicMemory", back_populates="conversation")


class EpisodicMemory(Base):
    __tablename__ = "episodic_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("context_clusters.id"), nullable=True)
    parent_message_id = Column(UUID(as_uuid=True), ForeignKey("episodic_memory.id"), nullable=True)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    # C6: one sitting = one session; a >session_gap_minutes silence opens a new
    # one (resolved at write time in src/memory/session.py). NULL on rows that
    # predate the migration. Feeds session-aware clustering (C5) and the
    # session-gap maintenance trigger (C7).
    session_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    # G16: turns of none-scoped ("incognito") conversations — invisible to every
    # other scope's retrieval, skipped by the derivative pipelines
    # (codex/procedural/clustering/batch-summary/reflection), readable only when
    # retrieval is explicitly scoped to their own conversation.
    is_private = Column(Boolean, default=False, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    # F14: 'original' = the timestamp came from the source (live turn or
    # provider export); 'synthetic_raw_import' = synthesized from file mtime
    # for a raw text dump — T-timelines caveat those dates.
    ts_provenance = Column(Text, nullable=False, default="original",
                           server_default="original")
    topic_tags = Column(ARRAY(Text), default=[])
    intent_tags = Column(ARRAY(Text), default=[])
    context_reliance = Column(Text, nullable=False)
    entropy_score = Column(Float, nullable=True)
    lossless_flag = Column(Boolean, nullable=True)  # NULL = not yet evaluated
    raw_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=True)
    # C1: measured fraction of the turn's must-preserve terms (NER entities +
    # figures + identifiers) retained by summary_text. Read-time representation
    # choice and budget degradation never trust a summary below threshold.
    # NULL = no summary or legacy pre-C1 summary.
    summary_coverage = Column(Float, nullable=True)
    # C3: one-line abstract (generated in the same LLM call as the summary) —
    # the third level of the raw → summary → abstract hierarchy. Used only by
    # budget degradation (never *preferred* by the read-time chooser).
    abstract_text = Column(Text, nullable=True)
    embedding = Column(Vector(1024), nullable=True)
    decay_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    is_archived = Column(Boolean, default=False)
    is_bookmarked = Column(Boolean, default=False)
    decay_immune = Column(Boolean, default=False)
    # F10 preserve/hybrid: temporary decay immunity that self-expires — the
    # decay UPDATEs skip rows whose window is still open. NULL = no window.
    # (decay_immune stays the permanent flag bookmarks own.)
    decay_immune_until = Column(DateTime(timezone=True), nullable=True)
    inject_raw = Column(Boolean, default=True)
    is_document = Column(Boolean, default=False)
    # C12 D7: this turn was a paste big enough to be promoted into its own
    # document/transcript conversation. The turn itself is NEVER rewritten —
    # what the user pasted stays verbatim — it is only skipped by the chunker
    # and by retrieval, because the promoted conversation is the readable copy.
    promoted_document_id = Column(UUID(as_uuid=True),
                                  ForeignKey("documents.id"), nullable=True)
    # G11: which batch summary covers this turn. NULL = not yet summarised, and
    # that is the batch summariser's selection predicate. Before this column the
    # worker had NO way to tell — its docstring claimed it skipped
    # already-summarised turns while nothing excluded them, so every cadence
    # pass re-ran the LLM over the same turns and appended another summary row,
    # which the batch-summary retrieval leg then injected as duplicates.
    # ON DELETE SET NULL: deleting a summary must make its turns eligible again,
    # never orphan the reference or block the delete.
    batch_summary_id = Column(UUID(as_uuid=True),
                              ForeignKey("batch_summaries.id",
                                         ondelete="SET NULL"),
                              nullable=True, index=True)
    idempotency_key = Column(Text, unique=True, nullable=False)

    conversation = relationship("Conversation", back_populates="episodic_turns")


class EpisodicChunk(Base):
    """C2: retrieval-grade chunks of document turns (is_document pastes),
    produced by workers/document_chunker.py with the shared chunker
    (src/memory/chunking.py). Documents are never injected whole anymore —
    the vector leg searches these embeddings directly, and BM25-found doc
    rows inject their keyword-relevant chunks. Visibility (decay, archive,
    privacy, scope) is enforced through the parent turn at query time.
    C3 extends this store to chunk-aware retrieval for ALL turns; C17
    re-embeds it at 1024 along with everything else."""
    __tablename__ = "episodic_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    turn_id = Column(UUID(as_uuid=True),
                     ForeignKey("episodic_memory.id", ondelete="CASCADE"),
                     nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)


class EpisodicClusterLink(Base):
    __tablename__ = "episodic_cluster_links"

    episodic_id = Column(UUID(as_uuid=True), ForeignKey("episodic_memory.id"), primary_key=True)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("context_clusters.id"), primary_key=True)

class MemorySlot(Base):
    __tablename__ = "memory_slots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slot_name = Column(Text, nullable=False)  # valid names per tier: services/slots.py
    content = Column(Text, default="")
    token_count = Column(Integer, default=0)
    version = Column(Integer, default=1)
    last_updated = Column(DateTime(timezone=True), default=utcnow)
    updated_by = Column(Text, nullable=False)  # user | agent | reflection | chat_command | mcp_edit | system
    is_active = Column(Boolean, default=True)
    # C9 (D5): three-tier slots. 'global' rows keep NULL anchors; 'project'
    # rows carry project_id; 'conversation' rows carry conversation_id.
    # Uniqueness = NULLS NOT DISTINCT index on (slot_name, scope_tier,
    # project_id, conversation_id) — the migration owns it.
    scope_tier = Column(Text, nullable=False, default="global", server_default="global")
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)

class ContextCluster(Base):
    __tablename__ = "context_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    tags = Column(ARRAY(Text), default=[])
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)
    embedding = Column(Vector(1024), nullable=True)
    
class CodexEntity(Base):
    __tablename__ = "codex_entities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name = Column(Text, nullable=False, unique=True)
    aliases = Column(ARRAY(Text), default=[])
    tags = Column(ARRAY(Text), default=[])
    # A7: entity_type is the structural node type (person/place/software/concept/
    # organization/entity), inferred from an entity's relations; the code graph
    # (E1b) sets it deterministically. description is the enriched "note body"
    # (Obsidian-style) written by the reflection enrichment worker; context_payload
    # is assembled from description + properties + links (both directions).
    entity_type = Column(Text, default="entity")
    description = Column(Text, default="")
    properties = Column(JSONB, default={})
    context_payload = Column(Text, default="")
    embedding = Column(Vector(1024), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=utcnow)
    # E1b (D3): one codex, namespaced. source ∈ conversation | static_analysis
    # | derived. Non-conversation rows are DERIVED memory: decay-exempt,
    # regenerable (bulk-rebuild safe, no CodexEvents), and excluded from
    # conversational retrieval unless the query's project matches.
    project_id = Column(UUID(as_uuid=True), nullable=True)
    source = Column(Text, nullable=False, default="conversation",
                    server_default="conversation")


class CodexEdge(Base):
    __tablename__ = "codex_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    target_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    relation = Column(Text, nullable=False)
    strength = Column(Float, default=1.0)
    source_batch = Column(UUID(as_uuid=True), nullable=False)
    confidence = Column(Text, default="pending")  # pending | active
    # A3: how much the extraction itself is trusted (0-1). Seeded by NER
    # grounding at write time (grounded high, grounding-rejected low), raised
    # by corroborating re-extractions. Orthogonal to strength (usage dynamics).
    extraction_confidence = Column(Float, default=1.0)
    # A8: polarity. False = the relation holds (X uses Y); True = it is negated /
    # absent (X does NOT use Y, X distrusts Y). Lets the graph store the negative
    # of any relation without doubling the controlled vocabulary. A negated edge
    # is a stored fact, not a navigable link.
    negated = Column(Boolean, default=False)
    valid_from = Column(DateTime(timezone=True), default=utcnow)
    valid_until = Column(DateTime(timezone=True), nullable=True)  # NULL = currently true
    # E1b (D3): mirrors codex_entities.source — static-analysis edges are
    # derived memory (decay-exempt, journal-free, regenerable).
    source = Column(Text, nullable=False, default="conversation",
                    server_default="conversation")


class CodexEvent(Base):
    __tablename__ = "codex_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    event_type = Column(Text, nullable=False)  # edge_added, edge_expired, property_updated, etc.
    payload = Column(JSONB, default={})
    timestamp = Column(DateTime(timezone=True), default=utcnow)
    batch_source = Column(UUID(as_uuid=True), nullable=False)
    compacted = Column(Boolean, default=False)


class CodexSnapshot(Base):
    __tablename__ = "codex_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id = Column(UUID(as_uuid=True), ForeignKey("codex_entities.id"), nullable=False)
    snapshot_ts = Column(DateTime(timezone=True), default=utcnow)
    last_event_id = Column(UUID(as_uuid=True), nullable=False)
    full_state = Column(JSONB, default={})


class ProceduralMemory(Base):
    __tablename__ = "procedural_memory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pattern_name = Column(Text, nullable=False)
    pattern_description = Column(Text, default="")
    topic_tags = Column(ARRAY(Text), default=[])
    trigger_conditions = Column(JSONB, default={})
    reinforcement_count = Column(Integer, default=1)
    confidence_score = Column(Float, default=0.0)
    first_observed = Column(DateTime(timezone=True), default=utcnow)
    last_observed = Column(DateTime(timezone=True), default=utcnow)
    is_active = Column(Boolean, default=True)
    source_batch_ids = Column(ARRAY(UUID(as_uuid=True)), default=[])
    embedding = Column(Vector(1024), nullable=True)
    # E1 (D1): coding conventions are procedural rows scoped to a project —
    # not a fourth pattern store. NULL = user-global pattern (today's rows).
    project_id = Column(UUID(as_uuid=True), nullable=True)


class Project(Base):
    """E1: projects are first-class. A registered repo root (or several —
    monorepo) whose code graph, facts, decisions, and tasks hang off this row.
    settings carries per-project knobs (ignore globs, hook_installed,
    unreachable flag)."""
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False, unique=True)
    slug = Column(Text, nullable=False, unique=True)
    roots = Column(ARRAY(Text), nullable=False)
    settings = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ProjectState(Base):
    """E1/E4: the living "where was I" row — goal, branch, last task, and the
    reconcile/session anchors the session-start block renders from."""
    __tablename__ = "project_state"

    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), primary_key=True)
    goal = Column(Text, nullable=True)
    current_branch = Column(Text, nullable=True)
    last_task_id = Column(UUID(as_uuid=True), nullable=True)
    last_session_at = Column(DateTime(timezone=True), nullable=True)
    last_reconciled_commit = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class Decision(Base):
    """E1/E8: bi-temporal decision memory (mirrors codex_edges' valid_from/
    valid_until; supersession is first-class via superseded_by — no event
    journal needed). decision_type ∈ decision | constraint | incident."""
    __tablename__ = "decisions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    decision = Column(Text, nullable=False)
    rationale = Column(Text, nullable=True)
    alternatives_rejected = Column(JSONB, default=[])
    files_affected = Column(ARRAY(Text), default=[])
    decision_type = Column(Text, nullable=False, default="decision")
    source_batch = Column(UUID(as_uuid=True), nullable=True)   # G17 provenance
    embedding = Column(Vector(1024), nullable=True)
    valid_from = Column(DateTime(timezone=True), default=utcnow)
    valid_until = Column(DateTime(timezone=True), nullable=True)
    superseded_by = Column(UUID(as_uuid=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ProjectTask(Base):
    """E1: lightweight task rows; commits link in via the reconciler (D5
    step 3), staleness (updated_at) feeds the agent's stale_work detector.
    daily_checklist is a VIEW over this table, not a table of its own."""
    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    title = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="pending")  # pending|active|done|dropped
    commit_hashes = Column(ARRAY(Text), default=[])
    files_changed = Column(ARRAY(Text), default=[])
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class Document(Base):
    """C12: the document REGISTRY. The document's *content* does not live here —
    it lives as an ordinary conversation (``conversation_id``, kind 'document'
    or 'transcript') whose turns are its sections, so every existing leg,
    summary, cluster and scope rule applies to it unchanged. This row is the
    identity: de-dup by ``sha256``, provenance for citations, the enable/disable
    toggles (``document_links``), and the knowledge-promotion latch.

    Replaces the v1 rag_documents/rag_chunks pair, whose only writer was an
    unstarted watchdog script and whose reader was a globally-unscoped lookup
    gated on five English nouns."""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True),
                             ForeignKey("conversations.id", ondelete="CASCADE"),
                             nullable=False, index=True)
    filename = Column(Text, nullable=False)
    file_type = Column(Text, nullable=False)     # pdf | docx | csv | md | ...
    kind = Column(Text, nullable=False, default="document")   # document|transcript
    origin = Column(Text, nullable=False, default="upload")   # upload|paste|watch_folder|project
    # Same bytes = same document, always. Re-upload enables the existing one
    # in the requesting conversation instead of ingesting a second copy.
    sha256 = Column(Text, nullable=False, unique=True)
    byte_size = Column(BigInteger, nullable=False, default=0)
    n_sections = Column(Integer, nullable=False, default=0)
    page_count = Column(Integer, nullable=True)   # NULL where meaningless
    source_path = Column(Text, nullable=True)
    # A pasted/blob document has no file to re-read, so its text lives here.
    # Exactly one of source_path / source_text is set. Without it the ENQUEUED
    # ingest — which is the path both live adapters take — failed every blob
    # with "source file is gone", because run_document_ingest re-parses from
    # source_path (found 2026-07-28; the suite only ever ran inline).
    source_text = Column(Text, nullable=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=True)
    # D4 latch: flips true the first time a SECOND distinct conversation enables
    # this document, and never flips back. The document's TEXT stays opt-in
    # forever; this governs only whether the codex/procedural knowledge derived
    # from it is denied to conversations that have not enabled it.
    knowledge_shared = Column(Boolean, nullable=False, default=False)
    shared_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(Text, nullable=False, default="pending")  # pending|ingesting|ready|failed
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class DocumentLink(Base):
    """C12 D3: one row per (document, conversation) that has EVER enabled it.

    ``enabled`` is a live toggle — flip it on for a few prompts, off when done;
    the document's text visibility follows it on the next turn. The row itself
    is never deleted on disable, because ``first_enabled_at`` is the evidence
    the promotion latch reads (two rows = a second conversation reached for
    this document = it is real working knowledge)."""
    __tablename__ = "document_links"

    document_id = Column(UUID(as_uuid=True),
                         ForeignKey("documents.id", ondelete="CASCADE"),
                         primary_key=True)
    conversation_id = Column(UUID(as_uuid=True),
                             ForeignKey("conversations.id", ondelete="CASCADE"),
                             primary_key=True)
    enabled = Column(Boolean, nullable=False, default=True)
    first_enabled_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class SessionReplay(Base):
    __tablename__ = "session_replays"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    event_sequence = Column(ARRAY(JSONB), default=[])
    created_at = Column(DateTime(timezone=True), default=utcnow)


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    # The bulletproof version:
    session_date = Column(Date, default=utcnow)
    topics_covered = Column(ARRAY(Text), default=[])
    decisions_made = Column(Text, default="")
    unresolved_items = Column(Text, default="")
    entities_updated = Column(ARRAY(Text), default=[])
    patterns_observed = Column(ARRAY(Text), default=[])


class IdempotencyKey(Base):
    __tablename__ = "idempotency_keys"

    key = Column(Text, primary_key=True)
    processed_at = Column(DateTime(timezone=True), default=utcnow)


class StoreMeta(Base):
    """G23 D1: store-level metadata, one row per key. Key 'embedding' holds
    the store's embedding identity {model, dim, stamped_at} — create_core()
    refuses to boot when settings disagree with it (src/memory/store_meta.py).
    Keys 'reembed:<table>' hold the re-embed runner's per-table progress
    stamps (kill-safe resume, src/memory/reembed.py)."""
    __tablename__ = "store_meta"

    key = Column(Text, primary_key=True)
    value = Column(JSONB, default={})
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class MaintenanceLedger(Base):
    """C7 D3: per-job schedule state for the in-process maintenance runtime.
    Survives restarts (feeds the overdue catch-up), doubles as the optimistic
    claim lock against a duplicate app instance, and is the surface Track D's
    maintenance agent reads."""
    __tablename__ = "maintenance_ledger"

    job_name = Column(Text, primary_key=True)
    last_started = Column(DateTime(timezone=True), nullable=True)
    last_finished = Column(DateTime(timezone=True), nullable=True)
    # running, ok, retrying, error, yielded — "yielded" (G4a) is NOT a failure:
    # the job stood down mid-flight for a returning user and was requeued with
    # its attempt count untouched.
    last_status = Column(Text, nullable=True)
    last_error = Column(Text, nullable=True)
    runs = Column(Integer, nullable=False, default=0, server_default="0")


class ColdStorage(Base):
    __tablename__ = "cold_storage"

    id = Column(UUID(as_uuid=True), primary_key=True)  # original episodic turn id
    archived_at = Column(DateTime(timezone=True), default=utcnow)
    raw_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=True)
    topic_tags = Column(ARRAY(Text), default=[])
    timestamp = Column(DateTime(timezone=True), nullable=False)
    # T3 (D12): carried from the episodic row so time-scoped retrieval can
    # honor privacy and resurrection can re-attach the turn. NULL
    # conversation_id = legacy pre-migration row → cite-only, never resurrected.
    conversation_id = Column(UUID(as_uuid=True), nullable=True)
    is_private = Column(Boolean, default=False, nullable=False)
    batch_id = Column(UUID(as_uuid=True), nullable=True)
    # C16: carried across at archive time so a resurrected turn - and the T3
    # cold leg itself - can be reached by MEANING. This is the one archive
    # table C17 never gave a vector, which is why the cold leg was still
    # querying `raw_text ILIKE ANY(%keyword%)` against a hardcoded English
    # stoplist: the last purely lexical query in retrieval.
    embedding = Column(Vector(1024), nullable=True)


class CuratedLabel(Base):
    __tablename__ = "curated_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    prompt = Column(Text, nullable=False)
    corrected_topic_labels = Column(ARRAY(Text), default=[])
    corrected_intent_labels = Column(ARRAY(Text), default=[])
    # v1: the single 3-way string (Zero_Shot | Long_Term_Memory | Real_Time_Search).
    corrected_context_reliance = Column(Text, nullable=False)
    # B1 v2: context-reliance is four INDEPENDENT signals, so corrections are a
    # set, not a choice. An empty list on a v2 row is meaningful ("none of the
    # four" = the derived Zero_Shot state), which is why schema_version and not
    # emptiness decides how a row is read. F9's feedback UI writes v2 rows.
    corrected_context_labels = Column(ARRAY(Text), default=[])
    schema_version = Column(Integer, nullable=False, default=1, server_default="1")
    created_at = Column(DateTime(timezone=True), default=utcnow)

class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_type = Column(Text, nullable=False)
    item_content = Column(JSONB, default={})
    # pending | approved | rejected | resolved (agent-settled, D1/D2)
    # | stale (referenced conversation deleted, C10)
    status = Column(Text, default="pending")
    created_at = Column(DateTime(timezone=True), default=utcnow)

class BatchSummary(Base):
    __tablename__ = "batch_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    # ⚠ WRITE-ONLY, and not usable as coverage provenance (G11, 2026-08-08).
    # These are positions in the *filtered, ordered list the producing run
    # happened to build* — a list that shifts between runs as decay advances —
    # not stable turn identifiers. Nothing reads them. Coverage is recorded on
    # the turn instead (`episodic_memory.batch_summary_id`). Kept rather than
    # dropped because they are harmless and a column drop is not this item's
    # business; do not start trusting them.
    start_turn_index = Column(Integer, nullable=False)
    end_turn_index = Column(Integer, nullable=False)
    summary_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class ConversationSummary(Base):
    """C4: ONE evolving summary per conversation — "the whole conversation so
    far, current" (never a batch_summaries range row). Maintained incrementally
    by the conversation_summary burst job; consumed by the assembler (active
    conversation, past the window condition) and the batch-summary retrieval
    leg (cross-conversation hits). T-track era digests read this shape as-is.
    Cascade delete = C10's conversation deletion takes the summary with it."""
    __tablename__ = "conversation_summaries"

    conversation_id = Column(UUID(as_uuid=True),
                             ForeignKey("conversations.id", ondelete="CASCADE"),
                             primary_key=True)
    summary_text = Column(Text, nullable=False)
    covers_through = Column(DateTime(timezone=True), nullable=True)
    covers_turns = Column(Integer, nullable=False, default=0)
    embedding = Column(Vector(1024), nullable=True)
    updated_at = Column(DateTime(timezone=True), default=utcnow)


class ImportRun(Base):
    """F10: one replay import of a provider export (or F14 raw dump).

    kind mirrors G23's manifest vocabulary: this is the REPLAY path (turns
    re-live the pipeline); scripts/ice_import.py's state-copy is the sibling.
    Progress counters heartbeat via updated_at; a `running` row older than
    the staleness window is treated as crashed and auto-aborted on the next
    start_import (resume rides the import_conversations hash ledger)."""
    __tablename__ = "import_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_path = Column(Text, nullable=False)
    source_format = Column(Text, nullable=False)   # chatgpt|claude|deepseek|jsonl|raw
    policy = Column(Text, nullable=False)          # hybrid|preserve|fast_forward|fresh
    kind = Column(Text, nullable=False, default="replay", server_default="replay")
    status = Column(Text, nullable=False, default="running",
                    server_default="running")      # running|completed|failed|aborted
    total_conversations = Column(Integer, nullable=False, default=0)
    total_turns = Column(Integer, nullable=False, default=0)
    done_conversations = Column(Integer, nullable=False, default=0)
    done_turns = Column(Integer, nullable=False, default=0)
    skipped_conversations = Column(Integer, nullable=False, default=0)
    failed_turns = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    report = Column(JSONB, nullable=True)


class ImportConversation(Base):
    """F10 D4: the hash-skip ledger — one row per FULLY replayed conversation
    (written after its last turn + clustering), keyed by the normalized-
    content hash. Re-running an export skips hashes present here; a
    mid-conversation kill leaves no row, so that conversation re-runs and
    its per-turn idempotency keys dedupe. conversation_id is deliberately
    NOT a FK: C10 deletion of an imported conversation keeps this row as a
    tombstone — re-imports do not resurrect what the user chose to forget."""
    __tablename__ = "import_conversations"

    content_hash = Column(Text, primary_key=True)
    # Nullable: the engine (import_conversations) can replay a source with no
    # ImportRun row (direct/headless calls, spec D1's signature); the REST/CLI
    # path always supplies one.
    import_id = Column(UUID(as_uuid=True), ForeignKey("import_runs.id"),
                       nullable=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), nullable=False)
    title = Column(Text, nullable=True)
    n_turns = Column(Integer, nullable=False, default=0)
    imported_at = Column(DateTime(timezone=True), default=utcnow)

class CodexRelationGap(Base):
    """G32/a1: the fact the extractor wanted to state but the vocabulary
    cannot express — kept instead of destroyed.

    `codex_extractor` filters every triplet whose relation is not in
    ALLOWED_RELATIONS. Measured on 300 real turns (2026-08-04, see
    PROVENANCE): **67.8% of produced relations fail that filter**, and the
    three most common are `is`, `has` and `includes` — ordinary English the
    197-word vocabulary lacks, not near-misses of it. Those rows were dropped
    with no log line, which is why nobody knew.

    The row keeps BOTH halves, and that is the point: `proposed_relation` is
    what the model actually said, so the accumulated table ranks what the
    vocabulary is missing — the input Z2's mini-experiment needs and today
    cannot get, because the evidence is deleted at the moment it is created.
    Keeping subject and object too means a relation added later can be
    replayed straight into the graph with **no new LLM call**: the fact was
    already extracted, it just had nowhere to go.

    Deliberately NOT a codex_edges row: an unmappable relation is not a fact
    the graph can hold, and writing it as one is exactly the confident-lie
    failure the enum measurement rejected (~78% of forced picks were wrong).
    """
    __tablename__ = "codex_relation_gaps"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # What the model asked for, normalized only for grouping (lowercase,
    # separators unified) — the raw form is kept alongside it because the
    # surface form is itself evidence about the model's habits.
    proposed_relation = Column(Text, nullable=False, index=True)
    raw_relation = Column(Text, nullable=False)
    subject = Column(Text, nullable=False)
    object = Column(Text, nullable=False)
    negated = Column(Boolean, default=False)
    # Provenance: enough to re-read the source turn when judging whether the
    # relation deserves to exist.
    batch_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    conversation_id = Column(UUID(as_uuid=True), nullable=True)
    # pending | promoted (the relation was added and this row replayed)
    # | dismissed (judged not worth a vocabulary entry)
    status = Column(Text, default="pending", index=True)
    # Set when the ladder found a mapping but it was not confident enough to
    # apply — records the near-miss without acting on it.
    suggested_relation = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
