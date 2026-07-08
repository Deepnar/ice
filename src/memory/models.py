import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date, Text,
    ForeignKey, ARRAY, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

def utcnow():
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    memory_scope_type = Column(Text, nullable=False, default="auto")  # none, auto, project, manual
    cluster_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    custom_filter = Column(Text, nullable=True)

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
    embedding = Column(Vector(384), nullable=True)
    decay_score = Column(Float, default=1.0)
    access_count = Column(Integer, default=0)
    is_archived = Column(Boolean, default=False)
    is_bookmarked = Column(Boolean, default=False)
    decay_immune = Column(Boolean, default=False)
    inject_raw = Column(Boolean, default=True)
    is_document = Column(Boolean, default=False)
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
    embedding = Column(Vector(384), nullable=True)


class EpisodicClusterLink(Base):
    __tablename__ = "episodic_cluster_links"

    episodic_id = Column(UUID(as_uuid=True), ForeignKey("episodic_memory.id"), primary_key=True)
    cluster_id = Column(UUID(as_uuid=True), ForeignKey("context_clusters.id"), primary_key=True)

class MemorySlot(Base):
    __tablename__ = "memory_slots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    slot_name = Column(Text, nullable=False)  # one of the seven predefined names
    content = Column(Text, default="")
    token_count = Column(Integer, default=0)
    version = Column(Integer, default=1)
    last_updated = Column(DateTime(timezone=True), default=utcnow)
    updated_by = Column(Text, nullable=False)  # user | reflection_worker
    is_active = Column(Boolean, default=True)

class ContextCluster(Base):
    __tablename__ = "context_clusters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    tags = Column(ARRAY(Text), default=[])
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow)
    embedding = Column(Vector(384), nullable=True)
    
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
    embedding = Column(Vector(384), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=utcnow)


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
    embedding = Column(Vector(384), nullable=True)


class RAGDocument(Base):
    __tablename__ = "rag_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    file_type = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=utcnow)
    token_count = Column(Integer, default=0)


class RAGChunk(Base):
    __tablename__ = "rag_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("rag_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=False)


class SentinelRule(Base):
    __tablename__ = "sentinel_rules"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False)
    description = Column(Text, default="")
    is_active = Column(Boolean, default=True)
    trigger_type = Column(Text, nullable=False)  # threshold, frequency, absence, contradiction, composite
    trigger_conditions = Column(JSONB, default={})
    action_type = Column(Text, nullable=False)   # notify, schedule_worker, create_review_item, log_event, propose_memory_update
    action_payload = Column(JSONB, default={})
    cooldown_seconds = Column(Integer, default=0)
    last_fired_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)


class SentinelEvent(Base):
    __tablename__ = "sentinel_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("sentinel_rules.id"), nullable=False)
    fired_at = Column(DateTime(timezone=True), default=utcnow)
    trigger_state = Column(JSONB, default={})
    action_taken = Column(Text, nullable=False)


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


class ColdStorage(Base):
    __tablename__ = "cold_storage"

    id = Column(UUID(as_uuid=True), primary_key=True)  # original episodic turn id
    archived_at = Column(DateTime(timezone=True), default=utcnow)
    raw_text = Column(Text, nullable=False)
    summary_text = Column(Text, nullable=True)
    topic_tags = Column(ARRAY(Text), default=[])
    timestamp = Column(DateTime(timezone=True), nullable=False)


class CuratedLabel(Base):
    __tablename__ = "curated_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    batch_id = Column(UUID(as_uuid=True), nullable=False)
    prompt = Column(Text, nullable=False)
    corrected_topic_labels = Column(ARRAY(Text), default=[])
    corrected_intent_labels = Column(ARRAY(Text), default=[])
    corrected_context_reliance = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow)

class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_type = Column(Text, nullable=False)
    item_content = Column(JSONB, default={})
    status = Column(Text, default="pending")  # pending, approved, rejected
    created_at = Column(DateTime(timezone=True), default=utcnow)

class BatchSummary(Base):
    __tablename__ = "batch_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False)
    start_turn_index = Column(Integer, nullable=False)
    end_turn_index = Column(Integer, nullable=False)
    summary_text = Column(Text, nullable=False)
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)