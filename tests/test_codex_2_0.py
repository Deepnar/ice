#!/usr/bin/env python3
"""Comprehensive integration test for Codex 2.0 – structural tests use
hard‑coded triplets, extraction uses background model on port 8002."""

import sys, os, uuid, json
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.db import SessionLocal
from src.memory.models import (
    EpisodicMemory, Conversation, CodexEntity, CodexEdge, CodexEvent
)
from src.workers.codex_extractor import (
    extract_triplets, handle_triplet, get_or_create_entity,
    PROPERTY_RELATIONS, MULTI_VALUED_RELATIONS, ALLOWED_RELATIONS,
)
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.classifier.schemas import ClassificationResult
from src.memory.embedder import get_embedder

# ---------------------------------------------------------------------------
def truncate_db():
    db = SessionLocal()
    from sqlalchemy import text
    db.execute(text(
        "TRUNCATE episodic_memory, conversations, codex_entities, codex_edges, "
        "codex_events, codex_snapshots, idempotency_keys RESTART IDENTITY CASCADE"
    ))
    db.commit()
    db.close()
    print("✅ Database truncated\n")

# ---------------------------------------------------------------------------
def test_extraction():
    """1 – Controlled‑vocabulary extraction (requires background model on 8002)."""
    print("=" * 60)
    print("1. EXTRACTION WITH CONTROLLED VOCABULARY")
    text = (
        "ICE is a memory middleware. It uses PostgreSQL for storage "
        "and Redis for task management. The system depends on Celery "
        "and is part of the Infinite Context Engine project. "
        "My character Kael is a fire mage from the northern kingdom. "
        "Kael's friend Lethe studies at the Arcane Academy."
    )
    triplets = extract_triplets(text)
    if not triplets:
        print("   ⚠️  Extraction returned 0 triplets (background model may be down).")
        print("   Skipping extraction validation – structural tests will still run.\n")
        return

    print(f"   Extracted {len(triplets)} triplets:")
    for t in triplets:
        print(f"      {t['subject']} --[{t['relation']}]--> {t['object']}")

    violations = [t for t in triplets if t["relation"] not in ALLOWED_RELATIONS]
    if violations:
        print(f"   ❌ {len(violations)} relation(s) outside vocabulary:")
        for t in violations:
            print(f"      {t['relation']}")
    else:
        print("   ✅ All relations are within controlled vocabulary")

    subjects = {t["subject"] for t in triplets}
    expected = {"ice", "postgresql", "redis", "celery", "kael"}
    found = subjects & expected
    missing = expected - found
    print(f"   Expected subjects present: {found}{' (all good)' if not missing else f' missing: {missing}'}")

def test_structural():
    """2 – Property, auto‑expiry, contradiction, multi‑valued (no model needed)."""
    print("\n" + "=" * 60)
    print("2. STRUCTURAL TESTS (hard‑coded triplets)")

    db = SessionLocal()
    batch = str(uuid.uuid4())

    # 2a – Property creation
    print("\n   2a. Property creation (name)")
    handle_triplet(db, "kael", "name", "Kael", batch)
    db.commit()

    # Fetch the newly created entity and give it a character tag (for MERA)
    entity = db.query(CodexEntity).filter_by(canonical_name="kael").first()
    entity.tags = ["character"]
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(entity, "tags")
    db.commit()
    print("      ✅ Added tag: character")

    assert entity is not None
    assert entity.properties.get("name") == "Kael"
    print(f"      ✅ name=Kael, properties={entity.properties}")

    # 2b – Property update
    print("\n   2b. Property update (rename Kael → Aroh)")
    handle_triplet(db, "kael", "name", "Aroh", str(uuid.uuid4()))
    db.commit()
    entity = db.query(CodexEntity).filter_by(canonical_name="kael").first()
    assert entity.properties.get("name") == "Aroh", f"Expected Aroh, got {entity.properties}"
    name_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id, CodexEdge.relation == "name"
    ).all()
    active = [e for e in name_edges if e.valid_until is None]
    expired = [e for e in name_edges if e.valid_until is not None]
    print(f"      ✅ Name updated. Active: {len(active)}, expired: {len(expired)}")

    # 2c – Single‑valued relation (works_at → Arcane Academy)
    print("\n   2c. Single‑valued (works_at → Arcane Academy)")
    handle_triplet(db, "kael", "works_at", "arcane academy", str(uuid.uuid4()))
    db.commit()
    works_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id, CodexEdge.relation == "works_at", CodexEdge.valid_until == None
    ).all()
    assert len(works_edges) == 1
    assert works_edges[0].confidence == "pending"
    print("      ✅ One pending works_at edge")

    # 2d – Single‑valued contradiction (works_at → Northern Kingdom)
    print("\n   2d. Contradiction (works_at → Northern Kingdom)")
    handle_triplet(db, "kael", "works_at", "northern kingdom", str(uuid.uuid4()))
    db.commit()
    works_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id, CodexEdge.relation == "works_at"
    ).all()
    active = [e for e in works_edges if e.valid_until is None]
    expired = [e for e in works_edges if e.valid_until is not None]
    assert len(active) == 1
    assert active[0].confidence == "active"          # immediate activation
    target_entity = db.get(CodexEntity, active[0].target_id)
    assert target_entity.canonical_name == "northern kingdom"
    print(f"      ✅ Active works_at = {target_entity.canonical_name} (confidence={active[0].confidence}), expired: {len(expired)}")
        # Give the entity a tag so MERA can find it
    # 2e – Multi‑valued (uses) – both edges should remain active
    print("\n   2e. Multi‑valued (uses)")
    handle_triplet(db, "kael", "uses", "fire magic", str(uuid.uuid4()))
    handle_triplet(db, "kael", "uses", "sword", str(uuid.uuid4()))
    db.commit()
    uses_edges = db.query(CodexEdge).filter(
        CodexEdge.source_id == entity.id, CodexEdge.relation == "uses", CodexEdge.valid_until == None
    ).all()
    assert len(uses_edges) == 2, f"Expected 2 active uses edges, got {len(uses_edges)}"
    t0 = db.get(CodexEntity, uses_edges[0].target_id)
    t1 = db.get(CodexEntity, uses_edges[1].target_id)
    print(f"      ✅ Both uses edges active ({t0.canonical_name}, {t1.canonical_name})")
    db.close()

# ---------------------------------------------------------------------------
def test_ner_and_graph():
    """3 – NER + vector matching + graph traversal (model + DB needed)."""
    print("\n" + "=" * 60)
    print("3. NER + VECTOR MATCHING + GRAPH TRAVERSAL")

    embedder = get_embedder()
    orchestrator = HybridRetrievalOrchestrator.__new__(HybridRetrievalOrchestrator)
    orchestrator.embedder = embedder
    orchestrator.db = SessionLocal()
    orchestrator.ner_model = orchestrator._load_ner_model()
    orchestrator.ner_tokenizer = orchestrator._load_ner_tokenizer()

    prompt = "What does Kael use in combat?"
    entities = orchestrator._extract_entities_with_ner(prompt)
    print(f"   NER extracted: {entities}")

    if entities:
        matched = orchestrator._match_entities_by_similarity(entities)
        print(f"   Matched entities: {[e.canonical_name for e in matched]}")
    else:
        print("   (NER found nothing – will skip vector match)")

    classification = ClassificationResult(
        topic_tags=["Creative_&_Media"], intent_tags=["Factual_Retrieval"],
        context_reliance="Long_Term_Memory", raw_probs=[0.9]*25, max_confidence=0.9,
        prompt=prompt
    )
    fragments = orchestrator._codex_graph(classification)
    if fragments:
        print(f"   Graph traversal returned {len(fragments)} fragment(s)")
        print(f"   Context: {fragments[0].text[:300]}...")
    else:
        print("   Graph traversal returned empty (no matched entities)")

    orchestrator.db.close()

# ---------------------------------------------------------------------------
def test_mera():
    """4 – MERA enumeration (requires background model on 8002)."""
    print("\n" + "=" * 60)
    print("4. MERA ENUMERATION")

    try:
        from src.retrieval.mera import is_mera_candidate, map_category_to_filters, enumerate_entities
    except ImportError:
        print("SKIP: MERA was removed (roadmap A4) — enumeration now lives in "
              "HybridRetrievalOrchestrator._codex_enumeration.")
        return
    db = SessionLocal()

    prompts = [
        "Who are the characters in the story?",
        "List all functions in the codebase.",
        "What tools does ICE use?",
        "Hello how are you?",
    ]
    for p in prompts:
        print(f"   is_mera_candidate('{p[:50]}...') → {is_mera_candidate(p)}")

    prompt = "Who are the characters?"
    filters = map_category_to_filters(db, prompt)
    print(f"   Filters for '{prompt}': {filters}")
    entities = enumerate_entities(db, filters.get("tags", []), filters.get("relations", []))
    print(f"   Enumerated {len(entities)} entities:")
    for e in entities:
        name = (e.properties or {}).get("name", e.canonical_name)
        print(f"      {name} (canonical: {e.canonical_name})")

    db.close()

# ---------------------------------------------------------------------------
def main():
    print("🧪 Codex 2.0 Comprehensive Integration Test\n")
    print("Prerequisites: background model on port 8002 for extraction + MERA\n")

    truncate_db()
    test_extraction()
    test_structural()
    test_ner_and_graph()
    test_mera()

    print("\n" + "=" * 60)
    print("✅ All Codex 2.0 tests completed.")

if __name__ == "__main__":
    main()