"""C2 behavioral test: document chunking + chunk-level retrieval, live DB.

Covers: shared-chunker extraction (codex aliases intact), chunking worker
(idempotency + catch-up), vector leg (doc rows excluded, chunks compete,
privacy through the parent join), BM25 doc rows injecting relevant chunks,
and the legacy no-chunks fallback (500-word cap, no more whole-doc dumps).

Inserts its own rows, deletes them after. Run:
    uv run python tests/test_document_chunking.py
"""
import os
import sys
import uuid
from types import SimpleNamespace
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.db import SessionLocal
from src.memory.models import Conversation, EpisodicMemory, EpisodicChunk
from src.memory.chunking import chunk_text
import src.workers.document_chunker as dc
from src.retrieval.orchestrator import HybridRetrievalOrchestrator

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failed += 1
        print(f"  FAIL  {name}")


db = SessionLocal()
conv = Conversation(memory_scope_type="auto")
conv_priv = Conversation(memory_scope_type="none")
db.add_all([conv, conv_priv])
db.commit()

# A document whose LAST section carries a distinctive fact — the failure mode
# C2 fixes is exactly "the fact is buried deep in a doc injected whole/mush".
filler = ("This section discusses ordinary project logistics and scheduling matters. " * 120)
marker_section = ("The migration codename is VERMILLION-KESTREL and it ships in week nine. " * 10)
doc_text = f"User: here is the doc\n\nAssistant: noted\n\n{filler}\n\n{marker_section}"

turn = EpisodicMemory(
    conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
    context_reliance="Long_Term_Memory", raw_text=doc_text, is_document=True,
    inject_raw=True, decay_score=1.0, embedding=[0.01] * 384,
    idempotency_key=f"test-c2-{uuid.uuid4()}",
)
legacy = EpisodicMemory(  # is_document but never chunked (pre-C2)
    conversation_id=conv.id, batch_id=uuid.uuid4(), timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
    context_reliance="Long_Term_Memory", raw_text="User: x\n\nAssistant: y\n\n" + ("legacy words " * 1500),
    is_document=True, inject_raw=True, decay_score=1.0, embedding=[0.01] * 384,
    idempotency_key=f"test-c2-{uuid.uuid4()}",
)
priv_doc = EpisodicMemory(
    conversation_id=conv_priv.id, batch_id=uuid.uuid4(), timestamp=datetime.now(timezone.utc),
    topic_tags=["Software_&_Tech"], intent_tags=["Factual_Retrieval"],
    context_reliance="Long_Term_Memory", is_private=True,
    raw_text="User: p\n\nAssistant: q\n\n" + ("the secret incognito ledger entry " * 300),
    is_document=True, inject_raw=True, decay_score=1.0, embedding=[0.01] * 384,
    idempotency_key=f"test-c2-{uuid.uuid4()}",
)
db.add_all([turn, legacy, priv_doc])
db.commit()

try:
    print("── shared chunker ──")
    chunks = chunk_text("A sentence here. " * 500)
    check("chunker splits long prose", len(chunks) > 2)
    from src.workers.codex_extractor import _chunk_text as codex_alias
    check("codex extractor still chunks via the shared module (alias)",
          codex_alias("A sentence here. " * 500) == chunks)

    print("── chunking worker ──")
    n = dc.run_chunk_turn(db, turn)
    check(f"document chunked into {n} chunks", n > 2)
    n2 = dc.run_chunk_turn(db, turn)
    check("idempotent: second run stores nothing", n2 == 0)
    n3 = dc.run_chunk_turn(db, priv_doc)
    check("private document chunked too (visibility via parent join)", n3 > 1)

    print("── catch-up heals the legacy doc ──")
    healed = dc.run_pending_documents(db, limit=10)
    check("catch-up chunked at least the legacy doc", healed >= 1)
    check("legacy doc now has chunks",
          db.query(EpisodicChunk).filter_by(turn_id=legacy.id).count() > 1)
    # restore the 'legacy = unchunked' state for the fallback test below
    db.query(EpisodicChunk).filter_by(turn_id=legacy.id).delete()
    db.commit()

    print("── vector leg: chunks compete, whole docs don't ──")
    orch = HybridRetrievalOrchestrator(db, dc.shared_embedder)
    q_emb = dc.shared_embedder.encode(
        "what is the migration codename?", convert_to_tensor=False).tolist()
    clf = SimpleNamespace(intent_tags=["Factual_Retrieval"], topic_tags=["Software_&_Tech"],
                          prompt="what is the migration codename?", max_confidence=0.9)
    frags = orch._vector_episodic(q_emb, clf, scope=None, conv_id=str(conv.id))
    texts = [f.text for f in frags]
    check("a CHUNK containing the buried fact is retrieved",
          any("VERMILLION-KESTREL" in t for t in texts))
    check("no whole-document fragment (nothing near full doc size)",
          all(len(t.split()) < 900 for t in texts))
    hit = next(f for f in frags if "VERMILLION-KESTREL" in f.text)
    check("chunk provenance points at the parent turn", hit.source_batch_id == str(turn.id))

    print("── vector chunks: privacy through the parent join ──")
    q2 = dc.shared_embedder.encode("secret incognito ledger", convert_to_tensor=False).tolist()
    global_chunks = orch._vector_chunks(q2, scope=None, conv_id=None)
    check("private doc's chunks invisible globally",
          not any("incognito ledger" in f.text for f in global_chunks))
    own_chunks = orch._vector_chunks(q2, scope=None, conv_id=str(conv_priv.id))
    check("private doc's chunks visible to its own conversation",
          any("incognito ledger" in f.text for f in own_chunks))

    print("── BM25-side: doc rows inject relevant chunks, never the whole doc ──")
    row = db.execute(__import__("sqlalchemy").text("""
        SELECT id, raw_text, summary_text, summary_coverage, lossless_flag,
               inject_raw, conversation_id, is_bookmarked, is_document,
               timestamp, 1.0 AS score
        FROM episodic_memory WHERE id = :tid
    """), {"tid": turn.id}).fetchone()
    frags = orch._rows_to_fragments([row], "episodic",
                                    prompt_text="migration codename week nine",
                                    classification=clf)
    check("doc fragment contains the keyword-relevant chunk",
          frags and "VERMILLION-KESTREL" in frags[0].text)
    # Structural bound: ≤2 chunks of ≤550 tokens (~415 words) + overlap ≈ ≤950
    # words, regardless of how big the document is (the old path injected ALL
    # of it via word_cap=999999).
    check("doc fragment is a bounded selection (≤2 chunks), not the whole doc",
          frags and len(frags[0].text.split()) <= 950
          and len(frags[0].text.split()) < len(doc_text.split()))

    row_legacy = db.execute(__import__("sqlalchemy").text("""
        SELECT id, raw_text, summary_text, summary_coverage, lossless_flag,
               inject_raw, conversation_id, is_bookmarked, is_document,
               timestamp, 1.0 AS score
        FROM episodic_memory WHERE id = :tid
    """), {"tid": legacy.id}).fetchone()
    frags = orch._rows_to_fragments([row_legacy], "episodic",
                                    prompt_text="legacy words",
                                    classification=clf)
    check("legacy doc without chunks: capped at ~500 words (no whole-dump)",
          frags and len(frags[0].text.split()) <= 501)

finally:
    db.rollback()
    for c in (conv, conv_priv):
        ids = [t.id for t in db.query(EpisodicMemory).filter_by(conversation_id=c.id)]
        if ids:
            db.query(EpisodicChunk).filter(EpisodicChunk.turn_id.in_(ids)).delete(
                synchronize_session=False)
        db.query(EpisodicMemory).filter_by(conversation_id=c.id).delete()
        db.query(Conversation).filter_by(id=c.id).delete()
    db.commit()
    db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
