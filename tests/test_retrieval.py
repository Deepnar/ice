#!/usr/bin/env python3
"""Retrieval smoke — seeds two marked turns + one codex entity, runs the
full orchestrator + assembler path, checks the seeded memory comes back.

House pattern: live Postgres, uniquely-marked fixture rows, cleanup in
`finally` — NEVER truncates. (The pre-G23 version of this file TRUNCATEd
episodic_memory/conversations/codex_* on every run and used the MiniLM-era
384 embedder — both fixed with C17.)

Run: uv run python tests/test_retrieval.py
"""
import os
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import text

from src.api.db import SessionLocal
from src.classifier.classifier import ClassificationResult
from src.memory.embedder import get_embedder
from src.memory.models import CodexEntity, Conversation, EpisodicMemory
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


HEX = uuid.uuid4().hex[:6]
MARK = "retrmark" + HEX.translate(str.maketrans("0123456789", "abcdefghij"))

embedder = get_embedder()
db = SessionLocal()
conv_id = None
turn_ids: list = []
entity_id = None

try:
    conv = Conversation(memory_scope_type="auto")
    db.add(conv)
    db.commit()
    conv_id = conv.id

    q = f"What is the {MARK} middleware?"
    t1_text = (f"User: What is the {MARK} middleware?\n\nAssistant: "
               f"{MARK} is a memory middleware for AI assistants.")
    t2_text = (f"User: I have a bug in my {MARK} setup\n\nAssistant: "
               "Check the indentation.")
    for body, lossless, summary in (
            (t1_text, False,
             f"{MARK} is a memory middleware providing context."),
            (t2_text, True, None)):
        user_half = body.split("\n\nAssistant:")[0][len("User: "):]
        turn = EpisodicMemory(
            conversation_id=conv_id, batch_id=uuid.uuid4(),
            timestamp=datetime.now(timezone.utc),
            topic_tags=["Software_&_Tech"], intent_tags=["Generation"],
            context_reliance="Long_Term_Memory", raw_text=body,
            lossless_flag=lossless, summary_text=summary,
            embedding=embedder.encode(
                user_half, convert_to_tensor=False).tolist(),
            idempotency_key=f"retr-{HEX}-{lossless}")
        db.add(turn)
        db.flush()
        turn_ids.append(turn.id)
    db.commit()

    entity = CodexEntity(
        canonical_name=f"{MARK}engine",
        aliases=[MARK],
        context_payload=f"{MARK} is an engine that uses PostgreSQL and pgvector.",
        embedding=embedder.encode(
            f"{MARK}engine", convert_to_tensor=False).tolist(),
        last_updated=datetime.now(timezone.utc))
    db.add(entity)
    db.commit()
    entity_id = entity.id

    orchestrator = HybridRetrievalOrchestrator(db, embedder)
    result = ClassificationResult(
        topic_tags=["Software_&_Tech"], intent_tags=["Generation"],
        context_reliance="Long_Term_Memory", raw_probs=[0.9] * 25,
        max_confidence=0.9, prompt=q)
    prompt_embedding = embedder.encode(q, convert_to_tensor=False).tolist()

    fragments = orchestrator.retrieve(
        classification=result, conversation_id=str(conv_id),
        prompt_embedding=prompt_embedding, scope=None)

    print(f"Retrieved {len(fragments)} fragments:")
    for f in fragments[:6]:
        print(f"  [{f.source_type}] score={f.score:.4f} text={f.text[:70]}…")

    check("retrieval returns fragments", len(fragments) > 0)
    check("the marked memory is retrieved",
          any(MARK in (f.text or "") for f in fragments))

    from src.api.prompt_assembler import assemble_prompt
    messages = assemble_prompt([], fragments, q)
    check("assembler injects the memory into the prompt",
          messages and any(MARK in (m.get("content") or "")
                           for m in messages))

finally:
    try:
        db.rollback()
        if entity_id is not None:
            db.execute(text("DELETE FROM codex_entities WHERE id = :i"),
                       {"i": entity_id})
        if turn_ids:
            db.execute(text("DELETE FROM episodic_memory WHERE id = ANY(:ids)"),
                       {"ids": turn_ids})
        if conv_id is not None:
            db.execute(text("DELETE FROM conversation_summaries "
                            "WHERE conversation_id = :c"), {"c": conv_id})
            db.execute(text("DELETE FROM conversations WHERE id = :c"),
                       {"c": conv_id})
        db.commit()
    finally:
        db.close()

print(f"\n{_passed} passed, {_failed} failed")
sys.exit(1 if _failed else 0)
