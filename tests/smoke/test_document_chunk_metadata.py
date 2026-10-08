"""Actual lexical SQL must reach its existing document chunk selector."""

import uuid
from types import SimpleNamespace as NS

from sqlalchemy import text

from src.api.db import SessionLocal
from src.api.prompt_budget import assemble_budgeted_prompt
from src.memory.models import Conversation, EpisodicChunk, EpisodicMemory
from src.memory.source import single_provenance
from src.memory.tokens import count
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator


def test_document_flag_reaches_lexical_selector_and_final_same_budget_prompt():
    cid = uuid.uuid4()
    value = "control-8391"
    good = f"Database connection example: password = '{value}'."
    bad = "The application user enters a password into a login form."
    raw = "Unrelated background. " * 4000 + "\n" + bad + "\n" + good
    query = "What is the password for the database?"
    with SessionLocal() as db:
        try:
            db.add(Conversation(id=cid)); db.flush()
            row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(), raw_text=raw,
                source_spans=single_provenance(raw, "assistant"), inject_raw=True,
                context_reliance="Long_Term_Memory",
                is_document=True, decay_score=1., idempotency_key=str(uuid.uuid4()))
            db.add(row); db.flush()
            db.add_all([EpisodicChunk(turn_id=row.id, chunk_index=i, chunk_text=chunk)
                        for i, chunk in enumerate((bad, good))])
            db.commit()
            orchestrator = HybridRetrievalOrchestrator(db, None)
            # Reproduce the prior missing-column shape through actual SQL.
            old_row = db.execute(text("SELECT id, raw_text, source_spans, inject_raw, conversation_id, "
                "timestamp, ts_provenance FROM episodic_memory WHERE id=:id"), {"id": row.id}).one()
            oversized = orchestrator._rows_to_fragments([old_row], "episodic", prompt_text=query)
            wrong_chunk = ContextFragment(text=bad, source_type="episodic", score=.1,
                token_count=count(bad), source_batch_id=str(row.id))
            before = orchestrator._enforce_token_budget([*oversized, wrong_chunk], max_tokens=1024)
            assert before and all(value not in fragment.text for fragment in before)
            assert str(row.id) in {fragment.source_batch_id for fragment in before}

            classification = NS(prompt=query, intent_tags=[], topic_tags=[])
            candidates = orchestrator._bm25_episodic(classification, None, str(cid))
            assert candidates and any(value in f.text for f in candidates)
            assert all(not f.covers_entire_source for f in candidates)
            after = orchestrator._enforce_token_budget(candidates, max_tokens=1024)
            assert after and any(value in f.text for f in after)
            prepared = assemble_budgeted_prompt(serving_window=2048, generation_reserve=300,
                memory_slots=[], retrieved_fragments=after, user_message=query)
            assert prepared.ledger.fits()
            assert value in "\n".join(m["content"] for m in prepared.messages)
        finally:
            db.rollback()
            db.query(EpisodicMemory).filter_by(conversation_id=cid).delete()
            db.query(Conversation).filter_by(id=cid).delete()
            db.commit()
