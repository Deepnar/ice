"""Re-embed covers every vector column, including composite-key note tables.

The 2026-10-15 finding: `reembed._validate_registry` refused ice_db because
batch_notes, cold_chunks and conversation_notes had no rule. Their rules key a
two-column primary key through a text id expression, so the proof is a real
UPDATE landing on the right row — not that the rule exists.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

OLD = [0.0] * 1024
NEW = [0.5] * 1024
NOTE_TABLES = ("batch_notes", "conversation_notes", "cold_chunks", "codex_facts")


class Encoder:
    def encode(self, texts, **_kwargs):
        return np.array([NEW for _ in texts])


def test_every_orm_vector_column_has_a_rule(empty_replay_database):
    from src.api.db import SessionLocal
    from src.memory import reembed
    with SessionLocal() as db:
        targets = reembed._validate_registry(db, None)   # raises on an unregistered column
    assert set(NOTE_TABLES) <= set(targets)


def test_composite_key_tables_and_facts_reembed_the_right_rows(empty_replay_database):
    from src.api.db import SessionLocal
    from src.memory import reembed
    from src.memory.models import (
        BatchNote, BatchSummary, CodexFact, ColdChunk, ColdStorage, Conversation,
        ConversationNote, ConversationSummary,
    )
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with SessionLocal() as db:
        conv = Conversation()
        db.add(conv)
        db.flush()
        summary = BatchSummary(id=uuid.uuid4(), conversation_id=conv.id, start_turn_index=0,
                               end_turn_index=1, summary_text="pending")
        cold = ColdStorage(id=uuid.uuid4(), raw_text="User: frozen\n\nAssistant: ok",
                           timestamp=now, conversation_id=conv.id)
        db.add_all([summary, cold, ConversationSummary(conversation_id=conv.id,
                                                        summary_text="pending", covers_turns=1)])
        db.flush()
        note = dict(text="A note body.", mode="source", recorded_range="unknown",
                    source_ids=[], batch_ids=[], embedding=OLD)
        db.add_all([BatchNote(summary_id=summary.id, ordinal=1, **note),
                    BatchNote(summary_id=summary.id, ordinal=2, **note),
                    ConversationNote(conversation_id=conv.id, ordinal=1, **note),
                    ColdChunk(id=uuid.uuid4(), turn_id=cold.id, chunk_index=0,
                              chunk_text="A frozen chunk.", embedding=OLD),
                    CodexFact(text="The user said a thing.", text_sha256=uuid.uuid4().hex,
                              speaker="user", modality="asserted", mentioned_at=now,
                              valid_from=now, embedding=OLD)])
        db.commit()
        report = reembed.run(db, embedder=Encoder(), tables=",".join(NOTE_TABLES), force=True)
        assert {t: report[t]["rows"] for t in NOTE_TABLES} == {
            "batch_notes": 2, "conversation_notes": 1, "cold_chunks": 1, "codex_facts": 1}
        db.expire_all()
        for model in (BatchNote, ConversationNote, ColdChunk, CodexFact):
            vectors = [row.embedding for row in db.query(model).all()]
            assert vectors and all(np.allclose(np.asarray(v), NEW) for v in vectors), model
        db.rollback()
        for model in (BatchNote, ConversationNote, ColdChunk, CodexFact):
            db.query(model).delete()
        db.query(ColdStorage).filter_by(id=cold.id).delete()
        db.query(BatchSummary).filter_by(id=summary.id).delete()
        db.query(ConversationSummary).filter_by(conversation_id=conv.id).delete()
        db.query(Conversation).filter_by(id=conv.id).delete()
        db.commit()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
