"""Independent source survives deletion; forgetting its last turn expires the edge."""

import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from src.api.db import SessionLocal
from src.memory.models import (
    CodexClaim, CodexClaimLink, CodexEdge, CodexEntity, CodexEvent,
    ColdStorage, Conversation, EpisodicMemory,
)
from src.memory.source import digest, single_provenance
from src.memory.support import verify_support
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.services.conversations import apply_forget, delete_conversation


def _turn(conversation_id, text):
    batch = uuid.uuid4()
    return EpisodicMemory(conversation_id=conversation_id, batch_id=batch,
        idempotency_key=f"source-delete-{batch}", context_reliance="Long_Term_Memory",
        raw_text=text, source_spans=single_provenance(text, "user"))


def _claim(turn):
    verdict = verify_support(turn.raw_text, turn.raw_text,
        scorer=lambda _pairs: [{"entailment": 0.99, "neutral": 0.005,
                                "contradiction": 0.005}])
    return CodexClaim(source_batch=turn.batch_id, episodic_id=turn.id,
        conversation_id=turn.conversation_id, role="user",
        raw_sha256=digest(turn.raw_text), start=0, end=len(turn.raw_text),
        sentence=turn.raw_text, sentence_sha256=digest(turn.raw_text),
        text=turn.raw_text, verification=asdict(verdict))


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
@pytest.mark.parametrize("cold_survivor", [False, True])
@pytest.mark.parametrize("primary_is_second", [False, True])
def test_delete_rebases_reinforced_edge_and_forget_expires_last_source(
        cold_survivor, primary_is_second):
    with SessionLocal() as db:
        first, second = Conversation(), Conversation()
        db.add_all([first, second])
        db.flush()
        first_id, second_id = first.id, second.id
        turn_a = _turn(first_id, "Mira lives in Berlin.")
        turn_b = _turn(second_id, "Mira still lives in Berlin.")
        db.add_all([turn_a, turn_b])
        db.flush()
        a_batch, b_batch, b_turn_id = turn_a.batch_id, turn_b.batch_id, turn_b.id
        src = CodexEntity(canonical_name=f"mira-{uuid.uuid4().hex}", source="conversation")
        tgt = CodexEntity(canonical_name=f"berlin-{uuid.uuid4().hex}", source="conversation")
        db.add_all([src, tgt])
        db.flush()
        src_id, tgt_id = src.id, tgt.id
        edge = CodexEdge(source_id=src_id, target_id=tgt_id, relation="lives_in",
            source_batch=b_batch if primary_is_second else a_batch,
            observed_batches=[a_batch, b_batch],
            confidence="active", strength=2.0, source="conversation")
        db.add(edge)
        db.flush()
        edge_id = edge.id
        for turn in (turn_a, turn_b):
            event_type = "edge_added" if turn.batch_id == edge.source_batch else "edge_strengthened"
            claim = _claim(turn)
            db.add(claim)
            db.flush()
            db.add(CodexClaimLink(claim_id=claim.id, edge_id=edge_id,
                relation_verification=asdict(verify_support(turn.raw_text,
                    "mira lives in berlin.", scorer=lambda _pairs: [{
                        "entailment": 0.99, "neutral": 0.005, "contradiction": 0.005}]))))
            db.add(CodexEvent(entity_id=src_id, event_type=event_type,
                payload={"edge_id": str(edge_id)}, batch_source=turn.batch_id))
        if cold_survivor:
            db.add(ColdStorage(id=b_turn_id, conversation_id=second_id,
                batch_id=b_batch, raw_text=turn_b.raw_text,
                source_spans=turn_b.source_spans, timestamp=turn_b.timestamp,
                is_private=False))
            db.delete(turn_b)
        db.commit()

        try:
            preview = delete_conversation(db, str(first_id), dry_run=True)
            manifest = delete_conversation(db, str(first_id))
            assert preview["codex"] == manifest["codex"]
            assert manifest["codex"]["edges_kept_corroborated"] == (not primary_is_second)
            assert manifest["codex"]["edges_support_pruned"] == 1
            assert manifest["codex"]["edges_expired"] == 0
            edge = db.get(CodexEdge, edge_id)
            assert edge.source_batch == b_batch and edge.observed_batches == [b_batch]
            assert edge.confidence == "pending" and edge.strength == 1.0
            assert edge.valid_until is None
            assert db.query(CodexEvent).filter_by(event_type="edge_source_rebased",
                entity_id=src_id).count() == (not primary_is_second)
            assert db.query(CodexClaim).filter_by(source_batch=a_batch).count() == 0
            assert db.query(CodexClaim).filter_by(source_batch=b_batch).count() == 1
            reader = HybridRetrievalOrchestrator(db, None)
            reader._prime_edge_times([edge])
            line = reader._fact_line(db.get(CodexEntity, src_id), edge,
                                     db.get(CodexEntity, tgt_id))
            assert "Mira still lives in Berlin." in line
            assert "speaker: user" in line and "Unverified" not in line

            result = apply_forget(db, {"turns": [{"id": str(b_turn_id)}]})
            db.commit()
            assert result["turns_deleted"] == 1 and result["edges_expired"] == 1
            assert db.get(CodexEdge, edge_id).valid_until is not None
            assert db.query(CodexClaim).filter_by(source_batch=b_batch).count() == 0
        finally:
            db.query(CodexClaimLink).filter_by(edge_id=edge_id).delete(synchronize_session=False)
            db.query(CodexClaim).filter(CodexClaim.source_batch.in_([a_batch, b_batch])).delete(
                synchronize_session=False)
            db.query(CodexEvent).filter(CodexEvent.entity_id == src_id).delete(
                synchronize_session=False)
            db.query(CodexEdge).filter_by(id=edge_id).delete(synchronize_session=False)
            db.query(CodexEntity).filter(CodexEntity.id.in_([src_id, tgt_id])).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter(EpisodicMemory.batch_id.in_([a_batch, b_batch])).delete(
                synchronize_session=False)
            db.query(ColdStorage).filter(ColdStorage.batch_id.in_([a_batch, b_batch])).delete(
                synchronize_session=False)
            db.query(Conversation).filter(Conversation.id.in_([first_id, second_id])).delete(
                synchronize_session=False)
            db.commit()


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_rebase_prefers_current_attributed_quote_over_newer_unlinked_source():
    with SessionLocal() as db:
        conversations = [Conversation() for _ in range(3)]
        db.add_all(conversations)
        db.flush()
        conv_ids = [conv.id for conv in conversations]
        turns = [_turn(conv.id, text) for conv, text in zip(conversations, (
            "Mira lives in Berlin.", "Mira still lives in Berlin.",
            "Mira has an apartment in Berlin."))]
        for year, turn in zip((2023, 2024, 2025), turns):
            turn.timestamp = datetime(year, 1, 1, tzinfo=timezone.utc)
        db.add_all(turns)
        db.flush()
        batches = [turn.batch_id for turn in turns]
        src = CodexEntity(canonical_name=f"mira-{uuid.uuid4().hex}", source="conversation")
        tgt = CodexEntity(canonical_name=f"berlin-{uuid.uuid4().hex}", source="conversation")
        db.add_all([src, tgt])
        db.flush()
        src_id, tgt_id = src.id, tgt.id
        edge = CodexEdge(source_id=src_id, target_id=tgt_id,
            relation="lives_in", source_batch=batches[0], observed_batches=batches,
            strength=3.0, confidence="active", source="conversation")
        db.add(edge)
        db.flush()
        edge_id = edge.id
        claim = _claim(turns[1])
        db.add(claim)
        db.flush()
        db.add(CodexClaimLink(claim_id=claim.id, edge_id=edge_id))
        for index, batch in enumerate(batches):
            db.add(CodexEvent(entity_id=src_id,
                event_type="edge_added" if index == 0 else "edge_strengthened",
                payload={"edge_id": str(edge_id)}, batch_source=batch))
        db.commit()

        try:
            delete_conversation(db, str(conv_ids[0]))
            edge = db.get(CodexEdge, edge_id)
            assert edge.source_batch == batches[1]
            reader = HybridRetrievalOrchestrator(db, None)
            reader._prime_edge_times([edge])
            line = reader._fact_line(db.get(CodexEntity, src_id), edge,
                                     db.get(CodexEntity, tgt_id))
            assert "Mira still lives in Berlin." in line
        finally:
            db.query(CodexClaimLink).filter_by(edge_id=edge_id).delete(synchronize_session=False)
            db.query(CodexClaim).filter(CodexClaim.source_batch.in_(batches)).delete(
                synchronize_session=False)
            db.query(CodexEvent).filter(CodexEvent.entity_id == src_id).delete(
                synchronize_session=False)
            db.query(CodexEdge).filter_by(id=edge_id).delete(synchronize_session=False)
            db.query(CodexEntity).filter(CodexEntity.id.in_([src_id, tgt_id])).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter(EpisodicMemory.batch_id.in_(batches)).delete(
                synchronize_session=False)
            db.query(Conversation).filter(Conversation.id.in_(conv_ids)).delete(
                synchronize_session=False)
            db.commit()
