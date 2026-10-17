"""A corroborated edge renders the source allowed by the current scope."""

import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.api.db import SessionLocal
from src.memory.models import (
    CodexClaim, CodexClaimLink, CodexEdge, CodexEntity, CodexEvent,
    ColdStorage, Conversation, EpisodicMemory,
)
from src.memory.source import digest, single_provenance
from src.memory.support import verify_support
from src.retrieval import orchestrator as retrieval_module
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.retrieval.timescope import TimeScope


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
@pytest.mark.parametrize("cold_second", [False, True])
def test_scoped_graph_uses_independent_warm_or_cold_source(monkeypatch, cold_second):
    with SessionLocal() as db:
        first, second = Conversation(), Conversation()
        db.add_all([first, second])
        db.flush()
        conv_ids = [first.id, second.id]
        turns = []
        for conv, sentence in ((first, "Mira lives in Berlin."),
                               (second, "Mira still lives in Berlin.")):
            batch = uuid.uuid4()
            turns.append(EpisodicMemory(conversation_id=conv.id, batch_id=batch,
                raw_text=sentence, source_spans=single_provenance(sentence, "user"),
                idempotency_key=f"codex-scope-{batch}",
                context_reliance="Long_Term_Memory"))
        db.add_all(turns)
        turns[0].timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)
        turns[1].timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        db.flush()
        batches = [turn.batch_id for turn in turns]
        ids = [turn.id for turn in turns]
        src = CodexEntity(canonical_name=f"mira-{uuid.uuid4().hex}", source="conversation")
        tgt = CodexEntity(canonical_name=f"berlin-{uuid.uuid4().hex}", source="conversation")
        db.add_all([src, tgt])
        db.flush()
        src_id, tgt_id = src.id, tgt.id
        edge = CodexEdge(source_id=src_id, target_id=tgt_id, relation="lives_in",
            source_batch=batches[0], observed_batches=batches,
            confidence="active", strength=2.0, source="conversation",
            valid_from=datetime(2023, 1, 1, tzinfo=timezone.utc))
        db.add(edge)
        db.flush()
        edge_id = edge.id
        for index, turn in enumerate(turns):
            verdict = verify_support(turn.raw_text, turn.raw_text,
                scorer=lambda _pairs: [{"entailment": 0.99, "neutral": 0.005,
                                        "contradiction": 0.005}])
            claim = CodexClaim(source_batch=turn.batch_id, episodic_id=turn.id,
                conversation_id=turn.conversation_id, role="user",
                raw_sha256=digest(turn.raw_text), start=0, end=len(turn.raw_text),
                sentence=turn.raw_text, sentence_sha256=digest(turn.raw_text),
                text=turn.raw_text, verification=asdict(verdict))
            db.add(claim)
            db.flush()
            db.add(CodexClaimLink(claim_id=claim.id, edge_id=edge_id,
                                  relation_verification=asdict(verdict)))
            db.add(CodexEvent(entity_id=src_id,
                event_type="edge_added" if index == 0 else "edge_strengthened",
                payload={"edge_id": str(edge_id)}, batch_source=turn.batch_id))
        if cold_second:
            db.add(ColdStorage(id=ids[1], conversation_id=conv_ids[1],
                batch_id=batches[1], raw_text=turns[1].raw_text,
                source_spans=turns[1].source_spans, timestamp=turns[1].timestamp,
                is_private=False))
            db.delete(turns[1])
        db.commit()

        monkeypatch.setattr(retrieval_module, "extract_entities",
                            lambda _prompt, _embedder: [src.canonical_name])

        def graph(scope, as_of=False):
            reader = HybridRetrievalOrchestrator(db, None)
            reader.use_fuzzy_match = False
            if as_of:
                reader._active_timescope = TimeScope(
                    mode="as_of", t1=datetime(2024, 1, 1, tzinfo=timezone.utc))
            reader._resolve_exclusion_sets(scope)
            return reader._codex_graph(SimpleNamespace(prompt="Where does Mira live?"),
                                       scope=scope)

        def enumeration(scope):
            reader = HybridRetrievalOrchestrator(db, None)
            reader._resolve_exclusion_sets(scope)
            entities, batches_allowed = reader._codex_scope_sets(scope)
            return reader._codex_enumeration("List all places", ["lives_in"],
                                             entities, batches_allowed)

        try:
            from_second = graph({"conversation_id": str(conv_ids[1])})
            assert from_second
            assert any("Mira still lives in Berlin." in f.text for f in from_second)
            assert all("Mira lives in Berlin." not in f.text for f in from_second)
            assert all(str(batches[0]) not in f.origin_batch_ids for f in from_second)
            assert any(str(batches[1]) in f.origin_batch_ids for f in from_second)
            from_second_enum = enumeration({"conversation_id": str(conv_ids[1])})
            assert any("Mira still lives in Berlin." in f.text for f in from_second_enum)
            assert all(str(batches[0]) not in f.origin_batch_ids
                       for f in from_second_enum)

            from_first = graph({"conversation_id": str(conv_ids[0])})
            assert any("Mira lives in Berlin." in f.text for f in from_first)
            assert all(str(batches[1]) not in f.origin_batch_ids for f in from_first)
            assert graph({"conversation_id": str(conv_ids[0])}, as_of=True)
            assert not graph({"conversation_id": str(conv_ids[1])}, as_of=True)

            exclude_first = graph({"exclude_conversation_ids": [str(conv_ids[0])]})
            assert any("Mira still lives in Berlin." in f.text for f in exclude_first)
            assert all(str(batches[0]) not in f.origin_batch_ids for f in exclude_first)

            exclude_second = graph({"conversation_id": str(conv_ids[1]),
                                    "exclude_conversation_ids": [str(conv_ids[1])]})
            assert not exclude_second
            assert not enumeration({"conversation_id": str(conv_ids[1]),
                                    "exclude_conversation_ids": [str(conv_ids[1])]})

            surviving = db.get(ColdStorage if cold_second else EpisodicMemory, ids[1])
            surviving.is_private = True
            db.commit()
            assert not graph({"conversation_id": str(conv_ids[1])})
            assert not enumeration({"conversation_id": str(conv_ids[1])})
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
            db.query(ColdStorage).filter(ColdStorage.batch_id.in_(batches)).delete(
                synchronize_session=False)
            db.query(Conversation).filter(Conversation.id.in_(conv_ids)).delete(
                synchronize_session=False)
            db.commit()
