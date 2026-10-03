"""G76: a quoted source can survive even when its proposed graph edge cannot."""

import json
import os
import uuid
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import support
from src.memory.claims import store_claims
from src.memory.models import (
    CodexClaim, CodexClaimLink, CodexEdge, CodexEntity, CodexEvent,
    Conversation, EpisodicMemory, IdempotencyKey,
)
from src.memory.relation_support import relation_proposition, verify_relation_pairs
from src.memory.source import chat_provenance, single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor as worker
from src.workers.idempotency import job_key


class Encoder:
    max_seq_length = 8192

    def tokenizer(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def encode(self, _text, **_kwargs):
        return np.array([1.0] + [0.0] * 1023)


def _scores(entailment):
    return {"entailment": entailment,
            "neutral": 1.0 - entailment - 0.001,
            "contradiction": 0.001}


def test_relation_pairs_keep_polarity_and_isolate_overlong_source():
    assert relation_proposition("Mira", "lives_in", "Berlin") == "Mira lives in Berlin."
    assert relation_proposition("Mira", "lives_in", "Berlin", negated=True) == (
        "It is false that Mira lives in Berlin.")
    calls = []

    def scorer(pairs):
        calls.append(pairs)
        if any(source == "too long" for source, _ in pairs):
            raise support.SupportInputError("complete input too long")
        return [_scores(0.99) for _ in pairs]

    verdicts = verify_relation_pairs([
        ("Mira lives in Berlin.", "Mira lives in Berlin."),
        ("too long", "Mira lives in London."),
        ("Mira works at Vertex Labs.", "Mira works at Vertex Labs."),
    ], scorer=scorer)
    assert [v.status for v in verdicts] == ["supported", "unknown", "supported"]
    assert len(calls) < 6  # one batch, then only the failing half is isolated

    def outage(_pairs):
        raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        verify_relation_pairs([("Mira lives in Berlin.", "Mira lives in Berlin.")],
                              scorer=outage)


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
@pytest.mark.parametrize("fields", [
    ("timer", "was disabled", None), ("timer", None, None),
    (None, "was disabled", None), (None, None, None),
])
def test_nullable_template_fields_retain_exact_claim_without_graph_edge(monkeypatch, fields):
    sentence = "The timer was disabled."
    response = json.dumps({"facts": [{**dict(zip(("subject", "relation", "object"), fields)),
                                      "source_sentence": sentence}]})
    completion = NS(choices=[NS(message=NS(content=response), finish_reason="stop")])
    monkeypatch.setattr(settings, "codex_sentence_claims", True)
    monkeypatch.setattr(settings, "codex_extraction_mode", "template")
    monkeypatch.setattr(worker, "bg_client", NS(chat=NS(completions=NS(
        create=lambda **_kwargs: completion))))
    monkeypatch.setattr(worker, "embedder", Encoder())
    monkeypatch.setattr(worker, "extract_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(worker, "serving_window", lambda *_args: 8192)
    monkeypatch.setattr(worker, "make_llm_reconciler", lambda: None)
    monkeypatch.setattr(worker, "store_claims", lambda db, row, sentences, *, encoder:
        store_claims(db, row, sentences, encoder=encoder,
            verifier=lambda source, claim: support.verify_support(source, claim,
                scorer=lambda _pairs: [_scores(0.99)])))
    batch = uuid.uuid4()
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        conversation_id = conversation.id
        db.add(EpisodicMemory(conversation_id=conversation_id, batch_id=batch,
            idempotency_key=f"source-only-{batch}", context_reliance="Long_Term_Memory",
            raw_text=sentence, source_spans=single_provenance(sentence, "user")))
        db.commit()
    try:
        worker.extract_codex(str(batch))
        with SessionLocal() as db:
            assert db.query(CodexClaim).filter_by(source_batch=batch).count() == 1
            assert db.query(CodexEdge).filter_by(source_batch=batch).count() == 0
            assert db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).count() == 1
            reader = HybridRetrievalOrchestrator(db, None)
            assert reader._codex_claims("timer", None,
                                        {"conversation_id": str(conversation_id)})
        completion.choices[0].message.content = response.replace(
            "The timer was disabled.", "The timer was enabled.")
        with pytest.raises(worker.ExtractionOutputError):
            worker.extract_triplets(sentence, source_sentences=[])
    finally:
        with SessionLocal() as db:
            db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).delete(
                synchronize_session=False)
            db.query(CodexClaim).filter_by(source_batch=batch).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter_by(batch_id=batch).delete(
                synchronize_session=False)
            db.query(Conversation).filter_by(id=conversation_id).delete(
                synchronize_session=False)
            db.commit()


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_writer_links_only_supported_relation_and_keeps_other_source(monkeypatch):
    monkeypatch.setattr(settings, "codex_sentence_claims", True)
    monkeypatch.setattr(worker, "embedder", Encoder())
    monkeypatch.setattr(worker, "make_llm_reconciler", lambda: None)
    monkeypatch.setattr(worker, "extract_entities", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(worker, "known_relations", lambda: [])
    monkeypatch.setattr(worker, "canonical_relation", lambda relation, **_kwargs: relation)
    monkeypatch.setattr(worker, "store_claims", lambda db, row, sentences, *, encoder:
        store_claims(db, row, sentences, encoder=encoder,
            verifier=lambda source, claim: support.verify_support(source, claim,
                scorer=lambda _pairs: [_scores(0.99)])))

    sources = {
        "Mira lives in Berlin.": ("mira", "lives_in", "berlin"),
        "Mira considered moving to London.": ("mira", "lives_in", "london"),
        "Only if approved: add Redis. Approval was later refused.":
            ("project", "added", "redis"),
        "The assistant suggested adding a cache, but the user had not decided.":
            ("project", "decided", "cache"),
    }

    def extract(text, *, source_sentences, **_kwargs):
        result = []
        for sentence in text.strip().split("\n\n"):
            source_sentences.append(sentence)
            subject, relation, obj = sources[sentence]
            result.append({"subject": subject, "relation": relation, "object": obj,
                           "source_sentence": sentence, "confidence": 0.9})
        return result

    def scorer(pairs):
        assert len(pairs) == 4
        return [_scores(0.99 if claim == "mira lives in berlin." else 0.01)
                for _source, claim in pairs]

    monkeypatch.setattr(worker, "extract_triplets", extract)
    monkeypatch.setattr(worker, "verify_relation_pairs", lambda pairs:
        verify_relation_pairs(pairs, scorer=scorer))
    user = "\n\n".join(list(sources)[:3])
    assistant = list(sources)[3]
    batch = uuid.uuid4()
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        conversation_id = conversation.id
        db.add(EpisodicMemory(conversation_id=conversation_id, batch_id=batch,
            idempotency_key=f"relation-{batch}", context_reliance="Long_Term_Memory",
            raw_text=f"User: {user}\n\nAssistant: {assistant}",
            source_spans=chat_provenance(user, assistant)))
        db.commit()

    try:
        worker.extract_codex(str(batch))
        worker.extract_codex(str(batch))  # completion key remains idempotent
        with SessionLocal() as db:
            assert db.query(CodexClaim).filter_by(source_batch=batch).count() == 4
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            assert len(edges) == 1 and edges[0].relation == "lives_in"
            links = db.query(CodexClaimLink).filter_by(edge_id=edges[0].id).all()
            assert len(links) == 1
            assert links[0].relation_verification["status"] == "supported"
            assert db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).count() == 1
            reader = HybridRetrievalOrchestrator(db, None)
            assert reader._codex_claims("London", None, {"conversation_id": str(conversation_id)})
            assert reader._codex_claims("cache", None, {"conversation_id": str(conversation_id)})
    finally:
        with SessionLocal() as db:
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            edge_ids = [edge.id for edge in edges]
            entity_ids = list({node_id for edge in edges
                               for node_id in (edge.source_id, edge.target_id)})
            if edge_ids:
                db.query(CodexClaimLink).filter(CodexClaimLink.edge_id.in_(edge_ids)).delete(
                    synchronize_session=False)
            if entity_ids:
                db.query(CodexEvent).filter(CodexEvent.entity_id.in_(entity_ids)).delete(
                    synchronize_session=False)
            db.query(CodexEdge).filter_by(source_batch=batch).delete(synchronize_session=False)
            if entity_ids:
                db.query(CodexEntity).filter(CodexEntity.id.in_(entity_ids)).delete(
                    synchronize_session=False)
            db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).delete(
                synchronize_session=False)
            db.query(CodexClaim).filter_by(source_batch=batch).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter_by(batch_id=batch).delete(
                synchronize_session=False)
            db.query(Conversation).filter_by(id=conversation_id).delete(synchronize_session=False)
            db.commit()


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                    reason="requires a disposable PostgreSQL database")
def test_explicit_sentence_claims_off_preserves_legacy_writer(monkeypatch):
    monkeypatch.setattr(settings, "codex_sentence_claims", False)
    monkeypatch.setattr(worker, "extract_triplets", lambda *_args, **_kwargs: [
        {"subject": "mira", "relation": "lives_in", "object": "berlin"}])
    monkeypatch.setattr(worker, "make_llm_reconciler", lambda: None)
    called = []
    monkeypatch.setattr(worker, "handle_triplet", lambda *_args, **_kwargs:
        called.append(True) or object())
    batch = uuid.uuid4()
    with SessionLocal() as db:
        conversation = Conversation()
        db.add(conversation)
        db.flush()
        conversation_id = conversation.id
        text = "Mira lives in Berlin."
        db.add(EpisodicMemory(conversation_id=conversation_id, batch_id=batch,
            idempotency_key=f"legacy-relation-{batch}", context_reliance="Long_Term_Memory",
            raw_text=text, source_spans=single_provenance(text, "user")))
        db.commit()
    try:
        worker.extract_codex(str(batch))
        assert called == [True]
        with SessionLocal() as db:
            assert db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).count() == 1
            assert db.query(CodexClaim).filter_by(source_batch=batch).count() == 0
    finally:
        with SessionLocal() as db:
            db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).delete(
                synchronize_session=False)
            db.query(EpisodicMemory).filter_by(batch_id=batch).delete(
                synchronize_session=False)
            db.query(Conversation).filter_by(id=conversation_id).delete(
                synchronize_session=False)
            db.commit()
