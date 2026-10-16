"""CODEX_V4 fact writer: verified v4 claims → facts, sources, entity links (spec §5.4).

Runs only in a disposable database; no model is called (stub embedder, no NLI —
the input is already-verified assertions, as the claim pass hands them over).

Run:  uv run --no-sync python tests/support/disposable_database.py -m pytest tests/test_codex_facts.py -q
"""

import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import text

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import facts
from src.memory.models import (
    CodexEntity, CodexEvent, CodexFact, CodexFactEntity, CodexFactSource, Conversation,
    EpisodicMemory, IdempotencyKey,
)
from src.memory.source import chat_provenance
from src.workers.maintenance_agent import merge_key

pytestmark = pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                                reason="requires a disposable PostgreSQL database")
WHEN = datetime(2026, 3, 1, tzinfo=timezone.utc)


class Encoder:
    def encode(self, value, convert_to_tensor=False):
        if isinstance(value, list):
            return np.ones((len(value), 1024), dtype=np.float32)
        return np.ones(1024, dtype=np.float32)


ENC = Encoder()


def _turn(db, user="I moved to Lisbon in March.", assistant="Nice.", private=False):
    conv = Conversation()
    db.add(conv)
    db.flush()
    batch = uuid.uuid4()
    turn = EpisodicMemory(conversation_id=conv.id, batch_id=batch, idempotency_key=f"facts-{batch}",
                          context_reliance="Long_Term_Memory", timestamp=WHEN,
                          raw_text=f"User: {user}\n\nAssistant: {assistant}",
                          source_spans=chat_provenance(user, assistant), is_private=private)
    db.add(turn)
    db.flush()
    return turn


def _a(text_, quote, speaker="user", modality="asserted", subject="user", relation="moved to",
       obj="Lisbon", source="w0"):
    return {"text": text_, "sources": [source], "premise": "p", "pair_tokens": 20,
            "cited_spans": [{"id": source, "role": speaker, "start": 0, "end": len(quote)}],
            "verification": {"status": "supported", "scores": {"entailment": 0.99}},
            "quote": quote, "speaker": speaker, "modality": modality, "subject": subject,
            "relation": relation, "object": obj}


def _links(db, fact):
    return {(r.role, db.get(CodexEntity, r.entity_id).canonical_name)
            for r in db.query(CodexFactEntity).filter_by(fact_id=fact.id)}


def test_fact_written_with_source_links_and_event():
    with SessionLocal() as db:
        turn = _turn(db)
        counts = facts.write_turn_facts(db, turn, {"assertions": [
            _a("The user said they moved to Lisbon in March.", "I moved to Lisbon in March")]},
            embedder=ENC)
        fact = db.query(CodexFact).filter_by(text="The user said they moved to Lisbon in March.").one()
        assert counts["new"] == 1 and fact.support_count == 1 and fact.confidence == 0.99
        assert fact.mentioned_at == WHEN and fact.valid_until is None and fact.modality == "asserted"
        assert _links(db, fact) == {("subject", "user"), ("object", "lisbon")}
        source = db.query(CodexFactSource).filter_by(fact_id=fact.id).one()
        assert source.source_batch == turn.batch_id and source.quote == "I moved to Lisbon in March"
        assert db.query(CodexEvent).filter(CodexEvent.payload["fact_id"].astext == str(fact.id),
                                           CodexEvent.event_type == "fact_added").count() == 2
        db.rollback()


def test_pronouns_resolve_by_speaker():
    with SessionLocal() as db:
        turn = _turn(db, user="You told me to rest.", assistant="I suggested resting.")
        facts.write_turn_facts(db, turn, {"assertions": [
            _a("The user said the assistant told them to rest.", "You told me to rest",
               subject="you", relation="told", obj="me"),
            _a("The assistant said it suggested resting.", "I suggested resting", speaker="assistant",
               subject="I", relation="suggested", obj="resting"),
        ]}, embedder=ENC)
        told = db.query(CodexFact).filter(CodexFact.relation == "told").one()
        said = db.query(CodexFact).filter(CodexFact.relation == "suggested").one()
        assert _links(db, told) == {("subject", "assistant"), ("object", "user")}
        assert ("subject", "assistant") in _links(db, said) and ("subject", "user") not in _links(db, said)
        db.rollback()


def test_exact_duplicate_adds_support_not_a_row_and_replay_is_idempotent():
    with SessionLocal() as db:
        one, two = _turn(db), _turn(db, user="As I said, I moved to Lisbon in March.")
        body = "The user said they moved to Lisbon in March."
        record = {"assertions": [_a(body, "I moved to Lisbon in March")]}
        facts.write_turn_facts(db, one, record, embedder=ENC)
        counts = facts.write_turn_facts(db, two, record, embedder=ENC)
        again = facts.write_turn_facts(db, two, record, embedder=ENC)       # same turn replayed
        rows = db.query(CodexFact).filter_by(text=body).all()
        assert len(rows) == 1 and rows[0].support_count == 2
        assert counts["supported"] == 1 and again["supported"] == 0
        assert db.query(CodexFactSource).filter_by(fact_id=rows[0].id).count() == 2
        # a different speaker saying the same words is a different fact
        facts.write_turn_facts(db, one, {"assertions": [_a(body, "I moved to Lisbon in March",
                                                           speaker="assistant")]}, embedder=ENC)
        assert db.query(CodexFact).filter_by(text=body).count() == 2
        db.rollback()


def test_post_rules_relabel_drop_and_flags():
    with SessionLocal() as db:
        turn = _turn(db, user="see https://platform.openai.com/docs please",
                     assistant="Here is my analysis of the plan.")
        counts = facts.write_turn_facts(db, turn, {"assertions": [
            _a("The assistant shared an analysis of the plan.", "Here is my analysis of the plan",
               speaker="assistant", modality="shared", subject="assistant", relation="analysed",
               obj="the plan"),
            _a("The user said requirements.txt describes requirements.txt.", "see", subject="requirements.txt",
               relation="describes", obj="requirements.txt"),
            _a("The user shared a link.", "see https://platform.openai.com/docs please", modality="shared",
               relation="shared link", obj="https://platform.open"),
        ]}, embedder=ENC)
        assert counts["dropped"] == 1 and counts["new"] == 2
        analysis = db.query(CodexFact).filter(CodexFact.relation == "analysed").one()
        assert analysis.modality == "asserted" and "relabel_shared" in analysis.flags
        link = db.query(CodexFact).filter(CodexFact.relation == "shared link").one()
        assert link.modality == "shared" and "object_cut" in link.flags     # user 'shared' kept
        db.rollback()


def test_long_values_are_not_entities():
    with SessionLocal() as db:
        turn = _turn(db, user="the loss went to 1.0789 after we raised curated weight to ten today")
        facts.write_turn_facts(db, turn, {"assertions": [
            _a("The user said the loss went to 1.0789.", "the loss went to 1.0789", subject="the loss",
               relation="went to", obj="1.0789 after we raised curated weight to ten today")]},
            embedder=ENC)
        fact = db.query(CodexFact).filter(CodexFact.relation == "went to").one()
        assert _links(db, fact) == {("subject", "loss")}
        db.rollback()


def test_entity_resolution_tiers():
    with SessionLocal() as db:
        base = f"multicon {uuid.uuid4().hex[:6]} conference"
        db.add(CodexEntity(canonical_name=base, source="conversation", aliases=[base],
                           properties={"merge_key": merge_key(base)}))
        db.add(CodexEntity(canonical_name="gpt 4o", source="conversation",
                           properties={"merge_key": merge_key("gpt 4o")}))
        code = f"orchestrator_{uuid.uuid4().hex[:6]}.py"
        db.add(CodexEntity(canonical_name=code, source="static_analysis"))
        db.flush()
        assert facts.resolve_entity(db, "GPT-4o", embedder=ENC).canonical_name == "gpt 4o"      # merge_key
        db.add(CodexEntity(canonical_name="devstral-small", source="conversation",
                           properties={"merge_key": merge_key("devstral-small")}))
        db.flush()
        # a different model's version suffix must stay apart; so does a plural (Tier 2's job)
        assert facts.resolve_entity(db, "devstral-small-2", embedder=ENC).canonical_name == "devstral-small-2"
        plural = facts.resolve_entity(db, base + "s", embedder=ENC)
        assert plural.canonical_name == base + "s"
        assert facts.resolve_entity(db, base.upper(), embedder=ENC).canonical_name == base       # exact
        assert facts.resolve_entity(db, code, embedder=ENC).source == "static_analysis"         # no clash
        far = facts.resolve_entity(db, "multicon", embedder=ENC)                                 # too far
        assert far.canonical_name == "multicon" and far.id == uuid.uuid5(facts.ENTITY_NAMESPACE, "multicon")
        assert facts.resolve_entity(db, "MultiCon", embedder=ENC).id == far.id                   # exact, case
        db.rollback()


def test_stage_reuses_summary_claims_and_skips_private(monkeypatch):
    from src.memory import summary_support as ss
    from src.workers import fact_extractor
    monkeypatch.setattr(settings, "summary_assertion_method", "v4")
    monkeypatch.setattr(fact_extractor, "get_embedder", lambda: ENC)
    monkeypatch.setattr(fact_extractor, "_claim_record",
                        lambda *_a, **_k: pytest.fail("claim pass called despite a v4 summary record"))
    with SessionLocal() as db:
        turn = _turn(db)
        turn.representation_verification = {"summary": {"method": ss.METHOD_V4, "assertions": [
            _a("The user said they moved to Lisbon in March.", "I moved to Lisbon in March")]}}
        private = _turn(db, private=True)
        db.commit()
        batch, private_batch = turn.batch_id, private.batch_id
    fact_extractor.extract_facts(batch)
    fact_extractor.extract_facts(batch)                                     # idempotent
    fact_extractor.extract_facts(private_batch)
    with SessionLocal() as db:
        rows = db.execute(text("SELECT count(*) FROM codex_fact_sources WHERE source_batch = :b"),
                          {"b": batch}).scalar()
        assert rows == 1
        assert db.execute(text("SELECT count(*) FROM codex_fact_sources WHERE source_batch = :b"),
                          {"b": private_batch}).scalar() == 0
        assert db.query(IdempotencyKey).count() >= 1


def test_stage_is_inert_unless_v4(monkeypatch):
    from src.workers import fact_extractor
    monkeypatch.setattr(settings, "summary_assertion_method", "v3")
    monkeypatch.setattr(fact_extractor, "SessionLocal", lambda: pytest.fail("touched the store under v3"))
    fact_extractor.extract_facts(uuid.uuid4())


def test_delete_conversation_expires_sole_facts_and_keeps_corroborated():
    from src.services.conversations import delete_conversation
    with SessionLocal() as db:
        gone, other = _turn(db, user="I keep bees on the roof."), _turn(db, user="I keep bees, still.")
        only_here = "The user said they keep bees on the roof."
        both = "The user said they keep bees."
        facts.write_turn_facts(db, gone, {"assertions": [
            _a(only_here, "I keep bees on the roof", relation="keeps", obj=f"roofhive-{uuid.uuid4().hex[:6]}"),
            _a(both, "I keep bees", relation="keeps", obj="bees")]}, embedder=ENC)
        facts.write_turn_facts(db, other, {"assertions": [
            _a(both, "I keep bees", relation="keeps", obj="bees")]}, embedder=ENC)
        db.commit()
        sole = db.query(CodexFact).filter_by(text=only_here).one()
        kept = db.query(CodexFact).filter_by(text=both).one()
        sole_id, kept_id = sole.id, kept.id
        roof = next(e for r, e in [(r.role, db.get(CodexEntity, r.entity_id))
                                    for r in db.query(CodexFactEntity).filter_by(fact_id=sole_id)]
                    if r == "object")
        roof_id = roof.id
        manifest = delete_conversation(db, str(gone.conversation_id), dry_run=True)
        assert manifest["codex"]["facts_expired"] == 1 and manifest["codex"]["facts_support_pruned"] == 1
        assert manifest["codex"]["fact_sources_deleted"] == 2
        assert db.get(CodexFact, sole_id).valid_until is None                    # dry run changed nothing
        delete_conversation(db, str(gone.conversation_id))
        db.expire_all()
        assert db.get(CodexFact, sole_id).valid_until is not None
        assert db.query(CodexFactSource).filter_by(fact_id=sole_id).count() == 0  # deleted quotes gone
        survivor = db.get(CodexFact, kept_id)
        assert survivor.valid_until is None and survivor.support_count == 1
        assert db.query(CodexFactSource).filter_by(fact_id=kept_id).one().source_batch == other.batch_id
        assert db.query(CodexEvent).filter(CodexEvent.event_type == "fact_expired",
                                           CodexEvent.payload["fact_id"].astext == str(sole_id)).count() >= 1
        assert "deleted_reason" in (db.get(CodexEntity, roof_id).properties or {})  # orphan expired
        user = db.query(CodexEntity).filter_by(canonical_name="user").one()
        assert "deleted_reason" not in (user.properties or {})                      # still linked


def test_forget_proposes_and_expires_facts():
    from src.services.conversations import apply_forget, propose_forget
    with SessionLocal() as db:
        turn = _turn(db, user="My locker code is 4417.")
        facts.write_turn_facts(db, turn, {"assertions": [
            _a("The user said their locker code is 4417.", "My locker code is 4417",
               relation="locker code is", obj="4417")]}, embedder=ENC)
        db.commit()
        proposal = propose_forget(db, str(turn.conversation_id), "locker code", embedder=ENC)
        listed = [f for f in proposal["facts"] if "locker code" in f["description"]]
        assert proposal["queued"] and listed
        fact_id = uuid.UUID(listed[0]["id"])
        assert db.get(CodexFact, fact_id).valid_until is None                    # proposal applies nothing
        result = apply_forget(db, {"facts": [{"id": str(fact_id)}], "proposed_by": "test"})
        db.commit()
        assert result["facts_expired"] == 1 and db.get(CodexFact, fact_id).valid_until is not None
