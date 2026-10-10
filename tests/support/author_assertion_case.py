"""A synthetic source-bound writer control; never imports personal corpus data."""

import json
import uuid

import numpy as np

from src.api.config import settings
from src.api.db import SessionLocal
from src.memory import support
from src.memory.assertion_support import review_assertions
from src.memory.claims import store_claims
from src.memory.models import (
    CodexClaim, CodexClaimLink, CodexEdge, CodexEntity, CodexEvent,
    Conversation, EpisodicMemory, IdempotencyKey, ReviewQueue,
)
from src.memory.relation_support import verify_relation_pairs
from src.memory.source import single_provenance
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor as worker
from src.workers.idempotency import job_key


SOURCE = ("Mira works at Atlas.\n\nMira and Niko both use Nimbus.\n\n"
          "Mira does not use Redis.\n\nDoes Niko work at Atlas?\n\n"
          "Should I say that Niko lives in Berlin?\n\n"
          "Here is a proposed alternate plot.\n\nNiko works at Atlas.")
FACTS = [
    ("Mira works at Atlas.", "mira", "works_at", "atlas", False),
    ("Mira and Niko both use Nimbus.", "mira", "uses", "nimbus", False),
    ("Mira does not use Redis.", "mira", "uses", "redis", True),
    ("Does Niko work at Atlas?", "niko", "works_at", "atlas", False),
    ("Should I say that Niko lives in Berlin?", "niko", "lives_in", "berlin", False),
    ("Niko works at Atlas.", "niko", "works_at", "atlas", False),
]
SCORES = dict(entailment=.99, neutral=.009, contradiction=.001)


class Encoder:
    max_seq_length = 8192

    def tokenizer(self, text, **_kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def encode(self, _text, **_kwargs):
        return np.array([1.] + [0.] * 1023)


def exercise_writer(monkeypatch, *, actual=False, receipt_sink=None, argument_only=False,
                    argument_case=False):
    """Inject proposals and positive NLI to isolate author-authority admission.

    Actual mode runs the real local E4B check through the otherwise unchanged
    transaction and readers; it does not qualify NuExtract or NLI accuracy.
    """
    source, facts = SOURCE, FACTS
    if argument_case:
        source = ("Mira applied for the Atlas team in Niko organization.\n\n"
                  "Niko was hired by Mira.\n\nMira does not use Redis.")
        facts = [
            ("Mira applied for the Atlas team in Niko organization.", "mira", "applied_for", "atlas team", False),
            ("Niko was hired by Mira.", "mira", "hired", "niko", False),
            ("Mira does not use Redis.", "mira", "uses", "redis", True),
            ("Mira applied for the Atlas team in Niko organization.", "atlas team", "applied_for", "niko organization", False),
            ("Niko was hired by Mira.", "niko", "hired", "mira", False),
            ("Mira does not use Redis.", "mira", "uses", "redis", False),
        ]
    expected_claims = len({fact[0] for fact in facts})
    monkeypatch.setattr(settings, "codex_sentence_claims", True)
    monkeypatch.setattr(settings, "codex_assertion_check_enabled", True)
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    monkeypatch.setattr(worker, "embedder", Encoder())
    monkeypatch.setattr(worker, "make_llm_reconciler", lambda: None)
    monkeypatch.setattr(worker, "store_claims", lambda db, row, sentences, *, encoder:
        store_claims(db, row, sentences, encoder=encoder, verifier=lambda source, claim:
            support.verify_support(source, claim, scorer=lambda _pairs: [SCORES])))
    monkeypatch.setattr(worker, "verify_relation_pairs", lambda pairs:
        verify_relation_pairs(pairs, scorer=lambda rows: [SCORES] * len(rows)))

    def extract(_text, *, source_sentences, **_kwargs):
        source_sentences.extend(f[0] for f in facts)
        return [dict(subject=s, relation=r, object=o, negated=n,
                     source_sentence=q, confidence=.9) for q, s, r, o, n in facts]

    monkeypatch.setattr(worker, "extract_triplets", extract)
    calls = []

    def review(items):
        calls.append(items)
        assert len(items) == 6
        assert all(i.source == source for i in items)
        assert all(i.subject and i.relation and i.object_name and type(i.negated) is bool for i in items)
        if actual:
            receipts = review_assertions(items)
            if receipt_sink is not None:
                receipt_sink(receipts)
            return receipts

        def sender(_body):
            return dict(done=True, done_reason="stop", prompt_eval_count=500,
                eval_count=200, message=dict(content=json.dumps(dict(decisions=[
                    dict(id=i + 1, source_act="assertion" if i < 3 or argument_only or argument_case else
                         "hypothetical" if i == 5 else "question",
                         supports_candidate=i < 3 or argument_only, reason="Synthetic boundary control.",
                         subject_matches=i < 3 if argument_only else True,
                         object_matches=True, qualifiers_match=True)
                    for i in range(6)]))))

        return review_assertions(items, sender=sender)

    monkeypatch.setattr(worker, "review_assertions", review)
    batch, cid = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        db.add(EpisodicMemory(conversation_id=cid, batch_id=batch, raw_text=source,
            source_spans=single_provenance(source, "user"),
            context_reliance="Long_Term_Memory", idempotency_key=f"assertion-{batch}"))
        db.commit()
    try:
        worker.extract_codex(str(batch))
        worker.extract_codex(str(batch))
        assert len(calls) == 1
        with SessionLocal() as db:
            claims = db.query(CodexClaim).filter_by(source_batch=batch).all()
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            assert len(claims) == expected_claims
            assert len(edges) == 3
            assert sum(bool(e.negated) for e in edges) == 1
            assert all(db.get(CodexEntity, e.source_id).canonical_name == "mira" for e in edges)
            links = db.query(CodexClaimLink).join(CodexClaim).filter(
                CodexClaim.source_batch == batch).all()
            assert len(links) == 3
            assert all(l.relation_verification["author_assertion"]["status"] == "supported" for l in links)
            assert sum(bool(c.verification.get("needs_author_context")) for c in claims) == 3
            from scripts.z1.seed_v3 import author_assertion_counts

            assert author_assertion_counts(db, batch) == dict(supported=3, not_asserted=3, unknown=0)
            assert author_assertion_counts(db, uuid.uuid4()) == dict(supported=0, not_asserted=0, unknown=0)
            reader = HybridRetrievalOrchestrator(db, None)
            fragments = reader._codex_claims("Niko Atlas", None, {"conversation_id": str(cid)})
            assert fragments
            assert any(source in f.text for f in fragments)
            assert db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).count() == 1
            # Linked graph rendering also retains original paragraph evidence.
            edge = next(e for e in edges if e.relation == facts[0][2])
            reader._prime_edge_times([edge])
            line = reader._fact_line(db.get(CodexEntity, edge.source_id), edge,
                                     db.get(CodexEntity, edge.target_id))
            assert facts[0][0] in line
            assert source in line  # Positive global proof cannot authorize only a paragraph.
            return dict(claims=expected_claims, admitted_relations=3, withheld_relations=3,
                        negative_relations=1, repeated_calls=0,
                        full_context_retrievable=True, linked_source_retrievable=True)
    finally:
        with SessionLocal() as db:
            edges = db.query(CodexEdge).filter_by(source_batch=batch).all()
            ids = [e.id for e in edges]
            entities = list({n for e in edges for n in (e.source_id, e.target_id)})
            if ids:
                db.query(CodexClaimLink).filter(CodexClaimLink.edge_id.in_(ids)).delete(synchronize_session=False)
            if entities:
                db.query(CodexEvent).filter(CodexEvent.entity_id.in_(entities)).delete(synchronize_session=False)
            db.query(ReviewQueue).filter(ReviewQueue.item_type == 'codex_reconciliation',
                ReviewQueue.item_content.contains({'new_batch_id': str(batch)})).delete(synchronize_session=False)
            db.query(CodexEdge).filter_by(source_batch=batch).delete(synchronize_session=False)
            if entities:
                db.query(CodexEntity).filter(CodexEntity.id.in_(entities)).delete(synchronize_session=False)
            db.query(IdempotencyKey).filter_by(key=job_key("codex", batch)).delete(synchronize_session=False)
            db.query(CodexClaim).filter_by(source_batch=batch).delete(synchronize_session=False)
            db.query(EpisodicMemory).filter_by(batch_id=batch).delete(synchronize_session=False)
            db.query(Conversation).filter_by(id=cid).delete(synchronize_session=False)
            db.commit()
