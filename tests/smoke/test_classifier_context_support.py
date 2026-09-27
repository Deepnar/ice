"""Actual classifier source reads cannot bypass summary support eligibility."""
import os
import uuid

import pytest
import torch

from src.api.db import SessionLocal
from src.classifier import templates
from src.classifier.classifier import PyTorchClassifier
from src.classifier.schema import load_schema
from src.memory.models import Conversation, EpisodicMemory
from src.memory.representation import verify_representations
from src.memory.source import single_provenance
from src.memory.support import verify_support


@pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"), reason="disposable DB required")
@pytest.mark.parametrize("state", ["supported", "contradicted", "unknown", "stale", "legacy"])
def test_sql_context_and_real_classify_encoder_path(state, monkeypatch):
    raw = "I selected amber for the crate. Keep the original decision."
    summary = "I selected amber for the crate."
    if state == "contradicted":
        summary = "I selected blue for the crate."
    classifier = PyTorchClassifier.__new__(PyTorchClassifier)
    classifier.active_schema = load_schema()
    classifier.template_version = templates.DEFAULT_VERSION
    classifier.tag_threshold = .65
    captured = []
    monkeypatch.setattr(classifier, "_encode", lambda text:
        captured.append(text) or torch.zeros(1, 1024))
    classifier.model = lambda _vec: torch.zeros(1, classifier.active_schema.total_width)
    cid, sid = uuid.uuid4(), uuid.uuid4()
    with SessionLocal() as db:
        db.add(Conversation(id=cid)); db.flush()
        row = EpisodicMemory(id=sid, conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=raw, summary_text=summary, summary_coverage=1., inject_raw=False,
            source_spans=single_provenance(raw, "user"), context_reliance="Long_Term_Memory",
            idempotency_key=str(uuid.uuid4()))
        db.add(row); db.flush()
        score = dict(entailment=.99, neutral=.005, contradiction=.005)
        if state == "contradicted":
            score = dict(entailment=.005, neutral=.005, contradiction=.99)
        elif state == "unknown":
            score = dict(entailment=.2, neutral=.7, contradiction=.1)
        if state != "legacy":
            row.representation_verification = verify_representations(row, summary, None,
                verifier=lambda source, claim: verify_support(source, claim,
                    scorer=lambda _pairs: [score]))
        if state == "stale":
            row.raw_text = "I corrected the crate decision to green."
            row.source_spans = single_provenance(row.raw_text, "user")
        expected = summary if state == "supported" else row.raw_text
        db.commit()
        try:
            assert classifier._get_context_turns(str(cid)) == expected
            result = classifier.classify("What color did I select?", conversation_id=str(cid))
            assert captured == [templates.render("What color did I select?", expected,
                                                  version=classifier.template_version)]
            assert len(result.raw_probs) == classifier.active_schema.total_width
            if state == "contradicted": assert "blue" not in captured[0]
            if state == "stale": assert "amber" not in captured[0]
        finally:
            db.query(EpisodicMemory).filter_by(id=sid).delete()
            db.query(Conversation).filter_by(id=cid).delete()
            db.commit()
