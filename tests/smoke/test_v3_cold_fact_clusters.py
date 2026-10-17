"""ICE v3 archived-source fact reach uses preserved cluster membership.

A CODEX_V4 fact is visible through its source turn, warm or archived; the
source's cluster membership, scope, privacy and time window decide. Authored
facts, controlled embeddings/classification/NER; no model calls. (Was
test_v3_cold_claim_clusters.py over v3 attributed claims.)
"""
import copy
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx
import numpy as np
import pytest
from sqlalchemy.engine import make_url
from transformers import AutoTokenizer

from src.api import db as database
from src.api import memory_preparation
from src.api.config import settings
from src.classifier.schemas import ClassificationResult
from src.memory import embedder as embedding_module
from src.memory import tokens
from src.memory.models import (CodexFact, CodexFactSource, ColdStorage, ContextCluster,
                               Conversation, EpisodicClusterLink, EpisodicMemory)
from src.memory.source import chat_provenance
from src.retrieval import orchestrator as deployed
from src.retrieval.timescope import TimeScope
from src.workers import decay
from src.workers.decay import apply_decay

VEC = [1.] + [0.] * 1023
NOW = datetime.now(timezone.utc)


class Encoder:
    max_seq_length = 8192

    def tokenizer(self, value, **_kwargs):
        return {"input_ids": list(range(len(value.split())))}

    def encode(self, value, **_kwargs):
        return np.array([VEC for _ in value] if isinstance(value, list) else VEC)


@pytest.fixture
def store(empty_replay_database, monkeypatch):
    assert os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
    assert make_url(settings.database_url).database == os.environ["ICE_TEST_DATABASE"]
    monkeypatch.setattr(tokens, "_tokenizer", AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Embedding-0.6B", local_files_only=True))
    monkeypatch.setattr(tokens, "_load_attempted", True)
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    monkeypatch.setattr(settings, "retrieval_strengthen_writes", False)
    monkeypatch.setattr(settings, "retrieval_rerank_enabled", False)
    monkeypatch.setattr(deployed, "extract_entities", lambda *_a, **_k: [])
    monkeypatch.setattr(decay, "SessionLocal", database.SessionLocal)
    monkeypatch.setattr(embedding_module, "get_embedder", lambda: Encoder())
    attempts = []
    def deny(*_args, **_kwargs):
        attempts.append(1)
        raise AssertionError("Unexpected HTTP in isolated cold-fact controls")
    with database.SessionLocal() as db, patch.object(httpx.Client, "request", deny), patch.object(
            httpx.AsyncClient, "request", deny):
        yield db
        db.rollback()
    assert not attempts


def cohort(db, *, cold=True, long_assistant=False):
    cid = uuid.uuid4()
    db.add(Conversation(id=cid, memory_scope_type="global"))
    db.flush()
    a, b = uuid.uuid4(), uuid.uuid4()
    db.add_all([ContextCluster(id=a, name="Atlas", conversation_id=cid, embedding=VEC),
                ContextCluster(id=b, name="Boreal", conversation_id=cid)])
    db.flush()
    rows, facts = {}, {}
    for name, memberships in [("a", [a]), ("b", [b]), ("both", [a, b]),
                               ("empty", []), ("unknown", None)]:
        user = f"Atlas uses PostgreSQL. It applies to the sandbox only. Cohort {name}."
        assistant = ("Recorded. " + "Unrelated commentary. " * 6000
                     if long_assistant else "Recorded.")
        cls = ColdStorage if cold else EpisodicMemory
        row = cls(id=uuid.uuid4(), conversation_id=cid, batch_id=uuid.uuid4(),
            raw_text=f"User: {user}\n\nAssistant: {assistant}",
            source_spans=chat_provenance(user, assistant), timestamp=NOW - timedelta(days=100),
            ts_provenance="original", embedding=VEC, inject_raw=True, lossless_flag=True,
            context_reliance="Long_Term_Memory",
            **({"cluster_ids": memberships} if cold else {
                "idempotency_key": str(uuid.uuid4()), "decay_score": 1.}))
        db.add(row); db.flush()
        if not cold:
            for membership in memberships or []:
                db.add(EpisodicClusterLink(episodic_id=row.id, cluster_id=membership))
            db.flush()
        sentence = "The user said Atlas uses PostgreSQL, for the sandbox only."
        fact = CodexFact(text=sentence, text_sha256=uuid.uuid4().hex, speaker="user",
            modality="asserted", embedding=VEC, mentioned_at=row.timestamp,
            valid_from=row.timestamp, confidence=.99)
        db.add(fact); db.flush()
        db.add(CodexFactSource(fact_id=fact.id, source_batch=row.batch_id,
            window_sha256=uuid.uuid4().hex, episodic_id=row.id, conversation_id=cid, window={},
            quote="Atlas uses PostgreSQL.", recorded_at=row.timestamp, verification={}))
        db.flush()
        rows[name], facts[name] = row, fact
    return cid, a, b, rows, facts


def identities(reader, scope, embedding):
    return {f.stored_artifact_ids[0][1] for f in reader._facts_leg(
        "PostgreSQL", embedding, scope)}


@pytest.mark.parametrize("embedding", [None, VEC])
@pytest.mark.parametrize("mode,expected", [
    ("ordinary", {"a", "b", "both", "empty", "unknown"}),
    ("matching", {"a", "both", "empty"}),
    ("explicit_matching", {"a", "both", "empty"}),
    ("disjoint", {"empty"}),
    ("exclude_b", {"a", "empty"}),
    ("include_a_exclude_b", {"a", "empty"}),
    ("explicit_empty", set()),
])
def test_cold_membership_candidates(store, embedding, mode, expected):
    cid, a, b, _rows, facts = cohort(store)
    scope = {"conversation_id": str(cid)}
    options = {"ordinary": {}, "matching": {"cluster_ids": [a]},
        "explicit_matching": {"cluster_ids": [a], "cluster_ids_explicit": True},
        "disjoint": {"cluster_ids": [uuid.uuid4()]},
        "exclude_b": {"exclude_cluster_ids": [b]},
        "include_a_exclude_b": {"cluster_ids": [a], "exclude_cluster_ids": [b]},
        "explicit_empty": {"cluster_ids": [], "cluster_ids_explicit": True}}
    scope.update(options[mode])
    reader = deployed.HybridRetrievalOrchestrator(store, Encoder())
    assert identities(reader, scope, embedding) == {str(facts[n].id) for n in expected}


@pytest.mark.parametrize("embedding", [None, VEC])
@pytest.mark.parametrize("mode,expected", [
    ("ordinary", {"a", "b", "both", "empty", "unknown"}),
    ("matching", {"a", "both", "empty", "unknown"}),
    ("exclude", {"a", "empty", "unknown"}),
    ("include_exclude", {"a", "empty", "unknown"}),
])
def test_warm_link_membership_unchanged(store, embedding, mode, expected):
    cid, a, b, _rows, facts = cohort(store, cold=False)
    scope = {"conversation_id": str(cid)}
    if mode in ("matching", "include_exclude"): scope["cluster_ids"] = [a]
    if mode in ("exclude", "include_exclude"): scope["exclude_cluster_ids"] = [b]
    reader = deployed.HybridRetrievalOrchestrator(store, Encoder())
    assert identities(reader, scope, embedding) == {str(facts[n].id) for n in expected}


# (v3 claims had a "stale" case — source text edited after extraction — because
# a claim re-checked its character offsets on read. Facts do not re-read their
# source; turns are not edited in place.)
@pytest.mark.parametrize("mode", ["empty_conversations", "other_conversation", "excluded_conversation",
    "empty_batches", "other_batch", "denied_batch", "private", "missing",
    "record_window", "event_window", "warm_private_override"])
def test_other_source_contracts_preserved(store, mode):
    cid, a, _b, rows, facts = cohort(store)
    row, fact = rows["a"], facts["a"]
    scope = {"conversation_id": str(cid), "cluster_ids": [a], "batch_ids": [row.batch_id]}
    reader = deployed.HybridRetrievalOrchestrator(store, Encoder())
    expected = {str(fact.id)}
    if mode == "empty_conversations": scope["conversation_ids"] = []; expected = set()
    if mode == "other_conversation": scope["conversation_id"] = str(uuid.uuid4()); expected = set()
    if mode == "excluded_conversation": scope["exclude_conversation_ids"] = [cid]; expected = set()
    if mode == "empty_batches": scope["batch_ids"] = []; expected = set()
    if mode == "other_batch": scope["batch_ids"] = [uuid.uuid4()]; expected = set()
    if mode == "denied_batch": reader._denied_batch_ids = {row.batch_id}; expected = set()
    if mode == "private": row.is_private = True; expected = set()
    if mode == "missing": store.delete(row); expected = set()
    if mode in ("record_window", "event_window"):
        reader._active_timescope = TimeScope(mode="range", t0=NOW - timedelta(days=2),
            t1=NOW, axis="source_record" if mode == "record_window" else "event")
        if mode == "record_window": expected = set()
    if mode == "warm_private_override":
        store.add(EpisodicMemory(id=row.id, conversation_id=cid, batch_id=row.batch_id,
            raw_text=row.raw_text, source_spans=row.source_spans, timestamp=row.timestamp,
            is_private=True, context_reliance="Long_Term_Memory", idempotency_key=str(uuid.uuid4())))
        expected = set()
    store.flush()
    assert identities(reader, scope, VEC) == expected


def test_actual_decay_preserves_cluster_fact_reach(store):
    cid, a, b, rows, facts = cohort(store, cold=False)
    ids = {name: (row.id, row.raw_text, row.source_spans) for name, row in rows.items()}
    for row in rows.values(): row.decay_score = .001; row.is_archived = True
    store.commit()
    try:
        scope = {"conversation_id": str(cid), "cluster_ids": [a], "exclude_cluster_ids": [b]}
        before = identities(deployed.HybridRetrievalOrchestrator(store, Encoder()), scope, VEC)
        apply_decay(); store.expire_all()
        for ident, raw, spans in ids.values():
            cold = store.get(ColdStorage, ident)
            assert cold.raw_text == raw and cold.source_spans == spans
            assert store.get(EpisodicMemory, ident) is None
        after_new = identities(deployed.HybridRetrievalOrchestrator(store, Encoder()), scope, VEC)
        assert before == after_new == {str(facts[n].id) for n in ("a", "empty", "unknown")}
    finally:
        store.rollback()
        store.query(CodexFact).filter(CodexFact.id.in_([f.id for f in facts.values()])).delete()
        store.query(ColdStorage).filter_by(conversation_id=cid).delete()
        store.query(ContextCluster).filter(ContextCluster.id.in_([a, b])).delete()
        store.query(Conversation).filter_by(id=cid).delete()
        store.commit()


@pytest.mark.parametrize("scope_kind", ["explicit", "inferred", "wide"])
def test_short_fact_reaches_native_prompt_when_its_source_cannot(store, scope_kind):
    cid, a, _b, rows, facts = cohort(store, long_assistant=True)
    # Keep one allowed source, so packing is measured without unrelated fixture competition.
    scope = {"conversation_id": str(cid), "batch_ids": [rows["a"].batch_id]}
    if scope_kind != "inferred":
        scope.update(cluster_ids=[a], cluster_ids_explicit=True)
    confidence = .1 if scope_kind == "wide" else 1.
    classification = ClassificationResult([], ["Factual_Retrieval"], "Long_Term_Memory",
        [0.] * 27, confidence, prompt="Which database did Atlas choose for its sandbox?", p_ltm=1.,
        head_confidences={"topic": confidence, "intent": confidence})
    new = memory_preparation.prepare_memory_context(db=store, classifier=NS(embedder=Encoder()),
        classification=copy.deepcopy(classification), conversation_id=cid,
        user_message=classification.prompt, scope=copy.deepcopy(scope),
        turn_count=80, total_tokens=10000, total_budget=2048, serving_window=8192,
        base_retrieve=True, memory_slots=[], bookmarked_texts=[])
    assert new.prepared.ledger.fits()
    assert new.prepared.messages[-1] == dict(role="user", content=classification.prompt)
    assert len(new.fragments) == 1
    fragment = new.fragments[0]
    assert fragment.stored_artifact_ids == (("codex_facts", str(facts["a"].id)),)
    assert fragment.text.startswith("[Fact; user, asserted;") and "sandbox only" in fragment.text
    assert "Unrelated commentary" not in fragment.text
    assert any(fragment.text in message["content"] for message in new.prepared.messages)
    assert not store.get(EpisodicMemory, rows["a"].id)
