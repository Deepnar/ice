"""CODEX_V4 read side: the facts leg, final-assembly skip, strength bonus, usage (§5.4).

Runs only in a disposable database; stub embeddings (one fixed direction), no
reranker, no model.

Run:  uv run --no-sync python tests/support/disposable_database.py -m pytest tests/test_codex_facts_read.py -q
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import numpy as np
import pytest

from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_budget import _fact_sources_in_prompt
from src.memory.models import (
    CodexFact, CodexFactSource, Conversation, EpisodicMemory,
)
from src.memory.source import chat_provenance
from src.memory.usage import record_graph_access
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator

pytestmark = pytest.mark.skipif(not os.getenv("ICE_TEST_DATABASE"),
                                reason="requires a disposable PostgreSQL database")
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
VEC = [1.0] + [0.0] * 1023


def _turn(db, text, private=False, when=T0):
    conv = Conversation()
    db.add(conv)
    db.flush()
    batch = uuid.uuid4()
    turn = EpisodicMemory(conversation_id=conv.id, batch_id=batch, idempotency_key=f"read-{batch}",
                          context_reliance="Long_Term_Memory", timestamp=when, is_private=private,
                          raw_text=f"User: {text}\n\nAssistant: ok",
                          source_spans=chat_provenance(text, "ok"))
    db.add(turn)
    db.flush()
    return turn


def _fact(db, text, *turns, when=T0, support=1, usage=0.0):
    fact = CodexFact(text=text, text_sha256=uuid.uuid4().hex, speaker="user", modality="asserted",
                     embedding=VEC, mentioned_at=when, valid_from=when, support_count=support,
                     usage=usage)
    db.add(fact)
    db.flush()
    for turn in turns:
        db.add(CodexFactSource(fact_id=fact.id, source_batch=turn.batch_id, window_sha256=uuid.uuid4().hex,
                               episodic_id=turn.id, conversation_id=turn.conversation_id,
                               window={}, quote=text[:20], recorded_at=when, verification={}))
    db.flush()
    return fact


def _leg(db, prompt, scope=None, embedding=None):
    o = HybridRetrievalOrchestrator(db, None)
    o._resolve_exclusion_sets(scope)
    o._facts_entities = lambda _p: []          # entity channel off: no NER model here
    return o, o._facts_leg(prompt, embedding, scope)


def test_visible_through_any_source_and_hidden_when_private_or_excluded():
    with SessionLocal() as db:
        a, b = _turn(db, "I adopted a cat named Miso."), _turn(db, "Miso the cat is three now.")
        secret = _turn(db, "I adopted a hamster named Pip.", private=True)
        shared = _fact(db, "The user said they adopted a cat named Miso.", a, b)
        hidden = _fact(db, "The user said they adopted a hamster named Pip.", secret)
        _, frags = _leg(db, "adopted cat Miso", scope={"conversation_id": str(b.conversation_id)})
        texts = " ".join(f.text for f in frags)
        assert "cat named Miso" in texts                                # via its SECOND source
        frag = next(f for f in frags if "Miso" in f.text)
        assert frag.leg == "facts" and frag.conversation_id is None
        assert frag.origin_fact_ids == (str(shared.id),)
        assert set(frag.fact_source_row_ids) == {str(b.id)}             # only the visible source
        _, frags = _leg(db, "adopted hamster Pip")
        assert not any(str(hidden.id) in f.origin_fact_ids for f in frags)
        _, frags = _leg(db, "adopted cat Miso", scope={"exclude_conversation_ids": [
            str(a.conversation_id), str(b.conversation_id)]})
        assert not any("Miso" in f.text for f in frags)
        _, frags = _leg(db, "adopted cat Miso", scope={"incognito": True})
        assert frags == []
        db.rollback()


def test_current_only_with_predecessor_clause():
    with SessionLocal() as db:
        old_t, new_t = _turn(db, "I live in Porto."), _turn(db, "I moved to Lisbon.", when=T0 + timedelta(days=40))
        old = _fact(db, "The user said they live in Porto.", old_t)
        new = _fact(db, "The user said they moved to Lisbon.", new_t, when=T0 + timedelta(days=40))
        old.valid_until, old.superseded_by = T0 + timedelta(days=40), new.id
        db.flush()
        _, frags = _leg(db, "where does the user live Porto Lisbon", embedding=VEC)
        texts = [f.text for f in frags]
        assert not any(t.startswith("[Fact") and "live in Porto." in t.split("]", 1)[1].split(" (earlier")[0]
                       for t in texts)                                  # replaced fact not served alone
        current = next(t for t in texts if "moved to Lisbon" in t)
        assert "; current]" in current and "(earlier: The user said they live in Porto. — until" in current
        db.rollback()


def test_keyword_and_meaning_channels():
    with SessionLocal() as db:
        turn = _turn(db, "The deploy key rotates every 90 days.")
        _fact(db, "The user said the deploy key rotates every 90 days.", turn)
        _, by_word = _leg(db, "deploy key rotation")
        assert any("deploy key" in f.text for f in by_word)
        _, by_vector = _leg(db, "zzz unrelated words", embedding=VEC)
        assert any("deploy key" in f.text for f in by_vector)
        _, neither = _leg(db, "zzz unrelated words")
        assert not any("deploy key" in f.text for f in neither)
        db.rollback()


def _frag(rows, leg="facts"):
    return ContextFragment("[Fact] x", "codex", 1.0, 3, leg=leg, fact_source_row_ids=tuple(rows),
                           origin_fact_ids=("f1",))


def _orig(row, complete=True):
    return ContextFragment("raw", "episodic", 1.0, 3, source_batch_id=row, covers_entire_source=complete)


def test_skip_only_when_every_source_is_in_the_prompt():
    assert _fact_sources_in_prompt(_frag(["r1"]), [_orig("r1")], set())
    assert _fact_sources_in_prompt(_frag(["r1", "r2"]), [_orig("r1")], {"r2"})          # recent window
    assert not _fact_sources_in_prompt(_frag(["r1", "r2"]), [_orig("r1")], set())
    assert not _fact_sources_in_prompt(_frag(["r1"]), [_orig("r1", complete=False)], set())  # an excerpt
    assert not _fact_sources_in_prompt(_frag(["r1"], leg="vector"), [_orig("r1")], set())


def test_strength_bonus_off_by_default_and_bounded(monkeypatch):
    o = HybridRetrievalOrchestrator(None, None)
    a = ContextFragment("a", "codex", 1.0, 1, leg="facts", origin_fact_ids=("a",))
    b = ContextFragment("b", "codex", 1.0, 1, leg="facts", origin_fact_ids=("b",))
    o._fact_strength = {"a": (1, 0.0), "b": (10, 50.0)}
    assert [f.score for f in o._fact_strength_bonus([a, b])] == [1.0, 1.0]
    monkeypatch.setattr(settings, "codex_fact_strength_bonus", 0.9)
    out = {f.text: f.score for f in o._fact_strength_bonus([a, b])}
    assert out["a"] == 1.0 and abs(out["b"] - 1.9) < 1e-9                # saturates at B
    o._fact_strength = {"b": (4, 0.0)}
    assert abs({f.text: f.score for f in o._fact_strength_bonus([a, b])}["b"] - 1.6) < 1e-9  # support = 2/3


def test_usage_counts_prompt_exposure_of_current_facts():
    with SessionLocal() as db:
        turn = _turn(db, "I drink oat milk.")
        live = _fact(db, "The user said they drink oat milk.", turn)
        gone = _fact(db, "The user said they drink cow milk.", turn)
        gone.valid_until = T0
        db.commit()
        frags = [NS(origin_fact_ids=(str(live.id),)), NS(origin_fact_ids=(str(gone.id),))]
        record_graph_access(db, frags, stage="prompt_prepared")
        record_graph_access(db, frags, stage="prompt_prepared")
        db.expire_all()
        assert db.get(CodexFact, live.id).usage == 2.0 and db.get(CodexFact, live.id).last_used_at
        assert db.get(CodexFact, gone.id).usage == 0.0


def test_entity_channel_finds_known_names_whole_word(monkeypatch):
    from src.memory.models import CodexEntity, CodexFactEntity
    from src.retrieval import orchestrator as orch
    monkeypatch.setattr(orch, "extract_entities", lambda *_a, **_k: [])     # micro-NER finds nothing
    with SessionLocal() as db:
        tag = uuid.uuid4().hex[:5]
        turn = _turn(db, "Orbit Lab is my side project.")
        fact = _fact(db, f"The user said Orbit Lab {tag} is their side project.", turn)
        ent = CodexEntity(canonical_name=f"orbit-lab {tag}", source="conversation")
        husk = CodexEntity(canonical_name=f"zephyr {tag} [merged:abc]", source="conversation",
                           properties={"merged_into": str(uuid.uuid4())})
        db.add_all([ent, husk])
        db.flush()
        db.add(CodexFactEntity(fact_id=fact.id, entity_id=ent.id, role="subject"))
        db.flush()
        o = HybridRetrievalOrchestrator(db, None)
        o._resolve_exclusion_sets(None)
        assert [e.id for e in o._facts_entities(f"what is Orbit Lab {tag} again?")] == [ent.id]
        frags = o._facts_leg(f"zzz Orbit-Lab {tag}?", None, None)
        assert any(str(fact.id) in f.origin_fact_ids for f in frags)
        assert o._facts_entities(f"orbit-labs{tag} and zephyr{tag}") == []          # not whole words
        assert o._facts_entities(f"zephyr {tag}") == []                             # merged husk never
        db.rollback()


def test_code_graph_anchor_renders_inside_its_project_only(monkeypatch):
    from src.memory.models import CodexEntity
    from src.retrieval import orchestrator as orch
    monkeypatch.setattr(orch, "extract_entities", lambda *_a, **_k: [])
    with SessionLocal() as db:
        project = uuid.uuid4()
        name = f"minipkg.core.greet{uuid.uuid4().hex[:4]}"
        db.add(CodexEntity(canonical_name=name, source="static_analysis", project_id=project,
                           entity_type="function", context_payload="defined at minipkg/core.py:4"))
        db.flush()
        o = HybridRetrievalOrchestrator(db, None)
        o._resolve_exclusion_sets(None)
        assert not any(f.leg == "code_graph" for f in o._facts_leg(f"what does {name} do", None))
        o._scope_project_id = project
        frags = o._facts_leg(f"what does {name} do", None)
        anchor = [f for f in frags if f.leg == "code_graph"]
        assert anchor and "minipkg/core.py:4" in anchor[0].text and anchor[0].origin_fact_ids == ()
        db.rollback()
