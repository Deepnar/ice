"""The harness must call retrieval the way production does — asserted, not assumed.

⚑ THIS IS THE TEST THAT DID NOT EXIST (G52/G54, TRAPS #43, TRAPS #44)

Nine days and ~2,000 scored requests were taken on a configuration production
never uses. Cross-checking did not catch it, because **every harness shared the
defect, so the harnesses agreed with each other.** Two instruments returning the
same answer feels like corroboration and is worthless when both call the system
the same wrong way. The disagreement only ever appeared against production, and
nothing was comparing to production.

TRAPS #43 wrote down the check that would have caught it:

  1. Find the production call site.
  2. Diff its arguments against the harness's, argument by argument.
  3. Run one request each way and diff the returned `source_type` counts.

Step 3 is this file. It is deliberately a comparison against the REAL
pre-retrieval path rather than a list of expected values: a test that pins
"scope must contain these keys" would have to be updated whenever production
changes, and would then be updated to match whatever production now does —
which is how the copies drifted in the first place.

Run: uv run pytest tests/test_harness_parity.py -q   (needs docker up + a seeded store)
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "z1"))

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import Conversation, EpisodicMemory  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """A live store, a classifier, and the write freeze every harness sets."""
    settings.retrieval_strengthen_writes = False
    settings.decay_strengthen_amount = 0.0

    from src.classifier.classifier import PyTorchClassifier
    from src.memory.embedder import get_embedder
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    db = SessionLocal()
    conv = (db.query(Conversation)
            .join(EpisodicMemory, EpisodicMemory.conversation_id == Conversation.id)
            .group_by(Conversation.id).first())
    if conv is None:
        db.close()
        pytest.skip("store not seeded — nothing to compare against")

    embedder = get_embedder()
    clf = PyTorchClassifier(model_path=settings.classifier_model_path,
                            schema_path=settings.label_schema_path)
    orch = HybridRetrievalOrchestrator(db, embedder)
    # The first retrieval of a process differs (lazily-built relation gloss
    # cache) — TRAPS #27. Burn one before anything is compared.
    import production_parity as pp
    pre = pp.build(db, "warm up", conv.id, clf, embedder)
    pp.retrieve(orch, pre)

    yield db, conv, clf, embedder, orch, pp
    db.close()


def _questions():
    """Real probe questions, not hand-written ones.

    ⚑ TRAPS #13, earned again while writing this file. The first version used
    three questions I made up, and the negative control below went green-when-
    it-should-be-red: none of them selected a cluster, so `scope=None` and the
    real scope returned identical fragments and the guard was vacuous. The
    generated probe set is drawn from the corpus and 90% of it selects a
    cluster, which is the distribution the defect actually lives in.
    """
    import json
    path = Path(__file__).resolve().parent.parent / \
        "experiments" / "curation_files" / "typed_probes.json"
    if not path.exists():
        pytest.skip("typed_probes.json absent — cannot use real questions")
    probes = json.loads(path.read_text())["probes"]
    # Spread across types: each exercises different legs.
    seen, out = set(), []
    for p in probes:
        t = p.get("probe_type")
        if t not in seen:
            seen.add(t)
            out.append(p["question"])
    return out or [p["question"] for p in probes[:5]]


QUESTIONS = _questions()


def test_scope_is_never_none(env):
    """Production passes a dict, always. `None` is a different call.

    `retrieve()` applies cluster ids only `if scope is not None`, so a None
    scope silently skips cluster-scoped retrieval. Measured on 30 probes:
    production sets cluster ids on 90% and the fragment set differs on 17%.
    """
    db, conv, clf, embedder, orch, pp = env
    for q in QUESTIONS:
        pre = pp.build(db, q, conv.id, clf, embedder)
        assert pre.scope is not None, "the harness must pass a scope dict"
        assert isinstance(pre.scope, dict)


def test_classification_sees_the_conversation(env):
    """The context prefix is on, and it demonstrably changes the result.

    Two-sided on purpose (TRAPS #5): asserting only that the call succeeds
    would pass even if `conversation_id` were ignored. The negative control is
    the same question classified WITHOUT the conversation — if that produces an
    identical result on every question, this test is vacuous and the prefix is
    not doing anything.
    """
    db, conv, clf, embedder, orch, pp = env
    differences = 0
    for q in QUESTIONS:
        with_conv = clf.classify(q, conversation_id=str(conv.id))
        without = clf.classify(q)
        if (set(with_conv.topic_tags or []) != set(without.topic_tags or [])
                or set(with_conv.intent_tags or []) != set(without.intent_tags or [])):
            differences += 1
    assert differences > 0, (
        "classifying with and without the conversation produced identical "
        "labels on every question — either the context prefix is disabled or "
        "this control is vacuous; both make the parity claim meaningless")


def test_harness_and_production_return_the_same_fragments(env):
    """The check TRAPS #43 prescribes and nobody ran.

    Drives the same question through the shared harness preamble and through a
    hand-built reproduction of `main.py`'s own sequence, then diffs the
    `source_type` counts. If they differ at all, the harness is measuring
    something other than ICE.
    """
    db, conv, clf, embedder, orch, pp = env
    from collections import Counter

    from src.api.memory_decision import (decide_memory_retrieval,
                                         derive_total_budget,
                                         estimate_recent_window_tokens)
    from src.api.context_ledger import effective_memory_budget
    from src.model_registry.registry import (find_best_model,
                                             get_model_context_window)
    from src.model_registry.runtime_probe import serving_window
    from src.retrieval.timescope import detect_timescope, to_scope_dict
    from src.services.scoping import resolve_retrieval_scope

    for q in QUESTIONS:
        # ── the harness path ──
        pre = pp.build(db, q, conv.id, clf, embedder)
        harness = Counter(f.source_type for f in pp.retrieve(orch, pre))

        # ── main.py's sequence, written out longhand ──
        scope = dict(resolve_retrieval_scope(db, conv) or {})
        result = clf.classify(q, conversation_id=str(conv.id))
        tscope = detect_timescope(q, p_ltm=getattr(result, "p_ltm", 0.0),
                                  p_temporal=getattr(result, "p_temporal", 0.0))
        ts = to_scope_dict(tscope)
        if ts:
            scope["timescope"] = ts
        result.prompt = q
        model_name, _ = find_best_model(result.topic_tags, result.intent_tags)
        raw = get_model_context_window(model_name)
        win = (serving_window(model_name, raw)
               if settings.context_use_serving_window else raw)
        budget = effective_memory_budget(
            derive_total_budget(win, settings), q,
            generation_reserve=settings.context_generation_reserve,
            floor=settings.context_budget_floor)
        tc, tt = pp.conversation_stats(db, conv.id)
        decision = decide_memory_retrieval(
            result, turn_count=tc, total_tokens=tt, settings=settings,
            recent_window_tokens=estimate_recent_window_tokens(tc, budget),
            timescope_mode=tscope.mode,
            coding_scope=bool(scope.get("project_id")))
        production = Counter()
        if decision.retrieve:
            result.context_reliance = "Long_Term_Memory"
            orch.set_budget_from_turn_count(
                tc, total_tokens=tt, classification=result, total_budget=budget)
            emb = embedder.encode(q, convert_to_tensor=False)
            frags = orch.retrieve(
                classification=result, conversation_id=str(conv.id),
                prompt_embedding=emb.tolist() if hasattr(emb, "tolist") else list(emb),
                scope=scope)
            production = Counter(f.source_type for f in frags)

        assert dict(harness) == dict(production), (
            f"harness and production disagree on {q!r}: "
            f"harness={dict(harness)} production={dict(production)}")


def test_scope_none_is_detectably_different(env):
    """The negative control, without which the test above proves nothing.

    If `scope=None` returned the same fragments as the real scope, the parity
    assertion would pass whatever the harness did. This pins that the thing
    being guarded against is actually detectable on this store — and if it ever
    stops being detectable, this test says so rather than quietly going green.
    """
    db, conv, clf, embedder, orch, pp = env
    from collections import Counter

    differing = 0
    for q in QUESTIONS:
        pre = pp.build(db, q, conv.id, clf, embedder)
        proper = Counter(f.source_type for f in pp.retrieve(orch, pre))

        orch.set_budget_from_turn_count(
            pre.turn_count, total_tokens=pre.total_tokens,
            classification=pre.classification, total_budget=pre.total_budget)
        broken = Counter(f.source_type for f in orch.retrieve(
            classification=pre.classification,
            conversation_id=pre.conversation_id,
            prompt_embedding=pre.prompt_embedding,
            scope=None))
        if dict(proper) != dict(broken):
            differing += 1

    assert differing > 0, (
        "scope=None returned identical fragments on every question, so the "
        "parity test above cannot detect the defect it exists for — this store "
        "has no clusters and no own-conversation summaries, and the guard is "
        "vacuous until it does")
