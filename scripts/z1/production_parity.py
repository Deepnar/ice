"""The ONE place a harness reproduces the chat proxy's pre-retrieval path.

⚑ WHY THIS FILE EXISTS (G52, G54, TRAPS #43, TRAPS #44)

Four harnesses — `score_typed`, `score_retrieval`, `answer_probes`,
`harvest_probe_context` — each carried their own copy of "the production
preamble". The copies drifted, every copy drifted the SAME way, and because
they all agreed with each other the disagreement with production stayed
invisible for nine days and ~2,000 scored requests.

Two divergences were found, and the smaller one got all the attention:

* **scope** — every copy passed `scope=None`; production passes a dict, always.
  `retrieve()` applies cluster ids only when the scope is not None, so the
  harnesses silently skipped cluster-scoped retrieval. Measured on 30 probes:
  production sets cluster ids on 90% of them and the returned fragment set
  differs on 17%.

* **⚑ classification context** — every copy called `classify(question)`;
  production calls `classify(user_message, conversation_id=...)`, which builds
  a context prefix from the conversation and embeds THAT. Measured on 100
  stratified probes: topic tags differ on 69, intent tags on 66, the RRF
  leg-blend weights on **65**, and the routed model on 42 — which changes the
  context window and therefore the memory budget. This is roughly four times
  the scope defect and nothing recorded it.

  ⚠ Worth stating in any write-up: the prefix is the conversation's recent
  turns, so a probe about turn 50 gets turn 145's context. There is no
  obviously right answer for an out-of-band probe. What is certain is that
  production ALWAYS has a prefix and the harnesses NEVER did, so they measured
  the classifier in a regime production never occupies. Passing the id models
  "the user asks this now, at the end of their conversation", which is the only
  regime that actually happens.

Two smaller ones fixed here too: the timescope was hardcoded `"current"`
instead of being detected (so the whole time-scoped subsystem never engaged,
including on the `temporal` probe type), and `classification.prompt` was
truncated to 2,000 chars where production sets the full message.

**Anything a harness needs before `retrieve()` belongs here, not in the
harness.** The reference is `src/api/main.py` lines ~400-610; when that
changes, this changes with it, and `tests/test_harness_parity.py` fails if it
does not.
"""
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import func

from src.api.config import settings
from src.api.context_ledger import effective_memory_budget
from src.api.memory_decision import (decide_memory_retrieval,
                                     derive_total_budget,
                                     estimate_recent_window_tokens)
from src.memory.models import Conversation, EpisodicMemory
from src.memory.tokens import estimate_from_chars
from src.model_registry.registry import find_best_model, get_model_context_window
from src.model_registry.runtime_probe import serving_window
from src.retrieval.timescope import detect_timescope, to_scope_dict
from src.services.scoping import resolve_retrieval_scope


@dataclass
class Preamble:
    """Everything production computes before it calls `retrieve()`."""
    classification: object
    scope: dict
    prompt_embedding: list
    conversation_id: str
    turn_count: int
    total_tokens: float
    total_budget: int
    model_name: str
    timescope_mode: str
    retrieve: bool          # what the B2 gate decided
    p_need_mem: float


def conversation_stats(db, conversation_id) -> tuple:
    """Live turn count + estimated token size, as `main.py` computes them."""
    turn_count = db.query(EpisodicMemory).filter_by(
        conversation_id=conversation_id).count()
    chars = db.query(
        func.coalesce(func.sum(func.length(EpisodicMemory.raw_text)), 0)
    ).filter_by(conversation_id=conversation_id).scalar() or 0
    return turn_count, estimate_from_chars(chars)


def build(db, question: str, conversation_id, classifier, embedder,
          stats: Optional[tuple] = None) -> Preamble:
    """Reproduce the chat path's pre-retrieval work, argument for argument.

    Mirrors `src/api/main.py` in order: resolve the scope from the conversation
    row, classify WITH the conversation, detect the timescope and fold it into
    the scope, route a model, derive the budget, run the B2 gate, then stamp
    `context_reliance`.
    """
    conv_row = db.query(Conversation).filter(
        Conversation.id == conversation_id).first()

    # main.py:421 — the scope is ALWAYS a dict. For an `auto` conversation it
    # is empty, and that is the correct empty, not a missing one.
    scope = dict(resolve_retrieval_scope(db, conv_row) or {})

    # main.py:448 — the conversation id is what turns on the context prefix.
    classification = classifier.classify(question,
                                         conversation_id=str(conversation_id))

    # main.py:456 — detected, never assumed. A non-current mode disables the
    # batch-summary leg and re-points cold storage, so hardcoding "current"
    # measured only one branch of a subsystem the `temporal` probes exist for.
    tscope = detect_timescope(
        question,
        p_ltm=getattr(classification, "p_ltm", 0.0),
        p_temporal=getattr(classification, "p_temporal", 0.0))
    ts_dict = to_scope_dict(tscope)
    if ts_dict:
        scope["timescope"] = ts_dict

    # main.py:556 — the full message, not a truncation. The search prompt and
    # the keyword set are both derived from this.
    classification.prompt = question

    model_name, _ = find_best_model(classification.topic_tags,
                                    classification.intent_tags)
    raw_window = get_model_context_window(model_name)
    window = (serving_window(model_name, raw_window)
              if settings.context_use_serving_window else raw_window)
    total_budget = effective_memory_budget(
        derive_total_budget(window, settings), question,
        generation_reserve=settings.context_generation_reserve,
        floor=settings.context_budget_floor)

    turn_count, total_tokens = stats if stats else conversation_stats(
        db, conversation_id)

    decision = decide_memory_retrieval(
        classification, turn_count=turn_count, total_tokens=total_tokens,
        settings=settings,
        recent_window_tokens=estimate_recent_window_tokens(turn_count, total_budget),
        timescope_mode=tscope.mode,
        coding_scope=bool(scope.get("project_id")))

    if decision.retrieve:
        classification.context_reliance = "Long_Term_Memory"

    vec = embedder.encode(question, convert_to_tensor=False)
    prompt_embedding = vec.tolist() if hasattr(vec, "tolist") else list(vec)

    return Preamble(
        classification=classification, scope=scope,
        prompt_embedding=prompt_embedding,
        conversation_id=str(conversation_id),
        turn_count=turn_count, total_tokens=total_tokens,
        total_budget=total_budget, model_name=model_name,
        timescope_mode=tscope.mode,
        retrieve=decision.retrieve, p_need_mem=decision.p_need_mem)


def retrieve(orchestrator, pre: Preamble):
    """Budget then retrieve, exactly as `main.py:597-606` does."""
    orchestrator.set_budget_from_turn_count(
        pre.turn_count, total_tokens=pre.total_tokens,
        classification=pre.classification, total_budget=pre.total_budget)
    return orchestrator.retrieve(
        classification=pre.classification,
        conversation_id=pre.conversation_id,
        prompt_embedding=pre.prompt_embedding,
        scope=pre.scope)


def provenance_fields(pre: Preamble) -> dict:
    """The fields whose ABSENCE let G52 and G54 survive nine days of runs.

    Every harness records resolved settings and a git sha, and not one recorded
    what it actually passed to `retrieve()`. These go in the run's `extra`.
    """
    return {
        "scope_keys": sorted(pre.scope.keys()),
        "scope_is_none": False,
        "classify_had_conversation_id": True,
        "timescope_mode": pre.timescope_mode,
        "coding_scope": bool(pre.scope.get("project_id")),
        "routed_model": pre.model_name,
        "total_budget": pre.total_budget,
        "b2_retrieve": pre.retrieve,
    }
