#!/usr/bin/env python3
"""Trace one ICE v2 LongMemEval retrieval without changing the store.

Run from the v2 worktree so imports and the 384-dimensional checkpoint resolve
to the evaluated system:

    uv run python /home/deepnar/Programs/ice/experiments/lme/trace_retrieval.py \
        --phase stratified --question-id 2b8f3739

The trace calls the real ``retrieve()`` method but replaces its strengthening
hook with a no-op, so access counts and decay scores are not modified. It wraps
each retrieval leg and each downstream stage to make candidate loss visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any


HARNESS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, os.getcwd())


def _fragment_summary(fragments: list[Any]) -> dict[str, Any]:
    return {
        "count": len(fragments),
        "tokens": sum(int(getattr(f, "token_count", 0) or 0) for f in fragments),
        "source_types": dict(Counter(getattr(f, "source_type", "<none>") for f in fragments)),
        "unique_texts": len({getattr(f, "text", "") for f in fragments}),
        "token_min": min((int(getattr(f, "token_count", 0) or 0) for f in fragments), default=0),
        "token_max": max((int(getattr(f, "token_count", 0) or 0) for f in fragments), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="stratified")
    parser.add_argument("--question-id", required=True)
    parser.add_argument(
        "--legacy-flattened", action="store_true",
        help="reproduce the retired UUID/one-conversation adapter for diagnosis",
    )
    args = parser.parse_args()

    from src.api.config import settings
    from src.api.db import SessionLocal
    from src.classifier.classifier import PyTorchClassifier
    from src.memory.models import EpisodicMemory
    from src.retrieval.orchestrator import HybridRetrievalOrchestrator

    from lme_run import (ADAPTER_VERSION, _to_pairs, auto_query_context,
                         estimate_tokens, instance_layout)

    corpus = json.loads((HARNESS_DIR / "data" / "longmemeval_s").read_text())
    instance = next((x for x in corpus if x["question_id"] == args.question_id), None)
    if instance is None:
        raise SystemExit(f"question id not found: {args.question_id}")

    pairs = [pair for session in instance["haystack_sessions"] for pair in _to_pairs(session)]
    total_tokens = sum(estimate_tokens(prompt + " " + response) for prompt, response in pairs)
    layout = instance_layout(instance, args.phase)
    if args.legacy_flattened:
        query_cid = uuid.uuid5(
            uuid.NAMESPACE_DNS, f"lme:{args.phase}:{args.question_id}"
        )
        classification_cid = query_cid
        retrieval_cid = query_cid
        scope = {"conversation_id": query_cid}
        budget_turns, budget_tokens = len(pairs), total_tokens
        layout_name = "flattened-sessions-v1"
    else:
        query_cid = layout["query_conversation_id"]
        retrieval_cid, scope = auto_query_context(query_cid)
        classification_cid = retrieval_cid
        budget_turns, budget_tokens = 0, 0
        layout_name = ADAPTER_VERSION

    classifier = PyTorchClassifier(
        model_path=settings.classifier_model_path,
        schema_path=settings.label_schema_path,
    )
    embedder = classifier.embedder
    question = instance["question"]
    classification = classifier.classify(
        question, conversation_id=classification_cid
    )
    embedding = embedder.encode(question, convert_to_tensor=False).tolist()

    db = SessionLocal()
    try:
        stored_pairs = db.query(EpisodicMemory).count()
        orchestrator = HybridRetrievalOrchestrator(db, embedder)
        orchestrator.set_budget_from_turn_count(
            budget_turns, budget_tokens, classification=classification
        )

        trace: dict[str, Any] = {"legs": {}, "stages": [], "controls": {}}
        control_fragments: dict[str, list[Any]] = {}

        leg_names = (
            "_bm25_episodic",
            "_vector_episodic",
            "_codex_graph",
            "_procedural_lookup",
            "_rag_lookup",
            "_batch_summary_lookup",
        )
        for name in leg_names:
            original = getattr(orchestrator, name)

            def wrapped_leg(*call_args, _name=name, _original=original, **call_kwargs):
                result = _original(*call_args, **call_kwargs)
                trace["legs"][_name] = _fragment_summary(result)
                return result

            setattr(orchestrator, name, wrapped_leg)

        stage_names = (
            "_apply_rrf",
            "_session_diversify",
            "_deduplicate",
            "_enforce_token_budget",
        )
        original_budget = orchestrator._enforce_token_budget
        original_deduplicate = orchestrator._deduplicate
        for name in stage_names:
            original = getattr(orchestrator, name)

            def wrapped_stage(*call_args, _name=name, _original=original, **call_kwargs):
                stage_input = call_args[0] if call_args else []
                if isinstance(stage_input, dict):
                    before = {
                        leg: _fragment_summary(fragments)
                        for leg, fragments in stage_input.items()
                    }
                else:
                    before = _fragment_summary(list(stage_input))
                result = _original(*call_args, **call_kwargs)
                if _name == "_session_diversify" and len(call_args) >= 2:
                    current_id = call_args[1]
                    string_id_result = _original(
                        stage_input, str(current_id), **call_kwargs
                    )
                    control_fragments["string_current_id"] = string_id_result
                    trace["controls"]["session_diversify_id_type"] = {
                        "live_current_id_type": type(current_id).__name__,
                        "fragment_conversation_id_type": (
                            type(getattr(stage_input[0], "conversation_id", None)).__name__
                            if stage_input else "<empty>"
                        ),
                        "live_result": _fragment_summary(result),
                        "string_id_result": _fragment_summary(string_id_result),
                    }
                trace["stages"].append({
                    "stage": _name,
                    "before": before,
                    "after": _fragment_summary(result),
                    "max_tokens": call_kwargs.get("max_tokens"),
                })
                return result

            setattr(orchestrator, name, wrapped_stage)

        # Retrieval normally strengthens returned episodic memories. That write is
        # irrelevant to candidate tracing and would make this diagnostic mutating.
        orchestrator._strengthen_retrieved = lambda _fragments: None

        final = orchestrator.retrieve(
            classification=classification,
            conversation_id=retrieval_cid,
            prompt_embedding=embedding,
            scope=scope,
        )
        if "string_current_id" in control_fragments:
            corrected_deduped = original_deduplicate(
                control_fragments["string_current_id"]
            )
            corrected_budgeted = original_budget(corrected_deduped)
            trace["controls"]["string_id_through_remaining_stages"] = {
                "after_deduplicate": _fragment_summary(corrected_deduped),
                "after_token_budget": _fragment_summary(corrected_budgeted),
            }

        payload = {
            "system": "ICE v2 on lme/v2-paper-eval",
            "adapter_layout": layout_name,
            "question_id": args.question_id,
            "question": question,
            "query_conversation_id": str(query_cid),
            "expected_pairs": len(pairs),
            "stored_pairs": stored_pairs,
            "store_complete": stored_pairs == len(pairs),
            "classification": {
                "topic_tags": list(classification.topic_tags or []),
                "intent_tags": list(classification.intent_tags or []),
                "context_reliance": str(classification.context_reliance),
                "max_confidence": classification.max_confidence,
                "fallback_threshold": settings.confidence_fallback_threshold,
            },
            "budget": {
                "retrieval_tokens": orchestrator.max_retrieval_tokens,
                "recent_tokens": orchestrator.recent_token_budget,
            },
            "scope": {key: [str(x) for x in value] if isinstance(value, list) else str(value)
                      for key, value in scope.items()},
            "trace": trace,
            "final": _fragment_summary(final),
        }
        print(json.dumps(payload, indent=2))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
