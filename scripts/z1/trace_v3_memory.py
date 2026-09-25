"""One private, disposable v3 source-to-prompt trace from recorded dialogue.

Run through tests/support/disposable_database.py. Each recorded assistant reply
enters the store only after that user's preflight retrieval and prompt assembly.
The JSON report contains source text, so the destination must be under logs/.
This is a development diagnostic, not the LME or LSREP evaluation harness.
"""

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.engine import make_url

from scripts.z1 import production_parity as pp
from src.api.config import settings
from src.api.db import SessionLocal
from src.api.prompt_assembler import bookmarked_turn_texts, conversation_summary_block
from src.api.prompt_budget import assemble_budgeted_prompt
from src.classifier.classifier import PyTorchClassifier
from src.ingestion.importer import _store_turn
from src.memory.models import Conversation, CodexClaim, CodexEdge, MemorySlot
from src.memory.tokens import count_messages
from src.memory.usage import evidence_after_eviction, record_graph_access
from src.model_registry.registry import get_model_context_window
from src.model_registry.runtime_probe import serving_window
from src.retrieval.orchestrator import HybridRetrievalOrchestrator
from src.workers import codex_extractor
from src.workers.post_flight import evaluate_turn


def _private_output(path: str) -> Path:
    target = Path(path).resolve()
    log_root = (Path(__file__).resolve().parents[2] / "logs").resolve()
    if not target.is_relative_to(log_root):
        raise ValueError("Trace output contains private dialogue and must stay under logs/")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _load_pairs(path: str, start: int, count: int) -> list[tuple[dict, dict]]:
    rows = [json.loads(line) for line in Path(path).open()]
    selected = rows[start:start + 2 * count]
    if len(selected) != 2 * count:
        raise ValueError("Requested window extends past the dialogue")
    pairs = []
    for offset in range(0, len(selected), 2):
        user, assistant = selected[offset:offset + 2]
        if (user.get("role"), assistant.get("role")) != ("user", "assistant"):
            raise ValueError(f"Expected user/assistant pair at source row {start + offset}")
        pairs.append((user, assistant))
    return pairs


def _fragment_record(fragment) -> dict:
    return {
        "leg": getattr(fragment, "leg", None),
        "source_type": fragment.source_type,
        "text": fragment.text,
        "tokens": fragment.token_count,
        "source_row": str(fragment.source_batch_id) if fragment.source_batch_id else None,
        "origin_batches": [str(i) for i in (fragment.origin_batch_ids or ())],
        "origin_edges": [str(i) for i in (fragment.origin_edge_ids or ())],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = _private_output(args.out)
    test_database = os.environ.get("ICE_TEST_DATABASE", "")
    if not test_database.startswith("ice_test_") or make_url(settings.database_url).database != test_database:
        raise RuntimeError("Trace must run in a disposable ICE_TEST_DATABASE")
    pairs = _load_pairs(args.source, args.start, args.pairs)

    classifier = PyTorchClassifier(model_path=settings.classifier_model_path,
                                   schema_path=settings.label_schema_path)
    embedder = classifier.embedder
    conversation_id = uuid.uuid4()
    db = SessionLocal()
    db.add(Conversation(id=conversation_id, memory_scope_type="auto"))
    db.commit()
    report = {"version": "v3", "source": Path(args.source).name,
              "source_start": args.start, "pairs_requested": args.pairs,
              "conversation_id": str(conversation_id), "turns": []}
    parse_response = codex_extractor.parse_extraction_response

    def capture_parse(content, finish_reason=None, *, template_mode=False,
                      allow_source_only=False):
        try:
            return parse_response(content, finish_reason, template_mode=template_mode,
                                  allow_source_only=allow_source_only)
        except Exception as exc:
            report.setdefault("invalid_extractions", []).append({
                "reason": str(exc), "finish_reason": finish_reason,
                "template_mode": template_mode, "model_output": content})
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            raise

    codex_extractor.parse_extraction_response = capture_parse
    try:
        for index, (user, assistant) in enumerate(pairs):
            question = user["content"]
            recorded_answer = assistant["content"]
            pre = pp.build(db, question, conversation_id, classifier, embedder)
            orchestrator = HybridRetrievalOrchestrator(db, embedder)
            fragments = pp.retrieve(orchestrator, pre) if pre.retrieve else []
            recent_budget = getattr(orchestrator, "recent_token_budget", None)
            if not pre.retrieve or recent_budget is None:
                from src.api.memory_decision import estimate_recent_window_tokens
                recent_budget = estimate_recent_window_tokens(pre.turn_count, pre.total_budget)
            summary_options = conversation_summary_block(
                db, str(conversation_id), pre.turn_count, pre.total_tokens,
                recent_budget, pre.prompt_embedding, include_options=True)
            window = (serving_window(pre.model_name, get_model_context_window(pre.model_name))
                      if settings.context_use_serving_window
                      else get_model_context_window(pre.model_name))
            prepared = assemble_budgeted_prompt(
                serving_window=window or 0,
                generation_reserve=settings.context_generation_reserve,
                safety_margin=settings.token_count_safety_margin,
                memory_slots=db.query(MemorySlot).filter_by(is_active=True).all(),
                retrieved_fragments=fragments, user_message=question, db_session=db,
                conversation_id=str(conversation_id),
                bookmarked_texts=bookmarked_turn_texts(db, conversation_id),
                classification=pre.classification, scope=pre.scope,
                max_recent_tokens=recent_budget,
                conversation_summary_text=summary_options[0] if summary_options else None,
                conversation_summary_options=summary_options)
            survivors = evidence_after_eviction(fragments, prepared.removed)
            record_graph_access(db, survivors, stage="prompt_prepared")
            if not prepared.ledger.fits():
                raise RuntimeError(f"Required prompt exceeds context window at pair {index}")
            before_edges = db.query(CodexEdge).count()
            before_claims = db.query(CodexClaim).count()
            turn = {"index": index, "source_row": args.start + 2 * index,
                    "question": question, "recorded_answer": recorded_answer,
                    "source_timestamp": user.get("timestamp"),
                    "preflight": pp.provenance_fields(pre),
                    "matched_graph_entities": len(orchestrator._last_matched_entities),
                    "retrieved": [_fragment_record(f) for f in fragments],
                    "selected": [_fragment_record(f) for f in survivors],
                    "prompt_messages": prepared.messages,
                    "prompt_tokens": count_messages(prepared.messages),
                    "evicted": prepared.removed, "complete": False}
            report["turns"].append(turn)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            timestamp = datetime.fromisoformat(user["timestamp"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            stored = _store_turn(db, conversation_id, question, recorded_answer,
                                 timestamp, f"v3-trace-{conversation_id}-{index}",
                                 "fresh", datetime.now(timezone.utc), "original",
                                 classifier, embedder)
            if stored is None:
                raise RuntimeError(f"Unexpected duplicate at pair {index}")
            batch_id, prompt, response = stored
            try:
                evaluate_turn(batch_id=str(batch_id), prompt=prompt, response=response,
                              conversation_id=str(conversation_id), model_used="recorded")
            except Exception as exc:
                turn["postflight_error"] = f"{type(exc).__name__}: {exc}"
                output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
                raise
            db.expire_all()
            turn["postflight"] = {
                "new_edges": db.query(CodexEdge).count() - before_edges,
                "new_claims": db.query(CodexClaim).count() - before_claims,
                "batch_id": str(batch_id),
            }
            turn["complete"] = True
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            print(f"pair {index + 1}/{args.pairs}: matched={turn['matched_graph_entities']} "
                  f"selected={len(survivors)} edges+={turn['postflight']['new_edges']} "
                  f"claims+={turn['postflight']['new_claims']}", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()
