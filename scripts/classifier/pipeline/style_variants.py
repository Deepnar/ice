#!/usr/bin/env python3
"""G28: paired meaning-preserving style probes for the live B2 gate.

Run only in a disposable database. Sources are synthetic and remain outside
both the classifier's last-three-turn view and the answerer's recent window.
This is a small diagnostic instrument,
not a population score or a parameter-tuning set. Keep it eval-only forever.

    uv run python tests/support/disposable_database.py \
        scripts/classifier/pipeline/style_variants.py
"""

import argparse
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sqlalchemy.engine import make_url

from src.api.config import settings
from src.api.db import SessionLocal
from src.api.memory_decision import decide_memory_retrieval, estimate_recent_window_tokens
from src.api.memory_preparation import prepare_memory_context
from src.api.prompt_assembler import get_recent_turns
from src.classifier.classifier import PyTorchClassifier
from src.memory.models import Conversation, EpisodicMemory
from src.memory.source import single_provenance
from src.retrieval.timescope import detect_timescope


# Each row in one group asks for the SAME information. A low flip rate alone
# is not success: the group must also get the correct retrieve/no-retrieve call.
GROUPS = (
    ("historical_gpu_command", True, (
        "Which exact command did I run to check GPU memory during our setup?",
        "ok so which exact command did i run to check GPU memory during our setup",
        "During our setup, what exact command did I run to check GPU memory?",
        "The GPU-memory check in our setup—what command did I run?",
        "At setup, which exact command did I run for GPU memory",
        "which exact comand did i run for gpu mem during our setup",
    )),
    ("historical_indexer_port", True, (
        "What port did we choose for the local indexer in our earlier setup?",
        "ok so what port did we choose for the local indexer in our earlier setup",
        "For the local indexer, what port did we choose in our earlier setup?",
        "Our earlier local-indexer setup used which port?",
        "at that local indexer setup, which port did we choose",
        "what port did we chosse for the local indexer in our earlier setup",
    )),
    ("generic_gpu_command", False, (
        "What command checks GPU memory from a terminal?",
        "ok so what command checks GPU memory from a terminal",
        "From a terminal, which command checks GPU memory?",
        "A command for checking GPU memory in a terminal—what is it?",
        "gpu memory terminal inspection command?",
        "what comand checks gpu mem from a terminal",
    )),
    ("self_contained_indexer_port", False, (
        "Our local indexer uses port 7619. Which port does it use?",
        "ok so our local indexer uses port 7619 which port does it use",
        "Given that our local indexer uses port 7619, what is its port?",
        "The port is 7619 for our local indexer—what port is it on?",
        "our local indexer: port 7619. which port does it use",
        "our local indexer uses port 7619 wich port does it use",
    )),
)


def _source_context(db):
    conversation = Conversation(id=uuid.uuid4())
    db.add(conversation)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    earlier_sitting, current_sitting = uuid.uuid4(), uuid.uuid4()
    texts = (
        "User: For this setup I checked GPU memory with gpu-scan --mem --device 7.",
        "User: The local indexer will listen on port 7619.",
        "Assistant: Those setup facts are recorded.",
        "User: Let's now discuss the shape of a short release note.",
        "Assistant: A release note can start with the user-facing change.",
        "User: Good, and keep the tone direct and plain.",
    )
    for index, raw in enumerate(texts):
        db.add(EpisodicMemory(
            conversation_id=conversation.id, batch_id=uuid.uuid4(),
            raw_text=raw, source_spans=single_provenance(
                raw, "assistant" if raw.startswith("Assistant:") else "user"),
            session_id=earlier_sitting if index < 3 else current_sitting,
            timestamp=start + timedelta(days=int(index >= 3), minutes=index),
            idempotency_key=f"g28-style-{uuid.uuid4()}",
            context_reliance="Long_Term_Memory",
        ))
    db.commit()
    total_tokens = sum(len(raw) / 4 for raw in texts)
    recent = get_recent_turns(
        db, str(conversation.id),
        max_tokens=int(estimate_recent_window_tokens(len(texts))),
    )
    visible = "\n".join(message["content"] for message in recent)
    if "gpu-scan" in visible or "7619" in visible:
        raise AssertionError("Historical answers leaked into the real recent window")
    if not recent:
        raise AssertionError("Recent-window fixture must contain the current sitting")
    return conversation.id, len(texts), total_tokens


def main():
    parser = argparse.ArgumentParser(description="G28 production-path style probes")
    parser.add_argument("--checkpoint", default=settings.classifier_model_path)
    parser.add_argument("--source-gate", action="store_true",
                        help="Run qualified current-source proof at a controlled window")
    parser.add_argument("--source-rescue", action="store_true",
                        help="Also qualify the unpromoted negative-rescue branch")
    parser.add_argument("--window", type=int, default=8192)
    args = parser.parse_args()
    if args.source_rescue and not args.source_gate:
        parser.error("--source-rescue requires --source-gate")
    if not os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_"):
        raise SystemExit("G28 style probes require a disposable database")
    if make_url(settings.database_url).database != os.environ["ICE_TEST_DATABASE"]:
        raise SystemExit("DATABASE_URL is not the disposable database")
    classifier = PyTorchClassifier(model_path=args.checkpoint)
    settings.memory_source_gate_enabled = args.source_gate
    settings.memory_source_rescue_enabled = args.source_rescue
    settings.retrieval_strengthen_writes = False
    with SessionLocal() as db:
        conversation_id, turns, total_tokens = _source_context(db)
        if args.source_gate:
            for turn in db.query(EpisodicMemory).filter_by(conversation_id=conversation_id):
                vector = classifier.embedder.encode(turn.raw_text, convert_to_tensor=False)
                turn.embedding = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            db.commit()
        all_correct = flip_groups = 0
        print(f"v3 checkpoint={args.checkpoint} "
              f"turns={turns} estimated_history_tokens={total_tokens:.0f}")
        print(f"stage={'final_source_gate' if args.source_gate else 'B2_prior'} "
              f"source_rescue={args.source_rescue} "
              f"controlled_window={args.window if args.source_gate else 'unused'} "
              "retention_writes=False")
        for name, want, prompts in GROUPS:
            decisions = []
            for prompt in prompts:
                result = classifier.classify(prompt, conversation_id=str(conversation_id))
                timescope = detect_timescope(prompt, p_ltm=result.p_ltm,
                                             p_temporal=result.p_temporal)
                decision = decide_memory_retrieval(
                    result, turn_count=turns, total_tokens=total_tokens,
                    settings=settings, timescope_mode=timescope.mode)
                retrieve = decision.retrieve
                if args.source_gate:
                    memory = prepare_memory_context(
                        db=db, classifier=classifier, classification=result,
                        conversation_id=conversation_id, user_message=prompt,
                        scope={"conversation_id": str(conversation_id)},
                        turn_count=turns, total_tokens=total_tokens, total_budget=4000,
                        serving_window=args.window, base_retrieve=retrieve,
                        memory_slots=[], bookmarked_texts=[])
                    retrieve = memory.retrieve
                decisions.append(retrieve)
                print(f"  {name}: retrieve={int(retrieve)} "
                      f"base_retrieve={int(decision.retrieve)} "
                      f"p_ltm={result.p_ltm:.3f} "
                      f"p_need={decision.p_need_mem:.3f} "
                      f"referential={int(decision.breakdown['referential'])} "
                      f"timescope={timescope.mode} :: {prompt}")
            correct = sum(value == want for value in decisions)
            all_correct += correct
            flip_groups += int(len(set(decisions)) > 1)
            print(f"{name}: correct={correct}/{len(prompts)} "
                  f"flip={int(len(set(decisions)) > 1)}")
        count = sum(len(prompts) for _, _, prompts in GROUPS)
        print(f"TOTAL correct={all_correct}/{count} flip_groups={flip_groups}/{len(GROUPS)}")


if __name__ == "__main__":
    main()
