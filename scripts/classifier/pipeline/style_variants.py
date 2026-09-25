#!/usr/bin/env python3
"""G28: paired meaning-preserving style probes for the live B2 gate.

Run only in a disposable database. Sources are synthetic and remain outside
the classifier's last-three-turn view. This is a small diagnostic instrument,
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
from src.api.memory_decision import decide_memory_retrieval
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
    texts = (
        "User: For this setup I checked GPU memory with gpu-scan --mem --device 7.",
        "User: The local indexer will listen on port 7619.",
        "User: Let's now discuss the shape of a short release note.",
        "Assistant: A release note can start with the user-facing change.",
        "User: Good, and keep the tone direct and plain.",
    )
    for index, raw in enumerate(texts):
        db.add(EpisodicMemory(
            conversation_id=conversation.id, batch_id=uuid.uuid4(),
            raw_text=raw, source_spans=single_provenance(
                raw, "assistant" if raw.startswith("Assistant:") else "user"),
            timestamp=start + timedelta(minutes=index),
            idempotency_key=f"g28-style-{uuid.uuid4()}",
            context_reliance="Long_Term_Memory",
        ))
    db.commit()
    return conversation.id, len(texts), sum(len(raw) / 4 for raw in texts)


def main():
    parser = argparse.ArgumentParser(description="G28 production-path style probes")
    parser.add_argument("--checkpoint", default=settings.classifier_model_path)
    args = parser.parse_args()
    if not os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_"):
        raise SystemExit("G28 style probes require a disposable database")
    if make_url(settings.database_url).database != os.environ["ICE_TEST_DATABASE"]:
        raise SystemExit("DATABASE_URL is not the disposable database")
    classifier = PyTorchClassifier(model_path=args.checkpoint)
    with SessionLocal() as db:
        conversation_id, turns, total_tokens = _source_context(db)
        all_correct = flip_groups = 0
        print(f"v3 checkpoint={args.checkpoint} "
              f"turns={turns} estimated_history_tokens={total_tokens:.0f}")
        for name, want, prompts in GROUPS:
            decisions = []
            for prompt in prompts:
                result = classifier.classify(prompt, conversation_id=str(conversation_id))
                timescope = detect_timescope(prompt, p_ltm=result.p_ltm,
                                             p_temporal=result.p_temporal)
                decision = decide_memory_retrieval(
                    result, turn_count=turns, total_tokens=total_tokens,
                    settings=settings, timescope_mode=timescope.mode)
                decisions.append(decision.retrieve)
                print(f"  {name}: retrieve={int(decision.retrieve)} "
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
