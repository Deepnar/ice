#!/usr/bin/env python3
"""Prepare private source-turn reviews for existing section-checkpoint questions.

Lexical suggestions are navigation aids for a human reviewer, never gold. The
reviewer must confirm answer support, original source turn(s), intervening
changes, and whether the answer was already available in recent history.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from scripts.z1.derive_retrieval_gt import build_idf, shortlist
from scripts.z1.run_meta import file_digest
from scripts.z1.seed_v3 import (CORPUS, DERIVED, TYPED, UNIFIED,
                                canonical_probe_id, load_plan, private_output)
from src.api.config import settings


def build_packet() -> dict:
    conversations, scheduled, quarantined, _ = load_plan()
    mapped = {p["probe_id"] for group in scheduled.values() for p in group
              if p["cutoff_kind"] == "native_checkpoint"}
    unified = json.loads(UNIFIED.read_text())["probes"]
    records = []
    sections = defaultdict(lambda: Counter())
    for probe in unified:
        if probe["source"] == "typed":
            continue
        slug, split = probe["conversation"], probe.get("split_turn")
        if not isinstance(split, int) or split > len(conversations[slug]):
            continue
        sections[(slug, split)]["native_questions"] += 1
        identity = canonical_probe_id(probe)
        if identity in mapped:
            sections[(slug, split)]["source_mapped_longterm"] += 1
            continue
        turns = conversations[slug][:split]
        old_rows = [{"turn_number": row["turn_number"],
                     "user_input": row["prompt"],
                     "ai_response": row["response"]} for row in turns]
        idf = build_idf(old_rows)
        question_rank = [t["turn_number"] for _, t in shortlist(
            probe["question"], old_rows, idf, 5)]
        answer_rank = [t["turn_number"] for _, t in shortlist(
            probe.get("expected_answer") or "", old_rows, idf, 5)]
        suggestions = sorted(set(question_rank + answer_rank))
        records.append({
            "probe_id": identity, "catalog_probe_id": probe["probe_id"],
            "source": probe["source"],
            "conversation": slug, "cutoff_turn": split,
            "question": probe["question"],
            "expected_answer": probe.get("expected_answer"),
            "candidate_sources": [{
                "turn": n,
                "age_at_cutoff": split - n,
                "outside_recent_history": split - n >= settings.recent_window_max_turns,
                "question_rank": question_rank.index(n) + 1 if n in question_rank else None,
                "answer_rank": answer_rank.index(n) + 1 if n in answer_rank else None,
                "timestamp": turns[n - 1]["timestamp"].isoformat(),
                "user": turns[n - 1]["prompt"],
                "assistant": turns[n - 1]["response"],
            } for n in suggestions],
            "reviewed_gold_turns": [],
            "answer_verdict": None,
            "reason": "",
        })
    return {
        "version": "v3", "kind": "native_checkpoint_source_review",
        "inputs": [file_digest(CORPUS), file_digest(UNIFIED),
                   file_digest(TYPED), file_digest(DERIVED)],
        "recent_window_turns": int(settings.recent_window_max_turns),
        "source_mapped_existing": len(mapped),
        "unlabeled_valid_checkpoint_questions": len(records),
        "out_of_history_checkpoint_questions": len(quarantined),
        # Candidate suggestions can miss the true source. Keep the complete
        # selected history once per conversation so a reviewer can inspect
        # every earlier turn, including one absent from both lexical top fives.
        "source_histories": {slug: [{"turn": row["turn_number"],
                                     "timestamp": row["timestamp"].isoformat(),
                                     "user": row["prompt"],
                                     "assistant": row["response"]}
                                    for row in turns]
                             for slug, turns in conversations.items()},
        "sections": [{"conversation": slug, "cutoff_turn": split, **counts}
                     for (slug, split), counts in sorted(sections.items())],
        "review_rule": (
            "Candidate sources are lexical suggestions, not gold. Inspect the "
            "original history through the section cutoff, record every necessary "
            "source turn, check whether the expected answer is supported and "
            "superseded, and mark a probe long-term only if recent turns alone "
            "cannot answer it. Never infer support from term overlap."),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = private_output(args.out)
    if path.exists():
        raise FileExistsError("review packet already exists")
    packet = build_packet()
    path.write_text(json.dumps(packet, indent=2, default=str) + "\n")
    counts = Counter(row["conversation"] for row in packet["records"])
    print(f"v3 source review: {len(packet['records'])} unlabeled questions, "
          f"{len(packet['sections'])} section checkpoints, "
          f"{packet['source_mapped_existing']} already mapped; "
          f"by conversation {dict(counts)}; {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
