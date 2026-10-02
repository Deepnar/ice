#!/usr/bin/env python3
"""Prepare private source-complete reviews for v3 long-term memory probes.

Existing source-linked questions are placed at real later section checkpoints.
An intervening turn may change the answer, and some source-first questions cite
assistant text. This packet exposes both issues before cloud scoring.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from scripts.z1.run_meta import file_digest
from scripts.z1.label_review import KNOWLEDGE_SCOPES, TASK_TYPES, blank_review
from scripts.z1.seed_v3 import (CORPUS, DERIVED, GENERATED, TYPED, UNIFIED, load_plan,
                                private_output)


def select(probes: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(lambda: defaultdict(list))
    for probe in probes:
        groups[probe["conversation"]][probe["split_turn"]].append(probe)
    rng = random.Random(seed)
    for sections in groups.values():
        for group in sections.values():
            group.sort(key=lambda p: p["probe_id"])
            rng.shuffle(group)
    if count == 0 or count >= len(probes):
        return sorted(probes, key=lambda p: p["probe_id"])
    chosen = []
    slugs = sorted(groups)
    section_order = {slug: sorted(groups[slug]) for slug in slugs}
    while len(chosen) < count:
        for slug in slugs:
            if len(chosen) >= count:
                break
            sections = section_order[slug]
            if not sections:
                continue
            split = sections.pop(0)
            chosen.append(groups[slug][split].pop())
            if groups[slug][split]:
                sections.append(split)
    return chosen


def build_packet(count: int, seed: int) -> dict:
    conversations, scheduled, _, ineligible = load_plan()
    probes = [probe for group in scheduled.values() for probe in group]
    selected = select(probes, count, seed)
    records = []
    for probe in selected:
        slug = probe["conversation"]
        source_split = probe["source_split_turn"]
        cutoff = probe["split_turn"]
        rows = conversations[slug]

        def source(turn_number: int) -> dict:
            turn = rows[turn_number - 1]
            return {"turn": turn_number,
                    "timestamp": turn["timestamp"].isoformat(),
                    "user": turn["prompt"], "assistant": turn["response"]}

        records.append({
            "probe_id": probe["probe_id"], "type": probe["probe_type"],
            "catalog_probe_id": probe.get("catalog_probe_id"),
            "conversation": slug, "source_split_turn": source_split,
            "cutoff_turn": cutoff, "question": probe["question"],
            "expected_answer": probe.get("expected_answer"),
            "reviewed_expected_answer": None,
            "gold_turns": probe["gold_turns"],
            "source_excerpt": probe.get("source_excerpt"),
            "source_role": probe.get("source_role"),
            "label_status": probe["label_status"],
            "anchor_entity": probe.get("anchor_entity"),
            "superseded_by": probe.get("superseded_by"),
            "gold_sources": [source(turn) for turn in probe["gold_turns"]],
            "intervening_turns": [source(turn) for turn in
                                  range(source_split + 1, cutoff + 1)],
            "verdict": None, "reason": "",
            **blank_review(),
        })
    return {
        "version": "v3", "kind": "longterm_probe_label_review",
        "inputs": [file_digest(CORPUS), file_digest(UNIFIED),
                   file_digest(TYPED), file_digest(DERIVED),
                   file_digest(GENERATED)],
        "sample_seed": seed, "longterm_candidates": len(probes),
        "no_delayed_window": len(ineligible),
        "review_rule": (
            "For each probe, set verdict to valid, invalid, or uncertain and give "
            "a concrete reason. Compare the expected answer with the original "
            "user statement, assistant reply, and all intervening turns. "
            "Confirm that the cited source supports the short answer and that "
            "the question is answerable at its checkpoint. A later user correction "
            "can supersede a current-value answer; a question explicitly asking "
            "what was said earlier may remain valid. Do not treat an assistant "
            "assertion alone as an independent user correction. Record review "
            "through the cutoff and whether recent history alone answers it. "
            "Declare knowledge_scope and task_types from the allowed values. "
            "Preserve expected_answer as catalog provenance; use "
            "reviewed_expected_answer for a source-backed correction, distinguishing "
            "required facts from optional context. Record reviewer and review_scope "
            "without claiming independent human qualification. Meaning-equivalent "
            "repeats may share question_family_id; review each cutoff separately."),
        "allowed_knowledge_scopes": sorted(KNOWLEDGE_SCOPES),
        "allowed_task_types": sorted(TASK_TYPES),
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30,
                        help="stratified candidates; 0 emits every long-term candidate")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n < 0:
        raise ValueError("--n cannot be negative")
    out = private_output(args.out)
    if out.exists():
        raise FileExistsError("review packet already exists")
    packet = build_packet(args.n, args.seed)
    out.write_text(json.dumps(packet, indent=2, default=str) + "\n")
    print(f"v3 long-term label packet: {len(packet['records'])} of "
          f"{packet['longterm_candidates']} candidates; {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
