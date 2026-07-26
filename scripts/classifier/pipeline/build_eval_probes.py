#!/usr/bin/env python3
"""Build the INDEPENDENT evaluation set from the user's own curation probes.

Why this exists. `train/val/test` are all carved out of the corpus that Gemma and
Qwen labeled, so they share those models' blind spots: where both labelers are
wrong in the same way, the held-out split is wrong the same way too, and the F1
table reports a high score for a model that learned the shared mistake. A split
cannot detect the bias of the process that produced it.

`data/labeled/probes_labeled_ltm.jsonl` is free of that problem. The prompts were
written **by the user**, months earlier, as evaluation probes for the Experiment-1
curation files — never seen by any labeler in this pipeline, and not authored by
the assistant either (unlike Pile B, which IS trained on and therefore cannot
serve as its own exam).

What the file actually contains, verified 2026-07-26: 708 rows but only **238
unique prompts** (the fine-tune run repeated them), and every row was force-set to
`Long_Term_Memory` — 64 of them carry model reasoning that concluded `Zero_Shot`,
which the user overrode. The override is right: these are *curation probes*, asked
at a checkpoint to test whether the system recalled the conversation, so needing
memory is true **by construction of what the probe is**.

Scope of the claim, deliberately narrow. This set asserts ONE label —
`Needs_Memory` — because that is the part its construction guarantees. The v1
topic/intent labels came from a weak 7B model and are carried only as a hint,
never scored. So the gate this set provides is exactly:

    "on 238 real memory-needing prompts the user actually typed,
     how often does the retrained head fire Needs_Memory?"

which is the single behaviour the whole memory system depends on.

Usage:
    uv run python scripts/classifier/pipeline/build_eval_probes.py
"""

import json
import os
import sys

from common import (CORPUS_RAW, DATA_DIR, ICEDEV_STITCHED, ROOT, read_jsonl,
                    stable_id, write_jsonl)

from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, NEEDS_MEMORY, TOPIC,
                                   load_schema)

SOURCE = os.path.join(ROOT, "data", "labeled", "probes_labeled_ltm.jsonl")
EVAL_PROBES = os.path.join(DATA_DIR, "eval_probes_independent.jsonl")


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit(f"missing {SOURCE}")
    schema = load_schema()

    # Every prompt already in the training corpus must be excluded — a probe the
    # model trained on proves nothing.
    corpus_text = set()
    for path in (CORPUS_RAW, ICEDEV_STITCHED,
                 os.path.join(DATA_DIR, "corpus_authored.jsonl")):
        for row in read_jsonl(path):
            corpus_text.add((row.get("text") or "").strip().lower())

    unique, overlap = {}, 0
    for row in read_jsonl(SOURCE):
        text = (row.get("prompt") or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in corpus_text:
            overlap += 1
            continue
        unique.setdefault(key, row)

    out = []
    for row in unique.values():
        text = row["prompt"].strip()
        v1 = row.get("label") or {}
        out.append({
            "id": stable_id("evalprobe", text),
            "source": "eval_probe_user_authored",
            "text": text,
            "context_text": None,
            # The ONLY asserted label. Guaranteed by what a curation probe is.
            "labels": {CONTEXT_RELIANCE: [NEEDS_MEMORY]},
            # Carried for inspection, never scored — a weak 7B model produced these.
            "hint": {TOPIC: v1.get("topic_labels", []),
                     INTENT: v1.get("intent_labels", [])},
            "origin": row.get("id"),
            "meta": {"held_out": True, "never_trained": True,
                     "asserted": "Needs_Memory by construction (curation probe)"},
        })

    write_jsonl(EVAL_PROBES, out)
    print(f"[eval-probes] {len(out)} independent probes → {EVAL_PROBES}")
    print(f"[eval-probes] dropped {overlap} rows already present in the training corpus")
    print(f"[eval-probes] asserts {NEEDS_MEMORY} only; topic/intent kept as an unscored hint")
    print(f"[eval-probes] schema v{schema.version}; these rows must NEVER enter train/val/test")
    summary = {"probes": len(out), "excluded_overlap": overlap,
               "asserted_label": NEEDS_MEMORY, "source": os.path.basename(SOURCE)}
    with open(EVAL_PROBES.replace(".jsonl", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    sys.exit(main())
