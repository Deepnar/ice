#!/usr/bin/env python3
"""Pile B — hand-authored rows whose labels ship WITH the prompt.

Two piles of non-organic data exist, and conflating them corrupts both:

* **Pile A** (`synth.py` → `corpus_synth.jsonl`) — bulk prompts a local model
  generates to pad a thin label. Generation drifts (ask for `Codebase_Query`,
  receive a `Code_Change`), so the label must be earned through the normal
  two-labeler path.
* **Pile B** (this module → `corpus_authored.jsonl`) — prompts written
  deliberately, one at a time, FOR a specific label combination. **The label is
  authored, not inferred, and these rows never go to the labelers.** Sending them
  to Qwen/Gemma would let two local models overrule a human-authored ground
  truth — and for the labels Pile B exists to fix, the labelers are precisely the
  thing that gets them wrong.

**Why Pile B is necessary at all — capability censoring.** A label gated on a
capability the data-collection environment lacked will look rare no matter how
much data you gather, and scaling the corpus cannot fix it. Measured on the
33,723 settled rows:

* `Codebase_Query` — 65 rows. The corpus is website chats where the assistant had
  no repo access, so "where is the retry logic in my project?" was a pointless
  question and was never asked. Under MCP it stops being pointless.
* **`Needs_Memory` across conversations — ZERO rows.** All 6,806 positives are
  within-conversation anaphora. Referring to a *different* conversation is futile
  when the assistant cannot see it, so nobody phrases it that way. This is the
  single case ICE exists to serve, and the corpus contains none of it.
* `Needs_Memory` + `Needs_Live_Info` together — ~137 rows. The flagship
  justification for the whole schema change ("what's the current price of the GPU
  I told you about"), thin because no web search existed to make it worth asking.
* `Meta_AI` about ICE's own memory — none. C10/C11 built `/forget`-style commands;
  the corpus predates having a system worth interrogating.

Prompts read as **typed by a human**: lowercase starts, typos, missing
punctuation, wildly varying length. Not polished prose — the corpus they join is
real chat traffic, and a block of tidy sentences would be trivially separable.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter

from common import DATA_DIR, ensure_data_dir, read_jsonl, stable_id, write_jsonl

from src.classifier.schema import CONTEXT_RELIANCE, INTENT, TOPIC, load_schema

CORPUS_AUTHORED = os.path.join(DATA_DIR, "corpus_authored.jsonl")


def make_row(text: str, topic, intent, ctx, note: str = "", context_text=None) -> dict:
    """One authored row, with every field populated rather than left null.

    Two fields deserve explanation because "empty" is a real value for them, not
    missing data:

    * ``labels.context_reliance = []`` is the **derived Zero_Shot state** — the
      prompt needs no memory, no live info, no time dimension, and no strong
      model. v2 deleted Zero_Shot as a label precisely so it would be expressed
      this way. ``meta.derived_zero_shot`` records that the emptiness was
      intended, so nobody later reads it as an unlabelled row.
    * ``context_text = None`` means the prompt stands alone. For the
      cross-conversation memory rows this is **load-bearing**: the referent must
      NOT be visible, or the row stops being an example of needing memory.
      ``meta.context = "none"|"attached"`` states which case it is.

    Everything else is filled: authored rows are their own single-turn
    conversation, so ``conversation_id`` is the row id (which also keeps
    ``build.py``'s conversation-grouped split from lumping them together) and
    ``turn_index`` is 0 for a standalone prompt, 1 when context is attached.
    ``ts`` stays null on purpose — an authored row has no real point in time, and
    inventing one would let synthetic timestamps leak into anything that later
    reasons about when something was said.
    """
    row_id = stable_id("authored", text)
    ctx = list(ctx)
    return {
        "id": row_id,
        "source": "authored",
        "provider": "authored",
        "text": text,
        "context_text": context_text,
        "conversation_id": row_id,
        "turn_index": 1 if context_text else 0,
        "ts": None,
        # Labels live IN the row — this is what separates Pile B from Pile A.
        "labels": {TOPIC: list(topic), INTENT: list(intent),
                   CONTEXT_RELIANCE: ctx},
        "authored_by": "assistant",
        "meta": {
            "pile": "B",
            "note": note,
            "derived_zero_shot": not ctx,
            "context": "attached" if context_text else "none",
            "signals": len(ctx),
        },
    }


def validate(rows, schema=None) -> list:
    """Every label must exist in the schema, and no row may be blank or duplicated.

    A typo in an authored label is invisible downstream — it silently becomes a
    label nothing trains on — so this fails loudly instead.
    """
    schema = schema or load_schema()
    problems, seen = [], set()
    for i, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        # Null_Noise is *defined* by being contentless ("ok", "...", keyboard
        # mash), so the length floor cannot apply to it — the floor exists to
        # catch a truncated row, not to reject the one label that looks like one.
        floor = 1 if "Null_Noise" in row["labels"][TOPIC] else 5
        if len(text) < floor:
            problems.append(f"row {i}: text too short")
        key = text.lower()
        if key in seen:
            problems.append(f"row {i}: duplicate text — {text[:60]}")
        seen.add(key)
        for head in (TOPIC, INTENT, CONTEXT_RELIANCE):
            for label in row["labels"][head]:
                if not schema.head(head).has(label):
                    problems.append(f"row {i}: {label!r} is not a {head} label")
        if not row["labels"][TOPIC] or not row["labels"][INTENT]:
            problems.append(f"row {i}: topic and intent must be non-empty")
    return problems


def report(path: str = CORPUS_AUTHORED) -> None:
    rows = list(read_jsonl(path))
    if not rows:
        print(f"[authored] {path} is empty")
        return
    schema = load_schema()
    problems = validate(rows, schema)
    print(f"[authored] {len(rows)} rows in {os.path.basename(path)}")
    if problems:
        print(f"[authored] ⚠ {len(problems)} problems:")
        for p in problems[:20]:
            print(f"    {p}")
    else:
        print("[authored] all labels valid, no duplicates")
    for head in (CONTEXT_RELIANCE, INTENT, TOPIC):
        c = Counter()
        for row in rows:
            for label in row["labels"][head]:
                c[label] += 1
            if head == CONTEXT_RELIANCE and not row["labels"][head]:
                c["(derived Zero_Shot)"] += 1
        print(f"  {head}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    notes = Counter(r["meta"].get("note", "?") for r in rows)
    print("  batches: " + ", ".join(f"{k} {v}" for k, v in notes.most_common()))


def add(new_rows, path: str = CORPUS_AUTHORED) -> int:
    """Append, skipping any text already present. Returns rows added."""
    ensure_data_dir()
    existing = list(read_jsonl(path))
    seen = {(r.get("text") or "").strip().lower() for r in existing}
    fresh = [r for r in new_rows if (r["text"].strip().lower() not in seen)]
    problems = validate(fresh)
    if problems:
        raise SystemExit("[authored] refusing to write:\n  " + "\n  ".join(problems[:20]))
    write_jsonl(path, existing + fresh)
    return len(fresh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="B1: inspect the hand-authored pile")
    ap.add_argument("--path", default=CORPUS_AUTHORED)
    ap.add_argument("--sample", type=int, default=0, help="print N random rows")
    args = ap.parse_args()
    report(args.path)
    if args.sample:
        import random
        rows = list(read_jsonl(args.path))
        random.shuffle(rows)
        print()
        for r in rows[:args.sample]:
            print(f"  {r['text']}")
            print(f"      {r['labels'][INTENT]} {r['labels'][CONTEXT_RELIANCE]}")
