#!/usr/bin/env python3
"""Stage 5 — turn settled labels into train/val/test splits.

Rewrites ``legacy/training/build_training_data.py``. Four jobs, each of which is
a decision the v1 pipeline never made:

1. **Join labels to text** and drop anything unsettled. Human-reviewed rows
   (``review_queue.jsonl`` with a ``decision``) override everything.

2. **Hit the ≥40% context-prefixed target (D3).** The v1 corpus was standalone
   prompts, so the v1 model had never once seen a prompt with its history
   attached. Standalone rows are down-sampled — never invented — until the share
   is met, and the shortfall is reported loudly if the corpus can't reach it.

3. **Synthesise the hard-negative pairs (D4.4)** — the payoff of the whole
   context-aware exercise. See ``hard_negative_pairs`` for the construction and
   why it needs no extra labeling call.

4. **Enforce the per-label floors (§4).** Report every label's positive count;
   a label under 150 positives is a coin flip, and the schema rule is to DROP it
   rather than ship a head that guesses. That drop is surfaced as an explicit
   instruction, never applied silently.

Split is stratified and **grouped by conversation**: two turns of the same
conversation must not straddle train and test, or the model is scored on
paraphrases of what it memorised.

Usage:
    uv run python scripts/classifier/pipeline/build.py
    uv run python scripts/classifier/pipeline/build.py --dry-run --limit 200
"""

import argparse
import json
import random
from collections import Counter, defaultdict

from common import (CORPUS_RAW, CORPUS_SYNTH, ICEDEV_STITCHED, LABELS_FINAL,
                    REVIEW_QUEUE, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT,
                    ensure_data_dir, read_jsonl, write_jsonl)

from src.api.memory_decision import REFERENTIAL_WORDS
from src.classifier import templates
from src.classifier.schema import (CONTEXT_RELIANCE, HIGH_COMPLEXITY, INTENT,
                                   NEEDS_MEMORY, TOPIC, load_schema)

SEED = 42
CONTEXT_TARGET = 0.40      # D3
PAIR_TARGET = 1000         # D4.4
LABEL_FLOOR = 300          # D4.3 — the goal
LABEL_DROP_FLOOR = 150     # §4 — below this, drop the label instead of training it
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def load_rows(paths) -> dict:
    rows = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("id"):
                rows[row["id"]] = row
    return rows


def load_labels() -> dict:
    """Settled labels, with human decisions taking precedence."""
    labels = {e["id"]: e for e in read_jsonl(LABELS_FINAL)}
    human = 0
    for entry in read_jsonl(REVIEW_QUEUE):
        decision = entry.get("decision")
        if not decision:
            continue
        merged = dict(entry.get("resolved_heads") or {})
        merged.update(decision)
        labels[entry["id"]] = {"id": entry["id"], "source": entry.get("source"),
                               "labels": merged, "agreement": "human"}
        human += 1
    if human:
        print(f"[build] {human} human-reviewed rows override the model labels")
    return labels


def _is_referential(text: str) -> bool:
    """Does this prompt lean on something outside itself?

    Reuses the live retrieval-time word list rather than a second private copy —
    if that list moves, the training pairs move with it.
    """
    words = {w.strip(".,!?;:'\"").lower() for w in text.split()}
    return bool(words & REFERENTIAL_WORDS)


def hard_negative_pairs(rows, labels, limit=PAIR_TARGET):
    """The same text, labeled once WITH context and once WITHOUT, where the
    reliance genuinely differs.

    Construction, and why it needs no second labeling pass: take a row the
    labelers saw WITH its conversation context and judged *not* to need memory —
    i.e. the context already supplied the referent — whose text is referentially
    ambiguous on its own ("so which of those should I pick?"). Strip the context
    and the referent is gone, so the twin necessarily DOES need memory. The label
    flip is entailed by the construction, not guessed.

    This is the pair that teaches context-awareness: identical text, opposite
    answer, and the only difference is whether the history was attached.
    """
    pairs = []
    for row_id, label in labels.items():
        if len(pairs) >= limit:
            break
        row = rows.get(row_id)
        if not row or not row.get("context_text"):
            continue
        ctx_labels = set(label["labels"].get(CONTEXT_RELIANCE, []))
        if NEEDS_MEMORY in ctx_labels:
            continue                     # already needs memory with context — no flip
        if not _is_referential(row.get("text", "")):
            continue                     # stripping context changes nothing
        twin = dict(label["labels"])
        twin[CONTEXT_RELIANCE] = sorted(ctx_labels | {NEEDS_MEMORY})
        pairs.append({
            "id": f"{row_id}_nocontext",
            "text": row["text"],
            "context_text": None,
            **twin,
            "source": row.get("source"),
            "conversation_id": row.get("conversation_id"),
            "labeler_agreement": "hard_negative_pair",
            "pair_of": row_id,
        })
    return pairs


def compose(rows, labels, context_target=CONTEXT_TARGET):
    """Join, then down-sample standalone rows until the context share is met."""
    joined = []
    for row_id, label in labels.items():
        row = rows.get(row_id)
        if not row:
            continue
        joined.append({
            "id": row_id,
            "text": row["text"],
            "context_text": row.get("context_text"),
            TOPIC: label["labels"].get(TOPIC, []),
            INTENT: label["labels"].get(INTENT, []),
            CONTEXT_RELIANCE: label["labels"].get(CONTEXT_RELIANCE, []),
            "source": row.get("source"),
            "conversation_id": row.get("conversation_id"),
            "labeler_agreement": label.get("agreement", "unknown"),
        })

    with_ctx = [r for r in joined if r["context_text"]]
    without = [r for r in joined if not r["context_text"]]
    share = len(with_ctx) / max(1, len(joined))
    print(f"[build] context-prefixed before balancing: {len(with_ctx)}/{len(joined)} "
          f"({share:.1%})")

    if share < context_target and with_ctx:
        # Keep every context row; trim standalone rows to hit the ratio. Never
        # the other way round — context rows are the scarce, valuable ones.
        keep = int(len(with_ctx) * (1 - context_target) / context_target)
        if keep < len(without):
            rng = random.Random(SEED)
            rng.shuffle(without)
            dropped = len(without) - keep
            without = without[:keep]
            print(f"[build] dropped {dropped} standalone rows to reach "
                  f"{context_target:.0%} context-prefixed")
    return with_ctx + without


def split(rows, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION):
    """Conversation-grouped, source-stratified split.

    Grouping is the part that matters: turns 4 and 5 of one conversation share a
    context prefix and often near-identical phrasing. Splitting them across
    train and test inflates every metric.
    """
    rng = random.Random(SEED)
    by_group = defaultdict(list)
    for row in rows:
        key = row.get("conversation_id") or row["id"]
        by_group[key].append(row)

    groups = list(by_group.values())
    rng.shuffle(groups)
    # Stratify by the group's dominant source so each split keeps the corpus mix.
    by_source = defaultdict(list)
    for group in groups:
        by_source[Counter(r.get("source") for r in group).most_common(1)[0][0]].append(group)

    train, val, test = [], [], []
    for source_groups in by_source.values():
        n = len(source_groups)
        n_val = int(n * val_fraction)
        n_test = int(n * test_fraction)
        for group in source_groups[:n_val]:
            val += group
        for group in source_groups[n_val:n_val + n_test]:
            test += group
        for group in source_groups[n_val + n_test:]:
            train += group
    for part in (train, val, test):
        rng.shuffle(part)
    return train, val, test


def label_report(rows, schema):
    """Per-label positives, and the drop/floor verdict for each."""
    print("\n[build] per-label positives (floor 300; <150 ⇒ DROP the label, §4):")
    verdicts = {}
    for head_name in (TOPIC, INTENT, CONTEXT_RELIANCE):
        print(f"  {head_name}:")
        for label in schema.labels(head_name):
            n = sum(1 for r in rows if label in r.get(head_name, []))
            if n < LABEL_DROP_FLOOR:
                verdict = "DROP"
            elif n < LABEL_FLOOR:
                verdict = "thin"
            else:
                verdict = "ok"
            verdicts[label] = (n, verdict)
            marker = {"ok": "   ", "thin": " ~ ", "DROP": " ! "}[verdict]
            print(f"   {marker}{label:26} {n:6}")

    drops = [lbl for lbl, (_n, v) in verdicts.items() if v == "DROP"]
    if drops:
        print("\n[build] ⚠ LABELS BELOW THE 150-POSITIVE FLOOR: " + ", ".join(drops))
        print("[build]   §4 rule: drop these from label_schema.json rather than "
              "train a coin-flip head. This is a DECISION, not an automatic edit —")
        print("[build]   remove them from the schema and re-run, or accept the risk "
              "explicitly.")
        if HIGH_COMPLEXITY in drops:
            print("[build]   (High_Complexity was flagged in the spec as the most "
                  "likely casualty — its only consumer, the F11/B3 cloud toggle, "
                  "does not exist yet.)")
    return verdicts


def main():
    ap = argparse.ArgumentParser(description="B1 stage 5: build train/val/test splits")
    ap.add_argument("--corpus", nargs="*",
                    default=[CORPUS_RAW, ICEDEV_STITCHED, CORPUS_SYNTH])
    ap.add_argument("--dry-run", action="store_true",
                    help="report composition without writing splits")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context-target", type=float, default=CONTEXT_TARGET)
    args = ap.parse_args()

    schema = load_schema()
    rows = load_rows(args.corpus)
    labels = load_labels()
    print(f"[build] corpus rows {len(rows)}, settled labels {len(labels)}")
    if args.limit:
        labels = dict(list(labels.items())[:args.limit])

    composed = compose(rows, labels, context_target=args.context_target)
    pairs = hard_negative_pairs(rows, labels)
    print(f"[build] hard-negative context pairs: {len(pairs)} (target {PAIR_TARGET})")
    if len(pairs) < PAIR_TARGET:
        print(f"[build] ⚠ short of the {PAIR_TARGET}-pair target — more multi-turn "
              f"context rows are the only fix (label more personal/icedev rows)")
    dataset = composed + pairs

    # Every row must render through the shared templates — if this ever fails,
    # training and inference have diverged again (D3).
    bad = 0
    for row in dataset:
        try:
            templates.render(row["text"], row.get("context_text"),
                             version=schema.template_version)
        except Exception:
            bad += 1
    print(f"[build] template render check: {len(dataset) - bad}/{len(dataset)} ok")

    with_ctx = sum(1 for r in dataset if r.get("context_text"))
    print(f"[build] final context-prefixed: {with_ctx}/{len(dataset)} "
          f"({with_ctx / max(1, len(dataset)):.1%}, target {args.context_target:.0%})")
    verdicts = label_report(dataset, schema)

    if args.dry_run:
        print("\n[build] dry run — no splits written")
        return

    train, val, test = split(dataset)
    ensure_data_dir()
    write_jsonl(TRAIN_SPLIT, train)
    write_jsonl(VAL_SPLIT, val)
    write_jsonl(TEST_SPLIT, test)
    print(f"\n[build] train {len(train)} / val {len(val)} / test {len(test)}")

    summary = {
        "rows": len(dataset), "train": len(train), "val": len(val), "test": len(test),
        "context_prefixed": with_ctx, "hard_negative_pairs": len(pairs),
        "schema_version": schema.version,
        "label_counts": {k: v[0] for k, v in verdicts.items()},
        "below_drop_floor": [k for k, v in verdicts.items() if v[1] == "DROP"],
        "sources": dict(Counter(r.get("source") for r in dataset)),
        "agreement": dict(Counter(r.get("labeler_agreement") for r in dataset)),
    }
    with open(TRAIN_SPLIT.replace("train.jsonl", "build_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[build] summary → {TRAIN_SPLIT.replace('train.jsonl', 'build_summary.json')}")


if __name__ == "__main__":
    main()
