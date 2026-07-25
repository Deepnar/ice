#!/usr/bin/env python3
"""Stage 1 — assemble the unlabeled v2 corpus.

Rewrites ``legacy/promt_extraction/*`` (which pulled 5k first-user-turns per
online dataset and wrote one file each). Three changes matter:

1. **Context rows.** v1 took only the FIRST user turn of each conversation, so
   the corpus was almost entirely standalone prompts — and the classifier that
   trained on it had never seen a prompt with its conversation history attached,
   which is exactly the case where "so which one should I use?" means something
   different. Every source here can emit turn *k* with its prior turns attached
   (D3 wants ≥40% context-prefixed rows overall).

2. **Diversity is this stage's job.** The online layer is where breadth comes
   from, so pulls are larger and *bucket-weighted*: a cheap keyword bucketer
   caps how much of the corpus any one theme can occupy. Without it the pull is
   dominated by coding help and roleplay, and thin topics stay thin.

3. **Personal exports come through F10's adapters** (``src/ingestion/formats.py``)
   rather than a bespoke parser — those already handle ChatGPT's mapping tree,
   Claude's branches, DeepSeek's subtree walk, and timestamp synthesis. B1 reads
   FILES; it never touches ``episodic_memory`` (the store was emptied at the C17
   cutover, and file-based extraction avoids polluting the live store anyway).

The v1 25k corpus is a fourth source (``--legacy``): its TEXT is reused, its
LABELS are dropped on the floor — the single-label 3-way context head IS the
defect B1 removes, so relabelling from scratch is the point (trap 1).

Usage:
    uv run python scripts/classifier/pipeline/extract.py --all
    uv run python scripts/classifier/pipeline/extract.py --online 15000 --personal
"""

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter

from common import (CORPUS_RAW, LEGACY_CORPUS, SIMULATION_DIR, coarse_bucket,
                    ensure_data_dir, is_usable_prompt, read_jsonl, stable_id,
                    write_jsonl)

from src.classifier import templates

SEED = 42
# No single coarse bucket may exceed this share of the online pull. Not a
# taxonomy claim — a diversity brake (see common.coarse_bucket).
BUCKET_CAP_FRACTION = 0.22
# Share of online rows that should carry conversation context.
ONLINE_CONTEXT_TARGET = 0.45


def _text_key(text: str) -> str:
    """Dedupe key: whitespace/case-normalized content hash, source-independent
    (the same prompt pulled fresh and reused from the 25k is ONE row)."""
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode()).hexdigest()[:20]


def _context_from(messages, upto: int) -> str:
    """Prior-context block for the message at *upto*.

    Delegates to the shared builder so the prefix is byte-comparable with what
    the live classifier constructs: user→assistant EXCHANGES (the unit the
    runtime stores), each capped at 150 words, last three, under a 500-word
    budget. *messages* is ``[(role, text), …]`` in order.
    """
    return templates.context_from_messages(messages[:upto])


# ── online datasets ─────────────────────────────────────────────────────────

def _iter_online(name: str, limit: int):
    """Yield (conversation_id, [turn texts in order]) for one HF dataset."""
    from datasets import load_dataset

    if name == "lmsys":
        ds = load_dataset("lmsys/chatbot_arena_conversations", split="train",
                          token=os.environ.get("HF_TOKEN"))
        for i, row in enumerate(ds):
            if i >= limit:
                break
            if row.get("language") != "English":
                continue
            conv = row.get("conversation_a") or []
            yield f"lmsys_{i}", [(t.get("role"), t.get("content")) for t in conv]

    elif name == "wildchat":
        ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= limit:
                break
            if row.get("language") != "English":
                continue
            conv = row.get("conversation") or []
            yield f"wildchat_{i}", [(t.get("role"), t.get("content")) for t in conv]

    elif name == "sharegpt":
        ds = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered",
                          data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                          split="train")
        for i, row in enumerate(ds):
            if i >= limit:
                break
            conv = row.get("conversations") or []
            roles = {"human": "user", "gpt": "assistant"}
            yield f"sharegpt_{i}", [(roles.get(t.get("from"), t.get("from")),
                                     t.get("value")) for t in conv]


def extract_online(target: int, seen: set) -> list:
    """Bucket-weighted pull across the three public datasets.

    Scans well past *target* and keeps rows only while their bucket has room, so
    the result is broad rather than "whatever the first N rows happened to be".
    """
    per_source = max(1, target // 3)
    rows, buckets = [], Counter()
    cap = max(1, int(target * BUCKET_CAP_FRACTION))
    want_context = int(target * ONLINE_CONTEXT_TARGET)
    have_context = 0

    for source in ("lmsys", "wildchat", "sharegpt"):
        kept = 0
        # Scan 6× the quota: the bucket cap rejects a lot, and rejects are cheap.
        try:
            stream = _iter_online(source, limit=per_source * 6)
        except Exception as exc:               # gated dataset, no token, offline
            print(f"  ! {source}: {str(exc)[:120]} — skipping")
            continue

        for conv_id, turns in stream:
            if kept >= per_source:
                break
            messages = [(role, text) for role, text in turns]
            for idx, (role, text) in enumerate(turns):
                if kept >= per_source:
                    break
                if role != "user" or not is_usable_prompt(text):
                    continue
                key = _text_key(text)
                if key in seen:
                    continue
                bucket = coarse_bucket(text)
                if buckets[bucket] >= cap and bucket != "general":
                    continue
                # Prefer context-bearing rows until the quota is met; a turn at
                # index 0 has no prior turns to attach.
                context = _context_from(messages, idx) if idx > 0 else ""
                if not context and have_context < want_context and idx == 0:
                    # Still take some standalone rows — real traffic has them.
                    if random.random() < 0.5:
                        continue
                seen.add(key)
                buckets[bucket] += 1
                kept += 1
                if context:
                    have_context += 1
                rows.append({
                    "id": stable_id(source, text, str(idx)),
                    "source": source,
                    "provider": None,
                    "text": text.strip(),
                    "context_text": context or None,
                    "conversation_id": conv_id,
                    "turn_index": idx,
                    "ts": None,
                    "meta": {"bucket": bucket},
                })
        print(f"  {source}: {kept} rows")

    print(f"  online buckets: {dict(buckets)}")
    print(f"  online context-prefixed: {have_context}/{len(rows)}")
    return rows


# ── personal exports (F10 adapters) ─────────────────────────────────────────

def _salvage_jsonl(path: str):
    """Re-read a JSONL export, dropping only the lines that don't parse."""
    from src.ingestion.formats import parse_jsonl

    good, skipped = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            good.append(line)
    try:
        return "jsonl", parse_jsonl(good, default_title=os.path.basename(path)), skipped
    except Exception:
        return "jsonl", [], skipped


def extract_personal(seen: set, per_file_cap: int = 4000) -> list:
    """Every user turn from the local exports, with its prior-turn context.

    This is the layer that makes context-aware classification real: these are
    genuine multi-turn conversations where a short follow-up genuinely depends on
    what came before. Consent for this is USER-REQUIRED and was given
    (2026-07-25); incognito/none-scoped material is excluded at source — export
    files carry no scope flags, so anything the user wants excluded must not be
    in ``data/simulation/`` in the first place.
    """
    from src.ingestion.formats import normalize_file

    rows = []
    if not os.path.isdir(SIMULATION_DIR):
        print(f"  ! {SIMULATION_DIR} missing — skipping personal layer")
        return rows

    for name in sorted(os.listdir(SIMULATION_DIR)):
        path = os.path.join(SIMULATION_DIR, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        try:
            fmt, convs = normalize_file(path)
        except ValueError as exc:
            # A truncated export (one unterminated line) should cost us that
            # line, not the file. normalize_file stays fail-loud for real
            # imports — corpus building is allowed to salvage.
            fmt, convs, skipped = _salvage_jsonl(path)
            if convs:
                print(f"  ~ {name}: skipped {skipped} malformed line(s) — salvaged")
            else:
                print(f"  ! {name}: {str(exc)[:100]} — skipping")
                continue
        except Exception as exc:
            print(f"  ! {name}: {str(exc)[:100]} — skipping")
            continue
        if not convs:
            print(f"  - {name}: format={fmt}, no conversations (raw needs the F14 slicer)")
            continue

        kept = 0
        for conv in convs:
            if kept >= per_file_cap:
                break
            messages = [(t.role, t.text) for t in conv.turns]
            for idx, turn in enumerate(conv.turns):
                if kept >= per_file_cap:
                    break
                if turn.role != "user" or not is_usable_prompt(turn.text):
                    continue
                key = _text_key(turn.text)
                if key in seen:
                    continue
                seen.add(key)
                kept += 1
                rows.append({
                    "id": stable_id("personal", turn.text, f"{conv.source_id}:{idx}"),
                    "source": "personal",
                    "provider": conv.provider,
                    "text": turn.text.strip(),
                    "context_text": _context_from(messages, idx) or None,
                    "conversation_id": conv.source_id,
                    "turn_index": idx,
                    # Real timestamps matter here: temporal weak supervision and
                    # FINAL's evolving-GT both want when a thing was said.
                    "ts": turn.ts.isoformat() if turn.ts else None,
                    "meta": {"title": conv.title, "format": fmt,
                             "ts_synthetic": conv.ts_synthetic},
                })
        print(f"  {name}: {kept} rows ({len(convs)} conversations, format={fmt})")
    return rows


# ── the v1 corpus: text reuse only ──────────────────────────────────────────

def extract_legacy(seen: set) -> list:
    """Reuse the 25,354-row v1 corpus TEXT. Labels are deliberately discarded."""
    rows = []
    if not os.path.exists(LEGACY_CORPUS):
        print(f"  ! {LEGACY_CORPUS} missing — skipping legacy layer")
        return rows
    counts = Counter()
    for row in read_jsonl(LEGACY_CORPUS):
        text = row.get("prompt") or row.get("text")
        if not is_usable_prompt(text):
            continue
        key = _text_key(text)
        if key in seen:
            continue
        seen.add(key)
        source = row.get("source", "legacy")
        counts[source] += 1
        rows.append({
            "id": stable_id(source, text, "legacy"),
            "source": source,
            "provider": None,
            "text": text.strip(),
            "context_text": None,      # v1 rows are standalone by construction
            "conversation_id": None,
            "turn_index": None,
            "ts": None,
            # Provenance, NOT a label — recorded so the audit can check that no
            # v1 label leaked into a v2 target.
            "meta": {"v1_reused_text": True, "bucket": coarse_bucket(text)},
        })
    print(f"  legacy corpus: {len(rows)} rows {dict(counts)}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="B1 stage 1: build the unlabeled v2 corpus")
    ap.add_argument("--out", default=CORPUS_RAW)
    ap.add_argument("--online", type=int, default=0,
                    help="target rows from LMSYS/WildChat/ShareGPT combined")
    ap.add_argument("--personal", action="store_true", help="include data/simulation/ exports")
    ap.add_argument("--legacy", action="store_true", help="reuse the v1 25k corpus text")
    ap.add_argument("--all", action="store_true", help="--online 15000 --personal --legacy")
    ap.add_argument("--per-file-cap", type=int, default=4000)
    args = ap.parse_args()

    if args.all:
        args.online = args.online or 15000
        args.personal = True
        args.legacy = True

    random.seed(SEED)
    ensure_data_dir()

    # Seed the dedupe set from any existing output so re-runs extend rather than
    # duplicate (this stage is re-run whenever a new export is added).
    seen, rows = set(), []
    for row in read_jsonl(args.out):
        seen.add(_text_key(row.get("text", "")))
        rows.append(row)
    if rows:
        print(f"Resuming from {len(rows)} existing rows in {args.out}")

    # Personal first: it is the smallest and most valuable layer, so it wins any
    # dedupe race against a public dataset that happens to contain the same text.
    if args.personal:
        print("Personal exports (F10 adapters):")
        rows += extract_personal(seen, per_file_cap=args.per_file_cap)
    if args.legacy:
        print("Legacy v1 corpus (text only):")
        rows += extract_legacy(seen)
    if args.online:
        print(f"Online datasets (target {args.online}):")
        rows += extract_online(args.online, seen)

    n = write_jsonl(args.out, rows)
    by_source = Counter(r["source"] for r in rows)
    with_ctx = sum(1 for r in rows if r.get("context_text"))
    print(f"\nWrote {n} rows → {args.out}")
    print(f"  by source: {dict(by_source)}")
    print(f"  context-prefixed: {with_ctx} ({with_ctx / max(n, 1):.1%})")
    summary = {"rows": n, "by_source": dict(by_source), "context_prefixed": with_ctx}
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
