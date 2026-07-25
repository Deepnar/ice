#!/usr/bin/env python3
"""Stage 2 — stitch the ICE development chats into ONE chronological conversation.

The whole of ICE was designed across a series of separate DeepSeek chats (ICE-1,
ICE-2, … — a new chat each time the previous one hit its context limit). They are
one continuous project conversation that a UI boundary happened to cut into
pieces: a decision made in ICE-2 is referenced in ICE-5, which is precisely the
"memory beyond the window" case ICE exists to serve.

Two consumers, one artifact (build it once, properly):

* **B1** — the richest available source of ``Codebase_Query`` / ``Code_Change``
  intent rows and of long-range ``Needs_Memory`` / ``Temporal_Recall`` rows, with
  real prior-turn context attached.
* **FINAL** — a genuine long-project memory test: months of continuous work with
  real callbacks, rather than a synthetic transcript.

Output is written in the JSONL dialogue shape that ``src/ingestion/formats.py``
already parses (``role``/``content``/``timestamp``/``conversation``/``title``), so
FINAL can feed it straight through the F10 import engine instead of re-parsing.

Usage:
    uv run python scripts/classifier/pipeline/stitch_icedev.py --list
    uv run python scripts/classifier/pipeline/stitch_icedev.py
    uv run python scripts/classifier/pipeline/stitch_icedev.py --include "project" "System Audit"
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone

from common import (ICEDEV_STITCHED, SIMULATION_DIR, ensure_data_dir,
                    is_usable_prompt, stable_id, write_jsonl)

from src.classifier import templates

# The explicit series. Matching on the naming convention the user actually used
# beats keyword-sniffing "does this chat mention ICE" — half the export mentions
# ICE in passing, and a wrong stitch silently corrupts a shared asset.
DEFAULT_PATTERN = r"^ICE-\d+$"
DEFAULT_EXPORT = os.path.join(SIMULATION_DIR, "deepseek.json")
STITCHED_ID = "icedev_stitched"


def _sort_key(conv):
    return conv.created_at or conv.updated_at or datetime.max.replace(tzinfo=timezone.utc)


def load_candidates(export_path: str):
    from src.ingestion.formats import normalize_file
    _fmt, convs = normalize_file(export_path)
    return sorted(convs, key=_sort_key)


def select(convs, pattern: str, include: list) -> list:
    rx = re.compile(pattern)
    extra = {name.lower() for name in (include or [])}
    picked = [c for c in convs
              if rx.match(c.title.strip())
              or any(name in c.title.lower() for name in extra)]
    return sorted(picked, key=_sort_key)


def stitch(picked) -> list:
    """Flatten the selected conversations into one ordered turn list.

    Chats are ordered by creation time and concatenated whole — turns are NOT
    interleaved by timestamp across chats. Two chats can overlap in wall-clock
    time (an aside opened while the main one was live); interleaving those would
    scramble two coherent threads into one incoherent one. Within a chat, the
    export's own order is authoritative.
    """
    out = []
    for conv in picked:
        for idx, turn in enumerate(conv.turns):
            out.append({
                "role": turn.role,
                "content": turn.text,
                "timestamp": turn.ts.isoformat() if turn.ts else None,
                "conversation": STITCHED_ID,
                "title": "ICE development (stitched)",
                # Provenance so a stitched turn can always be traced home.
                "segment": conv.title,
                "segment_index": idx,
            })
    return out


def to_pipeline_rows(turns: list) -> list:
    """User turns + their prior-turn context, in the shape stage 3+ consume."""
    rows = []
    texts = [t["content"] for t in turns]
    for idx, turn in enumerate(turns):
        if turn["role"] != "user" or not is_usable_prompt(turn["content"]):
            continue
        prior = texts[max(0, idx - templates.CONTEXT_TURNS):idx]
        rows.append({
            "id": stable_id("icedev", turn["content"], str(idx)),
            "source": "icedev",
            "provider": "deepseek",
            "text": turn["content"].strip(),
            "context_text": templates.truncate_context(prior) or None,
            "conversation_id": STITCHED_ID,
            "turn_index": idx,
            "ts": turn["timestamp"],
            "meta": {"segment": turn["segment"]},
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="B1 stage 2: stitch the ICE dev chats")
    ap.add_argument("--export", default=DEFAULT_EXPORT)
    ap.add_argument("--pattern", default=DEFAULT_PATTERN)
    ap.add_argument("--include", nargs="*", default=[],
                    help="extra chat titles (substring match) to fold in")
    ap.add_argument("--out", default=ICEDEV_STITCHED)
    ap.add_argument("--dialogue-out", default=None,
                    help="re-ingestible dialogue JSONL (default: alongside --out)")
    ap.add_argument("--list", action="store_true",
                    help="show every conversation in the export and exit")
    args = ap.parse_args()

    if not os.path.exists(args.export):
        raise SystemExit(f"export not found: {args.export}")

    convs = load_candidates(args.export)
    if args.list:
        for c in convs:
            print(f"{str(_sort_key(c))[:10]:12} {len(c.turns):5} turns  {c.title}")
        return

    picked = select(convs, args.pattern, args.include)
    if not picked:
        raise SystemExit(f"no conversations matched {args.pattern!r} — run --list")

    print(f"Stitching {len(picked)} conversations:")
    for c in picked:
        print(f"  {str(_sort_key(c))[:10]}  {len(c.turns):5} turns  {c.title}")

    turns = stitch(picked)
    rows = to_pipeline_rows(turns)

    ensure_data_dir()
    dialogue_out = args.dialogue_out or args.out.replace(".jsonl", "_dialogue.jsonl")
    write_jsonl(dialogue_out, turns)
    write_jsonl(args.out, rows)

    span = [t["timestamp"] for t in turns if t["timestamp"]]
    print(f"\nStitched conversation: {len(turns)} turns, {len(rows)} user rows")
    if span:
        print(f"  spans {min(span)[:10]} → {max(span)[:10]}")
    print(f"  dialogue (re-ingestible by F10): {dialogue_out}")
    print(f"  pipeline rows: {args.out}")
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as fh:
        json.dump({"segments": [c.title for c in picked], "turns": len(turns),
                   "user_rows": len(rows),
                   "span": [min(span), max(span)] if span else None}, fh, indent=2)


if __name__ == "__main__":
    main()
