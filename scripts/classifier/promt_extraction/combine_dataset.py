#!/usr/bin/env python3
"""
Combine all prompt sources into a single deduplicated dataset.

Input files (all in ../data/ relative to this script in scripts/):
  - clean_promts.json       → source: personal   (JSON array format)
  - personal_prompts.jsonl  → source: personal   (JSONL, has confidence field)
  - lmsys_prompts.jsonl     → source: lmsys      (JSONL, has id/source/label)
  - sharegpt_prompts.jsonl  → source: sharegpt   (JSONL, has id/source/label)
  - wildchat_prompts.jsonl  → source: wildchat   (JSONL, has id/source/label)

Output: ../data/dataset_unlabeled.jsonl

Final format per line:
  {"id": "personal_001", "source": "personal", "prompt": "...", "label": null}
"""

import json
import hashlib
import random
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR.parent / "data"
OUTPUT     = DATA_DIR / "dataset_unlabeled.jsonl"

SOURCES = [
    # (filename in data/,          source label,   format)
    ("clean_promts.json",          "personal"),
    ("personal_prompts.jsonl",     "personal"),
    ("lmsys_prompts.jsonl",        "lmsys"),
    ("sharegpt_prompts.jsonl",     "sharegpt"),
    ("wildchat_prompts.jsonl",     "wildchat"),
]

RANDOM_SEED = 42

# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_prompt(text: str) -> str:
    """
    - Collapse all newlines and tabs to a single space
    - Collapse multiple spaces to one
    - Strip leading/trailing whitespace
    - Remove null bytes and other control characters
    """
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)  # control chars except space
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def normalize_for_dedup(text: str) -> str:
    """Lowercase + collapse whitespace — for hash comparison only."""
    return re.sub(r"\s+", " ", text.lower().strip())


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode()).hexdigest()

# ── Loaders ────────────────────────────────────────────────────────────────────

def load_file(filepath: Path, default_source: str) -> list:
    """
    Load any of the supported formats and return list of
    {"source": str, "prompt": str} dicts.
    """
    if not filepath.exists():
        print(f"  ⚠  Not found, skipping: {filepath.name}")
        return []

    raw = filepath.read_text(encoding="utf-8").strip()
    if not raw:
        print(f"  ⚠  Empty file, skipping: {filepath.name}")
        return []

    entries = []

    if raw.startswith("["):
        # JSON array: clean_promts.json
        try:
            items = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"  ⚠  JSON parse error in {filepath.name}: {e}")
            return []
        for item in items:
            prompt = item.get("prompt", "").strip()
            if prompt:
                entries.append({
                    "source": item.get("source", default_source),
                    "prompt": prompt,
                })
    else:
        # JSONL: personal_prompts, lmsys, sharegpt, wildchat
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = item.get("prompt", "").strip()
            if prompt:
                entries.append({
                    "source": item.get("source", default_source),
                    "prompt": prompt,
                })

    return entries

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_entries   = []
    seen_hashes   = set()
    source_counts = {}   # kept per source
    source_dupes  = {}   # dupes per source

    for filename, default_source in SOURCES:
        filepath = DATA_DIR / filename
        print(f"Loading {filename}...")
        raw_entries = load_file(filepath, default_source)

        kept  = 0
        dupes = 0

        for entry in raw_entries:
            prompt = clean_prompt(entry["prompt"])

            # Drop if too short after cleaning
            if len(prompt) < 10:
                continue

            h = content_hash(prompt)
            if h in seen_hashes:
                dupes += 1
                continue

            seen_hashes.add(h)
            all_entries.append({
                "source": entry["source"],
                "prompt": prompt,
            })
            kept += 1

        source_counts[filename] = kept
        source_dupes[filename]  = dupes
        print(f"  → kept {kept}  |  dupes skipped {dupes}  |  raw total {len(raw_entries)}")

    print(f"\nShuffling {len(all_entries)} entries (seed={RANDOM_SEED})...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_entries)

    # Assign clean sequential IDs grouped by source, after shuffle
    source_counters = {}
    final_entries   = []
    for entry in all_entries:
        src = entry["source"]
        source_counters[src] = source_counters.get(src, 0) + 1
        final_entries.append({
            "id":     f"{src}_{source_counters[src]:03d}",
            "source": src,
            "prompt": entry["prompt"],
            "label":  None,
        })

    # Write output
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        for entry in final_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 48}")
    print(f"  Output: {OUTPUT}")
    print(f"{'─' * 48}")
    print(f"  {'Source file':<30}  {'Kept':>6}  {'Dupes':>6}")
    print(f"{'─' * 48}")
    for filename, _ in SOURCES:
        kept  = source_counts.get(filename, 0)
        dupes = source_dupes.get(filename, 0)
        print(f"  {filename:<30}  {kept:>6}  {dupes:>6}")
    print(f"{'─' * 48}")
    print(f"  {'TOTAL':.<30}  {len(final_entries):>6}")
    print(f"{'═' * 48}")

    # Per-source breakdown in final dataset
    final_by_source = {}
    for e in final_entries:
        final_by_source[e["source"]] = final_by_source.get(e["source"], 0) + 1
    print(f"\n  Final dataset breakdown by source:")
    for src, count in sorted(final_by_source.items()):
        print(f"    {src:<15} {count:>6}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()