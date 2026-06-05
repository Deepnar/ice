#!/usr/bin/env python3
"""
Reads failed prompts from the failed log, extracts their IDs,
and removes those IDs from the main unlabeled dataset.
Produces a new cleaned dataset file so the labeling script
never sees the permanently failing prompts again.
"""

import json
import sys
from pathlib import Path

# ---------- CONFIG (change if your paths differ) ----------
FAILED_FILE   = "/home/deepnar/Programs/ice/data/datasets/failed_prompts.jsonl"
INPUT_FILE    = "/home/deepnar/Programs/ice/data/labeled/dataset_cleaned.jsonl"
OUTPUT_FILE   = "/home/deepnar/Programs/ice/data/labeled/dataset_cleaned_filtered.jsonl"
# ----------------------------------------------------------

def load_failed_ids(failed_path):
    """Return a set of all unique prompt IDs that appear in the failed file."""
    failed_ids = set()
    if not Path(failed_path).exists():
        print(f"Failed file not found at {failed_path}, nothing to prune.")
        return failed_ids

    with open(failed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                failed_ids.add(entry["id"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line in failed file: {e}")
    return failed_ids

def filter_input(input_path, output_path, failed_ids):
    """Write only prompts whose ID is NOT in failed_ids to the output file."""
    total = 0
    kept = 0
    removed = 0

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                item = json.loads(line)
                if item["id"] in failed_ids:
                    removed += 1
                    continue
                fout.write(json.dumps(item) + "\n")
                kept += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line in input: {e}")

    return total, kept, removed

def main():
    print("Loading failed IDs...")
    failed_ids = load_failed_ids(FAILED_FILE)
    print(f"Found {len(failed_ids)} unique failed IDs.")

    if not failed_ids:
        print("Nothing to prune. Exiting.")
        return

    print(f"Filtering {INPUT_FILE} -> {OUTPUT_FILE} ...")
    total, kept, removed = filter_input(INPUT_FILE, OUTPUT_FILE, failed_ids)

    print(f"\nDone.")
    print(f"  Total prompts in original: {total}")
    print(f"  Removed (failed):          {removed}")
    print(f"  Kept (will be labeled):    {kept}")

    print(f"\nNew cleaned dataset: {OUTPUT_FILE}")
    print("Update your labeling script's INPUT_PATH to this file and restart.")

if __name__ == "__main__":
    main()