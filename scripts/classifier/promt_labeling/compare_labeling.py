#!/usr/bin/env python3
"""Compare label fields in two JSONL prompt labeling files.
Print prompts for mismatched entries.
"""

import json
from collections import defaultdict

FILE1 = "/home/deepnar/Programs/ice/data/datasets/test_labeled_prompts.jsonl"
FILE2 = "/home/deepnar/Programs/ice/data/datasets/labeled_prompts.jsonl"

def load_data(path):
    """Return dict: id -> {'prompt': ..., 'label': ...}"""
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pid = obj["id"]
            label = obj["label"]
            # Normalize label lists
            label["topic_labels"] = sorted(label.get("topic_labels", []))
            label["intent_labels"] = sorted(label.get("intent_labels", []))
            data[pid] = {"prompt": obj["prompt"], "label": label}
    return data

def labels_equal(lab1, lab2):
    """Compare topic, intent, context (excluding reasoning)."""
    return (
        lab1.get("topic_labels") == lab2.get("topic_labels")
        and lab1.get("intent_labels") == lab2.get("intent_labels")
        and lab1.get("context_reliance") == lab2.get("context_reliance")
    )

def main():
    print(f"Loading {FILE1} ...")
    data1 = load_data(FILE1)
    print(f"  Found {len(data1)} entries.")
    print(f"Loading {FILE2} ...")
    data2 = load_data(FILE2)
    print(f"  Found {len(data2)} entries.")

    ids1 = set(data1.keys())
    ids2 = set(data2.keys())

    only_in_file1 = ids1 - ids2
    only_in_file2 = ids2 - ids1
    common = ids1 & ids2

    matches = 0
    mismatches = []

    for pid in common:
        lab1 = data1[pid]["label"]
        lab2 = data2[pid]["label"]
        if labels_equal(lab1, lab2):
            matches += 1
        else:
            diffs = {}
            for field in ["topic_labels", "intent_labels", "context_reliance"]:
                v1 = lab1.get(field)
                v2 = lab2.get(field)
                if v1 != v2:
                    diffs[field] = (v1, v2)
            mismatches.append((pid, data1[pid]["prompt"], diffs))

    # Summary
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Common IDs: {len(common)}")
    print(f"  Matches: {matches}")
    print(f"  Mismatches: {len(mismatches)}")
    print(f"IDs only in test file: {len(only_in_file1)}")
    print(f"IDs only in main file: {len(only_in_file2)}")

    if mismatches:
        print("\n--- MISMATCH DETAILS (showing first 20) ---")
        for idx, (pid, prompt_text, diffs) in enumerate(mismatches[:]):
            print(f"\n[ Mismatch {idx+1} ] ID: {pid}")
            # Print truncated prompt for readability
            prompt_display = prompt_text[:300] + ("..." if len(prompt_text) > 300 else "")
            print(f"  Prompt: {prompt_display}")
            for field, (v1, v2) in diffs.items():
                print(f"  {field}:")
                print(f"    File1: {v1}")
                print(f"    File2: {v2}")
        if len(mismatches) > 20:
            print(f"\n  ... and {len(mismatches)-20} more mismatches.")

    if only_in_file1:
        print(f"\nIDs only in test file (first 10): {list(only_in_file1)[:10]}")
    if only_in_file2:
        print(f"\nIDs only in main file (first 10): {list(only_in_file2)[:10]}")

if __name__ == "__main__":
    main()