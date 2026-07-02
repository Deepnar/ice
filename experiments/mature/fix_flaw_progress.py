#!/usr/bin/env python3
"""Mark all Flaw probes as completed under the new 20‑split checkpoint set."""

import json
from pathlib import Path

MATURE_DIR = Path(__file__).parent
GENERATED_FILE = MATURE_DIR / "generated_probes.json"
PROGRESS_FILE = MATURE_DIR / "_corrected_gt_progress.txt"

FLAW_ID = "bb558b5f-5365-5bac-9ed0-07219025b5f2"

with open(GENERATED_FILE) as f:
    generated = json.load(f)

flaw_splits = generated.get(FLAW_ID, {})
if not flaw_splits:
    print("No Flaw entries in generated_probes.json")
    exit()

# Build a set of (cid, probe_id, checkpoint) for every probe at every split
new_entries = set()
for split_str, probes in flaw_splits.items():
    cp = int(split_str)
    for p in probes:
        pid = p["probe_id"]
        new_entries.add(f"{FLAW_ID}|{pid}|{cp}")

# Load existing progress, remove old Flaw lines
if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE) as f:
        lines = [line.strip() for line in f if line.strip() and FLAW_ID not in line]
else:
    lines = []

# Add new entries
lines.extend(sorted(new_entries))

with open(PROGRESS_FILE, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"Progress file updated – Flaw now has {len(new_entries)} completed entries.")