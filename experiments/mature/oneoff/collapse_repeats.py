#!/usr/bin/env python3
"""Collapse any phrase that repeats 3+ times in a row, keeping only the first."""

import json, re
from pathlib import Path

MATURE_DIR = Path(__file__).parent.parent  # -> mature/ (script now in mature/oneoff/)
CORRECTED_IN = MATURE_DIR / "intermediates" / "corrected_ground_truths.json"
CORRECTED_OUT = CORRECTED_IN

# This regex finds a sequence of (word + optional spaces) that repeats 3+ times.
# It's crude but catches the "completely and utterly still" pattern and similar.
REPEATED_PHRASE_RE = re.compile(r'(\b.{3,}?)\s*(\1\s*){3,}')

def collapse_repeats(text):
    """Replace any phrase repeated 3+ times consecutively with just one occurrence."""
    previous = None
    while previous != text:
        previous = text
        text = REPEATED_PHRASE_RE.sub(r'\1', text)
    return text

def main():
    with open(CORRECTED_IN) as f:
        data = json.load(f)

    total = 0
    cleaned = 0
    for conv_id, probes in data.items():
        for probe_id, checkpoints in probes.items():
            for cp, truth_text in checkpoints.items():
                total += 1
                new_text = collapse_repeats(truth_text)
                if new_text != truth_text:
                    checkpoints[cp] = new_text
                    cleaned += 1

    with open(CORRECTED_OUT, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Checked {total} answers, collapsed repeats in {cleaned}.")

if __name__ == "__main__":
    main()