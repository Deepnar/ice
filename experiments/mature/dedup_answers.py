#!/usr/bin/env python3
"""
Deduplicate bullet-point answers in corrected ground truths.
For any answer that is a bulleted list, removes duplicate bullet items,
keeping only the first occurrence. Non-bullet answers are left unchanged.
"""

import json, re
from pathlib import Path

MATURE_DIR = Path(__file__).parent
CORRECTED_IN = MATURE_DIR / "results" / "corrected_ground_truths.json"
CORRECTED_OUT = CORRECTED_IN

# Lines that look like a bullet item: optional whitespace, then * or - or •
BULLET_RE = re.compile(r'^\s*[*\-•]\s+')

def dedup_bullets(text):
    """If text looks like a bulleted list, deduplicate the bullet items."""
    lines = text.splitlines()
    # Check if the majority of non‑empty lines are bullet items
    bullet_lines = [i for i, l in enumerate(lines) if l.strip() and BULLET_RE.match(l)]
    if len(bullet_lines) < 3:   # not a bullet list
        return text

    seen = set()
    new_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            new_lines.append(line)
            continue
        if BULLET_RE.match(line):
            # Normalise the line to a canonical form (collapse whitespace, lowercase)
            key = re.sub(r'\s+', ' ', stripped).lower()
            if key not in seen:
                seen.add(key)
                new_lines.append(line)
            # else skip duplicate
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

def main():
    with open(CORRECTED_IN) as f:
        data = json.load(f)

    total_answers = 0
    cleaned_count = 0

    for conv_id, probes in data.items():
        for probe_id, checkpoints in probes.items():
            for cp, truth_text in checkpoints.items():
                total_answers += 1
                new_text = dedup_bullets(truth_text)
                if new_text != truth_text:
                    checkpoints[cp] = new_text
                    cleaned_count += 1

    with open(CORRECTED_OUT, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Checked {total_answers} answers, deduplicated {cleaned_count}.")

if __name__ == "__main__":
    main()