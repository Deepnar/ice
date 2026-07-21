#!/usr/bin/env python3
"""
Clean repetitive / stuttering output from corrected ground truths.
1. Split on double‑newlines → remove consecutive identical paragraph blocks.
2. Within each block, remove consecutive identical sentences.
"""

import json, re
from pathlib import Path

MATURE_DIR = Path(__file__).parent.parent  # -> mature/ (script now in mature/oneoff/)
CORRECTED_IN = MATURE_DIR / "intermediates" / "corrected_ground_truths.json"
CORRECTED_OUT = CORRECTED_IN   # overwrite original

# Split text into sentences
SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def clean_text(text):
    """Remove consecutive duplicate paragraphs, then consecutive duplicate sentences."""
    if not text:
        return text

    # ── Step 1: split into paragraphs (on two or more newlines) ──
    paragraphs = re.split(r'\n{2,}', text.strip())
    unique_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if not unique_paragraphs or p != unique_paragraphs[-1]:
            unique_paragraphs.append(p)

    # ── Step 2: within each paragraph, remove consecutive duplicate sentences ──
    cleaned_paragraphs = []
    for p in unique_paragraphs:
        sentences = SENTENCE_SPLIT.split(p.strip())
        if not sentences:
            cleaned_paragraphs.append(p)
            continue
        unique_sentences = [sentences[0]]
        for s in sentences[1:]:
            if s != unique_sentences[-1]:
                unique_sentences.append(s)
        cleaned_paragraphs.append(' '.join(unique_sentences))

    return '\n\n'.join(cleaned_paragraphs)


def main():
    with open(CORRECTED_IN) as f:
        data = json.load(f)

    total_answers = 0
    cleaned_count = 0

    for conv_id, probes in data.items():
        for probe_id, checkpoints in probes.items():
            for cp, truth_text in checkpoints.items():
                total_answers += 1
                new_text = clean_text(truth_text)
                if new_text != truth_text:
                    checkpoints[cp] = new_text
                    cleaned_count += 1

    with open(CORRECTED_OUT, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Checked {total_answers} answers, cleaned {cleaned_count}.")

if __name__ == "__main__":
    main()