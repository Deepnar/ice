#!/usr/bin/env python3
"""
Deep‑clean corrected ground truths by removing consecutively repeated
sentence blocks of ANY length (not just 1‑3 sentences).
Normalisation ignores punctuation, spaces, and minor typos so that
near‑identical blocks are collapsed to a single occurrence.
"""

import json, re
from pathlib import Path

MATURE_DIR = Path(__file__).parent.parent  # -> mature/ (script now in mature/oneoff/)
CORRECTED_IN = MATURE_DIR / "intermediates" / "corrected_ground_truths.json"
CORRECTED_OUT = CORRECTED_IN

# ── helpers ──────────────────────────────────────────────────────────────────
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]

def _key(sentence):
    """Normalise to alphanumeric only — catches '2nd protagonist' vs '2ndprotagonist'."""
    return re.sub(r'[^a-z0-9]', '', sentence.lower())

def _collapse_long_repeats(sentences):
    """
    Scan the list of sentences and collapse any sequence of 1‑N sentences
    that immediately repeats one or more times.  The longest possible
    repeating block is always preferred.
    """
    keys = [_key(s) for s in sentences]
    n = len(sentences)
    i = 0
    result = []          # list of original sentences we keep

    while i < n:
        best_len = 0      # length of the repeating block (in sentences)
        best_count = 0    # how many times it repeats (including the first)

        # Try every possible block length starting at position i
        max_len = (n - i) // 2   # need at least one full repetition
        for blen in range(1, min(max_len, 20) + 1):   # cap at 20 sentences
            pattern = keys[i : i + blen]
            count = 1
            j = i + blen
            while j + blen <= n and keys[j : j + blen] == pattern:
                count += 1
                j += blen
            if count >= 2 and blen > best_len:
                best_len = blen
                best_count = count

        if best_len > 0:
            # Keep the first occurrence of the block
            result.extend(sentences[i : i + best_len])
            i += best_len * best_count
        else:
            result.append(sentences[i])
            i += 1

    return result

def clean_text(text):
    sents = _sentences(text)
    if len(sents) < 2:
        return text
    compressed = _collapse_long_repeats(sents)
    return ' '.join(compressed)

def main():
    with open(CORRECTED_IN) as f:
        data = json.load(f)

    total = 0
    cleaned = 0
    for conv_id, probes in data.items():
        for probe_id, checkpoints in probes.items():
            for cp, truth_text in checkpoints.items():
                total += 1
                new_text = clean_text(truth_text)
                if new_text != truth_text:
                    checkpoints[cp] = new_text
                    cleaned += 1

    with open(CORRECTED_OUT, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Deep‑cleaned {cleaned} / {total} answers.")

if __name__ == "__main__":
    main()