#!/usr/bin/env python3
"""Align extracted entities to token positions and produce BIO training data.
Fixes: longest entities first, no overwriting of already‑labelled tokens."""

import json
import os
from transformers import AutoTokenizer

INPUT = "data/ner/extracted_entities.jsonl"
OUTPUT = "data/ner/training_data.jsonl"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

def find_entity_spans(text, entity):
    """Return list of (start_char, end_char) for all occurrences of entity in text (case‑insensitive)."""
    text_lower = text.lower()
    entity_lower = entity.lower()
    spans = []
    start = 0
    while True:
        idx = text_lower.find(entity_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(entity)))
        start = idx + 1
    return spans

def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with open(INPUT, "r", encoding="utf-8") as fin, open(OUTPUT, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line.strip())
            raw_text = obj["raw_text"]
            entities = obj.get("entities", [])

            # Filter out trivial entities
            # Filter out trivial entities
            # Filter out trivial entities
            entities = [e for e in entities if len(e) >= 2 and e.lower() not in {"user", "assistant", "question", "answer"}]

            # Filter out common English stop‑words that are never entities
            STOP_WORDS = {
                "the", "and", "this", "that", "with", "from", "have", "been",
                "was", "are", "were", "will", "would", "could", "should",
                "about", "also", "just", "like", "then", "than", "over",
                "into", "only", "more", "some", "such", "each", "every",
                "other", "many", "most", "its", "our", "his", "her", "they",
                "them", "these", "those", "class", "sentence", "object",
                "get", "plot", "friends", "deployed", "caching", "app",
                "win", "main", "first", "second", "third", "true", "false",
            }
            entities = [e for e in entities if e.lower() not in STOP_WORDS]

            # Deduplicate entity strings (still label every occurrence via span search)
            entities = list(set(entities))

            # Tokenize with offsets
            encoded = tokenizer(raw_text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
            offsets = encoded["offset_mapping"]
            labels = ["O"] * len(tokens)

            # Sort entities by length (longest first) to prevent short entity overwriting longer one
            entities.sort(key=lambda e: len(e), reverse=True)

            for entity in entities:
                spans = find_entity_spans(raw_text, entity)
                for start_char, end_char in spans:
                    # Find tokens that overlap this span
                    entity_tokens = []
                    for i, (tok_start, tok_end) in enumerate(offsets):
                        if tok_end <= start_char or tok_start >= end_char:
                            continue
                        entity_tokens.append(i)
                    if not entity_tokens:
                        continue
                    # Only label tokens that are still 'O' to avoid overwriting a longer entity
                    if labels[entity_tokens[0]] == "O":
                        labels[entity_tokens[0]] = "B-ENT"
                    for i in entity_tokens[1:]:
                        if labels[i] == "O":
                            labels[i] = "I-ENT"

            record = {"tokens": tokens, "labels": labels}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"BIO training data saved → {OUTPUT}")

if __name__ == "__main__":
    main()