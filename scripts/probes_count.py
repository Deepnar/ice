#!/usr/bin/env python3

import json

FILE = "/home/deepnar/Programs/ice/experiments/mature/generated_probes.json"

with open(FILE, "r") as f:
    data = json.load(f)

for conv_id, splits in data.items():

    split_turns = sorted(
        int(k)
        for k, v in splits.items()
        if len(v) > 0
    )

    total_probes = sum(
        len(v)
        for v in splits.values()
    )

    print("\n" + "="*70)
    print(conv_id)
    print("="*70)

    print(f"Splits: {len(split_turns)}")
    print(f"Probes: {total_probes}")
    print(f"Split turns: {split_turns}")