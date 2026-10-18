"""Per-turn source-gate decisions of two profile runs (memory_source_refinement events,
one per historical turn, in order). A differing decision counts against the gate
only when both runs showed it the same candidates; otherwise the input changed.

    uv run --no-sync python experiments/z2/compare_gate.py logs/<run-a>.log logs/<run-b>.log
"""
import re
import sys

FIELDS = ("action", "admitted", "base_retrieve", "candidates", "retrieve")


def events(path):
    rows = []
    for line in open(path):
        if "memory_source_refinement" in line:
            rows.append({k: (re.search(rf"\b{k}=(\S+)", line) or [None, None])[1] for k in FIELDS})
    return rows


def main():
    a, b = events(sys.argv[1]), events(sys.argv[2])
    print(f"events: {len(a)} vs {len(b)}")
    same = diff_same_input = diff_new_input = 0
    for turn, (x, y) in enumerate(zip(a, b), 1):
        if (x["action"], x["retrieve"]) == (y["action"], y["retrieve"]):
            same += 1
        elif (x["candidates"], x["base_retrieve"]) == (y["candidates"], y["base_retrieve"]):
            diff_same_input += 1
            print(f"  turn {turn}: SAME INPUT, decision changed {x} -> {y}")
        else:
            diff_new_input += 1
            print(f"  turn {turn}: input changed {x} -> {y}")
    print(f"identical decisions {same}/{min(len(a), len(b))}; changed on same input {diff_same_input}; "
          f"changed with different input {diff_new_input}")


if __name__ == "__main__":
    main()
