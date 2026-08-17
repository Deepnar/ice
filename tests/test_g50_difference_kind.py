#!/usr/bin/env python3
"""G50: `difference_kind` must type a name pair the way the store actually needs.

Every case below is a REAL pair from the arm-1 store (2026-08-17), not an
invented one — the point of the function is that it survives the corpus, and
hand-written probes agree with whoever wrote them.

Run: uv run python tests/test_g50_difference_kind.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workers.maintenance_agent import difference_kind, merge_key  # noqa: E402

# (a, b, expected, why) — cosines noted where they make the point
CASES = [
    # ── merge: pure formatting, deterministic, no judgement needed ──────────
    ("qwen coder 32b-q4", "qwen coder 32b q4", "merge", "0.9970 — punctuation"),
    ("gemma-4-e4b-q4", "gemma-4-e4b q4", "merge", "0.9939 — punctuation"),
    ("digital note taking", "digital note-taking", "merge", "0.9924 — hyphen"),
    ("18-month job search window", "18 month job search window", "merge", "hyphen"),
    ("u.s", "u.s.", "merge", "trailing period"),
    ("~18 gb", "~18gb", "merge", "digit/letter run split"),
    ("mixture of experts", "mixture-of-experts", "merge", "separators"),

    # ── reject: a token that CHANGES THE REFERENT ───────────────────────────
    ("8 gb", "4 gb", "reject:digits", "0.9852 — quantity"),
    ("qwen3-32b", "qwen3-30b", "reject:digits", "0.9860 — model size"),
    ("gemma-4-e4b", "gemma-4-e4b-q4", "reject:digits", "0.9851 — base vs quantised"),
    ("5 reddit.com", "1 reddit.com", "reject:digits", "0.9886"),
    ("~15 gb", "~17 gb", "reject:digits", "0.9388 — embeddings cannot separate"),
    ("1-2 percent", "1-4 percent", "reject:digits", "0.9470"),
    ("12th boards", "10th boards", "reject:digits", "quantity"),
    ("two sagas", "four sagas", "reject:wordnum", "0.9889 — spelled-out number"),
    ("two sagas", "three sagas", "reject:wordnum", "0.9883"),
    ("his father", "her father", "reject:gender", "0.9827 — different person"),
    ("her laptop", "his laptop", "reject:gender", "0.9844"),
    ("is rare", "was rare", "reject:tense", "0.9837 — a TEMPORAL claim"),
    ("was worth", "is worth", "reject:tense", "0.9849"),
    ("shiva without brahma", "brahma without shiva", "reject:permutation",
     "0.9639 — LIVE converse, TRAPS #26 entity twin"),
    ("see you", "you see", "reject:permutation", "converse"),
    ("a1 german certificate", "german a1 certificate", "reject:permutation", "0.9597"),

    # ── defer: genuinely ambiguous, belongs to G51 ──────────────────────────
    ("villainess", "villain", "defer", "0.9839 — morphology, needs judgement"),
    ("a 3d diagonal plane", "3d diagonal plane", "defer", "article added"),
    ("emotional validation", "validation", "defer", "a KIND of — G51 links it"),
    ("research focused programs", "research focused program", "defer", "plural"),
]


def main() -> int:
    failures = []
    for a, b, want, why in CASES:
        got = difference_kind(a, b)
        ok = got == want
        if not ok:
            failures.append((a, b, want, got, why))
        print(f"  {'ok ' if ok else 'FAIL'}  {want:<20} {a!r} | {b!r}"
              + ("" if ok else f"   ← got {got!r}"))

    # Symmetry: the verdict must not depend on argument order.
    asym = [(a, b, difference_kind(a, b), difference_kind(b, a))
            for a, b, _w, _y in CASES
            if difference_kind(a, b) != difference_kind(b, a)]

    # merge_key equality and a 'merge' verdict must agree, in both directions.
    disagree = [(a, b) for a, b, _w, _y in CASES
                if (merge_key(a) == merge_key(b)) != (difference_kind(a, b) == "merge")]

    print(f"\n  {len(CASES) - len(failures)}/{len(CASES)} verdicts correct")
    print(f"  symmetry violations: {len(asym)}")
    print(f"  merge_key/verdict disagreements: {len(disagree)}")
    for a, b, ab, ba in asym:
        print(f"      ASYMMETRIC {a!r}|{b!r}: {ab} vs {ba}")
    for a, b, want, got, why in failures:
        print(f"      FAILED {a!r}|{b!r}: wanted {want}, got {got}  ({why})")

    if failures or asym or disagree:
        print("\nFAILED")
        return 1
    print("\nPASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
