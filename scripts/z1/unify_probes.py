#!/usr/bin/env python3
"""Z1: fold all three probe sources into ONE set with one schema.

**The problem this fixes.** A conversation's probes live in three places written
by three different efforts, and no two carry the same fields — so no single
measurement can use all of them:

  | source | n | `gold_turns` | `target_leg` | `expected_fragments` |
  |---|---|---|---|---|
  | typed   | 444 | ✅ | ✗ | ✗ |
  | mature  |  93 | ✗ | ✅ | ✗ |
  | curated |  81 | ✗ | ✗ | ✅ |

`gold_turns` is what recall@k needs; `target_leg` is what the per-leg ablations
need. **174 of 618 probes have no turn anchor at all**, so today they can only
be scored by hand — which is why every mature evaluation was scored by a human
(`experiments/mature/manual_evaluate.py`).

**⚑ AND ONE CONVERSATION IS ANOTHER'S TAIL.** `cca73c87` is turns 1039–1119 of
`bb558b5f` — verified by matching prompt text, 81 of 81 exact. Its 128 probes
are really about `bb558b5f` content, so they are remapped here: **turn *n* →
turn *n + 1038*, 1-indexed.** Offsets +1037 and +1039 match only 6 turns each
(repeated short prompts), so the offset is not a guess.

⚠ **THIS SCRIPT DOES NOT INVENT GOLD TURNS.** Where a probe has no anchor it
says so and leaves the field empty. Deriving them is
`derive_retrieval_gt.py`'s job, and it takes care to avoid the circularity trap
(lexical shortlist against the *expected answer*, never ICE's own embedder). A
guessed anchor would make every retrieval number afterwards wrong invisibly.

  uv run python scripts/z1/unify_probes.py
  uv run python scripts/z1/unify_probes.py --write
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.probe_census import (  # noqa: E402
    _key,
    load_curated,
    load_mature,
    load_typed,
)

OUT = "experiments/curation_files/unified_probes.json"

# cca73c87 IS bb558b5f's tail. Verified 2026-08-26 by prompt-text match, 81/81.
ABSORBED = {"cca73c87": ("bb558b5f", 1038)}

# The reseed corpus. cca73c87 is deliberately absent — seeding it alongside
# bb558b5f would put the same 81 turns in the store twice.
CORPUS = ["bb558b5f", "ecc64aab", "355a5709"]


def normalise(p: dict) -> dict:
    """One schema, whatever the source. Missing stays missing."""
    src = p.get("_src")
    conv = p.get("_conv") or str(p.get("conversation") or "")[:8]
    gold = list(p.get("gold_turns") or [])
    # ⚑ `ENTER_TYPE` is a TEMPLATE PLACEHOLDER the curated generator never
    # filled — all 81 curated probes carry it literally. Counting it as a
    # probe_type inflated "85% have a type" to a number that included 81 rows
    # typed as the string "ENTER_TYPE". Missing is missing.
    ptype = p.get("probe_type")
    if ptype in ("ENTER_TYPE", "", None):
        ptype = None
    out = {
        "probe_id": f"{src}:{p.get('probe_id') or _key(p)[:40]}",
        "conversation": conv,
        "question": (p.get("user_injected_prompt") or p.get("question") or "").strip(),
        "expected_answer": (p.get("expected_answer") or p.get("answer") or "").strip(),
        "gold_turns": gold,
        "target_leg": p.get("target_leg"),
        "probe_type": ptype,
        "evidence": p.get("evidence"),
        "expected_fragments": p.get("ground_truth_expected_fragments") or [],
        "split_turn": p.get("_split"),
        "source": src,
        "source_file": p.get("_file"),
        "remapped_from": None,
    }
    if conv in ABSORBED:
        target, offset = ABSORBED[conv]
        out["remapped_from"] = {"conversation": conv, "offset": offset}
        out["conversation"] = target
        out["gold_turns"] = [t + offset for t in gold]
        if out["split_turn"] is not None:
            out["split_turn"] = out["split_turn"] + offset
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"write {OUT}")
    args = ap.parse_args()

    sources = {"curated": load_curated(), "mature": load_mature(), "typed": load_typed()}
    wanted = set(CORPUS) | set(ABSORBED)

    pool = []
    for name, by_conv in sources.items():
        for cid, probes in by_conv.items():
            if cid not in wanted:
                continue
            for p in probes:
                pool.append({**p, "_conv": cid})

    # Dedupe on question text — ids are checkpoint-scoped, so they would report
    # zero duplicates on a set that is 20% duplicate. Keep the SHALLOWEST split
    # and prefer the row that carries the most usable metadata, so a folded
    # probe does not lose its gold turns to an emptier twin.
    def richness(p):
        return (bool(p.get("gold_turns")), bool(p.get("target_leg")),
                bool(p.get("ground_truth_expected_fragments")))

    best: dict[str, dict] = {}
    for p in pool:
        k = _key(p)
        if not k:
            continue
        cur = best.get(k)
        if cur is None or sum(richness(p)) > sum(richness(cur)):
            best[k] = p

    unified = [normalise(p) for p in best.values()]

    print(f"raw rows        : {len(pool)}")
    print(f"distinct probes : {len(unified)}   (removed {len(pool)-len(unified)})\n")

    by_conv = Counter(p["conversation"] for p in unified)
    remapped = sum(1 for p in unified if p["remapped_from"])
    print("by conversation (after absorbing cca73c87 → bb558b5f):")
    for c, n in by_conv.most_common():
        print(f"   {c}: {n}")
    print(f"   ↳ {remapped} of bb558b5f's probes were remapped from cca73c87\n")

    print("COVERAGE — what each probe can actually score:")
    have_turns = sum(1 for p in unified if p["gold_turns"])
    have_leg = sum(1 for p in unified if p["target_leg"])
    have_frag = sum(1 for p in unified if p["expected_fragments"])
    have_type = sum(1 for p in unified if p["probe_type"])
    n = len(unified)
    for label, k in (("gold_turns  (recall@k, MRR)", have_turns),
                     ("target_leg  (per-leg ablations)", have_leg),
                     ("probe_type  (typed scoring)", have_type),
                     ("expected_fragments", have_frag)):
        print(f"   {label:34} {k:>4} / {n}  ({100*k/n:.0f}%)")
    print(f"\n   ⚠ {n-have_turns} probes have NO turn anchor — "
          f"run derive_retrieval_gt.py, do not guess")
    print(f"   by source: {dict(Counter(p['source'] for p in unified))}")

    # A remapped gold turn that lands outside the conversation is a bug in the
    # offset, and it must be loud rather than silently scored as a miss later.
    bad = [p for p in unified
           if p["conversation"] == "bb558b5f" and p["gold_turns"]
           and max(p["gold_turns"]) > 1119]
    print(f"   remapped turns beyond bb558b5f's 1,119: {len(bad)}"
          + ("  ⛔ CHECK THE OFFSET" if bad else "  ✓"))

    if args.write:
        with open(OUT, "w") as fh:
            json.dump({"corpus": CORPUS, "absorbed": ABSORBED,
                       "n": len(unified), "probes": unified}, fh, indent=1)
        print(f"\n→ {OUT}  (⚠ gitignored — TRAPS #40)")
    else:
        print(f"\n(dry run — pass --write to produce {OUT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
