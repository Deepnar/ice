#!/usr/bin/env python3
"""Z1: every probe that exists for a conversation, from all THREE sources, deduped.

**Why this exists.** Probes for one conversation are written in three different
places, by three different efforts, and nobody has ever counted them together:

  1. `experiments/curation_files/EC-*.json` → `evaluation_probes`
     Hand-written per CHECKPOINT. ⚠ The same probe is repeated in every
     checkpoint file of that conversation, so the raw count is inflated:
     `bb558b5f` has **113 rows that are 51 distinct probes**.
  2. `experiments/mature/intermediates/generated_probes.json`
     The mature experiment's generated set, keyed
     `conversation_uuid → split_turn → [probes]`. Only the FOUR mature
     conversations have these. Carries `target_leg`, which the curated ones do
     not.
  3. `experiments/curation_files/typed_probes.json`
     The Z1 typed set (444). ⚠ Covers ONLY `ecc64aab`/`355a5709`/`cca73c87`.

**And why the turn mapping matters.** A probe is answerable only if the turns it
asks about were actually seeded. Both source 1 and source 2 carry the split they
were written for, so truncating a conversation to N turns means keeping exactly
the probes whose split is ≤ N. Seeding 400 turns of a 1,119-turn conversation
and then scoring all 59 of its mature probes would mark the 40 that ask about
turns 400–1,119 as retrieval failures, when the answer was never in the store.
That is the [TRAPS #32](../../docs/TRAPS.md) shape: a metric that cannot see the
thing it is scoring.

  uv run python scripts/z1/probe_census.py
  uv run python scripts/z1/probe_census.py --conv bb558b5f --truncate 400
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

CURATION = "experiments/curation_files"
MATURE = "experiments/mature/intermediates/generated_probes.json"
TYPED = f"{CURATION}/typed_probes.json"


def _key(p: dict) -> str:
    """Dedupe key: the question text.

    Deliberately NOT `probe_id` — the ids are checkpoint-scoped (`23-GEN-01`),
    so the same question carries a different id in every file it appears in and
    keying on it would report zero duplicates on a set that is 55% duplicate.
    """
    return (p.get("user_injected_prompt") or p.get("question") or "").strip().lower()[:300]


def load_curated() -> dict[str, list[dict]]:
    """EC checkpoint probes, tagged with the split they were written for."""
    out: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(f"{CURATION}/EC-*.json")):
        try:
            d = json.loads(open(path).read())
        except Exception:
            continue
        # ⚠ Map by the FIELD, never the filename: EC-961862eb-FULL.json carries
        # original_conversation_id = bb558b5f.
        cid = (d.get("original_conversation_id") or "")[:8]
        split = d.get("split_turn_index")
        for p in (d.get("evaluation_probes") or []):
            out[cid].append({**p, "_split": split, "_src": "curated",
                             "_file": os.path.basename(path)})
    return out


def load_mature() -> dict[str, list[dict]]:
    """Mature generated probes, already keyed by split turn."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not os.path.exists(MATURE):
        return out
    for full, by_split in json.load(open(MATURE)).items():
        for split, probes in by_split.items():
            for p in probes:
                out[full[:8]].append({**p, "_split": int(split), "_src": "mature",
                                      "_file": os.path.basename(MATURE)})
    return out


def load_typed() -> dict[str, list[dict]]:
    """The Z1 typed set. Its `gold_turns` are the turns that must be seeded."""
    out: dict[str, list[dict]] = defaultdict(list)
    if not os.path.exists(TYPED):
        return out
    for p in json.load(open(TYPED)).get("probes", []):
        gold = p.get("gold_turns") or []
        out[str(p.get("conversation"))[:8]].append(
            {**p, "_split": max(gold) if gold else None, "_src": "typed",
             "_file": "typed_probes.json"})
    return out


def dedupe(probes: list[dict]) -> tuple[list[dict], int]:
    """Keep the FIRST occurrence of each question, and the SHALLOWEST split.

    Shallowest wins because a probe answerable from 336 turns is also
    answerable from 1,119 — recording the deeper split would drop it from a
    truncated corpus it is perfectly fine in.
    """
    best: dict[str, dict] = {}
    for p in probes:
        k = _key(p)
        if not k:
            continue
        cur = best.get(k)
        if cur is None:
            best[k] = p
        elif (p.get("_split") or 10**9) < (cur.get("_split") or 10**9):
            best[k] = p
    return list(best.values()), len(probes) - len(best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default=None, help="8-char id; default every conversation")
    ap.add_argument("--truncate", type=int, default=0,
                    help="keep only probes answerable from the first N turns")
    args = ap.parse_args()

    sources = {"curated": load_curated(), "mature": load_mature(), "typed": load_typed()}
    cids = sorted({c for s in sources.values() for c in s})
    if args.conv:
        cids = [c for c in cids if c == args.conv]

    print(f"{'conv':10} {'curated':>9} {'mature':>7} {'typed':>7} {'RAW':>6} "
          f"{'DISTINCT':>9} {'dupes':>6}")
    print("-" * 62)
    detail = {}
    for cid in cids:
        pools = {k: v.get(cid, []) for k, v in sources.items()}
        allp = [p for v in pools.values() for p in v]
        uniq, ndup = dedupe(allp)
        detail[cid] = uniq
        print(f"{cid:10} {len(pools['curated']):>9} {len(pools['mature']):>7} "
              f"{len(pools['typed']):>7} {len(allp):>6} {len(uniq):>9} {ndup:>6}")

    if args.conv and detail:
        uniq = detail[args.conv]
        print(f"\n── {args.conv}: {len(uniq)} distinct probes")
        print("   by source:", dict(Counter(p["_src"] for p in uniq)))
        legs = Counter(p.get("target_leg") for p in uniq if p.get("target_leg"))
        if legs:
            print("   by target_leg:", dict(legs))
        known = sorted(p["_split"] for p in uniq if p.get("_split") is not None)
        if known:
            print(f"   split range: {known[0]} → {known[-1]}"
                  f"  ({len(uniq) - len(known)} with no split recorded)")
            print("\n   answerable if the conversation is truncated at N turns:")
            for n in (100, 200, 300, 400, 500, 600, 800, 1000, 1200):
                if n > known[-1] + 200:
                    break
                keep = sum(1 for s in known if s <= n)
                print(f"     N={n:>5}  {keep:>4} of {len(uniq)} probes "
                      f"({100*keep/len(uniq):.0f}%)")
        if args.truncate:
            keep = [p for p in uniq
                    if p.get("_split") is None or p["_split"] <= args.truncate]
            print(f"\n   ⇒ at --truncate {args.truncate}: KEEP {len(keep)}, "
                  f"DROP {len(uniq)-len(keep)}")
            print("      kept by source:", dict(Counter(p["_src"] for p in keep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
