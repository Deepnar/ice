#!/usr/bin/env python3
"""Z1: seed the open relation vocabulary from the harvested candidates (G45).

**The problem this solves.** `canonical_relation` reuses a relation the graph
already holds, so incoming `using` becomes the stored `uses` instead of a second
node. On a FRESH store the graph holds nothing, so the first few hundred turns
canonicalise against an empty set and sprout exactly the synonym spread the
mechanism exists to prevent. G45's entry says the 3,244 harvested candidates
"become a one-time seed for the canonicalisation table" — this is that.

**What gets in, and what does not.** The harvest is raw model output and carries
its own junk. Kept only when a candidate is corroborated by several independent
models — agreement across arms is the closest thing to a ground truth available,
since eight different models inventing the same relation is evidence about the
data rather than about one model's habits.

Excluded on purpose:
  * `not`, `negated` — these are ICE's own polarity machinery leaking into the
    relation slot (A8 stores negation as a flag, not a relation word). Seeding
    them would make the leak canonical.
  * The harvest's `proposed opposite` column — it is naive suffixing
    (`haed_by`, `carrieed_by`, `becomeed_by`) and would seed malformed strings.
    Direction is handled by `_is_inverse_pair`, which needs no vocabulary.

Output: `data/relation_seed.json`, read by `known_relations()` and unioned with
whatever the live graph holds.

Run:
  uv run python scripts/z1/seed_relation_vocab.py --min-arms 4
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.run_meta import file_digest, run_meta  # noqa: E402

SRC = Path("experiments/curation_files/VOCAB_CANDIDATES.md")
OUT = Path("data/relation_seed.json")

# ICE's own polarity leaking into the relation slot — never a relation word.
_NEVER = {
    # ICE's own polarity leaking into the relation slot (A8 stores negation as
    # a flag, not a word).
    "not", "negated",
    # Placeholders and non-answers.
    "none", "null", "n_a", "na", "unknown", "other", "relation", "relates_to",
    # Copulas and bare prepositions: they carry no relation, and seeding them
    # gives canonicalisation a magnet that everything is vaguely close to.
    "is", "was", "be", "are", "were", "am", "about", "of", "to", "in", "for",
}

_ROW = re.compile(r"^\|\s*`?([a-z][a-z0-9_]*)`?\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-arms", type=int, default=4,
                    help="how many independent models must have produced it")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC.exists():
        print(f"missing {SRC} — nothing to seed from")
        return 1

    # ⚑ Parse ONLY the candidate table. The file opens with a per-ARM table
    # whose first column is a model name — `gemma4_e4b`, `granite4_micro` —
    # which matches a relation row exactly, so an unscoped parser seeds model
    # names as relations. Caught on the first dry run.
    lines, inside = [], False
    for raw in SRC.read_text().splitlines():
        if raw.startswith("## "):
            inside = raw.lower().startswith("## candidate relations")
            continue
        if inside:
            lines.append(raw)
    if not lines:
        print("could not find the '## Candidate relations' section")
        return 1

    kept, rejected = [], {"low_arms": 0, "low_count": 0, "blocked": 0, "malformed": 0}
    for line in lines:
        m = _ROW.match(line.strip())
        if not m:
            continue
        rel, count, arms = m.group(1), int(m.group(2)), int(m.group(3))
        if rel in _NEVER:
            rejected["blocked"] += 1
            continue
        if len(rel) < 3 or not rel[0].isalpha():
            rejected["malformed"] += 1
            continue
        if arms < args.min_arms:
            rejected["low_arms"] += 1
            continue
        if count < args.min_count:
            rejected["low_count"] += 1
            continue
        kept.append({"relation": rel, "count": count, "arms": arms})

    kept.sort(key=lambda r: (-r["arms"], -r["count"]))
    print(f"parsed {SRC}")
    print(f"  kept     {len(kept)}")
    for k, v in rejected.items():
        print(f"  rejected {k}: {v}")
    print(f"\n  top 20: {[r['relation'] for r in kept[:20]]}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": run_meta(script=__file__, args=vars(args),
                         settings_keys=["codex_relation_open_vocabulary",
                                        "codex_relation_canonical_threshold"],
                         inputs=[file_digest(SRC)],
                         extra={"kept": len(kept), "rejected": rejected,
                                "note": "seed for canonicalisation only — these "
                                        "are NOT a closed vocabulary and nothing "
                                        "is rejected for being absent from them"}),
        "relations": [r["relation"] for r in kept],
        "detail": kept,
    }, indent=1))
    print(f"\nwrote {OUT} ({len(kept)} relations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
