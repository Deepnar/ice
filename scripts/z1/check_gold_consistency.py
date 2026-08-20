#!/usr/bin/env python3
"""Deterministic ground-truth check: does each probe's gold match its own evidence?

**No model, no API, no cost.** 400 of 444 probes carry an `evidence` field that
was never used by any scorer. It comes in two shapes, and each admits an exact
check:

  · **verbatim quote** (`episodic_lookup` 232, `temporal` 13, `codex_multihop`
    12) — the quote must appear in the concatenated gold turns. If it does not,
    the gold turns are the wrong turns.
  · **numeric turn list** (`codex_multihop` 63, `procedural` 40,
    `summary_synthesis` 40) — the listed turns must be a SUBSET of
    `gold_turns`. A turn named as evidence but absent from the gold means the
    gold is incomplete, and every scorer measuring coverage against that gold
    is scoring against a hole.

This is the cheap half of the gold audit. `verify_gold.py` is the expensive
half (it asks a model whether the gold SUPPORTS the stated answer, which is a
different question and costs ~10k tokens a probe). Run this first and send only
the ambiguous remainder to the model.

Run:
  uv run python scripts/z1/check_gold_consistency.py
  uv run python scripts/z1/check_gold_consistency.py --write-clean
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text as _sql               # noqa: E402
from src.api.db import SessionLocal               # noqa: E402

PROBES = Path("experiments/curation_files/typed_probes.json")
MANIFEST = Path("experiments/curation_files/seeded_store.json")
CLEAN = Path("experiments/curation_files/typed_probes.clean.json")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default=str(PROBES))
    ap.add_argument("--write-clean", action="store_true",
                    help="write a probe set with the failures removed")
    ap.add_argument("--quote-prefix", type=int, default=60,
                    help="chars of the quote to match (typos beyond this are "
                         "the corpus's, not the label's)")
    args = ap.parse_args()

    probes = json.loads(Path(args.probes).read_text())["probes"]
    man = json.loads(MANIFEST.read_text())
    db = SessionLocal()
    raw = {r[0]: r[1] for r in db.execute(
        _sql("select id::text, raw_text from episodic_memory")).fetchall()}
    db.close()
    idx = {}
    for cid, d in man["conversations"].items():
        for t in d["turns"]:
            if raw.get(t["episodic_id"]):
                idx[(cid, int(t["turn_number"]))] = raw[t["episodic_id"]]

    tally = Counter()
    by_type = defaultdict(Counter)
    failures = []
    for p in probes:
        ptype = p["probe_type"]
        ev = p.get("evidence")
        gold = p.get("gold_turns") or []
        if not ev:
            tally["no evidence — UNCHECKABLE"] += 1
            by_type[ptype]["unchecked"] += 1
            continue
        ev = str(ev).strip()
        if re.fullmatch(r"[\d,\s]+", ev):
            # ⚑ THE INDEXING IS TYPE-DEPENDENT AND UNDOCUMENTED. Two readings
            # exist and picking one blindly destroys real probes:
            #   ABSOLUTE — `codex_multihop` (gold width 4): evidence values are
            #     turn numbers and appear directly in `gold_turns`.
            #   RELATIVE — `procedural` (gold width 14, windows starting 15,
            #     29, 43, …): evidence `[1, 3, 14]` against gold `[15..28]`
            #     means the 1st, 3rd and 14th turns OF THE WINDOW, i.e. 15, 17
            #     and 28. Read as absolute it looks like the gold is missing
            #     every evidence turn.
            # An absolute-first reading with a relative fallback satisfies both.
            # The first version of this check assumed absolute everywhere and
            # would have dropped 34 of 40 valid `procedural` probes.
            want = {int(x) for x in re.findall(r"\d+", ev)}
            if want <= set(gold):
                verdict = "consistent"                      # absolute reading
            elif want and all(1 <= v <= len(gold) for v in want):
                verdict = "consistent"                      # relative reading
            else:
                verdict = "GOLD MISSING EVIDENCE TURNS"
                failures.append((ptype, p["question"][:60],
                                 f"evidence {sorted(want)} fits neither absolute "
                                 f"nor relative reading of gold {gold}"))
        else:
            quotes = [q.strip() for q in ev.split("|||") if len(q.strip()) > 12]
            if not quotes:
                tally["evidence too short to check"] += 1
                by_type[ptype]["unchecked"] += 1
                continue
            blob = norm(" ".join(idx.get((p["conversation"], int(t)), "")
                                 for t in gold))
            hits = sum(1 for q in quotes
                       if norm(q)[:args.quote_prefix] in blob)
            if hits == len(quotes):
                verdict = "consistent"
            elif hits:
                verdict = "SOME quotes absent from gold"
                failures.append((ptype, p["question"][:60],
                                 f"{len(quotes)-hits} of {len(quotes)} quotes absent"))
            else:
                verdict = "QUOTE ABSENT FROM GOLD"
                failures.append((ptype, p["question"][:60],
                                 f"quote not in gold turns {gold}"))
        tally[verdict] += 1
        by_type[ptype]["consistent" if verdict == "consistent" else "BROKEN"] += 1

    tot = sum(tally.values())
    print(f"=== GOLD vs ITS OWN EVIDENCE — {tot} probes, no model used ===")
    for k, v in tally.most_common():
        print(f"  {k:<32} {v:>4}  ({100*v/tot:5.1f}%)")
    print("\nBY TYPE:")
    for t in sorted(by_type):
        c = by_type[t]
        n = sum(c.values())
        print(f"  {t:<20} n={n:>3}  consistent {c['consistent']:>3}  "
              f"BROKEN {c['BROKEN']:>3}  unchecked {c['unchecked']:>3}")
    print("\nEXAMPLES OF BROKEN GOLD:")
    for t, q, why in failures[:10]:
        print(f"  [{t}] {q}\n      {why}")

    if args.write_clean:
        badq = {f[1] for f in failures}
        keep = [p for p in probes if p["question"][:60] not in badq]
        CLEAN.write_text(json.dumps(
            {"probes": keep, "source": Path(args.probes).name,
             "dropped": len(probes) - len(keep),
             "reason": "gold inconsistent with its own evidence field"}, indent=1))
        print(f"\nwrote {CLEAN}  ({len(keep)} kept, "
              f"{len(probes)-len(keep)} dropped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
