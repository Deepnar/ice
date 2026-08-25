#!/usr/bin/env python3
"""Three-way blind sheet: ICE vs NuExtract3-bare vs NuExtract3-with-guards.

⚑ WHAT IS HIDDEN IS THE ARM. All three arms ran on the SAME 60 turns from the
G63 sweep, so a turn's difficulty cannot land on one arm. Facts are drawn from
turns where all three produced something, shuffled with a fixed seed, and the
key is written to a separate file.

⚑ WHY THREE AND NOT TWO. `nu_combined` adds the relation vocabulary, and G45
measured that **67.8% of real relations fall outside that list**. Forcing them
onto it is exactly how a wrong relation ends up looking correct — so the config
with the better SHAPE may have worse TRUTH. That is the question this round
exists to answer, and it cannot be answered from the sweep's counts.

⚠ INTERVAL. At 10 facts per arm the 95% interval on a rate is roughly ±19 pts.
That separates 10% from 70%; it does NOT separate two arms landing within ~20
points of each other. Use --per-arm 20 if the NuExtract3 arms need separating.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

IN = Path("experiments/curation_files/extractor_ab/g63_sweep.json")
OUT = Path("experiments/curation_files/extractor_ab")

ALL_ARMS = {
    "ice_baseline": "ICE (qwen3:4b-instruct, production path)",
    "nu_ref": "NuExtract3 (bare template + entity list)",
    "nu_combined": "NuExtract3 (+ vocabulary + canonicalisation + chunking)",
    "nu_vocab": "NuExtract3 + relation vocabulary ONLY",
    "nu_canonrule": "NuExtract3 + canonicalisation rule ONLY",
    "nu_chunked": "NuExtract3 + chunking ONLY",
    "nu_micro": "NuExtract3 + micro-NER entity list",
    "nu_tok1200": "NuExtract3 + 1200-token cap",
    "nu_novocab": "NuExtract3 + canonicalisation + chunking, NO vocabulary",
}
# default round; override with --arms
ARMS = {k: ALL_ARMS[k] for k in
        ("ice_baseline", "nu_ref", "nu_combined")}

RUBRIC = """correct     - true of the source, and the direction is right.
reversed    - subject and object are SWAPPED. Both entities and the relation are
              right, but the fact runs the other way.
wrong       - not supported by the source, or contradicts it.
vacuous     - true but says nothing (`have`, `are`, `in` joining two things
              pointlessly, or the object just restates the subject).
malformed   - subject or object is not a thing (a fragment, a clause), or the
              relation is a clause rather than a predicate.
unjudgeable - the source does not contain enough to decide."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arm", type=int, default=10)
    ap.add_argument("--seed", default="threeway-20260824")
    ap.add_argument("--arms", default="", help="comma-separated condition names")
    ap.add_argument("--out", default="THREEWAY", help="output file prefix")
    args = ap.parse_args()

    global ARMS
    if args.arms:
        ARMS = {k: ALL_ARMS[k] for k in args.arms.split(",") if k in ALL_ARMS}

    d = json.load(open(IN))
    conds = d["conditions"]
    missing = [a for a in ARMS if a not in conds]
    if missing:
        print(f"missing conditions in sweep artifact: {missing}")
        return 1

    by_arm_turn = {}
    for arm in ARMS:
        m = {}
        for f in conds[arm]["facts"]:
            m.setdefault(f["turn_id"], []).append(f)
        by_arm_turn[arm] = m

    shared = set(by_arm_turn[list(ARMS)[0]])
    for arm in ARMS:
        shared &= set(by_arm_turn[arm])
    shared = sorted(shared)
    if not shared:
        print("no turn has facts from all three arms")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(shared)

    # spread across as many turns as possible: 1-2 facts per arm per turn
    per_turn = 2
    n_turns = max(1, args.per_arm // per_turn)
    turns = shared[:n_turns]

    picked = []
    for t in turns:
        for arm in ARMS:
            pool = by_arm_turn[arm][t]
            picked += [{**f, "_arm": arm}
                       for f in rng.sample(pool, min(per_turn, len(pool)))]
    rng.shuffle(picked)

    turn_text = {}
    try:
        import os
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..")))
        from sqlalchemy import text as sqltext

        from src.api.db import SessionLocal
        db = SessionLocal()
        rows = db.execute(sqltext(
            "select id::text as id, raw_text from episodic_memory "
            "where id::text = any(:ids)"), {"ids": turns}).fetchall()
        db.close()
        turn_text = {r.id: r.raw_text for r in rows}
    except Exception as exc:
        print(f"warn: source turns unavailable ({type(exc).__name__})")

    L, key = [], []
    w = L.append
    w("# Three-way blind sheet — which extractor config is TRUE?\n")
    w("**⚠ GITIGNORED — raw corpus text. Never commit.**\n")
    w("## What to do\n")
    w(f"**{len(picked)} facts from {len(turns)} turns.** Read each turn, then "
      f"label every fact under it on the `YOUR LABEL:` line.\n")
    w(f"**Three different extractors produced these and you are not told which "
      "is which.** They all ran on these same turns. The key is in "
      "`{args.out}_KEY.md` — **do not open it until you are done.**\n")
    w("⚑ **Judge the fact against the turn, nothing else.** A fact can be neatly "
      "formatted and still false; that is exactly what this round is testing.\n")
    w("```text")
    w(RUBRIC)
    w("```\n")

    n = 0
    for t in turns:
        mine = [f for f in picked if f["turn_id"] == t]
        if not mine:
            continue
        w(f"\n---\n\n# Turn {t[:8]}\n")
        if turn_text.get(t):
            w(f"<details open><summary><b>source turn "
              f"({len(turn_text[t]):,} chars) — read first</b></summary>\n")
            w("```text")
            w(turn_text[t])
            w("```")
            w("</details>\n")
        for f in mine:
            n += 1
            w(f"\n**{n}.** `{f['subject']} --{f['relation']}--> {f['object']}`\n")
            w("**YOUR LABEL:** ______________\n")
            key.append((n, f))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.out}_SHEET.md").write_text("\n".join(L))

    K = [f"# KEY — {args.out}\n",
         "**Do not read until the sheet is filled in.**\n",
         "| # | fact | arm |", "|---|---|---|"]
    for i, f in key:
        K.append(f"| {i} | `{f['subject']} --{f['relation']}--> {f['object']}` "
                 f"| **{ARMS[f['_arm']]}** |")
    (OUT / f"{args.out}_KEY.md").write_text("\n".join(K))

    counts = {a: sum(1 for _, f in key if f["_arm"] == a) for a in ARMS}
    print(f"wrote {OUT}/{args.out}_SHEET.md — {n} facts from {len(turns)} turns")
    for a, c in counts.items():
        print(f"   {ARMS[a]:<52} {c}")
    print(f"wrote {OUT}/{args.out}_KEY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
