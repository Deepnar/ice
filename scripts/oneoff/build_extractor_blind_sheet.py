#!/usr/bin/env python3
"""Blind labelling sheet: ICE's facts vs NuExtract3's, source hidden.

⚑ WHAT IS BLIND HERE IS THE EXTRACTOR, not a model's verdict. Yesterday's sheet
hid the judge's answer because the question was "is the judge right". Today the
question is "which extractor is better", so the thing that must be hidden is
which extractor wrote each fact. The key is written to a separate file.

⚑ BALANCED AND PAIRED BY TURN. Equal facts from each extractor, drawn from the
SAME turns, so the reader judges both against one reading of one turn and no
turn's difficulty lands on one extractor. Facts are shuffled with a fixed seed.

Depends on `extractor_head_to_head.py` having written its artifact.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

IN = Path("experiments/curation_files/extractor_ab/head_to_head.json")
OUT = Path("experiments/curation_files/extractor_ab")

RUBRIC = """correct     - true of the source, and the direction is right.
reversed    - subject and object are SWAPPED. Both entities and the relation are
              right, but the fact runs the other way.
              e.g. source says "File 1 contains the villainess" and the fact
              says `villainess --contains--> file 1`.
wrong       - not supported by the source, or contradicts it.
vacuous     - true but says nothing (`have`, `are`, `in` joining two things
              pointlessly, or the object just restates the subject).
malformed   - subject or object is not a thing (a fragment, a clause), or the
              relation is a clause rather than a predicate.
unjudgeable - the source does not contain enough to decide."""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-side", type=int, default=10,
                    help="facts per extractor (total = 2x this)")
    ap.add_argument("--seed", default="blind-20260824")
    args = ap.parse_args()

    d = json.load(open(IN))
    facts = d["facts"]
    src_by_turn = {}
    by_turn = defaultdict(lambda: {"ice": [], "nu": []})
    for f in facts:
        by_turn[f["turn_id"]][f["src"]].append(f)

    # only turns where BOTH produced something — otherwise the pairing breaks
    usable = [t for t, v in by_turn.items() if v["ice"] and v["nu"]]
    rng = random.Random(args.seed)
    rng.shuffle(usable)

    per_turn_each = 2
    n_turns = max(1, args.per_side // per_turn_each)
    chosen_turns = usable[:n_turns]

    picked = []
    for t in chosen_turns:
        for side in ("ice", "nu"):
            pool = by_turn[t][side]
            picked += rng.sample(pool, min(per_turn_each, len(pool)))
    rng.shuffle(picked)

    # source text: pulled from the per-turn record if present, else the DB
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
            "where id::text = any(:ids)"), {"ids": chosen_turns}).fetchall()
        db.close()
        turn_text = {r.id: r.raw_text for r in rows}
    except Exception as exc:
        print(f"warn: could not load source turns ({type(exc).__name__})")

    L, key = [], []
    w = L.append
    w("# Which extractor is better — blind sheet\n")
    w("**⚠ GITIGNORED — raw corpus text. Never commit.**\n")
    w("## What to do\n")
    w(f"**{len(picked)} facts, drawn from {len(chosen_turns)} turns.** Each turn's "
      f"full text is below its facts. For every fact, write a label on the "
      f"`YOUR LABEL:` line.\n")
    w("**You are not told which extractor produced which fact** — that is the "
      "whole point. Half came from ICE's current extractor, half from "
      "NuExtract3, drawn from the same turns and shuffled. The key is in "
      "`BLIND_KEY.md`; **do not open it until you are done.**\n")
    w("```text")
    w(RUBRIC)
    w("```\n")
    w("---\n")

    n = 0
    for t in chosen_turns:
        mine = [f for f in picked if f["turn_id"] == t]
        if not mine:
            continue
        w(f"\n# Turn {t[:8]}\n")
        if turn_text.get(t):
            w(f"<details open><summary><b>source turn "
              f"({len(turn_text[t]):,} chars) — read this first</b></summary>\n")
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
    (OUT / "BLIND_SHEET.md").write_text("\n".join(L))

    K = ["# KEY — which extractor wrote each fact\n",
         "**Do not read until the sheet is filled in.**\n",
         "| # | fact | extractor |", "|---|---|---|"]
    for i, f in key:
        name = "ICE (qwen3:4b-instruct)" if f["src"] == "ice" else "NuExtract3"
        K.append(f"| {i} | `{f['subject']} --{f['relation']}--> {f['object']}` | **{name}** |")
    (OUT / "BLIND_KEY.md").write_text("\n".join(K))

    sides = {"ice": sum(1 for _, f in key if f["src"] == "ice"),
             "nu": sum(1 for _, f in key if f["src"] == "nu")}
    print(f"wrote {OUT}/BLIND_SHEET.md  ({n} facts from {len(chosen_turns)} turns; "
          f"ICE {sides['ice']} · NuExtract3 {sides['nu']})")
    print(f"wrote {OUT}/BLIND_KEY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
