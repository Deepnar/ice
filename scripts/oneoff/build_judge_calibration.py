#!/usr/bin/env python3
"""Build a BLIND human calibration sheet for a judgement run.

⚑ WHY BLIND. The point is to measure how far the judge is from a person, so the
person must not see the judge's answer first — showing it turns the task from
"label this" into "do you agree", and agreement is the thing being measured.
The judge's verdicts are written to a separate ANSWER KEY the reviewer does not
open until they are done.

⚑ WHY THE FULL TURN. The previous review sheet gave 110-char windows on turns
whose median length is 5,000 chars, and the maintainer correctly said that was
not enough to judge anything. Full source, no cap — the same thing the judge got.

Sampling is seeded and stratified in two blocks, both stated on the sheet:
  * BLOCK A - a uniform random draw. This is what estimates the true rates.
  * BLOCK B - drawn only from the judge's `reversed` calls, because `reversed`
    is the disputed label (46.1% vs 19.7% vs 32.0% across three judges) and a
    uniform draw would not put enough of it in front of a person.

  uv run python scripts/oneoff/build_judge_calibration.py --arm dir-false-run1-muse
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

JUD = Path("experiments/curation_files/judgements")
SRC = Path("experiments/curation_files/wrong_review")
OUT = Path("experiments/curation_files/calibration")

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
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n-random", type=int, default=10)
    ap.add_argument("--n-reversed", type=int, default=5)
    ap.add_argument("--seed", default="calib-20260823")
    args = ap.parse_args()

    verdicts = json.load(open(JUD / f"codex_quality_{args.arm}.json"))["verdicts"]

    # source turns: prefer the judgement's own turn_id join, else the dumped file
    src = {}
    for f in SRC.glob("*.wrong.json"):
        for v in json.load(open(f))["wrong"]:
            src[v["edge_id"]] = v["source"]
    for f in SRC.glob("*.all.json"):
        for v in json.load(open(f))["wrong"]:
            src[v["edge_id"]] = v["source"]

    pool = [v for v in verdicts if v["edge_id"] in src]
    if not pool:
        print("no source text available — run dump_wrong_verdicts.py first")
        return 1

    rng = random.Random(args.seed)
    block_a = rng.sample(pool, min(args.n_random, len(pool)))
    taken = {v["edge_id"] for v in block_a}
    rev_pool = [v for v in pool if v["label"] == "reversed" and v["edge_id"] not in taken]
    block_b = rng.sample(rev_pool, min(args.n_reversed, len(rev_pool)))

    OUT.mkdir(parents=True, exist_ok=True)
    L, key = [], []
    w = L.append

    w(f"# Judge calibration — `{args.arm}`\n")
    w("**⚠ GITIGNORED — contains raw corpus text. Never commit.**\n")
    w("## What to do\n")
    w("For each triplet below: read the source turn, then **write your own "
      "label** on the `YOUR LABEL:` line. Use the rubric — it is the same one "
      "the judge was given.\n")
    w("**Do not open the answer key until you are finished.** The judge's "
      "verdicts are in `ANSWER_KEY.md` in this folder. You are not being asked "
      "whether you agree with it — you are being asked what the right label is, "
      "so that how often it matches can be measured.\n")
    w(f"{len(block_a)} + {len(block_b)} = **{len(block_a)+len(block_b)} triplets.** "
      f"Coverage of only a small fraction, so this measures whether the judge is "
      f"roughly right, not a precise error rate.\n")
    w("```text")
    w(RUBRIC)
    w("```\n")
    w("---\n")

    for title, note, block in (
        ("BLOCK A — uniform random draw",
         "Unfiltered. These estimate the real rates, so judge them as they come.",
         block_a),
        ("BLOCK B — drawn only from triplets the judge called `reversed`",
         "`reversed` is the disputed label — three judges put it at 46.1%, 19.7% "
         "and 32.0%. A uniform draw would not show enough of it. ⚠ Expect these "
         "to be reversals IF the judge is right; that is exactly what is under test.",
         block_b)):
        w(f"\n# {title}\n")
        w(f"*{note}*\n")
        for i, v in enumerate(block, 1):
            neg = "  ⛔NEGATED" if v.get("negated") else ""
            w(f"\n---\n")
            w(f"## {title[6]}{i}.  `{v['subj']} --{v['rel']}--> {v['obj']}`{neg}\n")
            w(f"**YOUR LABEL:** ______________\n")
            w(f"<details><summary>source turn ({len(src[v['edge_id']]):,} chars)"
              f"</summary>\n")
            w("```text")
            w(src[v["edge_id"]])
            w("```")
            w("</details>\n")
            key.append((f"{title[6]}{i}", v))

    (OUT / f"{args.arm}.CALIBRATION.md").write_text("\n".join(L))

    K = [f"# ANSWER KEY — `{args.arm}`\n",
         "**Do not read until the calibration sheet is filled in.**\n",
         "| # | triplet | judge label | judge reason |", "|---|---|---|---|"]
    for tag, v in key:
        K.append(f"| {tag} | `{v['subj']} --{v['rel']}--> {v['obj']}` | "
                 f"**{v['label']}** | {v['why']} |")
    (OUT / "ANSWER_KEY.md").write_text("\n".join(K))

    print(f"wrote {OUT}/{args.arm}.CALIBRATION.md  ({len(key)} triplets)")
    print(f"wrote {OUT}/ANSWER_KEY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
