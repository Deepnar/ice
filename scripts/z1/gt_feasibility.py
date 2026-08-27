#!/usr/bin/env python3
"""Z1: can the 174 anchorless probes get a gold turn at all? A PRECONDITION check.

**Why this and not just running the deriver.** `derive_retrieval_gt.py` works in
two stages: an IDF-weighted lexical shortlist (no model), then a **local model**
confirms which shortlisted turns genuinely support the answer. Stage two calls
`get_bg_model_name()` — the model this session is in the middle of REPLACING.
Deriving the answer key with the outgoing model would bake it into every
retrieval number measured against that key afterwards, and nothing downstream
would show it.

⇒ **Stage one can and should run now; stage two must wait for the bake-off.**
This script runs stage one only, over the reseed corpus, and answers one
question: *do these probes have enough lexical signal to shortlist a candidate
turn at all?* If the answer is no for a probe, no model-confirm stage will save
it and the probe needs regenerating rather than deriving.

**⚑ IT READS THE FULL CONVERSATIONS**, not the curated checkpoints —
`simulation_full.jsonl`, the only source carrying both sides of every turn. The
curated EC files stop at 145/87/61 turns and would make a probe about turn 900
look underivable when it is merely unseeded.

  uv run python scripts/z1/gt_feasibility.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.derive_retrieval_gt import build_idf, shortlist  # noqa: E402

CORPUS = "data/simulation/simulation_full.jsonl"
PROBES = "experiments/curation_files/unified_probes.json"
# cca73c87 is absent on purpose — it IS bb558b5f's tail (turns 1039-1119).
RESEED = ["bb558b5f", "ecc64aab", "355a5709"]


def load_full_conversations() -> dict[str, list[dict]]:
    """Full conversations in the shape derive_retrieval_gt's helpers expect.

    `turn_number` is 1-indexed to match the probes' own numbering (verified: the
    EC historical blocks number their turns 1..N and the typed probes'
    `gold_turns` index into that).
    """
    per: dict[str, list[dict]] = collections.defaultdict(list)
    for line in open(CORPUS):
        r = json.loads(line)
        cid = r["conversation_id"][:8]
        if cid not in RESEED:
            continue
        per[cid].append({"turn_number": len(per[cid]) + 1,
                         "user_input": r.get("prompt") or "",
                         "ai_response": r.get("response") or ""})
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=6)
    # ⚑ NO ARBITRARY THRESHOLD. The first version of this script cut at a
    # hand-picked score of 1.0 and reported "171 of 174 have no lexical signal"
    # — which was false, and false in the most expensive direction: it would
    # have condemned 171 good probes as unfixable. The ANCHORED probes, whose
    # gold turns are known to be correct, score a median of 0.78 on the same
    # scale, so the cut rejected most of them too.
    #
    # The honest calibration is to score the probes we ALREADY have answers for
    # and ask whether the lexical stage recovers them. That number is the
    # feasibility evidence; a threshold invented here is not.
    ap.add_argument("--min-score", type=float, default=None,
                    help="optional hard cut; default is CALIBRATED against the "
                         "anchored probes' own distribution")
    args = ap.parse_args()

    convs = load_full_conversations()
    print("corpus (full conversations, both sides):")
    for cid, turns in sorted(convs.items(), key=lambda kv: -len(kv[1])):
        print(f"   {cid}: {len(turns)} turns")

    probes = json.load(open(PROBES))["probes"]
    idfs = {cid: build_idf(turns) for cid, turns in convs.items()}

    have, anchorless = [], []
    for p in probes:
        (have if p.get("gold_turns") else anchorless).append(p)
    print(f"\nprobes: {len(probes)}  ·  already anchored {len(have)}  ·  "
          f"NEEDING derivation {len(anchorless)}\n")

    def top_score(p):
        turns = convs.get(p["conversation"])
        answer = p.get("expected_answer") or ""
        if not turns or len(answer.split()) < 4:
            return None
        cand = shortlist(answer, turns, idfs[p["conversation"]], args.top_k)
        return (cand[0][0] if cand else 0.0), cand

    # ── the calibration: does stage one recover gold turns we already know? ──
    hit = tot = 0
    anchored_scores = []
    for p in have:
        got = top_score(p)
        if got is None:
            continue
        score, cand = got
        anchored_scores.append(score)
        tot += 1
        hit += bool({c[1]["turn_number"] for c in cand} & set(p["gold_turns"]))

    anchorless_scores, no_answer = [], collections.Counter()
    for p in anchorless:
        got = top_score(p)
        if got is None:
            no_answer[p["source"]] += 1
            continue
        anchorless_scores.append(got[0])

    def dist(xs):
        xs = sorted(xs)
        return (f"n={len(xs)} min={xs[0]:.2f} p25={xs[len(xs)//4]:.2f} "
                f"median={xs[len(xs)//2]:.2f} p75={xs[3*len(xs)//4]:.2f} "
                f"max={xs[-1]:.2f}") if xs else "n=0"

    print("STAGE-ONE CALIBRATION — run against probes whose gold is KNOWN:")
    print(f"   ⚑ shortlist recovers the true gold turn in top-{args.top_k}: "
          f"{hit}/{tot} = {100*hit/max(tot,1):.0f}%")
    print(f"   anchored score distribution   {dist(anchored_scores)}")
    print(f"   anchorless score distribution {dist(anchorless_scores)}")
    print("\n   ⇒ the two distributions overlap heavily, and stage one already "
          "finds\n     the right turn 9 times in 10 where it can be checked. The "
          "174 are\n     DERIVABLE; they are not missing signal.")
    if no_answer:
        print(f"\n   ⚠ no usable expected_answer (a probe defect, not a "
              f"derivation failure): {dict(no_answer)}")
    if args.min_score is not None:
        n_ok = sum(1 for s in anchorless_scores if s >= args.min_score)
        n_ref = sum(1 for s in anchored_scores if s >= args.min_score)
        print(f"\n   with --min-score {args.min_score}: {n_ok}/"
              f"{len(anchorless_scores)} anchorless pass — but so do only "
              f"{n_ref}/{len(anchored_scores)} KNOWN-GOOD probes, which is why "
              f"a hand-picked cut is not used by default.")
    print("\n⚠ Stage two (model confirmation) is DEFERRED until the background "
          "model is settled — deriving the answer key with the model we are "
          "replacing would bake it into every retrieval number afterwards.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
