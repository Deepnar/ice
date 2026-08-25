#!/usr/bin/env python3
"""Does asking direction as a FORCED CHOICE fix a model that is blind to it?

⚑ THE ONE VARIABLE IS FRAMING. Same model (CoE `Qwen3.6-35B-A3B`), same
triplets, same source turns. Only the question changes:

  free-form  - the full judge rubric, "label this triplet"     <- known to fail
  binary     - "which of these two is true: X->R->Y or Y->R->X, or neither"

The CoE model was measured blind to direction in free-form: on 50 sampled
triplets it answered `correct` while its own justification described the
reverse. If the binary framing recovers that, the defect is the TASK SHAPE, not
the model — and a cheap verification pass after extraction becomes viable,
because this gateway is free at ~2s/call.

Ground truth is muse-spark's labels, which are NOT gold: measured precision on
`reversed` is 75% (6/8) against the maintainer's own labels, recall 100% (6/6).
So read this as "does binary framing recover what muse found", and treat the
15 human-labelled triplets, reported separately, as the only real gold.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx                                       # noqa: E402

from src.api.config import settings                # noqa: E402

JUD = Path("experiments/curation_files/judgements")
SRC = Path("experiments/curation_files/wrong_review")
OUT = Path("experiments/curation_files/score_runs")

SYSTEM = """You decide which DIRECTION a fact runs in a conversation turn.

You get the source turn and two candidate statements, A and B, which are the \
same two things and the same relation with the subject and object exchanged.

Answer:
  "A"       - A is true of the source
  "B"       - B is true of the source
  "neither" - neither direction is supported, or the relation itself is wrong

Judge ONLY against the source text. Do not use outside knowledge.
Return ONLY {"answer": "A"|"B"|"neither", "why": "<max 12 words>"}"""

# the maintainer's own labels — the only gold in this project
HUMAN = {
    "A1": "wrong", "A2": "correct", "A3": "wrong", "A4": "reversed",
    "A5": "wrong", "A6": "malformed", "A7": "reversed", "A8": "wrong",
    "A9": "wrong", "A10": "wrong", "B1": "reversed", "B2": "reversed",
    "B3": "reversed", "B4": "reversed", "B5": "wrong",
}
HUMAN_TRIPLETS = {
    ("animator", "uses", "narrator"): "A1",
    ("undocumented neurodivergent person", "has_property", "traumatized"): "A2",
    ("goo", "made_by", "feather"): "A3",
    ("lock down", "lives_in", "fourteen_year_old"): "A4",
    ("arch", "endorses", "ai"): "A5",
    ("person", "is_large", "story"): "A6",
    ("emotional state", "lives_in", "flaw"): "A7",
    ("boy", "is", "orien"): "A8",
    ("story", "shares", "boy"): "A9",
    ("backup", "has_backup", "glm-4.7-flash"): "A10",
    ("villainess", "contains", "file 1"): "B1",
    ("girl", "is", "lethe"): "B2",
    ("representation", "felt", "people"): "B3",
    ("chance_to_hear", "was", "today"): "B4",
    ("the system", "was_available_as_output_of", "peace"): "B5",
}


def ask(client, model, src, v):
    """⚑ RANDOMISE which side the stored direction is on. Otherwise 'A' is
    always the extractor's output and a model with a position bias scores as
    if it understood the question."""
    stored = f"{v['subj']} --{v['rel']}--> {v['obj']}"
    swapped = f"{v['obj']} --{v['rel']}--> {v['subj']}"
    stored_is_a = random.Random(v["edge_id"]).random() < 0.5
    a, b = (stored, swapped) if stored_is_a else (swapped, stored)
    user = (f"SOURCE TURN:\n{src}\n\n"
            f"A) {a}\nB) {b}\n\nWhich direction is true of the source?")
    for attempt in range(4):
        try:
            r = client.post("/chat/completions", json={
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": 400,
                "response_format": {"type": "json_object"},
            }, timeout=180)
            r.raise_for_status()
            c = r.json()["choices"][0]["message"].get("content")
            if c:
                ans = str(json.loads(c).get("answer", "")).strip().lower()
                if ans in ("a", "b", "neither"):
                    # translate to a verdict about the STORED triplet
                    if ans == "neither":
                        return "neither", json.loads(c).get("why", "")
                    picked_stored = (ans == "a") == stored_is_a
                    return ("keep" if picked_stored else "flip",
                            json.loads(c).get("why", ""))
        except Exception:
            pass
        time.sleep(1.2 * (attempt + 1))
    return None, "failed"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["coe", "probe"], default="coe")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default="coe", help="suffix for the output file")
    args = ap.parse_args()
    if args.backend == "coe":
        b, k = settings.coe_api_base_url, settings.coe_api_key
        m = args.model or settings.coe_model
    else:
        b, k = settings.probe_api_base_url, settings.probe_api_key
        m = args.model or settings.probe_model
    V = json.load(open(JUD / "codex_quality_dir-false-run1-muse.json"))["verdicts"]
    src = {}
    for f in SRC.glob("*.wrong.json"):
        for x in json.load(open(f))["wrong"]:
            src[x["edge_id"]] = x["source"]

    pool = [v for v in V if v["edge_id"] in src
            and v["label"] in ("reversed", "correct")]
    rng = random.Random("probe-20260823")
    rev = [v for v in pool if v["label"] == "reversed"]
    cor = [v for v in pool if v["label"] == "correct"]
    sample = rng.sample(rev, min(60, len(rev))) + rng.sample(cor, min(40, len(cor)))
    rng.shuffle(sample)
    print(f"probing {len(sample)} triplets "
          f"({sum(1 for v in sample if v['label']=='reversed')} muse-reversed, "
          f"{sum(1 for v in sample if v['label']=='correct')} muse-correct) "
          f"· model {m}", flush=True)

    client = httpx.Client(base_url=b, headers={"Authorization": f"Bearer {k}"})
    rows, fails = [], 0
    for i, v in enumerate(sample, 1):
        ans, why = ask(client, m, src[v["edge_id"]], v)
        if ans is None:
            fails += 1
            continue
        rows.append({**{kk: v[kk] for kk in ("edge_id", "subj", "rel", "obj")},
                     "muse": v["label"], "binary": ans, "why": why})
        if i % 25 == 0:
            print(f"  {i}/{len(sample)}", flush=True)

    print(f"\n=== BINARY DIRECTION PROBE — {len(rows)} judged, {fails} failed ===")
    for lab in ("reversed", "correct"):
        sub = [r for r in rows if r["muse"] == lab]
        if not sub:
            continue
        c = Counter(r["binary"] for r in sub)
        want = "flip" if lab == "reversed" else "keep"
        hit = 100 * c[want] / len(sub)
        print(f"  muse said {lab:<9} n={len(sub):>3}  "
              f"binary said keep {c['keep']:>3} · flip {c['flip']:>3} · "
              f"neither {c['neither']:>3}   ⇒ agrees {hit:.0f}%")

    agree = sum(1 for r in rows
                if (r["muse"] == "reversed") == (r["binary"] == "flip")
                and r["binary"] != "neither")
    n = len([r for r in rows if r["binary"] != "neither"])
    if n:
        print(f"\n  OVERALL (excluding 'neither'): {agree}/{n} = {100*agree/n:.0f}%")

    gold = [r for r in rows if (r["subj"], r["rel"], r["obj"]) in HUMAN_TRIPLETS]
    if gold:
        print(f"\n=== vs the MAINTAINER'S OWN LABELS (the only gold), n={len(gold)} ===")
        ok = 0
        for r in gold:
            tag = HUMAN_TRIPLETS[(r["subj"], r["rel"], r["obj"])]
            h = HUMAN[tag]
            want = "flip" if h == "reversed" else "keep"
            good = r["binary"] == want
            ok += good
            print(f"  {tag:<4} human={h:<10} binary={r['binary']:<8} "
                  f"{'✓' if good else '✗'}  {r['subj'][:22]} --{r['rel']}--> {r['obj'][:18]}")
        print(f"  ⇒ {ok}/{len(gold)}")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"direction_binary_probe_{args.tag}.json"
    p.write_text(json.dumps({"model": m, "n": len(rows), "failed": fails,
                             "rows": rows}, indent=1))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
