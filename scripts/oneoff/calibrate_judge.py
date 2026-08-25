#!/usr/bin/env python3
"""Score any candidate judge against the maintainer's own 15 labels.

⚑ WHY THIS EXISTS. Judges have been swapped three times in two days — on
availability, on price, on speed — and each swap silently changes every number
downstream. This is the gate: a model does not become the judge because it is
up, it becomes the judge because it agrees with the one set of human labels
this project has.

The 15 labels came from a BLIND calibration sheet (the maintainer wrote a label
per triplet without seeing any model's answer). Measured so far:

    DeepSeek     12/15  80%
    muse-spark   11/15  73%
    Gemini        9/15  60%
    ChatGPT       7/15  47%

⚠ n=15. This separates "usable" from "broken", not 73% from 80%.

  uv run python scripts/oneoff/calibrate_judge.py --model mimo-v2.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx                                       # noqa: E402

from src.api.config import settings                # noqa: E402

JUD = Path("experiments/curation_files/judgements")
SRC = Path("experiments/curation_files/wrong_review")

# the maintainer's blind labels, 2026-08-23
HUMAN = {
    ("animator", "uses", "narrator"): "wrong",
    ("undocumented neurodivergent person", "has_property", "traumatized"): "correct",
    ("goo", "made_by", "feather"): "wrong",
    ("lock down", "lives_in", "fourteen_year_old"): "reversed",
    ("arch", "endorses", "ai"): "wrong",
    ("person", "is_large", "story"): "malformed",
    ("emotional state", "lives_in", "flaw"): "reversed",
    ("boy", "is", "orien"): "wrong",
    ("story", "shares", "boy"): "wrong",
    ("backup", "has_backup", "glm-4.7-flash"): "wrong",
    ("villainess", "contains", "file 1"): "reversed",
    ("girl", "is", "lethe"): "reversed",
    ("representation", "felt", "people"): "reversed",
    ("chance_to_hear", "was", "today"): "reversed",
    ("the system", "was_available_as_output_of", "peace"): "wrong",
}


def load_system() -> str:
    """The judge rubric, read from judge_codex.py so the two cannot drift."""
    txt = (Path("scripts/z1/judge_codex.py")).read_text()
    return txt.split('SYSTEM = """')[1].split('"""')[0].replace("\\\n", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", choices=["probe", "coe"], default="probe")
    args = ap.parse_args()

    if args.backend == "coe":
        base, key = settings.coe_api_base_url, settings.coe_api_key
    else:
        base, key = settings.probe_api_base_url, settings.probe_api_key

    system = load_system()
    verdicts = json.load(open(JUD / "codex_quality_dir-false-run1-muse.json"))["verdicts"]
    src = {}
    for f in SRC.glob("*.wrong.json"):
        for x in json.load(open(f))["wrong"]:
            src[x["edge_id"]] = x["source"]

    targets = []
    for v in verdicts:
        keyt = (v["subj"], v["rel"], v["obj"])
        if keyt in HUMAN and v["edge_id"] in src:
            targets.append((keyt, v, src[v["edge_id"]]))
    seen, uniq = set(), []
    for keyt, v, s in targets:
        if keyt not in seen:
            seen.add(keyt)
            uniq.append((keyt, v, s))
    print(f"calibrating {args.model} on {len(uniq)} of {len(HUMAN)} human-labelled triplets\n")

    client = httpx.Client(base_url=base, headers={"Authorization": f"Bearer {key}"})
    rows, t0 = [], time.time()
    for keyt, v, source in uniq:
        user = (f"SOURCE TURN:\n{source}\n\nTRIPLETS (1):\n"
                f"1. {v['subj']} --{v['rel']}--> {v['obj']}")
        got = None
        for attempt in range(4):
            try:
                r = client.post("/chat/completions", json={
                    "model": args.model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": 0.0, "max_tokens": 2400,
                    "response_format": {"type": "json_object"},
                }, timeout=180)
                r.raise_for_status()
                c = r.json()["choices"][0]["message"].get("content")
                if c:
                    got = json.loads(c)["verdicts"][0]
                    break
            except Exception:
                pass
            time.sleep(1.5 * (attempt + 1))
        lab = (got or {}).get("label", "FAILED")
        rows.append((keyt, HUMAN[keyt], lab, (got or {}).get("why", "")))

    ok = sum(1 for _, h, m, _ in rows if h == m)
    print(f"{'triplet':<50}{'HUMAN':<11}{'MODEL':<11}")
    print("-" * 74)
    for (s, r_, o), h, m, why in rows:
        trip = f"{s} --{r_}--> {o}"
        print(f"{trip[:48]:<50}{h:<11}{m:<11}{'OK' if h == m else 'x'}")
        if h != m and why:
            print(f"    model said: {why[:80]}")
    print(f"\n{args.model}: {ok}/{len(rows)} = {100*ok/len(rows):.0f}%  "
          f"({time.time()-t0:.0f}s total, {(time.time()-t0)/max(len(rows),1):.1f}s/call)")

    print("\nwhere it differs from the human, by direction:")
    for (h, m), n in Counter((h, m) for _, h, m, _ in rows if h != m).most_common():
        print(f"  human={h:<11} model={m:<11} x{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
