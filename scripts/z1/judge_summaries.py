#!/usr/bin/env python3
"""Z1 stage 2: is a summary FAITHFUL? — and can the judge even tell?

**The gap this closes.** `bg_model_bakeoff.py` scores summaries on
`summary_coverage`, `no_prose`, `no_abstract`, `empty`. Every one of those is a
PRESENCE or FORMAT check. None can see the failure that matters: a fluent,
correctly-formatted summary that says something the turn never said. That is the
one defect that corrupts memory rather than thinning it, because the summary
REPLACES the raw turn in the prompt when coverage clears the gate.

**⚑ THE JUDGE IS GATED BEFORE IT IS USED, and not by a human labelling set.**
`calibrate_judge.py` gates the graph judge against 15 maintainer labels. No such
set exists for summaries, and inventing one would be the author of the test
writing its answer key. So the gate here is **injected corruption**:

  * take a REAL summary of a REAL turn, produced by the production path;
  * corrupt it in a way we control — splice in a fabricated sentence, or delete
    the sentence carrying the must-terms;
  * ask the judge to score the corrupted version and the untouched one.

A judge that cannot flag a fabrication **we planted** cannot be trusted to flag
one we did not. The detection rate on planted defects is the judge's
sensitivity, and it is measured before any model is ranked. This is the same
logic as the falsification run against the bake-off itself: *would this number
look the same if the thing under test were broken?*

⚠ **Synthetic corruption is a FLOOR, not a proxy for real failure.** A spliced
sentence is more obvious than a subtly wrong paraphrase. A judge that fails here
is disqualified; one that passes is merely not disqualified.

  uv run python scripts/z1/judge_summaries.py --calibrate
  uv run python scripts/z1/judge_summaries.py --models ministral-3:8b,qwen3:4b-instruct
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx  # noqa: E402

from src.api.config import settings  # noqa: E402

OUT_DIR = "experiments/curation_files/bakeoff"

SYSTEM = """You judge whether a SUMMARY is faithful to the SOURCE TURN it summarises.

Return JSON: {"verdicts":[{"label":"...","why":"..."}]}

Labels, in priority order — use the FIRST that applies:
  fabricated  - the summary states something the source turn does not support.
                A claim, a number, a name, a relationship, an outcome that is
                not in the source. This is the most serious defect.
  incomplete  - nothing invented, but a fact central to the turn is missing.
  faithful    - everything stated is supported by the source, and the substance
                is preserved.

Judge ONLY against the source turn. Do not reward fluency, length or structure.
A summary that reads well and invents one detail is `fabricated`."""

FABRICATIONS = [
    " The user confirmed they had already deployed this to production on Tuesday.",
    " They mentioned their manager approved a budget of $40,000 for it.",
    " The assistant recommended switching to PostgreSQL 16 instead.",
    " This was the third time the same error had occurred that week.",
]


def judge_one(client, model, source: str, summary: str, retries: int = 3,
              system: str = None):
    # ⚑⚑ DO NOT TRUNCATE THE SOURCE. This function first sent `source[:4000]`
    # and `summary[:2000]`, and the maintainer caught it: **this project has
    # already killed one result this exact way** — the answer-level verdicts
    # (31-30, 58%/31% both_failed) are marked ⛔ dead in the Z1 index because
    # the judge truncated BOTH answers at 2,500 chars against a ~3,250 median.
    #
    # Measured here before the fix: the sampled turns run to **22,694 chars**,
    # median 3,572, and **3 of the 6 turns used in the first calibration were
    # over the cut** (9,292 and 10,655 among them). The judge therefore saw
    # ~38-43% of those sources while reading a summary of ALL of it — so a
    # summary faithfully describing the unseen tail is indistinguishable from
    # invention. That is almost certainly the whole of the 13-of-18
    # clean->fabricated result, which is now VOID.
    #
    # ⇒ The judge sees the entire source and the entire summary. If a turn ever
    # exceeds the judge's window, that must surface as a loud failure rather
    # than a silent slice.
    user = f"SOURCE TURN:\n{source}\n\nSUMMARY:\n{summary}"
    for attempt in range(retries):
        try:
            r = client.post("/chat/completions", json={
                "model": model,
                "messages": [{"role": "system", "content": system or SYSTEM},
                             {"role": "user", "content": user}],
                "temperature": 0.0, "max_tokens": 800,
                "response_format": {"type": "json_object"},
            }, timeout=120)
            r.raise_for_status()
            c = r.json()["choices"][0]["message"].get("content")
            if c:
                return json.loads(c)["verdicts"][0]
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {"label": "FAILED", "why": ""}


def make_summaries(model: str, n: int) -> list[dict]:
    """Real summaries, through the production path, for judging."""
    from src.memory.embedder import get_embedder
    from src.workers.post_flight import generate_summary
    from src.workers.turn_density import extract_key_terms

    from scripts.z1.bg_model_bakeoff import load_turns

    settings.background_model_name = model
    emb = get_embedder()
    out = []
    for t in load_turns(n):
        src = f"User: {t.get('prompt') or ''}\n\nAssistant: {t.get('response') or ''}"
        kt = extract_key_terms(src, emb)
        summary, coverage, _abstract = generate_summary(
            t.get("prompt") or "", t.get("response") or "", kt)
        if (summary or "").strip():
            out.append({"model": model, "source": src, "summary": summary,
                        "coverage": coverage})
    return out


def calibrate(client, judge: str, samples: list[dict]) -> dict:
    """Plant defects we control, and see whether the judge finds them."""
    rng = random.Random(20260826)
    rows = []
    for s in samples:
        # clean control — a real summary. ⚠ NOT ground truth: it may genuinely
        # contain invention, which is the whole thing we are trying to measure.
        rows.append((s, "clean", s["summary"], "faithful"))
        # fabrication — must come back `fabricated`
        sent = rng.choice(FABRICATIONS)
        body = s["summary"].rstrip()
        rows.append((s, "fabricated", body + sent, "fabricated"))
        # ⚑ THE VERBATIM CONTROL, and it is what makes the other two readable.
        # First run: the judge caught 18/18 planted fabrications but ALSO called
        # 13 of 18 clean summaries fabricated. Two explanations fit that
        # equally well and they have opposite consequences:
        #   (a) the judge cries fabrication at everything — useless, and 18/18
        #       detection is then meaningless rather than impressive;
        #   (b) the local models really do invent in most summaries.
        # A verbatim copy of the source CANNOT fabricate — every word is in the
        # source by construction. If the judge calls THAT fabricated it is (a).
        # If it calls it faithful, the clean flags deserve to be believed.
        rows.append((s, "verbatim", s["source"], "faithful"))

    got = Counter()
    hits = Counter()
    # ⚑ PER-MODEL BREAKDOWN OF THE CLEAN ARM. Gating the judge is a statement
    # about the JUDGE, so it pools every model's summaries — but the clean arm
    # is the only arm that is also a statement about the MODELS, and pooling it
    # hides the case that matters: one model inventing while another does not
    # averages into "summaries are ~70% fabricated" and names no culprit.
    per_model: dict[str, Counter] = {}
    for s, kind, text, want in rows:
        v = judge_one(client, judge, s["source"], text)
        lab = v.get("label", "FAILED")
        got[f"{kind}->{lab}"] += 1
        hits[kind] += (lab == want)
        if kind == "clean":
            per_model.setdefault(s["model"], Counter())[lab] += 1

    n_each = len(samples)
    detect = hits["fabricated"] / max(1, n_each)
    clean_ok = hits["clean"] / max(1, n_each)
    verbatim_ok = hits["verbatim"] / max(1, n_each)
    # ⚑ SENSITIVITY WITHOUT SPECIFICITY IS NOT A JUDGE. A model that answers
    # "fabricated" to everything scores 1.000 detection and is worthless. The
    # verbatim arm is the one that separates a strict judge from a broken one,
    # so it — not the clean arm — is the gate.
    usable = bool(detect >= 0.7 and verbatim_ok >= 0.7)
    if usable and clean_ok < 0.5:
        reading = ("judge is SOUND and the local summaries really do invent — "
                   f"only {clean_ok:.0%} of real summaries are faithful")
    elif not usable and verbatim_ok < 0.7:
        reading = ("judge cries fabrication even at a VERBATIM COPY of the "
                   "source, which cannot contain invention ⇒ the judge is "
                   "broken, and its 1.000 detection rate means nothing")
    else:
        reading = "judge passes; read the clean arm as a real measurement"
    return {"judge": judge, "n_per_arm": n_each,
            "fabrication_detected": round(detect, 3),
            "clean_called_faithful": round(clean_ok, 3),
            "verbatim_called_faithful": round(verbatim_ok, 3),
            "confusion": dict(got),
            "clean_by_model": {m: dict(c) for m, c in per_model.items()},
            "reading": reading,
            "USABLE": usable}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=None, help="default settings.probe_model")
    ap.add_argument("--models", default="qwen3:4b-instruct")
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--calibrate", action="store_true",
                    help="run the planted-defect gate and stop")
    args = ap.parse_args()

    judge = args.judge or settings.probe_model
    client = httpx.Client(base_url=settings.probe_api_base_url,
                          headers={"Authorization": f"Bearer {settings.probe_api_key}"})

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"generating real summaries through the production path: {models}")
    samples: list[dict] = []
    for m in models:
        got = make_summaries(m, args.turns)
        print(f"   {m}: {len(got)} summaries")
        samples.extend(got)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.calibrate:
        print(f"\n⚑ GATING THE JUDGE ({judge}) on planted defects — "
              f"{len(samples)} clean + {len(samples)} fabricated + "
              f"{len(samples)} verbatim-copy controls")
        res = calibrate(client, judge, samples)
        print(json.dumps(res, indent=1))
        print("\n" + ("✅ judge is USABLE for this task"
                      if res["USABLE"] else
                      "⛔ JUDGE FAILS THE GATE — do not rank models with it.")
              + "\n   " + res["reading"])
        path = f"{OUT_DIR}/judge_calibration_{stamp}.json"
        json.dump(res, open(path, "w"), indent=1)
        print(f"→ {path}")
        return 0

    print(f"\njudging {len(samples)} summaries with {judge} "
          f"(model identity hidden from the judge)")
    verdicts = []
    for s in samples:
        v = judge_one(client, judge, s["source"], s["summary"])
        verdicts.append({"model": s["model"], "label": v.get("label"),
                         "why": (v.get("why") or "")[:200],
                         "coverage": s["coverage"]})
    print(f"\n{'model':24}{'faithful':>10}{'incomplete':>12}{'fabricated':>12}{'failed':>8}")
    for m in models:
        c = Counter(v["label"] for v in verdicts if v["model"] == m)
        print(f"{m:24}{c.get('faithful',0):>10}{c.get('incomplete',0):>12}"
              f"{c.get('fabricated',0):>12}{c.get('FAILED',0):>8}")
    path = f"{OUT_DIR}/summary_verdicts_{stamp}.json"
    json.dump({"judge": judge, "verdicts": verdicts}, open(path, "w"), indent=1)
    print(f"\n→ {path}  (⚠ gitignored — copy what matters into PROVENANCE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
