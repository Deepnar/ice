#!/usr/bin/env python3
"""Z1: does the MUST-PRESERVE instruction manufacture the fabrication?

**The claim under test, measured 2026-08-26.** `summary_coverage` is the gate
deciding whether a summary REPLACES the raw turn in the prompt. Graded by a
judge that passed its own controls (planted fabrications 15/16, verbatim copies
16/16 faithful), **13 of 13 fabricated summaries cleared the gate and only 23 of
29 faithful ones did** — fabricated summaries score HIGHER coverage inside every
model tested.

The proposed mechanism is one sentence in `_summary_llm_call`:

    MUST-PRESERVE TERMS (every one of these must appear verbatim in your summary)

A model that cannot ground a must-term is being ordered to include it anyway, so
it invents context to carry it — and coverage rewards it for doing so. If that
is right, softening the order should reduce fabrication.

**⚑ ARM B DIFFERS FROM ARM A BY EXACTLY ONE SUBSTRING, AND IT IS PROVEN.**
The temptation is to copy `_summary_llm_call` and edit the prompt — which
re-implements the thing under test and lets a second, unnoticed difference in.
Instead both arms call the REAL production function, and arm B wraps the
background client so the assembled prompt is rewritten in flight. The rewrite
asserts it matched, so an arm that silently failed to differ cannot be reported
as a null.

⚠ **This changes NOTHING in `src/`.** It measures whether a change would be
worth proposing.

  uv run python scripts/z1/prompt_ab_fabrication.py --turns 70
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import httpx  # noqa: E402

from src.api.config import settings  # noqa: E402

OUT_DIR = "experiments/curation_files/bakeoff"

HARD = "MUST-PRESERVE TERMS (every one of these must appear verbatim in your summary):"
SOFT = ("KEY TERMS from this turn — preserve each one the turn actually supports, "
        "and simply omit any you cannot state faithfully. Never introduce a fact "
        "to accommodate a term:")

# ⚑ THE RETRY IS THE HARDER ORDER, AND THE FIRST VERSION LEFT IT ALONE.
# `generate_summary` calls `retry_on_coverage_miss`: when coverage misses, it
# re-asks with "include each of them verbatim this time". Softening only the
# first instruction meant arm B's retries carried the STRONGEST form of the very
# order under test — so on every turn that retried, arm B was arm A. The smoke
# run made that visible: arm B's coverage fell to 0.250, i.e. it was retrying
# constantly, so the confound was not rare but dominant.
HARD_RETRY = ("Your previous summary DROPPED these required terms — include "
              "each of them verbatim this time:")
SOFT_RETRY = ("Your previous summary omitted these key terms. Include any the "
              "turn genuinely supports; leave out any that would require "
              "inventing a detail:")


class _Rewriting:
    """Wraps bg_client.chat.completions so arm B's prompt differs by one string."""

    def __init__(self, inner, counter):
        self._inner, self._c = inner, counter

    def create(self, *args, **kwargs):
        msgs = kwargs.get("messages") or []
        for m in msgs:
            c = m.get("content") or ""
            hit = False
            if HARD in c:
                c = c.replace(HARD, SOFT)
                hit = True
            elif "MUST-PRESERVE" in c:
                # Loud: the anchor drifted and arm B is silently arm A.
                raise SystemExit("⛔ HARD anchor not found but MUST-PRESERVE "
                                 "present — the prompt changed; fix the anchor.")
            if HARD_RETRY in c:
                c = c.replace(HARD_RETRY, SOFT_RETRY)
                self._c["retries_rewritten"] += 1
                hit = True
            elif "DROPPED these required terms" in c:
                raise SystemExit("⛔ retry anchor drifted — arm B would carry the "
                                 "hard order on every retry.")
            if hit:
                m["content"] = c
                self._c["rewritten"] += 1
        return self._inner.create(*args, **kwargs)

    def __getattr__(self, k):
        return getattr(self._inner, k)


def run_arm(model: str, turns: list, soften: bool,
            term_cap: int = 0) -> tuple[list[dict], int, int]:
    """One arm. `soften` rewrites the instruction; `term_cap` changes HOW MANY
    must-terms are demanded. Only one of them is ever varied in a run — two
    changes and one number answers neither (Z1 index, rule 3)."""
    from src.memory.embedder import get_embedder
    from src.workers import post_flight as pf
    from src.workers.turn_density import extract_key_terms

    settings.background_model_name = model
    emb = get_embedder()
    counter = Counter()
    # ⚑ The knob, not the wording. `turn_max_must_terms` caps the list in BOTH
    # `extract_key_terms` (entities) and `must_terms` (the combined list), so
    # setting it here is the same lever production would pull.
    prev_cap = settings.turn_max_must_terms
    if term_cap:
        settings.turn_max_must_terms = term_cap
    original = pf.bg_client
    if soften:
        class _C:
            def __init__(self, inner):
                self.completions = _Rewriting(inner.chat.completions, counter)
        class _B:
            def __init__(self, inner):
                self.chat = _C(inner)
        pf.bg_client = _B(original)
    try:
        out = []
        for t in turns:
            src = f"User: {t.get('prompt') or ''}\n\nAssistant: {t.get('response') or ''}"
            kt = extract_key_terms(src, emb)
            summary, coverage, _a = pf.generate_summary(
                t.get("prompt") or "", t.get("response") or "", kt)
            if (summary or "").strip():
                from src.workers.turn_density import must_terms
                out.append({"source": src, "summary": summary,
                            "coverage": coverage, "terms": must_terms(kt)})
        counter["terms_median"] = int(statistics.median(
            [len(x["terms"]) for x in out])) if out else 0
        return out, counter["rewritten"], counter["retries_rewritten"]
    finally:
        pf.bg_client = original
        settings.turn_max_must_terms = prev_cap


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--judge", default=None)
    ap.add_argument("--turns", type=int, default=70)
    ap.add_argument("--vary", choices=["instruction", "terms"], default="instruction",
                    help="instruction = soften MUST-PRESERVE (refuted 2026-08-26: "
                         "fabrication 37%%->9%% but faithful 54%%->34%%). "
                         "terms = vary turn_max_must_terms instead")
    ap.add_argument("--terms-b", type=int, default=8,
                    help="arm B's must-term cap when --vary terms (prod is 25)")
    args = ap.parse_args()

    from scripts.z1.bg_model_bakeoff import load_turns
    from scripts.z1.judge_summaries import judge_one

    judge = args.judge or settings.probe_model
    turns = load_turns(args.turns)
    print(f"prompt A/B on {args.model} · {len(turns)} turns · judge {judge}\n")

    vary_terms = args.vary == "terms"
    print(f"── arm A: production ({settings.turn_max_must_terms} must-terms, "
          f"hard MUST-PRESERVE)")
    a, _, _ = run_arm(args.model, turns, soften=False)
    print(f"   {len(a)} summaries · median terms demanded "
          f"{int(statistics.median([len(x['terms']) for x in a])) if a else 0}")
    if vary_terms:
        print(f"── arm B: SAME prompt, must-terms capped at {args.terms_b}")
        b, n_rw, n_rt = run_arm(args.model, turns, soften=False,
                                term_cap=args.terms_b)
        med_b = int(statistics.median([len(x["terms"]) for x in b])) if b else 0
        med_a = int(statistics.median([len(x["terms"]) for x in a])) if a else 0
        print(f"   {len(b)} summaries · median terms demanded {med_b}")
        if med_b >= med_a:
            raise SystemExit(f"⛔ arm B demanded {med_b} terms vs arm A's {med_a} "
                             "— the cap did nothing. Aborting rather than "
                             "reporting a null.")
    else:
        print("── arm B: softened — preserve what the turn supports, omit the rest")
        b, n_rw, n_rt = run_arm(args.model, turns, soften=True)
        print(f"   {len(b)} summaries · prompt rewritten on {n_rw} calls "
              f"(of which {n_rt} were RETRY prompts — the harder order)")
        if n_rw == 0:
            raise SystemExit("⛔ arm B never rewrote a prompt — it IS arm A. "
                             "Aborting rather than reporting a null.")

    client = httpx.Client(base_url=settings.probe_api_base_url,
                          headers={"Authorization": f"Bearer {settings.probe_api_key}"})
    res = {}
    for name, arm in (("A_hard", a), ("B_soft", b)):
        print(f"\njudging arm {name} ({len(arm)}) …")
        labels = []
        for s in arm:
            v = judge_one(client, judge, s["source"], s["summary"])
            labels.append(v.get("label", "FAILED"))
        c = Counter(labels)
        n = max(1, len(arm))
        res[name] = {
            "n": len(arm), "counts": dict(c),
            "fabricated_rate": round(c.get("fabricated", 0) / n, 3),
            "faithful_rate": round(c.get("faithful", 0) / n, 3),
            "coverage_mean": round(statistics.mean([s["coverage"] for s in arm]), 4),
            "cleared_gate": sum(1 for s in arm
                                if s["coverage"] >= settings.turn_summary_coverage_threshold),
        }
        print(f"   {res[name]}")

    fa, fb = res["A_hard"]["fabricated_rate"], res["B_soft"]["fabricated_rate"]
    na, nb = res["A_hard"]["n"], res["B_soft"]["n"]
    # Two-proportion z, quoted with the interval rather than as a bare delta —
    # the standing rule after "+9.4 pts" turned out to be a sampling artifact.
    p = ((fa * na) + (fb * nb)) / max(1, na + nb)
    se = (p * (1 - p) * (1 / max(1, na) + 1 / max(1, nb))) ** 0.5
    z = (fa - fb) / se if se else 0.0
    print(f"\n⚑ fabrication  A(hard) {fa:.1%}  →  B(soft) {fb:.1%}   "
          f"delta {100*(fa-fb):+.1f} pts, z={z:.2f}")
    print("   " + ("✅ softening REDUCES fabrication — worth proposing as a src change"
                   if z >= 1.96 else
                   "⚠ NOT resolvable at this n — the prompt is not shown to be the cause"))
    print(f"   coverage A {res['A_hard']['coverage_mean']:.3f} → "
          f"B {res['B_soft']['coverage_mean']:.3f}  (expect B lower; that is the "
          f"POINT, not a regression)")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = f"{OUT_DIR}/prompt_ab_{stamp}.json"
    json.dump({"model": args.model, "judge": judge, "varied": args.vary,
               "terms_b": args.terms_b if vary_terms else None,
               "z": round(z, 3), "results": res}, open(path, "w"), indent=1)
    print(f"\n→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
