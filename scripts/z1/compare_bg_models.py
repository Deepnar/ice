#!/usr/bin/env python3
"""Z1/A12: compare background-model candidates on the two jobs ICE actually gives them.

**Judge-free by construction, which is what makes it affordable.** A12 splits the
background work into two families and each already has a measurable target:

  * **Extraction** (structured JSON — triplets). Measured as yield per turn, the
    share of turns that come back empty, and the **in-vocabulary rate** of the
    relations produced. That last one matters more than yield: G32's evidence run
    found **67.8%** of relations land outside the controlled vocabulary and are
    silently dropped, so a model that emits many triplets in words ICE cannot
    store is worse than one that emits fewer usable ones.
  * **Generation** (free text — summaries). Measured as `summary_coverage`, the
    fraction of MUST-PRESERVE terms the summary retained. ICE **already computes
    this on every summary** — A12 calls it "the cheapest measurement in this
    whole item", and it needs no ground truth and no judge.

**⚑ ONE PROBE PER FAMILY, and the limit is stated rather than hidden.** ICE gives
a background model about a dozen jobs (triplets, decisions, procedural patterns,
document-kind, four summarisation paths, reflection descriptions, cluster naming,
slot proposals, motifs). This measures **two**: codex triplets and the post-flight
summary. The defence is A12's own — *"the background work is TWO families, not six
tasks; this is the finding that makes the item tractable"* — and the backstop is
that the FULL SEED for a finalist exercises every one of those jobs, with
retrieval recall against the store it built as the task-grounded verdict. This
script only decides who earns a full seed.

*(A procedural-extraction probe was added here and removed: `extract_procedural`
is DB-bound with no text-level entry, and its call is free text — "one sentence
or NONE" — so it belongs to the generation family anyway, not to extraction.
Replicating its prompt here would have duplicated production logic into a test
and drifted from it.)*

**⚑ What this does NOT measure, deliberately.** Answer quality. No model writes
an answer here, because answer quality needs a judge and belongs to Z2. The
question this asks is narrower and more useful for Z1: *which model builds a
store that retrieval can use?* The task-grounded half of that — retrieval recall
against the store each arm produced — comes from `score_retrieval.py` run per
snapshot, not from here.

Same turns for every arm, in the same order, so the comparison is paired.

Run:
  uv run python scripts/z1/compare_bg_models.py --turns 12
  uv run python scripts/z1/compare_bg_models.py --turns 12 --models granite4:micro,qwen3.5:4b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.z1.derive_retrieval_gt import load_conversations  # noqa: E402
from src.api.config import settings  # noqa: E402

# PROVENANCE.md's shortlist, chosen by property (non-reasoning, sound
# quantization, structured-output discipline) rather than size — "a 7B was the
# worst arm and a 12B lost to a 4B". granite4:small-h (32.2B) is deliberately
# ABSENT: it is four times the size the shortlist asked for and contradicts the
# selection principle. Add it explicitly with --models if you want it measured.
# Raw per-arm output is SAVED, not just scored. The rough eyeball pass needs
# something to read, and this script measures in memory while store_report.py
# and sample_bg_output.py read the database — so without this the 8 arms would
# produce numbers with no samples behind them.
ARM_OUT = Path("experiments/curation_files/arm_output")

DEFAULT_ARMS = [
    "gemma4:26b-a4b-it-q4_K_M",   # the incumbent / current pin
    "granite4:micro",             # the 3B
    "granite4:tiny-h",            # 6.9B — the shortlist's "8B"
    "ministral-3:8b",
    "gemma4:e4b",
    "nemotron-mini:4b",
    "qwen3.5:4b",                 # incumbent to beat
    "qwen3:4b-instruct",          # incumbent to beat
]


def pick_turns(n: int) -> list[tuple[str, str]]:
    """Evenly spread across all three conversations, so one topic cannot
    dominate the verdict."""
    convs = load_conversations()
    out = []
    per = max(1, n // max(1, len(convs)))
    for _cid, meta in sorted(convs.items()):
        turns = [t for t in meta["turns"]
                 if len((t.get("user_input") or "").split()) > 30]
        step = max(1, len(turns) // per)
        for t in turns[::step][:per]:
            out.append(((t.get("user_input") or "").strip(),
                        (t.get("ai_response") or "").strip()))
    return out[:n]


def run_arm(model: str, turns: list, vocab: set) -> dict:
    from src.workers.codex_extractor import extract_triplets
    from src.workers.post_flight import generate_summary
    from src.workers.turn_density import extract_key_terms

    settings.background_model_name = model
    m = Counter()
    t0 = time.time()
    cov_scores = []
    samples = []

    for user, assistant in turns:
        text = f"User: {user}\n\nAssistant: {assistant}"
        try:
            tris = extract_triplets(text) or []
        except Exception as exc:
            m["extract_failed"] += 1
            print(f"    ! extract: {type(exc).__name__}: {str(exc)[:90]}")
            tris = []
        if not tris:
            m["empty_turns"] += 1
        m["triplets"] += len(tris)
        for t in tris:
            rel = str(t.get("relation") or "").strip().lower().replace(" ", "_")
            m["in_vocab" if rel in vocab else "out_of_vocab"] += 1

        summary = ""
        coverage = 0.0
        try:
            kt = extract_key_terms(user, assistant)
            summary, coverage, _abstract = generate_summary(
                user, assistant, kt, model_used=model)
            if summary:
                cov_scores.append(coverage)
            else:
                m["summary_empty"] += 1
        except Exception as exc:
            m["summary_failed"] += 1
            print(f"    ! summary: {type(exc).__name__}: {str(exc)[:90]}")

        samples.append({
            "source_turn": text[:2500],
            "triplets": [{"s": t.get("subject"), "r": t.get("relation"),
                          "o": t.get("object")} for t in tris],
            "summary": summary, "coverage": coverage,
        })

    elapsed = time.time() - t0
    n = max(1, len(turns))
    ARM_OUT.mkdir(parents=True, exist_ok=True)
    (ARM_OUT / f"{model.replace(':', '_').replace('/', '_')}.json").write_text(
        json.dumps({"model": model, "samples": samples}, indent=2))
    total_rel = m["in_vocab"] + m["out_of_vocab"]
    return {
        "model": model,
        "s_per_turn": round(elapsed / n, 1),
        "triplets_per_turn": round(m["triplets"] / n, 1),
        "empty_turns": f"{m['empty_turns']}/{n}",
        "in_vocab_pct": round(100 * m["in_vocab"] / total_rel) if total_rel else 0,
        "summary_coverage": round(sum(cov_scores) / len(cov_scores), 3) if cov_scores else 0.0,
        "summary_fail": m["summary_failed"] + m["summary_empty"],
        "extract_fail": m["extract_failed"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=60)
    ap.add_argument("--models", default=None, help="comma-separated override")
    args = ap.parse_args()

    from src.workers.codex_extractor import ALLOWED_RELATIONS
    vocab = {r.lower() for r in ALLOWED_RELATIONS}

    arms = ([m.strip() for m in args.models.split(",")] if args.models
            else DEFAULT_ARMS)
    turns = pick_turns(args.turns)
    print(f"{len(turns)} turns × {len(arms)} arms · vocabulary {len(vocab)} relations\n")

    original = settings.background_model_name
    rows = []
    try:
        for arm in arms:
            print(f"── {arm}")
            try:
                r = run_arm(arm, turns, vocab)
            except Exception as exc:
                print(f"   ARM FAILED: {type(exc).__name__}: {str(exc)[:120]}")
                continue
            rows.append(r)
            print(f"   {r['s_per_turn']}s/turn · {r['triplets_per_turn']} triplets "
                  f"· {r['in_vocab_pct']}% in-vocab · coverage {r['summary_coverage']}")
    finally:
        settings.background_model_name = original      # never leave the pin moved

    if not rows:
        return 1
    print(f"\n{'model':30}{'s/turn':>8}{'trip/turn':>11}{'in-vocab%':>11}"
          f"{'coverage':>10}{'empty':>8}{'fails':>7}")
    for r in sorted(rows, key=lambda x: (-x["in_vocab_pct"], -x["summary_coverage"])):
        print(f"{r['model']:30}{r['s_per_turn']:>8}{r['triplets_per_turn']:>11}"
              f"{r['in_vocab_pct']:>11}{r['summary_coverage']:>10}"
              f"{r['empty_turns']:>8}{r['summary_fail']+r['extract_fail']:>7}")
    print("\nRanked by in-vocabulary rate, then summary coverage. Yield alone is a "
          "trap: triplets in words the vocabulary lacks are dropped on write.")
    print("This is HALF the verdict — the other half is retrieval recall against "
          "the store each arm builds (score_retrieval.py per snapshot).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
