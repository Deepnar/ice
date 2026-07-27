#!/usr/bin/env python3
"""Stage 7b — the independent-probe gate (rev [2026-07-25d]'s second gate).

D5 asks "is the candidate worse than the model it replaces?". This asks the
different and harder question: **is it actually right?** — on prompts the
pipeline's own labelers never touched.

Why a second gate exists at all: train, val and test all descend from the same
two local labelers under the same rubric, and those labelers agree 90–98% on the
memory signals. They share their mistakes, so a held-out split encodes the same
misconception and no quantity of it can detect the bias of the process that
produced it. Pile B cannot serve as the exam either — it is trained on. These 207
probes were written by the USER, months earlier, for Experiment-1 curation, and
no labeler in this pipeline has seen them.

**The set asserts exactly one label, Needs_Memory, and carries no negatives** (a
curation probe is asked in order to test recall, so it needs memory by
construction). A recall-only set is trivially gamed by a head that fires on
everything — which is precisely the failure mode of training run 1. So the recall
number is meaningless alone and this script always reports it beside a
**false-fire control**: the same measurement over held-out rows whose gold
context labels contain no Needs_Memory. Read the two together, or not at all.

Both models are measured two ways:

* ``p_mem`` over its own calibrated threshold — the head's raw opinion;
* the DERIVED three-way landing on Long_Term_Memory — what actually gates
  retrieval in ``main.py`` today, and the only form the v1 baseline can express.

Usage:
    uv run python scripts/classifier/pipeline/eval_probes.py \
        --candidate models/classifier/ice_classifier_v4_schema2.pt
"""

import argparse
import json
import os

import torch
from common import DATA_DIR, TEST_SPLIT

from src.api.config import settings
from src.classifier import templates
from src.classifier.dataset import encode_rendered
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, LONG_TERM_MEMORY,
                                   NEEDS_LIVE_INFO, NEEDS_MEMORY,
                                   derive_context_reliance, load_schema)

PROBES = os.path.join(DATA_DIR, "eval_probes_independent.jsonl")


def _read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _probs_for(ckpt_path, rows, device, schema):
    """Run one checkpoint over *rows*, rendered the way THAT model was trained."""
    model, meta = load_checkpoint(ckpt_path, schema=schema)
    model.eval()
    active = getattr(model, "schema", schema)
    tmpl = int(meta.get("template_version", 1))
    input_dim = int(meta.get("input_dim", 384))
    threshold = float(meta.get("tag_threshold") or settings.classifier_threshold)

    rendered = [templates.render(r["text"], r.get("context_text"), version=tmpl)
                for r in rows]
    from src.memory.embedder import fit_width
    emb = fit_width(encode_rendered(rendered, device=device,
                                    show_progress=False), input_dim)

    with torch.no_grad():
        logits = model(emb)
    ctx_head = active.head(CONTEXT_RELIANCE)
    block = logits[:, ctx_head.slice]
    ctx = (torch.softmax(block, dim=1) if ctx_head.activation == "softmax"
           else torch.sigmoid(block))

    labels = list(active.labels(CONTEXT_RELIANCE))
    legacy = ctx_head.activation == "softmax"
    p_mem, fires = [], []
    for i in range(ctx.shape[0]):
        if legacy:
            # v1's softmax: Long_Term_Memory is a class, and p_ltm IS its mass.
            m = float(ctx[i, labels.index(LONG_TERM_MEMORY)])
            derived = labels[int(torch.argmax(ctx[i]).item())]
        else:
            m = float(ctx[i, labels.index(NEEDS_MEMORY)])
            derived, *_ = derive_context_reliance({
                NEEDS_MEMORY: m,
                NEEDS_LIVE_INFO: float(ctx[i, labels.index(NEEDS_LIVE_INFO)]),
            })
        p_mem.append(m)
        fires.append(derived == LONG_TERM_MEMORY)
    return {"p_mem": p_mem, "retrieval_fires": fires, "threshold": threshold,
            "schema_version": int(meta.get("schema_version", 1))}


def _rate(flags):
    return sum(1 for f in flags if f) / max(1, len(flags))


def main():
    ap = argparse.ArgumentParser(description="B1 stage 7b: independent probe gate")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default=settings.classifier_model_path)
    ap.add_argument("--probes", default=PROBES)
    ap.add_argument("--control", default=TEST_SPLIT,
                    help="split to draw not-Needs_Memory control rows from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    schema = load_schema()
    probes = _read_jsonl(args.probes)

    # The control: held-out rows the labelers judged to need NO memory. These
    # ARE labeler-derived, and that is fine — their only job is to catch a head
    # that fires indiscriminately, which does not require unbiased labels.
    control = [r for r in _read_jsonl(args.control)
               if NEEDS_MEMORY not in set(r.get(CONTEXT_RELIANCE) or [])]
    print(f"[probes] {len(probes)} independent probes (all assert Needs_Memory)")
    print(f"[probes] {len(control)} control rows (gold: no Needs_Memory)")

    results = {}
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        hit = _probs_for(path, probes, args.device, schema)
        ctl = _probs_for(path, control, args.device, schema)
        thr = hit["threshold"]
        results[name] = {
            "path": path,
            "schema_version": hit["schema_version"],
            "threshold": thr,
            "probe_recall_p_mem": _rate([p >= thr for p in hit["p_mem"]]),
            "probe_recall_retrieval": _rate(hit["retrieval_fires"]),
            "control_falsefire_p_mem": _rate([p >= thr for p in ctl["p_mem"]]),
            "control_falsefire_retrieval": _rate(ctl["retrieval_fires"]),
            "probe_mean_p_mem": sum(hit["p_mem"]) / max(1, len(hit["p_mem"])),
            "control_mean_p_mem": sum(ctl["p_mem"]) / max(1, len(ctl["p_mem"])),
        }

    print(f"\n{'':<26}{'baseline':>12}{'candidate':>12}{'delta':>10}")
    rows = [
        ("probe recall (p_mem)", "probe_recall_p_mem", True),
        ("probe recall (retrieval)", "probe_recall_retrieval", True),
        ("control false-fire (p_mem)", "control_falsefire_p_mem", False),
        ("control false-fire (retr.)", "control_falsefire_retrieval", False),
        ("probe mean p_mem", "probe_mean_p_mem", True),
        ("control mean p_mem", "control_mean_p_mem", False),
    ]
    for label, key, higher_better in rows:
        b, c = results["baseline"][key], results["candidate"][key]
        d = c - b
        mark = "" if (d > 0) == higher_better or abs(d) < 1e-9 else "  ←worse"
        print(f"{label:<26}{b:>12.3f}{c:>12.3f}{d:>+10.3f}{mark}")

    # Separation is the number that survives both halves: how much more often the
    # head fires on real memory prompts than on prompts that need no memory. A
    # head that fires on everything scores ~0 here however good its recall looks.
    print()
    for name in ("baseline", "candidate"):
        r = results[name]
        sep = r["probe_recall_retrieval"] - r["control_falsefire_retrieval"]
        results[name]["separation_retrieval"] = sep
        print(f"[probes] {name:<10} separation (probe recall − control false-fire) "
              f"= {sep:+.3f}")

    out_path = args.out or args.candidate.replace(".pt", "_probes.json")
    with open(out_path, "w") as fh:
        json.dump({"probes": len(probes), "control_rows": len(control),
                   "results": results}, fh, indent=2)
    print(f"[probes] report → {out_path}")


if __name__ == "__main__":
    main()
