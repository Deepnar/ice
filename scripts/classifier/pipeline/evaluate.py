#!/usr/bin/env python3
"""Stage 7 — score the candidate and run D5's non-regression gate.

Rewrites ``legacy/training/test_classifier.py`` and adds the gate the roadmap
asked for: the roadmap's own words were "we cannot know whether the retrain lands
better or worse", and this stage is that sentence made operational.

The gate scores BOTH models on IDENTICAL rows, each rendered the way IT was
trained (the old model through the frozen v1 template and the 384-dim MRL prefix,
the candidate through the v2 template at native 1024), and compares them ONLY on
the labels both can express:

* topic — all 11, unchanged between generations
* intent — the first 11; v2's two coding intents have no v1 counterpart and are
  excluded from the comparison (this is why they were appended last)
* context reliance — the DERIVED three-way, which is the interface every existing
  consumer actually reads. The candidate's four sigmoids collapse through the
  same derivation the runtime uses; the old model's softmax argmax is already
  three-way.

Honest caveat, printed with the verdict: the gold labels are v2 labels, produced
by the new methodology. The old model is being graded against a rubric it never
saw. That is unavoidable — no v1 gold exists for these rows — and it does tilt
the comparison toward the candidate. The gate is therefore a floor ("the new
model is not worse at the job the old one was doing"), not a proof of improvement.

Usage:
    uv run python scripts/classifier/pipeline/evaluate.py \
        --candidate models/classifier/ice_classifier_v4_schema2.pt
"""

import argparse
import json

import torch
from common import TEST_SPLIT
from train import per_label_f1

from src.api.config import settings
from src.classifier import templates
from src.classifier.dataset import ICEClassifierDataset
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, LONG_TERM_MEMORY,
                                   NEEDS_LIVE_INFO, NEEDS_MEMORY,
                                   REAL_TIME_SEARCH, TOPIC, ZERO_SHOT,
                                   derive_context_reliance, load_schema)

THREE_WAY = [ZERO_SHOT, LONG_TERM_MEMORY, REAL_TIME_SEARCH]


def _encode(rows, template_version, input_dim, device):
    """Render + encode a row list the way ONE model expects to see it."""
    from sentence_transformers import SentenceTransformer

    from src.memory.embedder import slice384
    model = SentenceTransformer(settings.embedding_model_name, device=device)
    rendered = [templates.render(r["text"], r.get("context_text"),
                                 version=template_version) for r in rows]
    emb = model.encode(rendered, convert_to_tensor=True, batch_size=256,
                       show_progress_bar=False).detach().cpu().float()
    if input_dim == 384 and emb.shape[1] != 384:
        emb = slice384(emb)
    return emb


def _gold_three_way(rows):
    """Collapse v2 gold context labels into the legacy three-way.

    Uses the SAME rule as the runtime derivation so both models are scored
    against the target the derivation would produce. A row that is both
    Needs_Memory and Needs_Live_Info collapses to Long_Term_Memory — the old
    taxonomy had no way to say "both", which is precisely the expressiveness B1
    added and precisely why this comparison is a floor rather than a verdict.
    """
    out = torch.zeros(len(rows), 3)
    for i, row in enumerate(rows):
        ctx = set(row.get(CONTEXT_RELIANCE, []))
        label, *_ = derive_context_reliance({
            NEEDS_MEMORY: 1.0 if NEEDS_MEMORY in ctx else 0.0,
            NEEDS_LIVE_INFO: 1.0 if NEEDS_LIVE_INFO in ctx else 0.0,
        })
        out[i, THREE_WAY.index(label)] = 1.0
    return out


def _shared_matrix(probs, schema, is_legacy):
    """Project either model's output onto the shared label space.

    Columns: 11 topic + 11 intent + 3 derived context = 25.
    """
    topic = probs[:, schema.slice(TOPIC)]
    intent = probs[:, schema.slice(INTENT)][:, :11]
    ctx = probs[:, schema.slice(CONTEXT_RELIANCE)]

    if is_legacy:
        three = ctx                       # already [Zero_Shot, LTM, RTS]
    else:
        three = torch.zeros(probs.shape[0], 3)
        labels = schema.labels(CONTEXT_RELIANCE)
        mem = ctx[:, labels.index(NEEDS_MEMORY)]
        live = ctx[:, labels.index(NEEDS_LIVE_INFO)]
        three[:, 0] = 1.0 - torch.maximum(mem, live)   # Zero_Shot
        three[:, 1] = mem
        three[:, 2] = live
    return torch.cat([topic, intent, three], dim=1)


class _SharedSchema:
    """Minimal schema-shaped view of the 25-column shared space, so the same
    per_label_f1 helper scores both models."""

    def __init__(self, v2):
        from src.classifier.schema import Head
        self.heads = (
            Head(TOPIC, v2.labels(TOPIC), ("",) * 11, "sigmoid", "multi_label", 0, 11),
            Head(INTENT, v2.labels(INTENT)[:11], ("",) * 11, "sigmoid", "multi_label", 11, 11),
            Head(CONTEXT_RELIANCE, tuple(THREE_WAY), ("",) * 3, "sigmoid", "single_label", 22, 3),
        )


def main():
    ap = argparse.ArgumentParser(description="B1 stage 7: evaluate + D5 gate")
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--baseline", default=settings.classifier_model_path)
    ap.add_argument("--test", default=TEST_SPLIT)
    ap.add_argument("--threshold", type=float, default=settings.classifier_threshold)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    schema = load_schema()
    test_ds = ICEClassifierDataset(args.test, schema=schema, device=args.device)
    rows, targets = test_ds.data, test_ds.labels
    print(f"[eval] {len(rows)} held-out rows")

    # ── candidate, on its own full label space ──
    candidate, cand_meta = load_checkpoint(args.candidate, schema=schema)
    x_cand = test_ds.embeddings
    with torch.no_grad():
        cand_probs = torch.sigmoid(candidate(x_cand))
    report, macro = per_label_f1(cand_probs, targets, schema, args.threshold)
    print(f"\n[eval] candidate macro-F1 by head: {macro}")
    for head in schema.heads:
        print(f"  {head.name}:")
        for label in head.labels:
            r = report[label]
            print(f"    {label:26} f1={r['f1']:.3f}  p={r['precision']:.3f}  "
                  f"r={r['recall']:.3f}  n={r['support']}")

    # ── D5 gate: same rows, each model rendered as it was trained ──
    print(f"\n[eval] D5 non-regression gate vs {args.baseline}")
    baseline, base_meta = load_checkpoint(args.baseline)
    base_schema = baseline.schema
    x_base = _encode(rows, base_meta.get("template_version", 1),
                     base_meta.get("input_dim", 384), args.device)
    with torch.no_grad():
        base_logits = baseline(x_base)
        base_probs = torch.cat([
            torch.softmax(base_logits[:, h.slice], dim=1) if h.activation == "softmax"
            else torch.sigmoid(base_logits[:, h.slice])
            for h in base_schema.heads], dim=1)

    shared = _SharedSchema(schema)
    gold = torch.cat([targets[:, schema.slice(TOPIC)],
                      targets[:, schema.slice(INTENT)][:, :11],
                      _gold_three_way(rows)], dim=1)

    cand_shared = _shared_matrix(cand_probs, schema, is_legacy=False)
    base_shared = _shared_matrix(base_probs, base_schema, is_legacy=True)

    cand_report, cand_macro = per_label_f1(cand_shared, gold, shared, args.threshold)
    base_report, base_macro = per_label_f1(base_shared, gold, shared, args.threshold)

    print(f"{'head':<20} {'baseline':>10} {'candidate':>10} {'delta':>8}")
    passed = True
    for head_name in (TOPIC, INTENT, CONTEXT_RELIANCE):
        b, c = base_macro[head_name], cand_macro[head_name]
        delta = c - b
        if delta < 0:
            passed = False
        print(f"{head_name:<20} {b:>10.4f} {c:>10.4f} {delta:>+8.4f}")

    overall_base = sum(base_macro.values()) / 3
    overall_cand = sum(cand_macro.values()) / 3
    print(f"{'OVERALL':<20} {overall_base:>10.4f} {overall_cand:>10.4f} "
          f"{overall_cand - overall_base:>+8.4f}")

    verdict = "PASS" if passed and overall_cand >= overall_base else "FAIL"
    print(f"\n[eval] D5 GATE: {verdict}")
    if verdict == "FAIL":
        print("[eval] Do NOT promote. The spec's rule is explicit: diagnose the DATA "
              "first (label counts, agreement rate, context share) — a head that "
              "regresses on the shared labels is user-visible immediately, because "
              "B2 and retrieval run on exactly those.")
    else:
        print("[eval] Caveat: gold labels are v2 labels; the baseline is graded on a "
              "rubric it never saw. Read this as 'not worse at the old job', not as "
              "proof of improvement.")

    out_path = args.out or args.candidate.replace(".pt", "_eval.json")
    with open(out_path, "w") as fh:
        json.dump({
            "candidate": args.candidate, "baseline": args.baseline,
            "rows": len(rows), "threshold": args.threshold,
            "candidate_macro_f1": macro, "candidate_per_label": report,
            "gate": {"verdict": verdict,
                     "baseline_macro": base_macro, "candidate_macro": cand_macro,
                     "baseline_overall": overall_base,
                     "candidate_overall": overall_cand,
                     "baseline_per_label": base_report,
                     "candidate_per_label": cand_report},
            "candidate_meta": {k: v for k, v in cand_meta.items()
                               if k not in ("head_labels",)},
        }, fh, indent=2, default=str)
    print(f"[eval] report → {out_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
