#!/usr/bin/env python3
"""Recalibrate B2's log-odds weights for the v2 classifier head.

**Why this is B1's job and not a later nicety.** B2 consumes the classifier as a
scalar (`p_ltm`) precisely so a retrain would not require rewriting it — that seam
worked, nothing broke. But *type* compatibility is not *distribution*
compatibility. v1's `p_ltm` was one class's share of a 3-way softmax, so it lived
in a compressed middle range; v2's is an independent sigmoid that saturates near 0
and 1. `decide_memory_retrieval` starts from `_logit(p_ltm)`, and the logit of a
saturating signal spans roughly ±4.6 where the old one spanned about ±2. Every
additive bump was sized against the old range, so relative to the new head they
are all mis-scaled — and the bumps exist to *compensate for a weak head*, which is
the thing B1 fixed.

The cost of leaving it is measured, not theoretical: on adversarial probes the v2
head beats v1 84% vs 78% with half the false alarms, yet **end-to-end through
`decide_memory_retrieval` the two tie at 80%**. The classifier improvement is
absorbed by B2's miscalibration instead of reaching the user.

**Methodology — Z1-prep's protocol applied to one stage.** Not a hand-edit: a
one-factor-at-a-time sweep, scored on data the classifier's labelers never
touched, reporting the curve rather than announcing a number.

The scoring set matters more than the search. The 207 user probes are ALL
positives, so tuning on them alone drives straight to always-retrieve — the exact
failure B2 exists to prevent. The set is therefore balanced explicitly:

  positives = 207 user curation probes + the hard set's memory-needing probes
  negatives = the hard set's must-stay-silent controls
              + held-out rows whose gold labels carry no Needs_Memory

The objective is **specificity subject to recall not falling below the shipped
baseline** — not balanced accuracy, which is the wrong metric for a silent gate
(see ``objective()``). Both probe families are reported separately: a knob that
helps one and hurts the other is overfitting, not tuning.

**Result when this was first run (2026-07-27): no change warranted.** The shipped
weights are already near-optimal for the v2 head; the only admissible move was
``ltm_bump_creative`` 0.7 → 0.35 for +0.005 specificity and 0.000 on both probe
families, i.e. noise, and it was not applied. So the prediction that motivated
this script was wrong — the end-to-end tie is not B2 miscalibration but B2
correctly spending specificity to buy recall. The script is kept because the
*measurement* is what settles the question, and Z1-prep should re-run it against
the synthetic ledger scorer rather than hand-editing these constants.

Usage:
    uv run python scripts/classifier/pipeline/tune_b2.py
    uv run python scripts/classifier/pipeline/tune_b2.py --apply   # prints the diff to make
"""

import argparse
import json
import os

import torch
from common import DATA_DIR, TEST_SPLIT

from src.api.config import settings
from src.api.memory_decision import decide_memory_retrieval
from src.classifier import templates
from src.classifier.dataset import encode_rendered
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, finalize_context_scalars,
                                   load_schema)
from src.classifier.schemas import ClassificationResult

HARD = os.path.join(DATA_DIR, "hard_probes_authored.jsonl")
USER = os.path.join(DATA_DIR, "eval_probes_independent.jsonl")
NEEDS_MEMORY = "Needs_Memory"

# Knobs, with the sweep grid for each. Ranges bracket the shipped default so a
# "no change" verdict is expressible.
GRID = {
    "ltm_prior_bias":           [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
    "ltm_decision_threshold":   [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "ltm_bump_referential":     [0.0, 0.25, 0.5, 0.75],
    "ltm_bump_low_confidence":  [0.0, 0.4, 0.8, 1.2],
    "ltm_bump_creative":        [0.0, 0.35, 0.7],
    # ``ltm_bump_reference`` was swept here and was one of the three knobs this
    # set could not move, because nothing in the scoring rows ever set
    # ``reference_signal`` — the sweep was measuring a constant. D8 measured it
    # properly (on the rows DI3's reference rule actually claimed), found it
    # net-negative, and deleted the knob along with DI3. 2026-07-27.
    "ltm_length_weight":        [0.0, 0.4, 0.8, 1.2],
}


def _read(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _classify_all(rows, ckpt, device):
    """Run the live classifier over rows, returning ClassificationResults."""
    schema = load_schema()
    model, meta = load_checkpoint(ckpt, schema=schema)
    model.eval()
    active = getattr(model, "schema", schema)
    tmpl = int(meta.get("template_version", 1))
    dim = int(meta.get("input_dim", 384))

    rendered = [templates.render(r["text"], r.get("context_text"), version=tmpl)
                for r in rows]
    emb = encode_rendered(rendered, device=device, show_progress=False)
    if dim == 384 and emb.shape[1] != 384:
        from src.memory.embedder import slice384
        emb = slice384(emb)
    with torch.no_grad():
        logits = model(emb)

    out = []
    for i, r in enumerate(rows):
        raw = []
        for h in active.heads:
            block = logits[i:i + 1, h.slice]
            pr = (torch.softmax(block, dim=1) if h.activation == "softmax"
                  else torch.sigmoid(block))
            raw.extend(pr.squeeze(0).tolist())
        res = ClassificationResult([], [], "Zero_Shot", raw, max(raw), r["text"])
        # topic_tags drive the creative bump; recover them from the head.
        from src.classifier.schema import TOPIC
        thead = active.head(TOPIC)
        thr = float(meta.get("tag_threshold") or settings.classifier_threshold)
        res.topic_tags = [lab for k, lab in enumerate(active.labels(TOPIC))
                          if raw[thead.offset + k] > thr] or []
        finalize_context_scalars(res, active)
        out.append(res)
    return out


def build_set(ckpt, device, max_neg=600):
    """(results, gold, family) over a BALANCED, labeler-independent set."""
    hard = _read(HARD)
    user = _read(USER)
    test = _read(TEST_SPLIT)

    rows, gold, fam = [], [], []
    for r in user:
        rows.append(r)
        gold.append(True)
        fam.append("user")
    for r in hard:
        want = NEEDS_MEMORY in (r["labels"].get(CONTEXT_RELIANCE) or [])
        rows.append(r)
        gold.append(want)
        fam.append("hard")
    negs = [r for r in test if NEEDS_MEMORY not in (r.get(CONTEXT_RELIANCE) or [])]
    for r in negs[:max_neg]:
        rows.append(r)
        gold.append(False)
        fam.append("split")

    print(f"[tune] scoring set: {sum(gold)} positives / {len(gold)-sum(gold)} negatives")
    print(f"[tune]   user probes {fam.count('user')} · hard {fam.count('hard')} "
          f"· held-out negatives {fam.count('split')}")
    return _classify_all(rows, ckpt, device), gold, fam, rows


def score(results, gold, fam, rows, cfg):
    """Balanced accuracy of the RETRIEVE decision, plus per-family accuracy."""
    tp = fp = tn = fn = 0
    per = {}
    for res, want, f, row in zip(results, gold, fam, rows):
        # Conversation stats: probes carry no history, so use a mid-conversation
        # operating point. Rows that shipped a context prefix get a larger one.
        turns = 4 if row.get("context_text") else 2
        tokens = 1200.0 if row.get("context_text") else 500.0
        d = decide_memory_retrieval(res, turn_count=turns, total_tokens=tokens,
                                    settings=cfg)
        ok = (d.retrieve == want)
        per.setdefault(f, [0, 0])
        per[f][0] += int(ok)
        per[f][1] += 1
        if want and d.retrieve:
            tp += 1
        elif want:
            fn += 1
        elif d.retrieve:
            fp += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {"balanced_acc": (tpr + tnr) / 2, "recall": tpr, "specificity": tnr,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "per_family": {k: v[0] / max(1, v[1]) for k, v in per.items()}}


def objective(s, mode, recall_floor):
    """The number coordinate descent maximises.

    **Balanced accuracy is the wrong objective for this decision**, which is why it
    is not the default. B2 gates whether memory is consulted, and the two errors
    are not symmetric:

    * a **false negative** means retrieval silently does not run. The answer is
      quietly worse, nothing signals it, and the user cannot distinguish "ICE had
      nothing" from "ICE never looked". This is the failure the system exists to
      prevent.
    * a **false positive** means a wasted retrieval round-trip and some irrelevant
      context. It costs latency and tokens, and the assembler's budget bounds the
      damage.

    Weighting those equally optimises a metric nobody cares about. ``recall_first``
    therefore maximises specificity **subject to recall not falling below the
    shipped baseline** — strictly fewer false alarms, never fewer catches. A pass
    that cannot find such a point correctly reports "no change".
    """
    if mode == "balanced":
        return s["balanced_acc"]
    if s["recall"] < recall_floor - 1e-9:
        return -1.0            # inadmissible: it lost a catch
    return s["specificity"]


class Cfg:
    """A mutable settings stand-in — decide_memory_retrieval only reads attrs."""
    def __init__(self, base):
        for k in dir(base):
            if k.startswith("ltm_") or k in ("confidence_fallback_threshold",
                                             "temporal_label_threshold"):
                setattr(self, k, getattr(base, k))


def main():
    ap = argparse.ArgumentParser(description="B1→B2 recalibration sweep")
    ap.add_argument("--checkpoint", default=settings.classifier_model_path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--rounds", type=int, default=2,
                    help="coordinate-descent passes over the knob list")
    ap.add_argument("--objective", default="recall_first",
                    choices=["recall_first", "balanced"],
                    help="recall_first (default): maximise specificity subject to "
                         "NOT losing recall vs the shipped defaults. balanced: "
                         "plain balanced accuracy — reported for comparison, but "
                         "it is the wrong objective for a silent gate (see below)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    results, gold, fam, rows = build_set(args.checkpoint, args.device)

    cfg = Cfg(settings)
    base = score(results, gold, fam, rows, cfg)
    print(f"\n[tune] SHIPPED defaults: balanced_acc {base['balanced_acc']:.4f} "
          f"(recall {base['recall']:.3f} / specificity {base['specificity']:.3f}) "
          f"TP{base['tp']} FP{base['fp']} TN{base['tn']} FN{base['fn']}")
    print(f"[tune]   per family: {base['per_family']}")

    recall_floor = base["recall"]
    if args.objective == "recall_first":
        print(f"[tune] objective: maximise specificity with recall >= "
              f"{recall_floor:.3f} (shipped baseline). Rationale in objective().")
    chosen = {k: getattr(cfg, k) for k in GRID}

    for rnd in range(args.rounds):
        print(f"\n[tune] ── coordinate pass {rnd + 1} ──")
        for knob, values in GRID.items():
            cur = getattr(cfg, knob)
            row = []
            best_v, best_s = cur, None
            # Current value FIRST and a strict > below, so a tie resolves to "no
            # change". Without this, an inert knob (one that never flips a
            # decision on this set) gets silently zeroed by tie-breaking, which
            # reads as a finding and is not one.
            for v in ([cur] + [x for x in values if x != cur]):
                setattr(cfg, knob, v)
                s = score(results, gold, fam, rows, cfg)
                o = objective(s, args.objective, recall_floor)
                row.append((v, o))
                if best_s is None or o > best_s:
                    best_v, best_s = v, o
            setattr(cfg, knob, best_v)
            chosen[knob] = best_v
            marks = "  ".join(
                f"{v}={'INADM' if a < 0 else format(a, '.4f')}"
                f"{'*' if v == best_v else ''}" for v, a in row)
            flag = "" if best_v == cur else f"   → {cur} → {best_v}"
            print(f"  {knob:<26} {marks}{flag}")

    final = score(results, gold, fam, rows, cfg)
    print(f"\n[tune] TUNED: balanced_acc {final['balanced_acc']:.4f} "
          f"(recall {final['recall']:.3f} / specificity {final['specificity']:.3f}) "
          f"TP{final['tp']} FP{final['fp']} TN{final['tn']} FN{final['fn']}")
    print(f"[tune]   per family: {final['per_family']}")
    print(f"[tune]   delta balanced_acc: {final['balanced_acc'] - base['balanced_acc']:+.4f}")

    print("\n[tune] changed knobs:")
    any_change = False
    for k in GRID:
        old, new = getattr(settings, k), chosen[k]
        if old != new:
            any_change = True
            print(f"    {k}: {old} → {new}")
    if not any_change:
        print("    (none — the shipped defaults already win this objective)")

    # Overfitting check: did both probe families move the same way?
    print("\n[tune] per-family sanity (a knob that helps one and hurts the other "
          "is overfitting):")
    for f in sorted(base["per_family"]):
        b, a = base["per_family"][f], final["per_family"][f]
        print(f"    {f:<8} {b:.3f} → {a:.3f}  {a - b:+.3f}")

    if args.apply:
        print("\n[tune] --apply prints the change for you to make in config.py; it "
              "does not edit settings behind your back:")
        for k in GRID:
            if getattr(settings, k) != chosen[k]:
                print(f"    {k}: float = {chosen[k]}")


if __name__ == "__main__":
    main()
