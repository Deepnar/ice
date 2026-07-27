#!/usr/bin/env python3
"""Score the hand-authored adversarial probes and PRINT THE FAILURES.

A pass rate alone would repeat the mistake this whole diagnosis exists to
correct. The point of an authored probe is that it comes with a stated reason
(``why``), so a failure is readable: you learn *which boundary* the head cannot
draw, not merely that a number is low. This script therefore prints every miss
with the model's actual output beside the assertion.

It runs the REAL inference path — ``templates.render`` at the checkpoint's own
template version, the checkpoint's own ``tag_threshold``, and
``finalize_context_scalars`` for the derived scalars — so what is measured is
what the proxy would do, not a reimplementation of it. Context is injected
directly rather than read from the DB (probes carry their own).

Usage:
    uv run python scripts/classifier/pipeline/score_hard_probes.py \
        --checkpoint models/classifier/ice_classifier_v4_schema2.pt
"""

import argparse
import collections
import json
import os

import torch
from common import DATA_DIR

from src.api.config import settings
from src.classifier import templates
from src.classifier.dataset import encode_rendered
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, TOPIC,
                                   finalize_context_scalars, load_schema)
from src.classifier.schemas import ClassificationResult

PROBES = os.path.join(DATA_DIR, "hard_probes_authored.jsonl")


def load_probes(path):
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser(description="B1: score the adversarial probes")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--probes", default=PROBES)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--show", default="fail", choices=["fail", "all", "none"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    schema = load_schema()
    model, meta = load_checkpoint(args.checkpoint, schema=schema)
    model.eval()
    active = getattr(model, "schema", schema)
    tmpl = int(meta.get("template_version", 1))
    input_dim = int(meta.get("input_dim", 384))
    thr = float(meta.get("tag_threshold") or settings.classifier_threshold)

    # Per-label thresholds for the context head, if the sweep stamped them.
    fit = (meta.get("threshold_fit") or {}).get("per_label") or {}
    ctx_head = active.head(CONTEXT_RELIANCE)
    ctx_thr = {lab: float(fit.get(str(ctx_head.offset + i), thr))
               for i, lab in enumerate(ctx_head.labels)}

    probes = load_probes(args.probes)
    print(f"[hard] {len(probes)} probes · checkpoint schema v{meta.get('schema_version')} "
          f"· tag_threshold {thr}")
    print("[hard] context thresholds: "
          + ", ".join(f"{k}={v}" for k, v in ctx_thr.items()))

    rendered = [templates.render(p["text"], p.get("context_text"), version=tmpl)
                for p in probes]
    from src.memory.embedder import fit_width
    emb = fit_width(encode_rendered(rendered, device=args.device,
                                    show_progress=False), input_dim)
    with torch.no_grad():
        logits = model(emb)

    probs_by_head = {}
    for head in active.heads:
        block = logits[:, head.slice]
        probs_by_head[head.name] = (torch.softmax(block, dim=1)
                                    if head.activation == "softmax"
                                    else torch.sigmoid(block))

    per_cat = collections.defaultdict(lambda: [0, 0])
    head_stats = {h: [0, 0] for h in (TOPIC, INTENT, CONTEXT_RELIANCE)}
    failures = []

    for i, p in enumerate(probes):
        pred = {}
        for head in active.heads:
            row = probs_by_head[head.name][i]
            labels = list(active.labels(head.name))
            if head.name == CONTEXT_RELIANCE:
                pred[head.name] = [name for j, name in enumerate(labels)
                                   if float(row[j]) >= ctx_thr[name]]
            else:
                tags = [name for j, name in enumerate(labels) if float(row[j]) > thr]
                if not tags:      # _tags_above's argmax fallback
                    tags = [labels[int(torch.argmax(row).item())]]
                pred[head.name] = tags

        # Derived scalars, via the real derivation.
        raw = []
        for head in active.heads:
            raw.extend(probs_by_head[head.name][i].tolist())
        res = ClassificationResult([], [], "Zero_Shot", raw, max(raw), p["text"])
        finalize_context_scalars(res, active)

        probe_ok = True
        detail = []
        for head_name, asserted in p["labels"].items():
            if head_name not in pred:
                continue
            got, want = set(pred[head_name]), set(asserted)
            if head_name == CONTEXT_RELIANCE:
                ok = got == want           # exact: silence matters as much as firing
            else:
                # topic/intent are multi-label and genuinely fuzzy; require the
                # asserted labels to be PRESENT, don't punish extras.
                ok = want.issubset(got) if want else not got
            head_stats[head_name][1] += 1
            head_stats[head_name][0] += int(ok)
            if not ok:
                probe_ok = False
                detail.append(f"{head_name}: want {sorted(want) or '∅'} got {sorted(got) or '∅'}")

        per_cat[p["category"]][1] += 1
        per_cat[p["category"]][0] += int(probe_ok)
        if not probe_ok:
            failures.append((p, detail, res))

    print(f"\n{'head':<20}{'pass':>8}{'total':>8}{'rate':>8}")
    for h, (ok, tot) in head_stats.items():
        print(f"{h:<20}{ok:>8}{tot:>8}{ok/max(1,tot):>8.0%}")

    print(f"\n{'category':<42}{'pass':>6}{'of':>5}")
    for c in sorted(per_cat):
        ok, tot = per_cat[c]
        flag = "  ← ALL FAIL" if ok == 0 and tot else ("" if ok == tot else "  ←")
        print(f"{c:<42}{ok:>6}{tot:>5}{flag}")

    total_ok = sum(v[0] for v in per_cat.values())
    print(f"\n[hard] probes fully correct: {total_ok}/{len(probes)} "
          f"({total_ok/len(probes):.0%})")

    if args.show != "none" and failures:
        print(f"\n{'='*78}\nFAILURES ({len(failures)})\n{'='*78}")
        for p, detail, res in failures:
            t = " ".join(p["text"].split())
            print(f"\n[{p['category']}]")
            print(f"  prompt : {t[:200]}{'…' if len(t) > 200 else ''}")
            if p.get("context_text"):
                print("  context: (present)")
            for d in detail:
                print(f"  MISS   : {d}")
            print(f"  p_mem={res.p_ltm:.2f} p_temporal={res.p_temporal:.2f} "
                  f"p_live={res.p_rts:.2f} p_complex={res.p_complex:.2f} "
                  f"→ derived={res.context_reliance}")
            print(f"  why    : {p['why']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"checkpoint": args.checkpoint, "probes": len(probes),
                       "fully_correct": total_ok,
                       "by_head": {k: v for k, v in head_stats.items()},
                       "by_category": {k: v for k, v in per_cat.items()}},
                      fh, indent=2)
        print(f"\n[hard] report → {args.out}")


if __name__ == "__main__":
    main()
