#!/usr/bin/env python3
"""E12 — do the labels B1 added actually reach a decision?

A label can be present in the taxonomy, trained, stored on every result, and
still change nothing. B1's consumer audit found `Codebase_Query` and `p_complex`
in that state; this stage measures the two labels whose fate is still open, so
the decision is made on numbers rather than on reading the call graph and
guessing how often the code path is taken.

**Code_Change.** Its only effect system-wide is one row in the orchestrator's
intent→leg-weight PROFILES, and the blend divides every profile's contribution
by `len(active_intents)`. So the question is not "is it wired up" (it is) but
"how much of the intended nudge survives contact with a real multi-label row",
which is a measurement: how often the head fires it, how many intents ride
along, and what the leg weights actually come out as with and without it.

**Temporal_Recall.** B1 D7 OR's it with T2's deterministic detector: either one
adds `ltm_bump_timescope` (+3.0) exactly once. An OR-arm only earns its place if
it fires on rows the other arm misses — if the label only ever agrees with the
detector, it is decoration on an already-made decision. So the measurement is
the *disagreement*: rows where the label fires alone, whether those rows really
need memory, and whether the OR flips the final retrieve decision at all.

Usage:
    uv run python scripts/classifier/pipeline/audit_labels.py
    uv run python scripts/classifier/pipeline/audit_labels.py --splits test val
"""

import argparse
import json
import os
from collections import Counter

import torch
# `common` first — importing it runs the sys.path.insert + chdir(ROOT) that
# every `src.` import below depends on. Not isort-ordered, deliberately.
from common import DATA_DIR, TEST_SPLIT, VAL_SPLIT

from src.api.config import settings
from src.api.memory_decision import decide_memory_retrieval
from src.classifier.dataset import ICEClassifierDataset
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, NEEDS_MEMORY,
                                   TOPIC, finalize_context_scalars, load_schema)
from src.classifier.schemas import ClassificationResult
from src.retrieval.timescope import detect_timescope

CODE_CHANGE = "Code_Change"
TEMPORAL_RECALL = "Temporal_Recall"

# Mirrors orchestrator._compute_dynamic_weights' PROFILES + blend, so the leg
# weights reported here are the ones retrieval would actually use.
PROFILES = [
    ({"Factual_Retrieval", "Utility_Formatting"},
     {"vector": 1.2, "bm25": 0.8, "codex": 0.1, "procedural": 0.1}),
    ({"Troubleshooting", "Strategic_Planning"},
     {"vector": 1.0, "bm25": 0.8, "codex": 0.3, "procedural": 1.2}),
    ({"Generation", "Ideation", "Open_Exploration"},
     {"vector": 0.6, "bm25": 0.6, "codex": 1.2, "procedural": 0.1}),
    ({"Emotional_Processing", "Analysis_&_Summarization", "Decision_Making"},
     {"vector": 1.1, "bm25": 0.6, "codex": 0.9, "procedural": 0.0}),
    ({"Casual_Banter", "Null_Noise"},
     {"vector": 0.5, "bm25": 0.2, "codex": 0.0, "procedural": 0.0}),
    ({CODE_CHANGE},
     {"procedural": 1.2, "codex": 1.0, "vector": 0.6, "bm25": 0.6}),
]
BASE = {"vector": 1.0, "bm25": 1.0, "codex": 0.5, "procedural": 0.2}
_BY_INTENT = {i: w for tags, w in PROFILES for i in tags}


def blend(intent_tags):
    """The orchestrator's per-intent blend, verbatim."""
    n = len(intent_tags) if intent_tags else 1
    out = {leg: 0.0 for leg in BASE}
    for tag in intent_tags:
        prof = _BY_INTENT.get(tag) or BASE
        for leg, w in prof.items():
            out[leg] += w / n
    return out if any(out.values()) else dict(BASE)


def _results(rows, embeddings, model, active, threshold):
    with torch.no_grad():
        logits = model(embeddings)
    out = []
    for i, row in enumerate(rows):
        raw = []
        for head in active.heads:
            block = logits[i:i + 1, head.slice]
            probs = (torch.softmax(block, dim=1) if head.activation == "softmax"
                     else torch.sigmoid(block))
            raw.extend(probs.squeeze(0).tolist())
        res = ClassificationResult([], [], "Zero_Shot", raw, max(raw),
                                   row.get("text", ""))
        for head_name, attr in ((TOPIC, "topic_tags"), (INTENT, "intent_tags")):
            head = active.head(head_name)
            labels = list(active.labels(head_name))
            tags = [lab for k, lab in enumerate(labels)
                    if raw[head.offset + k] > threshold]
            if not tags:
                block = raw[head.offset:head.offset + head.width]
                tags = [labels[block.index(max(block))]]
            setattr(res, attr, tags)
        finalize_context_scalars(res, active)
        out.append(res)
    return out


def _retrieve(res, row, settings_obj, timescope_mode=None):
    total = (len(row.get("text") or "") + len(row.get("context_text") or "")) / 4.0
    return decide_memory_retrieval(res, turn_count=0, total_tokens=total,
                                   settings=settings_obj,
                                   timescope_mode=timescope_mode).retrieve


def main():
    ap = argparse.ArgumentParser(description="E12: do B1's new labels reach a decision?")
    ap.add_argument("--splits", nargs="+", default=["test", "val"],
                    choices=["test", "val"])
    ap.add_argument("--checkpoint", default=settings.classifier_model_path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "e12_label_audit.json"))
    args = ap.parse_args()

    schema = load_schema()
    model, meta = load_checkpoint(args.checkpoint, schema=schema)
    model.eval()
    active = getattr(model, "schema", schema)
    threshold = float(meta.get("tag_threshold") or settings.classifier_threshold)

    rows, embs = [], []
    for split in args.splits:
        ds = ICEClassifierDataset({"test": TEST_SPLIT, "val": VAL_SPLIT}[split],
                                  schema=schema, device=args.device,
                                  show_progress=False)
        rows.extend(ds.data)
        embs.append(ds.embeddings)
    res = _results(rows, torch.cat(embs), model, active, threshold)
    print(f"[e12] {len(rows)} held-out rows · checkpoint schema "
          f"v{meta.get('schema_version')} · tag_threshold {threshold}")

    report = {"rows": len(rows), "checkpoint": args.checkpoint}

    # ── Code_Change ───────────────────────────────────────────────────────
    fired = [i for i, r in enumerate(res) if CODE_CHANGE in r.intent_tags]
    gold_cc = [i for i, r in enumerate(rows) if CODE_CHANGE in (r.get("intent") or [])]
    tp = len(set(fired) & set(gold_cc))
    companions = Counter()
    n_intents = Counter()
    for i in fired:
        n_intents[len(res[i].intent_tags)] += 1
        for t in res[i].intent_tags:
            if t != CODE_CHANGE:
                companions[t] += 1

    # How much of the profile actually survives the blend, per firing row.
    deltas = []
    for i in fired:
        with_cc = blend(res[i].intent_tags)
        without = blend([t for t in res[i].intent_tags if t != CODE_CHANGE]
                        or [CODE_CHANGE])
        deltas.append(max(abs(with_cc[leg] - without[leg]) for leg in BASE))

    print(f"\n══ {CODE_CHANGE}")
    print(f"   head fires it on          : {len(fired)}/{len(rows)} rows "
          f"({len(fired) / len(rows):.2%})")
    print(f"   gold has it on            : {len(gold_cc)} rows   "
          f"(precision {tp / max(1, len(fired)):.2f}, recall {tp / max(1, len(gold_cc)):.2f})")
    if fired:
        mean_n = sum(k * v for k, v in n_intents.items()) / len(fired)
        alone = n_intents.get(1, 0)
        print(f"   intents on those rows     : mean {mean_n:.2f}  "
              f"(alone on {alone} rows = {alone / len(fired):.1%})")
        print(f"   → its profile is divided by that, so it delivers "
              f"~{1 / mean_n:.0%} of its intended weight")
        print(f"   biggest leg-weight change : mean "
              f"{sum(deltas) / len(deltas):.3f} (base weights are ~0.2–1.2)")
        print(f"   rides with                : "
              f"{', '.join(f'{t}×{c}' for t, c in companions.most_common(4))}")
    report[CODE_CHANGE] = {
        "fires": len(fired), "gold": len(gold_cc),
        "precision": tp / max(1, len(fired)), "recall": tp / max(1, len(gold_cc)),
        "mean_intents_when_fired": (sum(k * v for k, v in n_intents.items())
                                    / len(fired)) if fired else 0,
        "mean_max_leg_delta": (sum(deltas) / len(deltas)) if deltas else 0,
    }

    # ── Temporal_Recall: does the OR-arm fire where the detector doesn't? ──
    thr = settings.temporal_label_threshold
    label_fires, det_fires, both, label_only, det_only = [], [], [], [], []
    for i, row in enumerate(rows):
        lab = res[i].p_temporal >= thr
        det = detect_timescope(row.get("text", ""),
                               p_ltm=res[i].p_ltm).mode != "current"
        if lab:
            label_fires.append(i)
        if det:
            det_fires.append(i)
        if lab and det:
            both.append(i)
        elif lab:
            label_only.append(i)
        elif det:
            det_only.append(i)

    gold_mem = [NEEDS_MEMORY in set(r.get(CONTEXT_RELIANCE) or []) for r in rows]
    lo_right = sum(1 for i in label_only if gold_mem[i])
    do_right = sum(1 for i in det_only if gold_mem[i])

    print(f"\n══ {TEMPORAL_RECALL}  (D7 OR-arm, threshold {thr})")
    print(f"   label p_temporal ≥ {thr}     : {len(label_fires)} rows")
    print(f"   T2 detector fires          : {len(det_fires)} rows")
    print(f"   BOTH (label is redundant)  : {len(both)}")
    print(f"   label ONLY (its unique add): {len(label_only)}  "
          f"— {lo_right} of them truly need memory "
          f"({lo_right / max(1, len(label_only)):.0%})")
    print(f"   detector ONLY              : {len(det_only)}  "
          f"— {do_right} truly need memory "
          f"({do_right / max(1, len(det_only)):.0%})")

    # Does the OR change the final answer? Compare the real decision with the
    # label arm live vs disabled (threshold pushed out of reach).
    off = settings.model_copy(update={"temporal_label_threshold": 1.01})
    flips = changed_right = 0
    for i, row in enumerate(rows):
        mode = "as_of" if i in det_fires else None
        a = _retrieve(res[i], row, settings, timescope_mode=mode)
        b = _retrieve(res[i], row, off, timescope_mode=mode)
        if a != b:
            flips += 1
            changed_right += int(a == gold_mem[i])
    print(f"   rows where the OR-arm flips the retrieve decision: {flips}"
          + (f"  ({changed_right} of them to the CORRECT answer)" if flips else ""))

    # ── where the label DOES belong: T2's seam, measured both directions ──
    # The bump above is inert because a question about the past needs memory by
    # definition, so the label is a SUBSET of the decision it was wired to. But
    # subset ≠ useless: only ~20% of memory queries are time-shaped, so this is
    # the only signal ICE has for "is this about the past". Measure the two
    # Track T consumers that could use that (roadmap T5).
    from src.retrieval import timescope as _ts
    gold_t = [TEMPORAL_RECALL in set(r.get(CONTEXT_RELIANCE) or []) for r in rows]
    now = _ts.datetime.now(_ts.timezone.utc)

    def _parses(text):
        clean = _ts._INLINE_CODE.sub(" ", _ts._FENCED_CODE.sub(" ", text))
        return bool(_ts._scan_expressions(clean, now)), clean

    # (a) as a FILTER on the gate — precision. The gate's p_ltm arm is a loose
    #     proxy for "is this temporal"; requiring p_temporal too should tighten
    #     it. A false window is the expensive error: it hides everything outside.
    print("\n   (a) T2 joint gate, as a FILTER — precision of the windows it opens:")
    gate_rows = {}
    for name, keep in ([("today", None)]
                       + [(f"AND p_temporal>={t}", t) for t in (0.5, 0.7, 0.85)]):
        idx = [i for i in det_fires if keep is None or res[i].p_temporal >= keep]
        hit = sum(1 for i in idx if gold_t[i])
        gate_rows[name] = {"fires": len(idx), "really_temporal": hit,
                           "precision": hit / max(1, len(idx))}
        print(f"       {name:<22} fires {len(idx):>4}   really temporal "
              f"{hit:>3} = {hit / max(1, len(idx)):.0%}")
    # ⚠ recorded so nobody re-tests it: as an extra OR-arm it is worthless.
    refused = [i for i, r in enumerate(rows)
               if i not in det_fires and (_parses(r.get("text", ""))[0]
                                          or any(c.search(_parses(r.get("text", ""))[1])
                                                 for c in _ts._EVOLUTION_CUES))]
    admitted = [i for i in refused if res[i].p_temporal >= thr]
    ok = sum(1 for i in admitted if gold_mem[i])
    print(f"       (as an extra OR-arm instead: {len(refused)} refused rows, "
          f"would admit {len(admitted)}, {ok} correctly — don't.)")

    # (b) the bigger half — time questions the parser never sees at all, where
    #     the ranker defaults to boosting *recent* on a question about the past.
    temporal_rows = [i for i, g in enumerate(gold_t) if g]
    no_parse = [i for i in temporal_rows if not _parses(rows[i].get("text", ""))[0]]
    print(f"\n   (b) gold-temporal rows with NO parseable date: {len(no_parse)}"
          f"/{len(temporal_rows)} ({len(no_parse) / max(1, len(temporal_rows)):.0%})")
    print("       These never reach the gate (early return), so mode stays")
    print("       `current` and _recency_params applies +0.25 toward NOW — "
          "backwards")
    print("       for a question about the past. The flat branch it wants "
          "already")
    print("       exists (range/evolution); it is just unreachable without a "
          "parse.")
    for i in no_parse[:4]:
        print(f"         · {rows[i].get('text', '')[:78]!r}")

    report[TEMPORAL_RECALL] = {
        "gate_as_filter": gate_rows,
        "gate_as_or_arm_refused": len(refused),
        "gate_as_or_arm_would_admit": len(admitted),
        "gate_as_or_arm_admitted_correctly": ok,
        "gold_temporal_rows": len(temporal_rows),
        "gold_temporal_without_parseable_date": len(no_parse),
        "threshold": thr, "label_fires": len(label_fires),
        "detector_fires": len(det_fires), "both": len(both),
        "label_only": len(label_only), "label_only_truly_needs_memory": lo_right,
        "detector_only": len(det_only), "detector_only_truly_needs_memory": do_right,
        "decision_flips": flips, "flips_to_correct": changed_right,
    }

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[e12] report → {args.out}")


if __name__ == "__main__":
    main()
