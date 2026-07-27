#!/usr/bin/env python3
"""D8 — DI3's sentence: measure the v2 head against DI3 on the rows DI3 intercepts.

B1's spec (D8) fixes the order as **promote → measure → delete**: the fast-path
heuristics only earn their place if they beat the model *on the rows they
actually take over*, so the comparison has to run against a promoted checkpoint.
That happened 2026-07-27, which is what unblocks this script.

**Slice definition — the rows DI3 intercepts, not the rows a label says are
noisy.** ``run_di3`` evaluates five rules in order and returns the FIRST that
fires, so a row belongs to the ``code`` slice only if it clears the code
threshold *and* did not already trip ``noise``. This script reproduces that
ordering exactly (it imports the live thresholds and signal functions rather
than restating them) so each slice is precisely the population that path is
responsible for. Rows no rule claims are reported as ``passed_to_ml`` — the
majority, and DI3's actual main behaviour.

**conversation_length is 0. Always.** ``run_di3`` takes it, and three of the five
rules branch on it (code: >0, sentiment: >5, reference: >10), but a grep over
``src/`` finds **no caller that passes it**: ``main.py:315``,
``services/retrieval_svc.py:96`` and ``ingestion/importer.py:312`` all call
``classify()`` without it, so it defaults to 0 through every production path.
The Long_Term_Memory branches of the code and sentiment rules are therefore
unreachable in the live system, and the reference rule's documented two-tier
threshold (0.2 → 0.1 past ten turns) has never once used its second tier. This
script measures what actually runs — and reports the counterfactual too, so the
deletion decision does not rest on a bug that could be fixed.

**Two comparisons per slice, because they answer different questions:**

* *tags* — topic and intent micro-F1 of DI3's hardcoded lists vs the head's
  thresholded tags. This is what feeds the orchestrator's leg weights.
* *the retrieval decision* — the one that matters. Both arms are pushed through
  the real ``decide_memory_retrieval`` (B2), with DI3's arm carrying the
  ``_DI3_PRIORS`` scalar that ``finalize_context_scalars`` gives a fast-path
  result, and the ML arm carrying the real head probabilities. Accuracy is
  against gold ``Needs_Memory``.

The ``reference`` rule is scored differently on purpose: it emits no tags (it
returns empty lists, and ``classify()`` falls through to the MLP for them), so
its only effect is ``reference_signal=True`` → B2's ``ltm_bump_reference``. Its
slice therefore compares the ML result *with* that bump against the same result
*without* it — which is simultaneously the measurement of whether the knob earns
its keep.

Usage:
    uv run python scripts/classifier/pipeline/eval_di3.py
    uv run python scripts/classifier/pipeline/eval_di3.py --splits test val
"""

import argparse
import json
import os

import torch
# `common` first, and deliberately not isort-ordered: importing it is what runs
# the sys.path.insert + chdir(ROOT) that every `src.` import below depends on.
# Every sibling stage does the same; do not let a formatter "fix" this.
from common import DATA_DIR, TEST_SPLIT, VAL_SPLIT

from src.api.config import settings
from src.api.memory_decision import decide_memory_retrieval
from src.classifier import templates
from src.classifier.dataset import ICEClassifierDataset, encode_rendered
from src.classifier.model import load_checkpoint
from src.classifier.schema import (CONTEXT_RELIANCE, INTENT, NEEDS_MEMORY,
                                   TOPIC, finalize_context_scalars, load_schema)
from src.classifier.schemas import ClassificationResult
from src.retrieval.timescope import detect_timescope

HARD_PROBES = os.path.join(DATA_DIR, "hard_probes_authored.jsonl")
USER_PROBES = os.path.join(DATA_DIR, "eval_probes_independent.jsonl")

# ─────────────────────────────────────────────────────────────────────────────
# FROZEN COPY OF DI3, as it stood at commit d981ca9 (the last commit before this
# script's verdict deleted it).
#
# Verbatim from the deleted src/classifier/di3_signals.py, di3_config.py and
# di3.py. It is frozen here for the same reason templates.py freezes the v1
# prompt strings: a measurement that justifies a deletion has to stay runnable
# after the deletion, or the finding becomes an assertion nobody can re-check.
# Do NOT "improve" any of it — it is a historical record, not live code.
# ─────────────────────────────────────────────────────────────────────────────
_SENTIMENT_WORDS = {
    "feel", "felt", "feeling", "frustrated", "upset", "angry", "happy",
    "sad", "love", "hate", "excited", "worried", "scared", "tired",
    "overwhelmed", "depressed", "anxious", "stressed", "hopeless",
    "grateful", "thankful", "annoyed", "irritated", "confused", "lost",
}
_CODE_FEATURES = {
    "```": 0.4, "=": 0.1, "==": 0.1, "!=": 0.1, ">": 0.1, "<": 0.1,
    "def": 0.1, "class": 0.1, "function": 0.1, "import": 0.1,
    "{": 0.1, "}": 0.1, ";": 0.1,
    "print": 0.05, "return": 0.05, "if": 0.05, "else": 0.05,
    "for": 0.05, "while": 0.05,
}
_META_KEYWORDS = {"you", "your", "model"}
_META_PHRASES = {"prompt", "prompting"}
_META_PATTERNS = {"how do i prompt", "what model", "which model",
                  "how should i prompt"}
_KEYBOARD_MASH = {"asdf", "qwerty", "asdfghjkl", "zzzzzzzz", "asd;fkj"}
_REFERENCE_WEIGHTS = {"this": 0.15, "that": 0.1, "it": 0.05,
                      "these": 0.1, "those": 0.1, "the": 0.05}
# The DI3_* settings, at their shipped defaults (none were ever set in .env).
PRE_D8_REFERENCE_BUMP = 1.2   # settings.ltm_bump_reference, deleted by D8

# T2's joint gate had a `reference_signal` arm, and `detect_timescope` no longer
# accepts the kwarg — so unlike everything else here, that half cannot be
# re-run. These are the numbers it produced on test+val (9,441 rows) at commit
# d981ca9, scored with the real per-row p_ltm. All 49 changed rows were long
# pasted documents (p_ltm 0.00–0.16) whose LENGTH had accumulated enough "the"s
# to clear the density threshold, so the arm's removal deletes 49 false temporal
# windows and no true ones.
PRE_D8_TIMESCOPE_NONCURRENT = 465
PRE_D8_TIMESCOPE_FLIPPED = 49


def extract_signals(text: str) -> dict:
    low, words = text.lower(), text.lower().split()
    code = min(sum(w for t, w in _CODE_FEATURES.items() if t in low), 1.0)

    sent = sum(0.1 for w in words if w in _SENTIMENT_WORDS)
    sent += sum(0.2 for p in ("i feel", "i'm feeling") if p in low)
    if any(p in low for p in ("i'm", "im")):
        sent += sum(0.15 for w in words if w in _SENTIMENT_WORDS)

    meta = (sum(0.1 for k in _META_KEYWORDS if k in low)
            + sum(0.15 for p in _META_PHRASES if p in low)
            + sum(0.2 for p in _META_PATTERNS if p in low))

    stripped = text.strip()
    noise = 0.0
    if len(stripped) < 5:
        noise += 0.2
    if not any(c.isalpha() for c in stripped):
        noise += 0.6
    if any(m in stripped.lower() for m in _KEYBOARD_MASH):
        noise += 0.3
    if len(set(stripped)) <= 3 and len(stripped) > 3:
        noise += 0.2

    ref = sum(_REFERENCE_WEIGHTS.get(w, 0.0) for w in words)
    return {"code_density": code, "sentiment_density": min(sent, 1.0),
            "meta_density": min(meta, 1.0), "noise_density": min(noise, 1.0),
            "reference_density": min(ref, 1.0)}


# The five rules in evaluation order, with the fixed output each returned.
# ``ctx_at_len0`` is what production got (conversation_length was never passed);
# ``ctx_if_long`` is the unreachable branch, measured as a counterfactual.
RULES = [
    {"name": "noise",     "signal": "noise_density", "threshold": 0.8,
     "topic": ["Null_Noise"], "intent": ["Casual_Banter"],
     "ctx_at_len0": "Zero_Shot", "ctx_if_long": "Zero_Shot", "confidence": 0.95},
    {"name": "code",      "signal": "code_density", "threshold": 0.3,
     "topic": ["Software_&_Tech"], "intent": ["Generation"],
     "ctx_at_len0": "Zero_Shot", "ctx_if_long": "Long_Term_Memory", "confidence": 0.90},
    {"name": "sentiment", "signal": "sentiment_density", "threshold": 0.4,
     "topic": ["Lifestyle_&_Health", "Social_&_Relationships"],
     "intent": ["Emotional_Processing"],
     "ctx_at_len0": "Zero_Shot", "ctx_if_long": "Long_Term_Memory", "confidence": 0.85},
    {"name": "meta",      "signal": "meta_density", "threshold": 0.2,
     "topic": ["Meta_AI"], "intent": ["Factual_Retrieval"],
     "ctx_at_len0": "Zero_Shot", "ctx_if_long": "Zero_Shot", "confidence": 0.90},
    # Reference emitted no tags — scored as a bump, see module docstring.
    {"name": "reference", "signal": "reference_density", "threshold": 0.2,
     "topic": [], "intent": [],
     "ctx_at_len0": "Long_Term_Memory", "ctx_if_long": "Long_Term_Memory",
     "confidence": 0.70},
]


def firing_rule(text: str) -> str:
    """Which DI3 rule claimed *text* — first match wins, exactly as run_di3 did."""
    signals = extract_signals(text)
    for rule in RULES:
        if signals[rule["signal"]] > rule["threshold"]:
            return rule["name"]
    return "passed_to_ml"


class SetScore:
    """Micro-averaged P/R/F1 over multi-label set predictions."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.exact = self.n = 0

    def add(self, pred, gold):
        pred, gold = set(pred), set(gold)
        self.tp += len(pred & gold)
        self.fp += len(pred - gold)
        self.fn += len(gold - pred)
        self.exact += int(pred == gold)
        self.n += 1

    @property
    def f1(self):
        denom = 2 * self.tp + self.fp + self.fn
        return (2 * self.tp / denom) if denom else 0.0

    @property
    def exact_rate(self):
        return self.exact / self.n if self.n else 0.0


def _binary(name, pred, gold):
    """Accuracy / precision / recall of a boolean decision over a slice."""
    tp = sum(1 for p, g in zip(pred, gold) if p and g)
    fp = sum(1 for p, g in zip(pred, gold) if p and not g)
    fn = sum(1 for p, g in zip(pred, gold) if not p and g)
    tn = sum(1 for p, g in zip(pred, gold) if not p and not g)
    n = max(1, len(gold))
    return {"label": name, "acc": (tp + tn) / n,
            "prec": tp / max(1, tp + fp), "rec": tp / max(1, tp + fn),
            "fires": (tp + fp) / n}


def _ml_results(rows, embeddings, model, active, threshold):
    """Real head probabilities → ClassificationResult, the production way."""
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
            if not tags:  # _tags_above's argmax fallback
                block = raw[head.offset:head.offset + head.width]
                tags = [labels[block.index(max(block))]]
            setattr(res, attr, tags)
        finalize_context_scalars(res, active)
        out.append(res)
    return out


def _di3_result(rule, text):
    """What run_di3 hands back for a fast-path hit, scalars and all."""
    res = ClassificationResult(
        topic_tags=list(rule["topic"]), intent_tags=list(rule["intent"]),
        context_reliance=rule["ctx"], raw_probs=[0.0] * 25,
        max_confidence=rule["confidence"], prompt=text)
    # classify() calls _finalize_confidence on fast-path results, which lands in
    # finalize_context_scalars' all-zero branch → the _DI3_PRIORS scalar.
    finalize_context_scalars(res, LIVE_SCHEMA)
    return res


def _retrieves(res, row, reference_signal=False, bump=None):
    """Run the real B2 decision for one row.

    ``reference_signal`` no longer exists on ClassificationResult (D8 removed it
    along with ``settings.ltm_bump_reference``), so the pre-D8 arm is reproduced
    here by adding the bump to a settings shim — which keeps this script runnable
    after the deletion it justifies.
    """
    total_tokens = (len(row.get("text") or "")
                    + len(row.get("context_text") or "")) / 4.0
    cfg = settings
    if reference_signal and bump:
        cfg = settings.model_copy(
            update={"ltm_prior_bias": settings.ltm_prior_bias + bump})
    return decide_memory_retrieval(res, turn_count=0, total_tokens=total_tokens,
                                   settings=cfg).retrieve


def main():
    ap = argparse.ArgumentParser(description="D8: DI3 vs the v2 head, per slice")
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["test", "val"])
    ap.add_argument("--checkpoint", default=settings.classifier_model_path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "d8_di3_slices.json"))
    args = ap.parse_args()

    global LIVE_SCHEMA
    LIVE_SCHEMA = load_schema()
    model, meta = load_checkpoint(args.checkpoint, schema=LIVE_SCHEMA)
    model.eval()
    active = getattr(model, "schema", LIVE_SCHEMA)
    threshold = float(meta.get("tag_threshold") or settings.classifier_threshold)
    print(f"[d8] checkpoint {args.checkpoint} "
          f"(schema v{meta.get('schema_version')}, tag_threshold {threshold})")

    rows, embs = [], []
    for split in args.splits:
        path = {"test": TEST_SPLIT, "val": VAL_SPLIT}[split]
        # Reuses the split's cached embeddings — same renderer, same template
        # version, so this is the exact input the head was trained and served on.
        ds = ICEClassifierDataset(path, schema=LIVE_SCHEMA, device=args.device,
                                  show_progress=False)
        rows.extend(ds.data)
        embs.append(ds.embeddings)
        print(f"[d8] {split}: {len(ds.data)} rows")
    embeddings = torch.cat(embs)

    ml = _ml_results(rows, embeddings, model, active, threshold)

    buckets = {}
    for i, row in enumerate(rows):
        buckets.setdefault(firing_rule(row.get("text", "")), []).append(i)

    report = {"checkpoint": args.checkpoint, "splits": args.splits,
              "total_rows": len(rows), "tag_threshold": threshold,
              "conversation_length_in_production": 0, "slices": {}}

    print(f"\n{'slice':<14}{'rows':>7}{'share':>8}")
    for name in [r["name"] for r in RULES] + ["passed_to_ml"]:
        idx = buckets.get(name, [])
        print(f"{name:<14}{len(idx):>7}{len(idx) / len(rows):>8.1%}")

    for rule in RULES:
        name = rule["name"]
        idx = buckets.get(name, [])
        if not idx:
            report["slices"][name] = {"rows": 0}
            continue

        gold_mem = [NEEDS_MEMORY in set(rows[i].get(CONTEXT_RELIANCE) or [])
                    for i in idx]
        entry = {"rows": len(idx), "gold_needs_memory": sum(gold_mem)}

        # ── the retrieval decision, both arms through real B2 ──────────────
        ml_retr = [_retrieves(ml[i], rows[i]) for i in idx]
        if name == "reference":
            # No tags to compare; the rule's entire effect is the bump.
            with_bump = [_retrieves(ml[i], rows[i], reference_signal=True,
                                    bump=PRE_D8_REFERENCE_BUMP) for i in idx]
            entry["decision"] = {
                "di3_on (ml + reference bump)": _binary("retrieve", with_bump, gold_mem),
                "di3_off (ml alone)": _binary("retrieve", ml_retr, gold_mem),
            }
            entry["bump_flips"] = sum(1 for a, b in zip(with_bump, ml_retr) if a != b)
        else:
            for branch in ("ctx_at_len0", "ctx_if_long"):
                rule_ctx = dict(rule, ctx=rule[branch])
                di3_retr = [_retrieves(_di3_result(rule_ctx, rows[i].get("text", "")),
                                       rows[i]) for i in idx]
                entry.setdefault("decision", {})[
                    f"di3 ({branch})"] = _binary("retrieve", di3_retr, gold_mem)
            entry["decision"]["ml (di3 deleted)"] = _binary("retrieve", ml_retr, gold_mem)

            # ── tags ──────────────────────────────────────────────────────
            tags = {}
            for head_name, key in ((TOPIC, "topic"), (INTENT, "intent")):
                di3_s, ml_s = SetScore(), SetScore()
                for i in idx:
                    gold = rows[i].get(key) or []
                    di3_s.add(rule["topic"] if head_name == TOPIC else rule["intent"], gold)
                    ml_s.add(getattr(ml[i], f"{key}_tags"), gold)
                tags[key] = {"di3_f1": di3_s.f1, "ml_f1": ml_s.f1,
                             "di3_exact": di3_s.exact_rate, "ml_exact": ml_s.exact_rate}
            entry["tags"] = tags

        report["slices"][name] = entry

    # ── print ─────────────────────────────────────────────────────────────
    for rule in RULES:
        name = rule["name"]
        entry = report["slices"][name]
        print(f"\n══ slice: {name}  ({entry['rows']} rows"
              + (f", {entry['gold_needs_memory']} gold Needs_Memory)" if entry["rows"] else ")"))
        if not entry["rows"]:
            print("   (empty — this path never fires on held-out data)")
            continue
        if "tags" in entry:
            for key, t in entry["tags"].items():
                verdict = "MODEL" if t["ml_f1"] >= t["di3_f1"] else "DI3"
                print(f"   {key:<7} micro-F1   DI3 {t['di3_f1']:.3f}   "
                      f"model {t['ml_f1']:.3f}   → {verdict}")
        print("   retrieval decision (vs gold Needs_Memory):")
        for arm, m in entry["decision"].items():
            print(f"     {arm:<28} acc {m['acc']:.3f}  prec {m['prec']:.3f}  "
                  f"rec {m['rec']:.3f}  fires {m['fires']:.1%}")
        if "bump_flips" in entry:
            print(f"   rows the bump flips: {entry['bump_flips']}")

    # ── the reference rule's OTHER consumer ───────────────────────────────
    # `reference_signal` does not only feed B2 — T2's joint gate reads it as the
    # third of four alternatives (`?` → interrogative → reference_signal →
    # p_ltm≥0.5). Deleting DI3 pins it False, so that arm goes dead and some
    # prompts stop resolving to a non-current TimeScope. Measured with the REAL
    # p_ltm, because the p_ltm arm sits directly below it in the elif chain and
    # rescues most of what the reference arm was carrying — scoring this at
    # p_ltm=0 would overstate the loss.
    without = sum(detect_timescope(row.get("text", ""),
                                   p_ltm=ml[i].p_ltm).mode != "current"
                  for i, row in enumerate(rows))
    report["timescope"] = {"non_current_now": without,
                           "non_current_pre_d8_measured": PRE_D8_TIMESCOPE_NONCURRENT,
                           "rows_that_changed_measured": PRE_D8_TIMESCOPE_FLIPPED}
    print("\n══ T2 joint gate (the reference rule's other consumer)")
    print(f"   non-current TimeScope now (arm removed)      : {without}")
    print(f"   non-current TimeScope pre-D8 (measured)      : "
          f"{PRE_D8_TIMESCOPE_NONCURRENT}")
    print(f"   rows that changed                            : "
          f"{PRE_D8_TIMESCOPE_FLIPPED}  "
          f"({PRE_D8_TIMESCOPE_FLIPPED / len(rows):.2%}, all false positives)")

    # ── the regression gate for THIS change ───────────────────────────────
    # Neither `score_hard_probes.py` nor `eval_probes.py` can detect a D8
    # regression: both call `load_checkpoint` directly and never go through
    # `classify()`, so DI3 was invisible to them in the first place. The only
    # instrument that sees this change is the end-to-end retrieve decision on the
    # two independent probe sets, scored the way production makes it — which is
    # what this section does, for both the pre-D8 and post-D8 pipelines.
    hard = [json.loads(l) for l in open(HARD_PROBES) if l.strip()]
    user = [json.loads(l) for l in open(USER_PROBES) if l.strip()]
    probe_rows = ([(r, NEEDS_MEMORY in (r["labels"].get(CONTEXT_RELIANCE) or []))
                   for r in hard]
                  + [(r, True) for r in user])  # user probes are all positives

    ds_rendered = [templates.render(r.get("text", ""), r.get("context_text"),
                                    version=int(meta.get("template_version", 2)))
                   for r, _ in probe_rows]
    probe_emb = encode_rendered(ds_rendered, device=args.device, show_progress=False)
    probe_ml = _ml_results([r for r, _ in probe_rows], probe_emb, model, active,
                           threshold)

    gold = [g for _, g in probe_rows]
    post = [_retrieves(probe_ml[i], probe_rows[i][0])
            for i in range(len(probe_rows))]
    # Pre-D8: DI3 intercepts first; a reference hit becomes the bump instead.
    pre = []
    for i, (row, _) in enumerate(probe_rows):
        name = firing_rule(row.get("text", ""))
        if name == "reference":
            pre.append(_retrieves(probe_ml[i], row, reference_signal=True,
                                  bump=PRE_D8_REFERENCE_BUMP))
        elif name != "passed_to_ml":
            rule = next(r for r in RULES if r["name"] == name)
            pre.append(_retrieves(_di3_result(dict(rule, ctx=rule["ctx_at_len0"]),
                                              row.get("text", "")), row))
        else:
            pre.append(_retrieves(probe_ml[i], row))

    report["probe_gate"] = {
        "rows": len(probe_rows), "positives": sum(gold),
        "pre_d8": _binary("retrieve", pre, gold),
        "post_d8": _binary("retrieve", post, gold),
        "rows_changed": sum(1 for a, b in zip(pre, post) if a != b),
    }
    print("\n══ end-to-end retrieve decision on the 311 independent probes")
    print("   (the only gate that can see D8 — the other two bypass classify())")
    for arm in ("pre_d8", "post_d8"):
        m = report["probe_gate"][arm]
        print(f"   {arm:<9} acc {m['acc']:.3f}  prec {m['prec']:.3f}  "
              f"rec {m['rec']:.3f}  fires {m['fires']:.1%}")
    print(f"   rows whose decision changes: {report['probe_gate']['rows_changed']}")

    # The standing weakness, re-measured. A missed memory-needing probe is
    # SILENT — retrieval never runs and nothing signals it — so this count is
    # re-taken after every change to B2, the head, or (as here) the pre-classifier.
    # Split by family: the two sets are not comparable and prior sessions quoted
    # the hard-set number ("7 of 49"), so reporting only the pooled count would
    # read as a regression when it is a different denominator.
    n_hard = len(hard)
    fams = {"hard (adversarial)": range(0, n_hard),
            "user (curation)": range(n_hard, len(probe_rows))}
    print("\n   ⚠ silent misses (needs memory, retrieval never runs):")
    silent_all = []
    for fam, rng in fams.items():
        pos = [i for i in rng if gold[i]]
        miss = [i for i in pos if not post[i]]
        was = sum(1 for i in pos if not pre[i])
        silent_all.extend(miss)
        report["probe_gate"].setdefault("silent_misses", {})[fam] = {
            "misses": len(miss), "positives": len(pos), "pre_d8": was}
        print(f"     {fam:<20} {len(miss):>3}/{len(pos):<4} [was {was} pre-D8]")
    for i in silent_all[:8]:
        print(f"       · {probe_rows[i][0].get('text', '')[:86]}")

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[d8] report → {args.out}")


if __name__ == "__main__":
    main()
