"""Does restoring `concept`/`object` fix the codex grounding whitelist, or wreck it?

⚑ THE QUESTION. `_background_labels()` excludes `concept` and `object` because
they are junk magnets — measured, and right for the two consumers the background
tier was built for (`turn_density.extract_key_terms`, `clustering`), which both
want PRECISION. The codex grounding whitelist wants the opposite: PROVENANCE
2026-08-03, on why the micro-NER stayed on that path — "a long noisy list gives
the model more legal subjects and it discards the junk itself, while a short
clean list FORBIDS REAL FACTS." Running the 293-turn NuNER arm on the
precision-tuned list would confound "NuNER lost" with "the label list lost".
This settles the label list first, at n=small, before that arm runs.

⚑ NO LLM IS INVOLVED, DELIBERATELY. Run-to-run variance on the extraction
pipeline is ±60% at temperature 0 (PROVENANCE 2026-08-03: the same arm scored
104 and 166 triplets on the same turns), so a small run cannot resolve anything
downstream of the model. It CAN resolve a label-set question, because that
failure is categorical: the NER either returns a string or it does not. So this
probe stops at the confirmed-entity list — which is also where the evidence
actually is. `_ground_triplets` grounds a term iff its normalised form equals,
or is a token-subset of, a confirmed entity; a term absent from the list cannot
ground, mechanically. Confidence is the consequence, not the measurement.

⚑ SAMPLING IS A RULE, NOT AN INTENTION. Sampling failed twice in the previous
session: `ORDER BY id LIMIT 30` drew 25 of 30 turns from one conversation, and a
`length BETWEEN 300 AND 2000` cap then dropped 238 of 293 turns UNEVENLY (it
kept 45 of one conversation and 2 of 87 of another, because median turn length
runs 2540 / 6253 / 5264 chars). So:
  · STRATIFIED across all three conversations — they are three different
    registers (narrative/philosophical, tech/hardware, LinkedIn/education) and
    `concept` behaves differently in each. Sampling one register measures one.
  · STRATIFIED BY POSITION inside each conversation (evenly spaced indices), so
    a register drift late in a long conversation is not invisible.
  · NO LENGTH CAP. Whatever the turn is, it goes in.

Usage:
    uv run python scripts/oneoff/probe_grounding_label_set.py
    uv run python scripts/oneoff/probe_grounding_label_set.py --per-conv 5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.z1.derive_retrieval_gt import load_conversations       # noqa: E402
from src.memory.embedder import get_embedder                        # noqa: E402
from src.retrieval import ner_utils                                 # noqa: E402
from src.retrieval.ner_utils import (_background_labels,            # noqa: E402
                                     _clean_entity, _merge_word_spans,
                                     _NER_STOP, extract_entities)

#: A HIGHLIGHTER, NOT A METRIC. These are function/verb words that in this
#: repo's measured failures led a bled span (`finally got`, `gave episodic`,
#: `My CGPA`, `uses PostgreSQL`). A span starting with one is worth looking at
#: by eye; the eye decides, not this list. Deliberately tiny and closed — a
#: bigger one would start making the judgement instead of pointing at it.
_BLEED_LEAD = {
    "my", "his", "her", "their", "our", "your", "its",
    "is", "was", "are", "were", "be", "been",
    "uses", "used", "use", "gave", "give", "got", "get", "has", "have", "had",
    "check", "checks", "make", "makes", "made", "and", "but", "not", "the",
    "a", "an", "finally", "just", "really", "very", "so",
}


def sample_turns(per_conv: int) -> list:
    """Evenly spaced turns from every conversation. No length cap, no ORDER BY."""
    convs = load_conversations()
    picked = []
    for cid in sorted(convs):
        turns = convs[cid]["turns"]
        n = len(turns)
        # Evenly spaced across the WHOLE conversation: for k picks, take the
        # midpoints of k equal bands. Deterministic and reproducible; covers
        # early/mid/late rather than clustering wherever ids happen to sort.
        idxs = [int((2 * j + 1) * n / (2 * per_conv)) for j in range(per_conv)]
        for j in idxs:
            t = turns[min(j, n - 1)]
            text = f"{t.get('user_input') or ''}\n\n{t.get('ai_response') or ''}".strip()
            picked.append({"conv": cid, "turn": t.get("turn_number"),
                           "chars": len(text), "text": text})
    return picked


def background_with_labels(text: str, labels) -> list:
    """Mirror of `_extract_background`, keeping each span's LABEL.

    ⚠ A MIRROR IS AN ADJACENT SYSTEM UNTIL PROVEN OTHERWISE. The caller asserts
    this returns the same entity strings, in the same order, as the production
    `extract_entities(..., tier="background")` — if it ever drifts, the label
    histogram is describing something production does not do and the run says
    so instead of quietly reporting it.
    """
    from src.api.config import settings
    model = ner_utils._load_background_ner()
    if model is None:
        return []
    chunk = int(settings.background_ner_chunk_words)
    threshold = float(settings.background_ner_threshold)

    found = []
    words = text.split()
    for i in range(0, len(words), chunk):
        piece = " ".join(words[i:i + chunk])
        spans = model.predict_entities(piece, list(labels), threshold=threshold)
        for s in _merge_word_spans(spans, piece):
            found.append((s["text"].strip(), s.get("label", "?")))

    cleaned, seen = [], set()
    for name, label in found:
        name = _clean_entity(name)
        low = name.lower()
        if name and low not in _NER_STOP and low not in seen:
            seen.add(low)
            cleaned.append((name, label))
    return cleaned


def bleed_flags(names) -> list:
    """Spans whose first token is a known bleed-leader. A highlighter."""
    out = []
    for n in names:
        head = n.split()[0].lower() if n.split() else ""
        if head in _BLEED_LEAD and len(n.split()) > 1:
            out.append(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-conv", type=int, default=3,
                    help="turns sampled per conversation (evenly spaced)")
    ap.add_argument("--show", type=int, default=40,
                    help="entities printed per config per turn")
    args = ap.parse_args()

    embedder = get_embedder()
    turns = sample_turns(args.per_conv)
    narrow = _background_labels()
    wide = _background_labels(extra_types=["concept", "object"])
    print(f"labels: narrow {len(narrow)} · wide {len(wide)} "
          f"(+{sorted(set(wide) - set(narrow))})")
    print(f"sampled {len(turns)} turns, {args.per_conv} per conversation, "
          f"evenly spaced, NO length cap\n")

    agg = {"micro": Counter(), "nuner_narrow": Counter(), "nuner_wide": Counter()}
    hist = {"nuner_narrow": Counter(), "nuner_wide": Counter()}
    totals = {k: 0 for k in agg}
    bleeds = {k: [] for k in agg}
    added_by_wide = []
    mirror_ok = True

    for t in turns:
        print("=" * 74)
        print(f"conv {t['conv']} · turn {t['turn']} · {t['chars']} chars")

        micro = extract_entities(t["text"], embedder, tier="preflight")
        nn = extract_entities(t["text"], embedder, tier="background", labels=narrow)
        nw = extract_entities(t["text"], embedder, tier="background", labels=wide)

        # The mirror must agree with production before its labels mean anything.
        for tag, prod, labels in (("narrow", nn, narrow), ("wide", nw, wide)):
            mirrored = background_with_labels(t["text"], labels)
            names = [n for n, _ in mirrored]
            if names != prod:
                mirror_ok = False
                print(f"  ⚠ MIRROR DRIFT ({tag}): {len(names)} vs production "
                      f"{len(prod)} — label histogram NOT trustworthy")
            key = "nuner_narrow" if tag == "narrow" else "nuner_wide"
            for _, lab in mirrored:
                hist[key][lab] += 1

        for key, ents in (("micro", micro), ("nuner_narrow", nn),
                          ("nuner_wide", nw)):
            totals[key] += len(ents)
            agg[key].update(e.lower() for e in ents)
            bleeds[key].extend(bleed_flags(ents))

        gained = [e for e in nw if e.lower() not in {x.lower() for x in nn}]
        lost = [e for e in nn if e.lower() not in {x.lower() for x in nw}]
        added_by_wide.extend(gained)

        print(f"  micro-NER        {len(micro):3d}  {micro[:args.show]}")
        print(f"  NuNER narrow     {len(nn):3d}  {nn[:args.show]}")
        print(f"  NuNER wide       {len(nw):3d}  {nw[:args.show]}")
        print(f"  ↳ wide ADDS      {len(gained):3d}  {gained[:args.show]}")
        if lost:
            print(f"  ↳ wide LOSES     {len(lost):3d}  {lost[:args.show]}")
        for key, label in (("micro", "micro-NER"), ("nuner_narrow", "NuNER narrow"),
                           ("nuner_wide", "NuNER wide")):
            b = bleed_flags({"micro": micro, "nuner_narrow": nn,
                             "nuner_wide": nw}[key])
            if b:
                print(f"  ⚑ bleed? {label:<13} {b[:12]}")

    print("\n" + "=" * 74)
    print("TOTALS across the sample")
    for key in ("micro", "nuner_narrow", "nuner_wide"):
        n = len(turns)
        print(f"  {key:<13} {totals[key]:5d} entities "
              f"({totals[key]/n:6.1f}/turn) · "
              f"{len(agg[key]):4d} distinct · "
              f"{len(bleeds[key]):3d} bleed-flagged")

    print("\nLABEL HISTOGRAM (NuNER) — a store typed mostly `concept` changes")
    print("which pairs G50's duplicate detector ever sees: its cosine channel")
    print("joins on entity_type equality, which removed 454 of 2,247 pairs.")
    for key in ("nuner_narrow", "nuner_wide"):
        tot = sum(hist[key].values()) or 1
        top = ", ".join(f"{k} {v} ({100*v/tot:.0f}%)"
                        for k, v in hist[key].most_common(10))
        print(f"  {key:<13} {top}")

    print(f"\nWHAT `concept`+`object` ADD ({len(added_by_wide)} total, "
          f"{len(set(a.lower() for a in added_by_wide))} distinct) — READ THESE:")
    for name, c in Counter(a.lower() for a in added_by_wide).most_common(60):
        print(f"    {c:3d}x  {name}")

    if not mirror_ok:
        print("\n⚠ MIRROR DRIFTED — the label histogram is not trustworthy.")
        return 1
    print("\nmirror OK: label histogram is of the same extraction production runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
