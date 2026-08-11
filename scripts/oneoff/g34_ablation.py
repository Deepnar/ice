"""G34 — measure candidate relation detectors before designing one.

Read-only. No DB writes, no LLM. Reproduces the 2026-08-08 baseline recorded in
PROVENANCE.md ("G34 — the relation detector, measured") and runs three candidate
fixes against the same probes, so the design decision is made on numbers rather
than on which idea sounds best.

The baseline finding this exists to fix: absolute cosine is ANTI-correlated with
relational content. `"ok"` clears 197/197 relations at floor 0.45 (top-1 0.844)
while `"who inspired Kael"` clears 43 (top-1 0.575) — short contentless strings
sit near the centroid and are therefore close to everything.

Candidates:
  base  absolute cosine >= floor                         (today)
  B     centre both sides on the vocabulary centroid, then absolute cosine
  A     z-score within the prompt's OWN similarity distribution
  A+B   centre, then z-score

C (require a resolved entity before running channel 2) is a GATE, orthogonal to
all of these, and its effect depends on the store's entity set rather than on
the vocabulary — measured separately, not here.

Two probe sets, deliberately:
  * the ten PROVENANCE prompts, which have known labels and let this be
    compared against the recorded baseline;
  * every user turn in the curated corpus, unlabelled, for the firing-RATE
    distribution. A detector that fires on ~100% of real traffic is broken
    whatever the labelled probes say, and real prompts cannot agree with the
    author the way hand-written ones do (TRAPS #13).

Style variants hold meaning fixed and vary only form (G28): a design whose
decision flips on a question mark has traded one bet for another.

Run:  uv run python scripts/oneoff/g34_ablation.py
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.memory.embedder import get_embedder          # noqa: E402
from src.workers.codex_extractor import ALLOWED_RELATIONS  # noqa: E402

FLOOR = 0.45          # settings.codex_relation_sim_floor at the time of measure
TOP_K = 5             # settings.codex_relation_top_k

# (prompt, expected relation substring or None if nothing should fire)
PROBES = [
    ("ok", None),
    ("hello", None),
    ("thanks, that helped", None),
    ("what is 2 + 2", None),
    ("write me a haiku about rain", None),
    ("what does Kael own", "own"),
    ("what is Kael using for the ritual", "use"),
    ("what does Kael use for the ritual", "use"),
    ("who is Rika married to", "married"),
    ("who inspired Kael", "inspired"),
]

# G28: same meaning, varied form. Every row must reach the same decision.
STYLE_VARIANTS = [
    ["who inspired Kael", "who inspired kael?", "ok so like who inspired Kael",
     "Kael was inspired by whom", "who insipred Kael"],
    ["ok", "ok.", "okay", "k", "OK"],
]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def fired(sims, mode, centred_sims, z_floor=2.5):
    """Relations this design would return, as indices."""
    if mode == "base":
        idx = np.where(sims >= FLOOR)[0]
        return idx[np.argsort(-sims[idx])][:TOP_K]
    if mode == "B":
        idx = np.where(centred_sims >= FLOOR)[0]
        return idx[np.argsort(-centred_sims[idx])][:TOP_K]
    s = sims if mode == "A" else centred_sims
    sd = s.std()
    if sd == 0:
        return np.array([], dtype=int)
    z = (s - s.mean()) / sd
    idx = np.where(z >= z_floor)[0]
    return idx[np.argsort(-z[idx])][:TOP_K]


def main():
    emb = get_embedder()
    rels = sorted(ALLOWED_RELATIONS)
    gloss = unit(emb.encode([r.replace("_", " ") for r in rels],
                            convert_to_tensor=False, show_progress_bar=False))
    centroid = unit(gloss.mean(axis=0))
    gloss_c = unit(gloss - np.outer(gloss @ centroid, centroid))

    def analyse(text):
        p = unit(emb.encode(text, convert_to_tensor=False))
        p_c = unit(p - (p @ centroid) * centroid)
        return gloss @ p, gloss_c @ p_c

    modes = ["base", "B", "A", "A+B"]

    print("=" * 78)
    print("LABELLED PROBES — count fired (want: 0 for negatives, >=1 correct for positives)")
    print("=" * 78)
    print(f"{'prompt':38} {'want':9} " + " ".join(f"{m:>8}" for m in modes))
    score = {m: {"false_fire": 0, "hit": 0, "pos": 0} for m in modes}
    for text, want in PROBES:
        sims, csims = analyse(text)
        cells = []
        for m in modes:
            f = fired(sims, m, csims)
            names = [rels[i] for i in f]
            if want is None:
                score[m]["false_fire"] += len(f)
                cells.append(f"{len(f):>8}")
            else:
                score[m]["pos"] += 1
                ok = any(want in n for n in names)
                score[m]["hit"] += int(ok)
                cells.append(f"{len(f):>6}{'*' if ok else ' '} ")
        print(f"{text[:38]:38} {str(want or 'NONE'):9} " + " ".join(cells))

    print("\n  * = the expected relation was among those returned")
    print(f"\n{'design':8} {'spurious fires on 5 negatives':32} {'positives hit'}")
    for m in modes:
        s = score[m]
        print(f"{m:8} {s['false_fire']:>10} (of {5*len(rels)} possible)   "
              f"{s['hit']}/{s['pos']}")

    # ---- real traffic: firing rate, unlabelled -----------------------------
    prompts = []
    for f in glob.glob("experiments/curation_files/*.json"):
        if f.endswith("derived_gt.json"):
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        for t in (d.get("historical_context_block") or [])[:12]:
            u = (t.get("user_input") or "").strip()
            if u:
                prompts.append(u[:400])
    prompts = prompts[:300]

    if prompts:
        print("\n" + "=" * 78)
        print(f"REAL CORPUS TURNS (n={len(prompts)}) — relations fired per prompt")
        print("=" * 78)
        print(f"{'design':8} {'mean':>8} {'median':>8} {'% firing':>10} {'% firing >=1':>14}")
        for m in modes:
            counts = []
            for text in prompts:
                sims, csims = analyse(text)
                counts.append(len(fired(sims, m, csims)))
            counts = np.array(counts)
            print(f"{m:8} {counts.mean():>8.2f} {np.median(counts):>8.1f} "
                  f"{100*(counts > 0).mean():>9.1f}% {100*(counts >= 1).mean():>13.1f}%")

    # ---- threshold sweep ---------------------------------------------------
    # Comparing designs at ONE threshold is invalid: centring changes the scale
    # of the similarities, so 0.45 means something different per design. Each
    # design gets its own sweep and is judged at its own best operating point.
    print("\n" + "=" * 78)
    print("THRESHOLD SWEEP — each design at its own best operating point")
    print("=" * 78)
    print("  neg = mean relations fired across the 5 negatives (want 0)")
    print("  pos = positives whose expected relation was returned (want 5)")
    print("  rate = % of 300 real corpus turns firing anything\n")

    cache = {t: analyse(t) for t, _ in PROBES}
    corpus_cache = [analyse(t) for t in prompts[:120]] if prompts else []

    for mode in modes:
        raw = mode in ("base", "A")
        grid = np.arange(0.30, 0.86, 0.05) if mode in ("base", "B") else np.arange(1.0, 5.1, 0.5)
        print(f"  {mode}")
        for thr in grid:
            def f_at(sims, csims):
                s = sims if raw else csims
                if mode in ("base", "B"):
                    idx = np.where(s >= thr)[0]
                    return idx[np.argsort(-s[idx])][:TOP_K]
                sd = s.std()
                if sd == 0:
                    return np.array([], dtype=int)
                z = (s - s.mean()) / sd
                idx = np.where(z >= thr)[0]
                return idx[np.argsort(-z[idx])][:TOP_K]

            neg, hit, npos = [], 0, 0
            for text, want in PROBES:
                sims, csims = cache[text]
                names = [rels[i] for i in f_at(sims, csims)]
                if want is None:
                    neg.append(len(names))
                else:
                    npos += 1
                    hit += int(any(want in n for n in names))
            rate = (100 * np.mean([len(f_at(s, c)) > 0 for s, c in corpus_cache])
                    if corpus_cache else float("nan"))
            print(f"    thr={thr:>5.2f}  neg={np.mean(neg):>5.2f}  "
                  f"pos={hit}/{npos}  rate={rate:>5.1f}%")

    # ---- G28 invariance ----------------------------------------------------
    print("\n" + "=" * 78)
    print("STYLE INVARIANCE (G28) — same meaning, varied form; want identical decisions")
    print("=" * 78)
    for group in STYLE_VARIANTS:
        print(f"\n  group: {group[0]!r}")
        for m in modes:
            decisions = []
            for text in group:
                sims, csims = analyse(text)
                decisions.append(tuple(rels[i] for i in fired(sims, m, csims)))
            n_distinct = len(set(decisions))
            counts = [len(d) for d in decisions]
            flag = "OK " if n_distinct == 1 else "FLIP"
            print(f"    {m:6} {flag} distinct={n_distinct}/{len(group)}  counts={counts}")


if __name__ == "__main__":
    main()
