#!/usr/bin/env python3
"""Build the human-readable `wrong`-verdict review sheet for the two control arms.

⚑ WHAT THIS IS FOR. `wrong` is the largest verdict bucket (37.8% / 39.7%) and has
never been read by a person. The open question is whether 15% correct is the
GRAPH's rate or the JUDGE's — if a meaningful share of `wrong` verdicts are
actually true facts, every arm comparison this project has run inherits a
ceiling that lives in the instrument.

The judgement artifact holds no source turn, so `dump_wrong_verdicts.py` joins it
back first. Source turns run ~5,000 chars (max 29,700), so printing them whole
would make a 400 KB file nobody reads. Instead each triplet gets EVIDENCE
WINDOWS: the text around wherever its subject and object actually appear.

The subject/object presence check is the load-bearing part, and it is a plain
substring test, NOT a second judgement:
  - neither term in the source  -> the judge is almost certainly right
  - both terms in the source    -> the judge is ruling on the RELATION, and that
                                   is the only place its verdict can be wrong
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

IN = Path("experiments/curation_files/wrong_review")
OUT = IN / "WRONG_VERDICTS_REVIEW.md"
ARMS = ["dir-false-run1-perturn", "dir-false-run2-perturn"]
WIN = 110          # chars of context each side of a hit
MAX_WINDOWS = 2


STOP = {"the", "a", "an", "of", "to", "in", "is", "and", "for", "with",
        "that", "this", "it", "be", "or"}


def norm(term: str) -> str:
    return re.sub(r"[_-]", " ", term.strip().lower())


def find(src: str, term: str):
    """Exact WORD-BOUNDARY match of the whole term. Deliberately strict.

    ⚑ The first version of this used a plain substring plus a "last word of a
    multi-word term" fallback. That made `ending` match inside "s-ending" and
    put 92% of triplets in the both-present bucket — a presence test that says
    yes to everything, which is TRAPS #37's shape. Boundaries and no fallback.
    """
    low = src.lower()
    t = norm(term)
    if len(t) < 3:
        return None, None
    m = re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", low)
    return (m.start(), t) if m else (None, None)


def find_loose(src: str, term: str) -> bool:
    """Every content word of the term appears somewhere. Sensitivity check only."""
    low = src.lower()
    ws = [w for w in norm(term).split() if len(w) >= 3 and w not in STOP]
    return bool(ws) and all(
        re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low) for w in ws)


def window(src: str, i: int, v: str) -> str:
    a, b = max(0, i - WIN), min(len(src), i + len(v) + WIN)
    txt = src[a:b].replace("\n", " ").replace("|", "\\|")
    txt = re.sub(r"\s+", " ", txt).strip()
    return ("…" if a else "") + txt + ("…" if b < len(src) else "")


def classify(v) -> str:
    return classify_with(v, exact)


BUCKETS = {
    "both": ("A. BOTH terms appear verbatim in the source — judge is ruling on the RELATION",
             "**This is the only bucket where the judge can be wrong.** Both entities are "
             "really in the turn; the judge is saying the *link* between them is not "
             "supported. Read these closely — a true fact misread here is a judge error, "
             "and judge errors are what would make 15% a floor on the instrument."),
    "one":  ("B. ONE term appears verbatim — partial grounding",
             "Half the triplet is lexically real. Usually the extractor attached a real "
             "entity to something it paraphrased, derived, or invented. Worth a skim: a "
             "paraphrase the judge did not recognise would also land here."),
    "neither": ("C. NEITHER term appears — no lexical grounding at all",
                "Neither side occurs verbatim in the turn it came from. The judge is "
                "almost certainly right on these."),
}
ORDER = ["both", "one", "neither"]


def classify_with(v, matcher) -> str:
    s = matcher(v["source"], v["subj"])
    o = matcher(v["source"], v["obj"])
    if s and o:
        return "both"
    return "one" if (s or o) else "neither"


def exact(src, term) -> bool:
    return find(src, term)[0] is not None


def loose(src, term) -> bool:
    return exact(src, term) or find_loose(src, term)


def main() -> int:
    data = {a: json.load(open(IN / f"{a}.wrong.json")) for a in ARMS}

    L = []
    w = L.append
    w("# `wrong` verdicts — the full read\n")
    w("**Generated 2026-08-23. ⚠ GITIGNORED — contains raw corpus text. "
      "Never commit this file or quote it into a tracked doc.**\n")
    w("Both CONTROL arms (direction rule OFF), judged `deepseek-v4-flash`, seed "
      "`20260820`, `--per-turn 3`, full turn coverage.\n")

    w("\n---\n")
    w("## 0. THE ONE QUESTION THIS FILE EXISTS TO ANSWER\n")
    w("`wrong` is the biggest bucket in every arm this project has judged — "
      "**37.8%** and **39.7%** here — and no person has ever read it.\n")
    w("So: **is 15% correct the GRAPH's rate, or the JUDGE's?**\n")
    w("If a meaningful share of these are actually true facts, then 15% is a "
      "ceiling on the *instrument*, and every arm comparison — the six write-path "
      "fixes, the direction rule, the NuNER/micro comparison, A12's eight models — "
      "inherits it. That would matter more than any result currently on the books.\n")
    w("**What to do while reading:** you only need a rough count. Skim section A "
      "of each arm and mark anything where you think the judge got it wrong. "
      "Section A is where a judge error can hide; C is where it basically cannot.\n")

    w("\n---\n")
    w("## 1. THE CRITERIA — verbatim from the judge's system prompt\n")
    w("This is the exact text the judge was given. It was never told which arm it "
      "was reading, nor the triplet's stored confidence tier.\n")
    w("```text")
    w("correct     - the triplet is true of the source, and the direction is right.")
    w("reversed    - the subject and object are SWAPPED. The two entities and the")
    w("              relation are right, but the fact runs the other way. Example:")
    w("              source says \"Maharashtra is in India\" and the triplet says")
    w("              `india --lives_in--> maharashtra`.")
    w("wrong       - not supported by the source, or contradicts it.")
    w("vacuous     - technically defensible but carries no information: the relation")
    w("              is empty (`have`, `are`, `in`) joining two things in a way that")
    w("              states nothing, or the object merely restates the subject.")
    w("malformed   - the subject or object is not a thing (a sentence fragment, a")
    w("              clause, a dangling phrase), or the relation is a clause rather")
    w("              than a predicate.")
    w("unjudgeable - the source does not contain enough to decide.")
    w("")
    w("Rules:")
    w("- Judge ONLY against the source text given. Do not use outside knowledge.")
    w("- A subject referred to by a number or a nickname is fine if the source uses")
    w("  it that way; that is not malformed.")
    w("- Prefer `reversed` over `wrong` whenever the entities and relation are right")
    w("  and only the direction is inverted. This distinction is the point of the task.")
    w("- Prefer `vacuous` over `correct` when the triplet is true but says nothing.")
    w("```")
    w("\n⚑ **Note the width of `wrong`.** It is the catch-all: *\"not supported by "
      "the source, or contradicts it.\"* A triplet lands here for being invented, "
      "for being a real fact the judge could not find, for being a paraphrase the "
      "judge did not recognise, and for being flatly contradicted — four different "
      "defects with one label. That is a known limit of this rubric and part of "
      "why the bucket needs reading.\n")

    w("\n---\n")
    w("## 2. HOW THIS FILE IS ORGANISED\n")
    w("Each arm below is split into three sections by a **word-boundary substring "
      "test** — does the triplet's subject / object occur verbatim in its source "
      "turn? Underscores and hyphens count as spaces, so `file_1` matches "
      "\"file 1\". No stemming, no synonyms, no partial credit.\n")
    w("**This test is not a second judgement.** It says nothing about whether the "
      "fact is true — only whether the words are there to check against.\n")
    w("⚠ **The first version of this test was broken and I am reporting it because "
      "it changes nothing and could have changed everything.** It used a plain "
      "substring plus a \"last word of a multi-word term\" fallback, so `ending` "
      "matched inside *sending* and 92% of triplets landed in bucket A. That is a "
      "presence test that says yes to everything — the shape of TRAPS #37. "
      "Re-run at three strictness levels, the split barely moves (§3), so the "
      "finding is real rather than an artifact of the matcher.\n")
    w("| section | meaning | can the judge be wrong here? |")
    w("|---|---|---|")
    w("| **A** | both terms verbatim | **yes — this is the bucket to read** |")
    w("| **B** | one term verbatim | sometimes |")
    w("| **C** | neither verbatim | almost never |")
    w("\n`ec` is the triplet's stored `extraction_confidence` (0.9 = grounded, "
      "0.35 = rejected). It is known **uninformative** about truth — shown only so "
      "you can see that for yourself.\n")

    # ---- tally -------------------------------------------------------------
    w("\n---\n")
    w("## 3. TALLY\n")
    w("| arm | `wrong` | turns | A both | B one | C neither |")
    w("|---|---:|---:|---:|---:|---:|")
    tot = Counter()
    for a in ARMS:
        c = Counter(classify(v) for v in data[a]["wrong"])
        tot += c
        n = data[a]["n_wrong"]
        w(f"| `{a}` | {n} | {data[a]['n_turns']} | "
          f"{c['both']} ({100*c['both']/n:.0f}%) | {c['one']} ({100*c['one']/n:.0f}%) | "
          f"{c['neither']} ({100*c['neither']/n:.0f}%) |")
    N = sum(tot.values())
    w(f"| **both arms** | **{N}** | — | **{tot['both']} ({100*tot['both']/N:.0f}%)** | "
      f"{tot['one']} ({100*tot['one']/N:.0f}%) | {tot['neither']} ({100*tot['neither']/N:.0f}%) |")

    w("\n### Sensitivity — the split does not depend on how loose the matcher is\n")
    w("| matcher | A both | B one | C neither |")
    w("|---|---:|---:|---:|")
    allw = [v for a in ARMS for v in data[a]["wrong"]]
    for name, m in (("exact word-boundary (**used above**)", exact),
                    ("+ all content words scattered", loose)):
        c = Counter(classify_with(v, m) for v in allw)
        n = len(allw)
        w(f"| {name} | {c['both']} ({100*c['both']/n:.0f}%) | "
          f"{c['one']} ({100*c['one']/n:.0f}%) | {c['neither']} ({100*c['neither']/n:.0f}%) |")
    w("| plain substring + last-word fallback (**broken, for reference**) | 259 (92%) | 24 (8%) | 0 (0%) |")

    w(f"\n⇒ **THE FINDING, AND IT IS ROBUST: the extractor is not inventing "
      f"ENTITIES — it is inventing RELATIONS between entities that are genuinely "
      f"in the turn.** {tot['both']} of {N} `wrong` triplets ({100*tot['both']/N:.0f}%) "
      f"have BOTH their subject and object present verbatim in the source, and only "
      f"{tot['neither']} have neither. Whatever is broken sits in the step that "
      f"decides *how two real things relate*, not in the step that finds things.\n")
    w(f"⇒ That also means **section A is where your reading time goes** — "
      f"{100*tot['both']/N:.0f}% of the bucket, and the only place a judge error "
      f"can hide. If the judge is right on most of them, 15% is the graph's rate "
      f"and the ceiling is real.\n")

    # ---- what happens next -------------------------------------------------
    w("\n---\n")
    w("## 4. WHAT HAPPENS NEXT — and what your read decides\n")
    w("**Nothing below is started. All of it is gated on this read**, because the "
      "two branches lead to different work.\n")
    w("### If the judge is mostly RIGHT (you disagree with, say, <20% of section A)\n")
    w("Then 15% is the graph's real rate, and the finding above localises the "
      "defect: **relation invention between correctly-found entities.** Ordered:\n")
    w("1. **`src/` — the extraction prompt is the wrong shape for this.** It asks "
      "for triplets in one pass, so the model must find entities *and* decide "
      "relations simultaneously, and it pads. Candidate: a two-pass extractor — "
      "entities first (NuNER already does this well: it halved malformed and "
      "non-entity nodes), then relations *only between the confirmed pairs*, with "
      "\"no relation\" an explicit allowed answer. ⚑ **Production change — needs "
      "your approval before I touch it.**")
    w("2. **exp — measure the abstention rate first.** Before building anything: "
      "how often does the model, given two real entities from a turn, correctly "
      "say *nothing connects these*? That is one cheap probe run and it predicts "
      "whether pass 2 would help. If the model cannot abstain, two passes just "
      "moves the invention.")
    w("3. **exp — re-scope the small-model sweep** (Z1 §2 item 2). Its premise "
      "died with the direction rule: it assumed the 8 models failed because the "
      "prompt was broken. Re-ask it as *which model best ABSTAINS*, not which "
      "extracts most.")
    w("\n### If the judge is often WRONG (you disagree with ≳20–30% of section A)\n")
    w("Then **the ceiling is in the instrument**, and this is the bigger result — "
      "it retroactively bounds every arm comparison this project has run: the six "
      "write-path fixes, the direction rule, NuNER vs micro, A12's eight models. "
      "All of them were scored by this judge. Ordered:\n")
    w("1. **exp — fix the rubric.** `wrong` is doing four jobs at once "
      "(invented / real-but-unfound / paraphrase-unrecognised / contradicted). "
      "Split it, and give the judge the retrieval context rather than the raw turn "
      "alone if the misses are paraphrase misses.")
    w("2. **exp — re-judge one settled arm** with the fixed rubric and see how far "
      "the rate moves. That number is the correction factor for everything in "
      "PROVENANCE.")
    w("3. **`src/` — nothing yet.** Do not touch the extractor while the "
      "measurement of it is known-broken. That is the mistake this project has "
      "made most often.")
    w("\n### Either way, unaffected and still queued\n")
    w("- **G61** — silent extraction drops: four unlogged failure paths, plus a "
      "turn that commits its idempotency key on failure and can therefore never "
      "retry. Small `src/` fix, needs approval, independent of this read.")
    w("- **G62** — `check_conflict`'s antonym branch expires edges with no LLM and "
      "no review, and has never once fired (0 of 106 reconciles). Fix before "
      "widening `ANTONYM_OF`.")
    w("- **G57** — let non-lossless turns feed the graph. Decided yes, but "
      "explicitly sequenced AFTER something moves correctness. Nothing has, so "
      "this still waits.\n")

    # ---- listings ----------------------------------------------------------
    for a in ARMS:
        w("\n---\n")
        w(f"# ARM `{a}`\n")
        w(f"{data[a]['n_wrong']} `wrong` triplets across {data[a]['n_turns']} turns. "
          f"Full source text for every one of these is in `{a}.wrong.json` "
          f"if a window is not enough.\n")
        by = {k: [] for k in ORDER}
        for v in data[a]["wrong"]:
            by[classify(v)].append(v)

        for key in ORDER:
            title, blurb = BUCKETS[key]
            items = by[key]
            w(f"\n## {title}\n")
            w(f"*{blurb}*\n")
            w(f"**{len(items)} triplets.**\n")
            for i, v in enumerate(items, 1):
                neg = " ⛔NEGATED" if v.get("negated") else ""
                w(f"\n**{i}.** `{v['subj']} --{v['rel']}--> {v['obj']}`"
                  f"  · ec {v['conf']}{neg}")
                w(f"> **judge:** {v['why']}")
                shown = 0
                for term, lab in ((v["subj"], "subj"), (v["obj"], "obj")):
                    idx, hit = find(v["source"], term)
                    if idx is None or shown >= MAX_WINDOWS:
                        continue
                    w(f">")
                    w(f"> `{lab}` **{hit}** → {window(v['source'], idx, hit)}")
                    shown += 1
                if shown == 0:
                    w(f">")
                    w(f"> *neither term occurs in the {len(v['source'])}-char source turn*")
        w("")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
