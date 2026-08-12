#!/usr/bin/env python3
"""Z1/Z2: mine the dropped relations from every arm into vocabulary CANDIDATES.

**What this is for.** G32 measured **67.8%** of the relations the extractor
produces landing outside the 197-word controlled vocabulary and being silently
destroyed. The eight-arm run turned that from a statistic into a corpus: every
arm logged what it produced and what was thrown away, so the union of those is
the best available evidence for what the vocabulary is actually MISSING.

**⚑ A low in-vocabulary rate is not a model defect, and ranking on it was wrong
(user, 2026-08-12).** The 197-word list is known-broken; a model scoring badly
against it may be producing better relations than the list can hold. And once
constrained decoding is switched on the model cannot leave the vocabulary at
all, so in-vocab becomes 100% by construction and stops measuring anything. The
run's value is this harvest, not that ranking.

**Three things it does that a plain frequency count does not:**

  1. **Rejects malformed output rather than counting it as a missing word.** The
     extractor's prompt renders the vocabulary as labelled category groups, and
     several models echo those headers back as relations — `# category:
     property: identity`. `codex_extractor` already detects that case
     deliberately (see its note on why it does not auto-remap). Bare
     punctuation, colon-prefixed fragments and whole sentences go the same way.
     The junk RATE is reported per arm, because it measures contract adherence,
     which is a real quality signal unlike in-vocab.
  2. **Groups surface forms that mean one thing.** `is_used_by` / `used by` /
     `uses` are one concept in three costumes; normalising first stops the
     ranking being decided by spelling.
  3. **⚑ Proposes the OPPOSITE of every candidate (user's request).** A
     vocabulary with `uses` but not `used_by`, or `likes` but not `dislikes`,
     forces the extractor to invert subject and object to say the obvious thing
     — and G32 already found the deterministic matcher inverting direction
     (`is_used → uses`) as a live hazard. Broadening by pairs closes that.

Output is a ranked, categorised proposal for a human to accept or reject. It
does NOT touch `ALLOWED_RELATIONS` — the vocabulary decision is Z2's and the
user's.

Run:
  uv run python scripts/z1/harvest_vocabulary.py
  uv run python scripts/z1/harvest_vocabulary.py --min-count 3 --top 120
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

LOGS = "experiments/curation_files/arm_stores/*.seed.log"
OUT = Path("experiments/curation_files/VOCAB_CANDIDATES.md")

_DROP_RE = re.compile(r"dropped=(\d+) kept=\d+ relations=\[(.*?)\]")

# Opposite-pair templates. Morphological first (cheap and reliable), then a
# small semantic list for the antonyms morphology cannot reach. Deliberately
# short: this proposes candidates for review, it does not decide the vocabulary,
# so a missing pair costs a suggestion rather than correctness.
_SEMANTIC_OPPOSITES = {
    "likes": "dislikes", "loves": "hates", "wants": "rejects",
    "starts": "ends", "began": "ended", "opens": "closes",
    "includes": "excludes", "adds": "removes", "creates": "destroys",
    "enables": "disables", "allows": "forbids", "accepts": "refuses",
    "supports": "opposes", "increases": "decreases", "gains": "loses",
    "remembers": "forgets", "trusts": "distrusts", "helps": "hinders",
    "before": "after", "above": "below", "parent_of": "child_of",
    "causes": "prevents", "solves": "breaks", "improves": "worsens",
}


def is_junk(rel: str) -> tuple[bool, str]:
    r = rel.strip()
    if not r:
        return True, "empty"
    low = r.lower()
    if low.startswith("#") or low.startswith("category:"):
        return True, "category-header echoed as a relation"
    if r.startswith(":"):
        return True, "colon-prefixed fragment"
    if not re.match(r"^[a-z]", low):
        return True, "does not start with a letter"
    if len(r) < 3:
        return True, "too short"
    if len(low.split()) > 4 or len(r) > 40:
        return True, "sentence rather than a relation"
    if re.search(r'["\']', r):
        return True, "contains a quote mark"
    return False, ""


def normalise(rel: str) -> str:
    """Case, separators, and leading helper verbs — the same shape
    `normalize_relation` uses in production, so candidates arrive in the form
    the extractor would actually emit."""
    r = rel.strip().lower()
    r = re.sub(r"[\s\-]+", "_", r)
    r = re.sub(r"[^a-z0-9_]", "", r)
    r = re.sub(r"^(is|are|was|were|be|been|has|have|had|does|do|did)_", "", r)
    return r.strip("_")


def opposite_of(rel: str) -> str | None:
    """The counter-relation, so the vocabulary broadens in PAIRS."""
    if rel in _SEMANTIC_OPPOSITES:
        return _SEMANTIC_OPPOSITES[rel]
    for a, b in _SEMANTIC_OPPOSITES.items():
        if rel == b:
            return a
    # Morphological inverse: the passive/active direction pair. This is the one
    # G32 named as a live hazard — the deterministic matcher was inverting
    # direction (`is_used` -> `uses`) because only one side existed.
    if rel.endswith("_by"):
        return rel[:-3]
    if rel.endswith("s") and not rel.endswith("ss"):
        return f"{rel[:-1]}ed_by"
    if rel.startswith("not_"):
        return rel[4:]
    return f"not_{rel}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--top", type=int, default=150)
    args = ap.parse_args()

    from src.workers.codex_extractor import ALLOWED_RELATIONS
    try:
        from src.workers.codex_extractor import _ALL_CATEGORIES_GROUPED as GROUPS
    except Exception:
        GROUPS = {}
    vocab = {r.lower() for r in ALLOWED_RELATIONS}

    counts: Counter = Counter()
    per_arm: dict[str, Counter] = defaultdict(Counter)
    junk_reasons: Counter = Counter()
    arm_totals: Counter = Counter()
    arm_junk: Counter = Counter()

    files = sorted(glob.glob(LOGS))
    for f in files:
        arm = os.path.basename(f).replace(".seed.log", "")
        for line in open(f, errors="ignore"):
            m = _DROP_RE.search(line)
            if not m:
                continue
            arm_totals[arm] += int(m.group(1))
            for raw in re.findall(r"'([^']*)'", m.group(2)):
                bad, why = is_junk(raw)
                if bad:
                    junk_reasons[why] += 1
                    arm_junk[arm] += 1
                    continue
                n = normalise(raw)
                if not n or n in vocab:
                    continue
                counts[n] += 1
                per_arm[arm][n] += 1

    L: list[str] = []
    def w(s=""):
        L.append(s)

    w("# Vocabulary candidates — harvested from all arms\n")
    w(f"Sources: {len(files)} arm logs · "
      f"{sum(arm_totals.values())} relations dropped · "
      f"**{len(counts)} distinct candidates** after junk removal and "
      f"normalisation, against a live vocabulary of {len(vocab)}.\n")
    w("> A low in-vocabulary rate is not a model defect — the 197-word list is "
      "known-broken, and under constrained decoding in-vocab is 100% by "
      "construction. This harvest is what the run was *for*.\n")

    w("\n## Contract adherence per arm (junk rate)\n")
    w("*Malformed output — echoed category headers, punctuation, sentences. "
      "This measures whether the model honoured the output contract, which IS "
      "a quality signal.*\n")
    w("| arm | dropped | junk | junk % |")
    w("|---|---|---|---|")
    for arm, tot in arm_totals.most_common():
        j = arm_junk[arm]
        w(f"| {arm} | {tot} | {j} | {100*j//max(1,tot)}% |")

    w("\n## Why output was rejected\n")
    for why, n in junk_reasons.most_common():
        w(f"- **{n}** — {why}")

    w("\n## Candidate relations, with their OPPOSITES\n")
    w("*Ranked by how many arms independently produced it — a relation several "
      "different models reach for is evidence about the domain, not one model's "
      "quirk. The opposite is proposed so the vocabulary broadens in pairs: a "
      "list with `uses` but no `used_by` forces the extractor to invert subject "
      "and object, which G32 already found happening.*\n")
    w("| candidate | count | arms | proposed opposite | opposite already in vocab? |")
    w("|---|---|---|---|---|")
    shown = 0
    for rel, n in counts.most_common():
        if n < args.min_count:
            continue
        arms = sum(1 for a in per_arm if per_arm[a][rel])
        opp = opposite_of(rel)
        have = "yes" if opp and opp in vocab else "no"
        w(f"| `{rel}` | {n} | {arms} | `{opp}` | {have} |")
        shown += 1
        if shown >= args.top:
            break

    w(f"\n*Showing {shown} of {len(counts)} candidates "
      f"(min-count {args.min_count}).*\n")

    if GROUPS:
        w("\n## Where they would go — the vocabulary's existing structure\n")
        w("*`property:` single fact about an entity · `single-valued:` one "
          "target at a time (a contradiction expires the old edge) · "
          "`multi-valued:` many targets coexist. A candidate's category decides "
          "its write semantics, so this is a decision per relation, not a bulk "
          "import.*\n")
        for cat in sorted(GROUPS):
            members = GROUPS[cat]
            w(f"- **{cat}** — {len(members)} today: "
              f"{', '.join(sorted(members)[:6])}…")

    w("\n---\n")
    w("**Next:** accept/reject per row, assign a category, and add accepted "
      "pairs together. Owner: Z2's vocabulary decision — this file is evidence, "
      "not a change.\n")

    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}")
    print(f"  {sum(arm_totals.values())} dropped · {len(counts)} distinct candidates "
          f"· {sum(junk_reasons.values())} junk rejected")
    print(f"  top: {[r for r, _ in counts.most_common(12)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
