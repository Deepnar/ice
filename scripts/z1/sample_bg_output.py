#!/usr/bin/env python3
"""Z1/A12: random samples of what the background model actually produced, for review.

**Why sampling and not another metric.** `store_report.py` measures everything
that can be counted, and says so honestly: `summary_coverage` is term presence,
not faithfulness; the in-vocabulary rate is storability, not correctness. Both
blind spots have the same shape — **a number cannot tell you whether the output
is TRUE**. G32 measured the relation enum rescuing 100% of relations while ~78%
of what it rescued was wrong, and every one of those wrong relations would have
counted as a success in the metric.

So the last check is a human (or a model) reading real output. This script draws
the sample; it deliberately does not judge, because who judges is a decision with
a privacy consequence:

  * **local model** — the content never leaves the machine; weaker judgement;
  * **cloud agent** — better judgement, but the samples are summaries and facts
    derived from personal conversations, so that is personal content leaving the
    machine. The user's standing decision (2026-08-10) was transcripts stay
    local, judging may go to the cloud. This is the boundary case, so it is
    surfaced rather than assumed.

Follows the protocol Z2's entry now carries: one yes/no question per artifact,
silence means correct, the excerpt is centred on the evidence, and the SOURCE
turn is shown beside the output so the reviewer can check faithfulness rather
than plausibility.

Run:
  uv run python scripts/z1/sample_bg_output.py --n 8
  uv run python scripts/z1/sample_bg_output.py --n 8 --md
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from src.api.db import SessionLocal  # noqa: E402

OUT = Path("experiments/curation_files/BG_QUALITY_SAMPLE.md")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="samples per section")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--md", action="store_true", help="write markdown to a file")
    args = ap.parse_args()
    random.seed(args.seed)
    db = SessionLocal()
    L: list[str] = []

    def w(s=""):
        L.append(s)
        if not args.md:
            print(s)

    w(f"# Background-model output — quality sample\n")
    w(f"*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")
    w("## The question, per item\n")
    w("> **Is this output TRUE to the source turn shown with it?**\n")
    w("- **Yes** → say nothing.\n- **No** → write the item number.\n")
    w("Not judging style, length or usefulness. Only: does it misstate the source.\n")
    w("**Stop at 3 No's in any section** — that section's job needs work and the "
      "rest of the reading is wasted.\n")

    # ── 1. summaries against their source turn ──────────────────────────────
    w("\n---\n\n## A · Summaries (faithfulness)\n")
    rows = db.execute(text("""
        SELECT raw_text, summary_text, summary_coverage, abstract_text
          FROM episodic_memory
         WHERE summary_text IS NOT NULL AND summary_text <> ''
         ORDER BY random() LIMIT :n"""), {"n": args.n}).all()
    for i, r in enumerate(rows, 1):
        w(f"\n### A{i}.  (coverage {r[2]})\n")
        w(f"**Source turn:**\n\n> {(r[0] or '')[:900].strip()}…\n")
        w(f"**Summary produced:**\n\n> {(r[1] or '').strip()}\n")
        if r[3]:
            w(f"**Abstract:** {r[3].strip()}\n")

    # ── 2. codex facts against the turn they came from ──────────────────────
    w("\n---\n\n## B · Codex facts (is the extracted fact true?)\n")
    rows = db.execute(text("""
        SELECT s.canonical_name, e.relation, t.canonical_name, e.negated, m.raw_text
          FROM codex_edges e
          JOIN codex_entities s ON s.id = e.source_id
          JOIN codex_entities t ON t.id = e.target_id
          LEFT JOIN episodic_memory m ON m.batch_id = e.source_batch
         ORDER BY random() LIMIT :n"""), {"n": args.n}).all()
    for i, r in enumerate(rows, 1):
        neg = " (NEGATED)" if r[3] else ""
        w(f"\n### B{i}.  `{r[0]} --{r[1]}--> {r[2]}`{neg}\n")
        w(f"**From this turn:**\n\n> {(r[4] or '(source turn not found)')[:700].strip()}…\n")

    # ── 3. relations the vocabulary could not store ─────────────────────────
    w("\n---\n\n## C · Dropped relations (was the vocabulary wrong, or the model?)\n")
    w("*Each of these was extracted and then discarded because the 197-relation "
      "vocabulary has no place for it. The question here is different: **should "
      "the vocabulary have had this word?** These feed Z2's repair.*\n")
    rows = db.execute(text("""
        SELECT proposed_relation, subject, object, count(*) OVER (PARTITION BY proposed_relation)
          FROM codex_relation_gaps ORDER BY random() LIMIT :n"""), {"n": args.n}).all()
    for i, r in enumerate(rows, 1):
        w(f"- **C{i}.** `{r[1]} --{r[0]}--> {r[2]}`   *(this relation dropped {r[3]}×)*")

    # ── 4. procedural patterns ──────────────────────────────────────────────
    w("\n---\n\n## D · Procedural patterns (is this a real habit, or a one-off?)\n")
    rows = db.execute(text("""
        SELECT pattern_name, pattern_description, reinforcement_count
          FROM procedural_memory ORDER BY random() LIMIT :n"""), {"n": args.n}).all()
    for i, r in enumerate(rows, 1):
        w(f"- **D{i}.** {r[0] or '(unnamed)'} — {(r[1] or '').strip()[:300]} "
          f"*(seen {r[2]}×)*")

    # ── 5. cluster names ────────────────────────────────────────────────────
    w("\n---\n\n## E · Cluster names (does the name describe its members?)\n")
    rows = db.execute(text("""
        SELECT c.name, c.description,
               (SELECT count(*) FROM episodic_cluster_links l WHERE l.cluster_id = c.id)
          FROM context_clusters c ORDER BY random() LIMIT :n"""), {"n": args.n}).all()
    for i, r in enumerate(rows, 1):
        w(f"- **E{i}.** **{r[0]}** — {(r[1] or '').strip()[:220]} *({r[2]} turns)*")
    if not rows:
        w("*(no clusters yet — clustering runs at the end of seeding)*")

    w("\n---\n\n**Report only the item numbers that were No.**\n")

    if args.md:
        OUT.write_text("\n".join(L))
        print(f"wrote {OUT}  ({len(L)} lines)")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
