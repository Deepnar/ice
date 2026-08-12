#!/usr/bin/env python3
"""Z1/A12: measure EVERY background job from the store it produced. No LLM calls.

**Why this exists.** `compare_bg_models.py` can only reach the two background
jobs that have a text-level entry point (triplets, post-flight summary). The
rest — batch and conversation summaries, reflection descriptions, cluster
naming, procedural patterns, the density decision — are DB-bound, so a cheap
direct-call comparison cannot see them.

It does not need to. **A full seed runs all of them, and each leaves a trace in
the store.** Reading those traces is free, deterministic and takes seconds, so
after each arm is seeded this gives a per-job report for that model without a
single extra token.

**⚑ EVERY NUMBER HERE IS DETERMINISTIC. Nothing is judged, by a model or by an
agent.** `summary_coverage` is term presence, the in-vocabulary rate is string
membership in the controlled vocabulary, the rest are counts. That is what makes
the loop repeatable and free — and it is also the limit worth stating plainly:

  * coverage measures whether the must-preserve terms SURVIVED, not whether the
    summary is TRUE. A summary can carry every required term and still misstate
    what happened.
  * the in-vocabulary rate measures whether a relation is STORABLE, not whether
    it is CORRECT. G32 measured the enum rescuing 100% of relations while ~78%
    of what it rescued was wrong.

Judging truth needs a judge, and that is Z2's eyeball pass — deliberately not
automated here, because a cheap wrong judge is worse than an honest gap.

Run:
  uv run python scripts/z1/store_report.py
  uv run python scripts/z1/store_report.py --arm gemma4-26b --save
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sqlalchemy import text  # noqa: E402

from src.api.db import SessionLocal  # noqa: E402

OUT = Path("experiments/curation_files/store_reports")

# job -> (label, SQL returning a single scalar). Grouped by the background job
# that produces the row, so a zero points at which worker is silent rather than
# just saying the store is thin.
QUERIES = [
    ("ingest",     "turns",                "SELECT count(*) FROM episodic_memory"),
    ("ingest",     "turns_lossless",       "SELECT count(*) FROM episodic_memory WHERE lossless_flag"),
    ("chunker",    "chunks",               "SELECT count(*) FROM episodic_chunks"),
    ("chunker",    "turns_chunked",        "SELECT count(DISTINCT turn_id) FROM episodic_chunks"),
    ("summary",    "turns_summarised",     "SELECT count(*) FROM episodic_memory WHERE summary_text IS NOT NULL AND summary_text <> ''"),
    ("summary",    "with_abstract",        "SELECT count(*) FROM episodic_memory WHERE abstract_text IS NOT NULL AND abstract_text <> ''"),
    ("codex",      "entities",             "SELECT count(*) FROM codex_entities"),
    ("codex",      "edges",                "SELECT count(*) FROM codex_edges"),
    ("codex",      "edges_negated",        "SELECT count(*) FROM codex_edges WHERE negated"),
    ("codex",      "edges_active",         "SELECT count(*) FROM codex_edges WHERE confidence = 'active'"),
    ("codex",      "orphan_entities",      "SELECT count(*) FROM codex_entities e WHERE NOT EXISTS (SELECT 1 FROM codex_edges x WHERE x.source_id = e.id OR x.target_id = e.id)"),
    ("codex",      "relation_gaps",        "SELECT count(*) FROM codex_relation_gaps"),
    ("codex",      "distinct_gap_relations", "SELECT count(DISTINCT proposed_relation) FROM codex_relation_gaps"),
    ("procedural", "patterns",             "SELECT count(*) FROM procedural_memory"),
    ("clustering", "clusters",             "SELECT count(*) FROM context_clusters"),
    ("clustering", "cluster_links",        "SELECT count(*) FROM episodic_cluster_links"),
    ("batch_summ", "batch_summaries",      "SELECT count(*) FROM batch_summaries"),
    ("batch_summ", "conversation_summaries", "SELECT count(*) FROM conversation_summaries"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    db = SessionLocal()
    vals = {}
    for job, name, sql in QUERIES:
        try:
            vals[name] = db.execute(text(sql)).scalar() or 0
        except Exception as exc:
            print(f"  ! {name}: {str(exc)[:90]}")
            vals[name] = None

    cov = db.execute(text(
        "SELECT avg(summary_coverage), min(summary_coverage), "
        "count(*) FILTER (WHERE summary_coverage < 1.0) "
        "FROM episodic_memory WHERE summary_coverage IS NOT NULL")).first()
    mean_cov, min_cov, imperfect = (cov or (None, None, 0))

    turns = vals.get("turns") or 1
    rel_total = (vals.get("edges") or 0) + (vals.get("relation_gaps") or 0)
    in_vocab = round(100 * (vals.get("edges") or 0) / rel_total) if rel_total else 0

    print(f"\n{'='*62}\nSTORE REPORT{f'  ·  arm {args.arm}' if args.arm else ''}\n{'='*62}")
    last = None
    for job, name, _ in QUERIES:
        if job != last:
            print(f"\n[{job}]")
            last = job
        v = vals.get(name)
        per = f"   ({v/turns:.2f}/turn)" if v and name not in ("turns",) else ""
        print(f"  {name:24} {str(v):>7}{per}")

    print(f"\n[derived — the numbers that decide an arm]")
    print(f"  {'relation in-vocab %':24} {in_vocab:>7}   "
          f"(edges kept vs gaps logged; storable, NOT correct)")
    print(f"  {'mean summary_coverage':24} "
          f"{round(mean_cov, 4) if mean_cov is not None else 'n/a':>7}   "
          f"(term presence, NOT faithfulness)")
    print(f"  {'worst coverage':24} "
          f"{round(min_cov, 4) if min_cov is not None else 'n/a':>7}")
    print(f"  {'summaries below 1.0':24} {imperfect:>7}")
    orph = vals.get("orphan_entities") or 0
    ents = vals.get("entities") or 1
    print(f"  {'orphan entity %':24} {round(100*orph/ents):>7}   "
          f"(G33: property values that became nodes with no edges)")

    silent = [j for j, n, _ in QUERIES if vals.get(n) == 0
              and j in ("chunker", "summary", "codex", "procedural",
                        "clustering", "batch_summ")]
    if silent:
        print(f"\n⚠ jobs with a ZERO count — a silent worker, not a thin store: "
              f"{sorted(set(silent))}")

    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        dest = OUT / f"{args.arm or 'store'}.json"
        dest.write_text(json.dumps(
            {"arm": args.arm, "utc": datetime.now(timezone.utc).isoformat(),
             "counts": vals, "in_vocab_pct": in_vocab,
             "mean_summary_coverage": float(mean_cov) if mean_cov is not None else None,
             "min_summary_coverage": float(min_cov) if min_cov is not None else None},
            indent=2))
        print(f"\nwrote {dest}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
