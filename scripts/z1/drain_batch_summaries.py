"""Run the batch summariser to completion and ASSERT what it produced.

⚑ WHY THIS IS A SCRIPT AND NOT A LINE IN THE DRIVER. `batch_summarize()` is
called by `seed_store.py` inside a try/except. On 2026-08-17 that call failed
**once, transiently**, the exception was swallowed, and `batch_summaries` stayed
at 0 for the whole arm — which made `summary_synthesis` unscoreable and was
misread for a session as "the summariser is broken". It was not; it was never
run. A seed that ends with zero summaries must fail loudly, here, before anyone
scores anything.

⚑ AND WHY IT COUNTS THE REMAINDER WITH THE PRODUCTION PREDICATE. The same day,
"130 turns eligible" was reported from a hand-written query that omitted
`lossless_flag`; the true remainder was **3**. The eligibility check below is a
deliberate mirror of `batch_summarizer.batch_summarize`'s own filter — if that
filter changes and this one does not, the drain will under-report and the
mismatch is the bug. Keep them in step.

Usage:
    uv run python scripts/z1/drain_batch_summaries.py --expect-min 1
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sqlalchemy import or_  # noqa: E402

from src.api.config import settings  # noqa: E402
from src.api.db import SessionLocal  # noqa: E402
from src.memory.models import BatchSummary, EpisodicMemory  # noqa: E402
from src.workers.batch_summarizer import batch_summarize  # noqa: E402


def _eligible(db) -> int:
    """Turns the summariser would still select. Mirror of its own filter."""
    cutoff = datetime.now(timezone.utc) - timedelta(
        days=settings.batch_summary_age_days)
    return db.query(EpisodicMemory).filter(
        EpisodicMemory.is_private == False,           # noqa: E712
        EpisodicMemory.batch_summary_id.is_(None),
        or_(EpisodicMemory.decay_score < 0.3,
            EpisodicMemory.timestamp < cutoff),
        EpisodicMemory.lossless_flag == False,        # noqa: E712
        EpisodicMemory.is_document == False,          # noqa: E712
    ).count()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expect-min", type=int, default=1,
                    help="fail if fewer than this many summaries exist after")
    ap.add_argument("--passes", type=int, default=3,
                    help="re-run while the eligible remainder is still falling")
    ap.add_argument("--bg-model", default=None,
                    help="⚑ PASS THE ARM'S MODEL. batch_summarize() resolves "
                         "its model through settings.background_model_name, "
                         "and .env pins that to gemma4:26b-a4b-it-q4_K_M — a "
                         "model A12 ranked third and which no arm uses. "
                         "seed_store overrides it in-process, so its own "
                         "internal call is correct; THIS script is a separate "
                         "process and would silently summarise an arm's turns "
                         "with a different model than the arm ran on, making "
                         "the arms incomparable on summary_synthesis.")
    args = ap.parse_args()

    if args.bg_model:
        settings.background_model_name = args.bg_model
        print(f"background model pinned for this drain: {args.bg_model}")
    else:
        print(f"⚠ NO --bg-model: summaries would be written by "
              f"{settings.background_model_name!r} (from .env), which may not "
              f"be the model this arm was seeded with.")

    db = SessionLocal()
    try:
        before = db.query(BatchSummary).count()
        eligible_before = _eligible(db)
        total = db.query(EpisodicMemory).count()
        lossless = db.query(EpisodicMemory).filter(
            EpisodicMemory.lossless_flag == True).count()   # noqa: E712
        print(f"turns {total} · lossless (never summarised BY DESIGN) {lossless}")
        print(f"batch_summaries before {before} · eligible turns {eligible_before}")

        remainder = eligible_before
        for i in range(1, args.passes + 1):
            if remainder == 0:
                break
            batch_summarize()
            db.expire_all()
            new_remainder = _eligible(db)
            print(f"  pass {i}: eligible {remainder} -> {new_remainder} · "
                  f"summaries {db.query(BatchSummary).count()}")
            if new_remainder == remainder:
                print("  remainder stopped falling — stopping")
                remainder = new_remainder
                break
            remainder = new_remainder

        after = db.query(BatchSummary).count()
        covered = db.query(EpisodicMemory).filter(
            EpisodicMemory.batch_summary_id.isnot(None)).count()
        print(f"batch_summaries after {after} (+{after - before}) · "
              f"turns covered {covered} · eligible remaining {remainder}")

        if after < args.expect_min:
            print(f"FAIL: {after} summaries, expected at least {args.expect_min}. "
                  f"summary_synthesis is UNSCOREABLE on this store — do not "
                  f"score it and do not report a number for it.")
            return 1
        print("OK")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
