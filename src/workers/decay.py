"""Decay Worker – applies access-weighted memory decay and archival.

Runs on the maintenance runtime's cadence (maintenance_intervals
["decay_episodic"], 1.5h). Catch-up after downtime is closed-form (C7 D5):
the runtime passes ``cycles`` = whole intervals elapsed since this job's last
ledger finish, and one UPDATE applies ``rate ** cycles`` — the multipliers are
exponential, so any gap collapses into a single statement.
"""

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import JSONB

from src.api.config import settings
from src.api.db import SessionLocal

logger = structlog.get_logger("ice.workers.decay")


# G9: the DAILY targets are settings (decay_daily_*); the per-cycle rates are
# derived here, because 0.9968 is unreadable as a tuning target and 0.95/day
# is not. Functions rather than module constants so a Z1 sweep of the daily
# target is picked up — a module-level derivation would freeze at import.

# Fallback used only when the cadence is missing or nonsensical; it is the
# shipped 1.5h cadence expressed as cycles/day.
_FALLBACK_CYCLES_PER_DAY = 16.0
_SECONDS_PER_DAY = 86_400.0


def cycles_per_day(job: str = "decay_episodic") -> float:
    """How many times *job* runs in a day, from its configured cadence.

    NOT a setting, and deliberately so. This job applies its multiplier once
    per run, so for the daily target to actually compound to itself the count
    has to equal 86400 / cadence. A separate knob could disagree with the
    cadence, and then `0.95/day` would simply be false — the decay would be
    whatever the arithmetic happened to produce. There is no value of this
    number that is correct while disagreeing, which is what makes it derived
    rather than tunable (G9 left it as a setting; the coupling is closed here).

    Change the cadence and the daily target is preserved automatically, which
    is the property tests/test_dynamics_invariants.py pins.
    """
    seconds = settings.maintenance_intervals.get(job)
    if not seconds or seconds <= 0:
        logger.warning("decay_cadence_unresolved", job=job, configured=seconds,
                       using=_FALLBACK_CYCLES_PER_DAY,
                       reason="maintenance_intervals has no usable cadence for "
                              "this job; the daily decay target cannot be honoured")
        return _FALLBACK_CYCLES_PER_DAY
    return _SECONDS_PER_DAY / float(seconds)


def per_cycle(daily: float, job: str = "decay_episodic") -> float:
    """Per-cycle multiplier that compounds to *daily* over one day."""
    return daily ** (1.0 / cycles_per_day(job))


def rate_unaccessed() -> float:
    return per_cycle(settings.decay_daily_unaccessed)


def rate_accessed() -> float:
    return per_cycle(settings.decay_daily_accessed)


def rate_creative() -> float:
    return per_cycle(settings.decay_daily_creative)


def apply_decay(cycles: int = 1):
    """Decay old turns, archive stale ones, move to cold storage.

    ``cycles`` (clamped to [1, 96]; rate**96 ≈ 0.73 for the unaccessed rate)
    compresses missed runs into one pass — thresholds and the creative floor
    apply once at the end, exactly as they would after N sequential runs.
    """
    cycles = max(1, min(int(cycles), settings.runtime_cycles_cap))
    db = SessionLocal()
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=7)

        # T3 (D11) freeze fix: the three decay UPDATEs no longer filter
        # is_archived = FALSE — an archived row's score used to freeze at
        # ~0.1 forever, so nothing could ever cross the cold line and
        # cold_storage was unreachable. Archived rows now keep decaying at
        # their access-class rate (onward to cold), and the symmetric
        # un-archive clause below restores strengthen-driven recoveries.

        # F10: decay_immune_until is the self-expiring immunity window
        # (preserve/hybrid imports) — an open window skips decay, an expired
        # one needs no sweeper because the filter re-admits the row here.
        # Access-weighted decay: unaccessed non-creative turns
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND (decay_immune_until IS NULL OR decay_immune_until < :now)
              AND is_bookmarked = FALSE
              AND access_count = 0
              AND NOT ('Creative_&_Media' = ANY(topic_tags))
        """), {"rate": rate_unaccessed(), "cutoff": cutoff,
               "cycles": cycles, "now": now})

        # Access-weighted decay: previously-accessed non-creative turns
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND (decay_immune_until IS NULL OR decay_immune_until < :now)
              AND is_bookmarked = FALSE
              AND access_count > 0
              AND NOT ('Creative_&_Media' = ANY(topic_tags))
        """), {"rate": rate_accessed(), "cutoff": cutoff,
               "cycles": cycles, "now": now})

        # Slow decay for creative turns (1% per day)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = decay_score * POWER(:rate, :cycles)
            WHERE timestamp < :cutoff
              AND decay_immune = FALSE
              AND (decay_immune_until IS NULL OR decay_immune_until < :now)
              AND is_bookmarked = FALSE
              AND 'Creative_&_Media' = ANY(topic_tags)
        """), {"rate": rate_creative(), "cutoff": cutoff,
               "cycles": cycles, "now": now})

        # Creative floor (G9: settings.decay_creative_floor)
        db.execute(text("""
            UPDATE episodic_memory
            SET decay_score = :creative_floor
            WHERE 'Creative_&_Media' = ANY(topic_tags)
              AND decay_score < :creative_floor
        """), {"creative_floor": settings.decay_creative_floor})

        # T3 (D11) un-archive: a row whose score recovered above the archive
        # line (write-on-read strengthening — retrieval under a time window
        # reaches archived rows now) comes back. Symmetric with the archive
        # step; also the automatic probation reversal for resurrected rows.
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = FALSE
            WHERE is_archived = TRUE AND decay_score >= :archive_threshold
        """), {"archive_threshold": settings.decay_archive_threshold})

        # Archive turns below threshold
        db.execute(text("""
            UPDATE episodic_memory
            SET is_archived = TRUE
            WHERE decay_score < :archive_threshold AND is_archived = FALSE
        """), {"archive_threshold": settings.decay_archive_threshold})

        # Move extremely stale archived turns to cold_storage. T3 (D12): the
        # cold row carries conversation_id / is_private / batch_id so
        # time-scoped retrieval can honor privacy and resurrection can
        # re-attach the turn.
        cold_rows = db.execute(text("""
            SELECT id, raw_text, summary_text, topic_tags, timestamp,
                   conversation_id, is_private, batch_id, embedding, source_spans, ts_provenance,
                   summary_coverage, representation_verification, abstract_text, lossless_flag, inject_raw, session_id, intent_tags, context_reliance, idempotency_key
            FROM episodic_memory
            WHERE is_archived = TRUE AND decay_score < :cold_threshold
            FOR UPDATE
        """), {"cold_threshold": settings.decay_cold_threshold}).fetchall()

        for row in cold_rows:
            db.execute(text("""
                INSERT INTO cold_storage (id, archived_at, raw_text, summary_text,
                                          topic_tags, timestamp, conversation_id,
                                          is_private, batch_id, embedding, source_spans, ts_provenance,
                                          summary_coverage, representation_verification, abstract_text, lossless_flag, inject_raw, session_id, intent_tags, context_reliance, idempotency_key)
                VALUES (:id, :now, :raw, :summary, :tags, :ts, :conv, :priv, :batch,
                        :emb, :source_spans, :ts_provenance,
                        :summary_coverage, :representation_verification, :abstract_text, :lossless_flag, :inject_raw, :session_id, :intent_tags, :context_reliance, :idempotency_key)
                ON CONFLICT (id) DO UPDATE SET
                    archived_at = EXCLUDED.archived_at,
                    raw_text = EXCLUDED.raw_text,
                    summary_text = EXCLUDED.summary_text,
                    topic_tags = EXCLUDED.topic_tags,
                    timestamp = EXCLUDED.timestamp,
                    conversation_id = EXCLUDED.conversation_id,
                    is_private = EXCLUDED.is_private,
                    batch_id = EXCLUDED.batch_id,
                    embedding = EXCLUDED.embedding,
                    source_spans = EXCLUDED.source_spans,
                    ts_provenance = EXCLUDED.ts_provenance,
                    summary_coverage = EXCLUDED.summary_coverage,
                    representation_verification = EXCLUDED.representation_verification,
                    abstract_text = EXCLUDED.abstract_text,
                    lossless_flag = EXCLUDED.lossless_flag,
                    inject_raw = EXCLUDED.inject_raw,
                    session_id = EXCLUDED.session_id,
                    intent_tags = EXCLUDED.intent_tags,
                    context_reliance = EXCLUDED.context_reliance,
                    idempotency_key = EXCLUDED.idempotency_key
            """).bindparams(bindparam("source_spans", type_=JSONB),
                            bindparam("representation_verification", type_=JSONB)), {
                **{key: getattr(row, key) for key in (
                    'summary_coverage', 'representation_verification', 'abstract_text',
                    'lossless_flag', 'inject_raw', 'session_id', 'intent_tags',
                    'context_reliance', 'idempotency_key')},
                "id": row.id,
                "now": datetime.now(timezone.utc),
                "raw": row.raw_text,
                "source_spans": row.source_spans,
                "ts_provenance": row.ts_provenance,
                "summary": row.summary_text,
                "tags": row.topic_tags,
                "ts": row.timestamp,
                "conv": row.conversation_id,
                "priv": bool(row.is_private),
                "batch": row.batch_id,
                # C16: the vector rides along. Re-embedding on resurrect would
                # need the model at decay time and would drift from what the
                # live row was matched on.
                "emb": row.embedding,
            })
            db.execute(text("DELETE FROM episodic_memory WHERE id = :id"), {"id": row.id})

        db.commit()
        logger.info("decay_cycle_complete", archived=len(cold_rows), cycles=cycles)

    except Exception as exc:
        db.rollback()
        logger.error("decay_failed", error=str(exc))
        raise
    finally:
        db.close()
