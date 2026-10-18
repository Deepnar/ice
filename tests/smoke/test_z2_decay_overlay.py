"""Z2 decay overlay (spec D8/D10, §5 item 5): decay replayed from the event log.

Pure replay checks on synthetic events; the store-level check (default profile
reproduces a real checkpoint exactly) is `decay_overlay.py --validate`."""

from datetime import datetime, timedelta, timezone

import pytest

from experiments.z2 import decay_overlay as O
from src.api.config import settings
from src.workers.decay import per_cycle

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)


def row(day=0, tags=(), immune=False, bookmarked=False):
    return dict(timestamp=T0 + timedelta(days=day), topic_tags=list(tags), decay_immune=immune,
                decay_immune_until=None, is_bookmarked=bookmarked)


def decay(day, cycles, changed=(), job="decay_episodic"):
    return dict(event="decay", job=job, kwargs=dict(cycles=cycles), changed=list(changed),
                source_time=str(T0 + timedelta(days=day)))


def strengthen(*ids):
    return dict(event="strengthen", episodic_ids=list(ids), enabled=True)


def test_default_follows_the_jobs_arithmetic():
    rows = dict(a=row(), b=row(), c=row(tags=[O.CREATIVE]), d=row(day=9))
    events = [strengthen("b"), decay(10, 16)]
    s = O.trajectory(events, rows, {}, [], {})["episodic"]
    assert s["a"]["score"] == 1.0 * per_cycle(settings.decay_daily_unaccessed) ** 16
    assert s["b"]["score"] == 1.0 * per_cycle(settings.decay_daily_accessed) ** 16 and s["b"]["access"] == 1
    assert s["c"]["score"] == 1.0 * per_cycle(settings.decay_daily_creative) ** 16
    assert s["d"]["score"] == 1.0                     # under 7 days old at the pass


def test_immune_and_bookmarked_rows_never_decay_and_null_flags_do_not_either():
    rows = dict(a=row(immune=True), b=row(bookmarked=True), c=dict(row(), decay_immune=None))
    s = O.trajectory([decay(30, 96)], rows, {}, [], {})["episodic"]
    assert all(s[k]["score"] == 1.0 for k in "abc")


def test_rows_archive_then_go_cold_and_neutral_keeps_them_warm():
    rows = dict(a=row())
    events = [decay(8 + k, 96) for k in range(30)]
    live = O.trajectory(events, rows, {}, [], {})["episodic"]["a"]
    assert live["archived"] and not live["present"]
    neutral = O.trajectory(events, {}, dict(a=dict(timestamp=T0, topic_tags=[])), [],
                           O.PROFILES["neutral"])["episodic"]["a"]
    assert neutral == dict(score=1.0, access=0, archived=False, present=True)


def test_resurrection_restores_probation_only_where_the_row_was_cold():
    rows = dict(a=row())
    events = [decay(8 + k, 96) for k in range(30)] + [
        dict(event="resurrect", episodic_ids=["a"], score=settings.timescope_probation_score)]
    live = O.trajectory(events, rows, {}, [], {})["episodic"]["a"]
    assert live == dict(score=settings.timescope_probation_score, access=1, archived=False, present=True)
    kept = O.trajectory(events, rows, {}, [], O.PROFILES["neutral"])["episodic"]["a"]
    assert kept["score"] == 1.0 and kept["access"] == 0


def test_codex_usage_counts_applied_exposures_and_decays_only_live_changed_facts():
    events = [dict(event="fact_usage", fact_ids=["f", "g"], applied_ids=["f"]),
              decay(1, 16, changed=[("f", 1.0, 0.99)], job="decay_codex")]
    usage = O.trajectory(events, {}, {}, ["f", "g"], {})["usage"]
    assert usage == dict(f=1.0 * per_cycle(settings.codex_decay_daily, job="decay_codex") ** 16, g=0.0)
    assert O.trajectory(events, {}, {}, ["f", "g"], O.PROFILES["neutral"])["usage"] == dict(f=1.0, g=0.0)


def test_old_logs_without_applied_ids_and_unknown_profile_keys_are_refused():
    with pytest.raises(ValueError, match="applied_ids"):
        O.trajectory([dict(event="fact_usage", fact_ids=["f"])], {}, {}, ["f"], {})
    with pytest.raises(ValueError, match="cannot replay"):
        O.trajectory([], {}, {}, [], dict(procedural_stale_days=30))


def test_procedural_off_undoes_decay_retirements():
    events = [decay(1, 1, changed=[("p", True, False)], job="decay_procedural")]
    assert O.trajectory(events, {}, {}, [], {})["reactivate"] == []
    assert O.trajectory(events, {}, {}, [], O.PROFILES["neutral"])["reactivate"] == ["p"]
