"""Z2 decay overlay: per-profile decay recomputed from the replay's event log (spec D8/D10).

The replay runs with live default decay and logs every strengthening, fact-usage
increment, cold resurrection and decay pass (`replay.py` → `events.jsonl`). Decay
is a pure function of those events (V3_POST_RESEED_REPAIR §2.5 feasibility read):

- episodic: a pass multiplies `decay_score` by `rate ** cycles` for rows older than
  7 days at the pass's source time, not immune, not bookmarked; the rate class is
  creative / accessed (`access_count > 0`) / unaccessed; then the creative floor,
  un-archive, archive and cold thresholds. Strengthening adds a capped amount and
  counts an access; a resurrection re-inserts a cold row at the probation score.
- codex: usage +1 per applied exposure; a pass multiplies the facts it changed live
  (exactly the valid ones with usage > 0) by `rate ** cycles`.
- procedural: retirement is an absolute-age cutoff; the overlay can only keep the
  live result or undo it (`procedural: off`).

`trajectory` replays the events under a profile into a per-row target. Under the
default profile the target must equal the checkpoint store exactly (`validate`, §5
item 5). `apply` writes a target onto a RESTORED COPY of a checkpoint: scores,
access counts and archive flags; a row the profile keeps warm that went cold live is
moved back from cold storage. A row the profile would send cold that is live in the
store is refused for that row and listed — never approximated (spec §4).

    uv run --no-sync python experiments/z2/decay_overlay.py --events <replay>/events.jsonl \
        --checkpoint cp-25 --database ice_z2_replay_b_r1 --validate
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

POLICY = "ice-z2-decay-overlay-1"
CREATIVE = "Creative_&_Media"
AGE_DAYS = 7                                    # decay.apply_decay's hard-coded cutoff
EPISODIC_KEYS = ("decay_daily_unaccessed", "decay_daily_accessed", "decay_daily_creative",
                 "decay_creative_floor", "decay_archive_threshold", "decay_cold_threshold",
                 "decay_strengthen_amount", "timescope_probation_score")
CODEX_KEYS = ("codex_decay_daily",)
# neutral = no decay at all: every multiplier 1.0, so scores stay at their capped
# 1.0, nothing archives or goes cold, usage is the raw exposure count, and no
# procedure is retired. Ablation conditions all read with it (D10).
PROFILES = {
    "default": {},
    "neutral": dict(decay_daily_unaccessed=1.0, decay_daily_accessed=1.0, decay_daily_creative=1.0,
                    codex_decay_daily=1.0, procedural="off"),
}


def resolve(profile):
    """Full parameter set: live settings overlaid with the profile's overrides."""
    from src.api.config import settings
    unknown = set(profile) - set(EPISODIC_KEYS) - set(CODEX_KEYS) - {"procedural"}
    if unknown:
        raise ValueError(f"profile keys the overlay cannot replay: {sorted(unknown)}")
    if profile.get("procedural", "live") not in ("live", "off"):
        raise ValueError("procedural is 'live' or 'off'")
    params = {k: getattr(settings, k) for k in EPISODIC_KEYS + CODEX_KEYS}
    return {**params, "procedural": "live", **profile}


def rates(params):
    """Per-cycle multipliers, derived exactly as the jobs derive them."""
    from src.workers.decay import per_cycle
    return dict(unaccessed=per_cycle(params["decay_daily_unaccessed"]),
                accessed=per_cycle(params["decay_daily_accessed"]),
                creative=per_cycle(params["decay_daily_creative"]),
                codex=per_cycle(params["codex_decay_daily"], job="decay_codex"))


def load_events(path, checkpoint):
    """Events up to and including the checkpoint `checkpoint` (e.g. 'cp-25')."""
    out = []
    for line in Path(path).read_text().splitlines():
        event = json.loads(line)
        out.append(event)
        if event["event"] == "checkpoint" and event["store"] == f"{checkpoint}.sql":
            return out
    raise ValueError(f"{checkpoint} not in {path}")


def _when(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


def trajectory(events, episodic, cold, facts, profile):
    """Replay *events* under *profile*.

    episodic: {id: dict(timestamp, topic_tags, decay_immune, decay_immune_until,
    is_bookmarked)} for rows in the store; cold: {id: dict(timestamp, topic_tags)} for
    rows in cold storage; facts: ids in codex_facts. Returns per-row targets."""
    p = resolve(profile)
    r = rates(p)
    meta = {i: dict(row) for i, row in episodic.items()}
    for i, row in cold.items():                 # a row that went cold was decaying: no flags
        meta.setdefault(i, dict(row, decay_immune=False, decay_immune_until=None, is_bookmarked=False))
    state = {i: dict(score=1.0, access=0, archived=False, present=True) for i in meta}
    usage = {i: 0.0 for i in facts}
    retired = set()
    for e in events:
        kind = e["event"]
        if kind == "strengthen" and e.get("enabled", True):
            for i in e["episodic_ids"]:
                s = state.get(i)
                if s and s["present"]:
                    s["access"] += 1
                    s["score"] = min(1.0, s["score"] + p["decay_strengthen_amount"])
        elif kind == "resurrect":
            for i in e["episodic_ids"]:
                s = state.get(i)
                if s and not s["present"]:
                    s.update(score=p["timescope_probation_score"], access=1, archived=False, present=True)
                    # the resurrect INSERT writes these flags; the store row holds what it wrote
                    for flag in ("decay_immune", "is_bookmarked", "decay_immune_until"):
                        meta[i][flag] = episodic.get(i, {}).get(flag)
        elif kind == "fact_usage":
            if "applied_ids" not in e:
                raise ValueError("event log predates applied_ids; re-run the replay")
            for i in e["applied_ids"]:
                if i in usage:
                    usage[i] += 1
        elif kind == "decay" and e["job"] == "decay_episodic":
            now, cycles = _when(e["source_time"]), e["kwargs"]["cycles"]
            cutoff = now - timedelta(days=AGE_DAYS)
            for i, s in state.items():
                if not s["present"]:
                    continue
                m = meta[i]
                creative = CREATIVE in (m["topic_tags"] or [])
                eligible = (_when(m["timestamp"]) < cutoff and m["decay_immune"] is False
                            and (m["decay_immune_until"] is None or _when(m["decay_immune_until"]) < now)
                            and m["is_bookmarked"] is False)
                if eligible:
                    rate = r["creative"] if creative else r["accessed"] if s["access"] > 0 else r["unaccessed"]
                    s["score"] = s["score"] * rate ** cycles
                if creative and s["score"] < p["decay_creative_floor"]:
                    s["score"] = p["decay_creative_floor"]
                if s["archived"] and s["score"] >= p["decay_archive_threshold"]:
                    s["archived"] = False
                if s["score"] < p["decay_archive_threshold"] and not s["archived"]:
                    s["archived"] = True
                if s["archived"] and s["score"] < p["decay_cold_threshold"]:
                    s["present"] = False
        elif kind == "decay" and e["job"] == "decay_codex":
            cycles = e["kwargs"]["cycles"]
            for i, _before, _after in e["changed"]:      # live: exactly the valid facts with usage
                if i in usage:
                    usage[i] = usage[i] * r["codex"] ** cycles
        elif kind == "decay" and e["job"] == "decay_procedural":
            retired |= {i for i, before, after in e["changed"] if before is True and after is False}
    return dict(policy=POLICY, params=p, episodic=state, usage=usage,
                reactivate=sorted(retired) if p["procedural"] == "off" else [])


# ── store access ───────────────────────────────────────────────────────────

def read_store(db):
    from sqlalchemy import text
    episodic = {str(r.id): dict(r._mapping) for r in db.execute(text(
        "SELECT id, timestamp, topic_tags, decay_immune, decay_immune_until, is_bookmarked, "
        "decay_score, access_count, is_archived FROM episodic_memory"))}
    cold = {str(r.id): dict(r._mapping) for r in db.execute(text(
        "SELECT id, timestamp, topic_tags, decay_score FROM cold_storage"))}
    facts = {str(r.id): r.usage for r in db.execute(text("SELECT id, usage FROM codex_facts"))}
    return episodic, cold, facts


def validate(target, episodic, cold, facts, tolerance=0.0):
    """Default-profile target vs the live store: every difference, plus the max."""
    bad, worst = [], 0.0
    for i, row in episodic.items():
        s = target["episodic"][i]
        diff = abs(s["score"] - (row["decay_score"] if row["decay_score"] is not None else 0.0))
        worst = max(worst, diff)
        if not s["present"] or diff > tolerance or s["access"] != (row["access_count"] or 0) \
                or s["archived"] != bool(row["is_archived"]):
            bad.append(("episodic", i, s, {k: row[k] for k in ("decay_score", "access_count", "is_archived")}))
    for i in cold:
        if target["episodic"][i]["present"]:
            bad.append(("cold", i, target["episodic"][i], "in cold storage"))
    for i, live in facts.items():
        diff = abs(target["usage"][i] - live)
        worst = max(worst, diff)
        if diff > tolerance:
            bad.append(("codex", i, target["usage"][i], live))
    return dict(rows=len(episodic), cold=len(cold), facts=len(facts), mismatches=len(bad),
                max_abs_diff=worst, examples=bad[:10])


def apply(db, target, episodic, cold):
    """Write *target* onto a restored copy. Returns counts and refused rows."""
    from sqlalchemy import text
    from src.api.config import settings
    if os.environ.get("ICE_TEST_DATABASE") != db.get_bind().url.database or \
            db.get_bind().url.database == "ice_db":
        raise RuntimeError("an overlay writes only to a restored scratch copy")
    counts = dict(updated=0, warmed_from_cold=0, facts=0, reactivated=0)
    refused = []
    for i, s in target["episodic"].items():
        if i in episodic:
            if not s["present"]:
                refused.append(i)                   # profile sends it cold; live kept it
                continue
            db.execute(text("UPDATE episodic_memory SET decay_score = :s, access_count = :a, "
                            "is_archived = :z WHERE id = CAST(:i AS uuid)"),
                       dict(s=s["score"], a=s["access"], z=s["archived"], i=i))
            counts["updated"] += 1
        elif i in cold and s["present"]:
            _warm(db, i, s)
            counts["warmed_from_cold"] += 1
    for i, u in target["usage"].items():
        db.execute(text("UPDATE codex_facts SET usage = :u WHERE id = CAST(:i AS uuid)"), dict(u=u, i=i))
        counts["facts"] += 1
    if target["reactivate"]:
        counts["reactivated"] = db.execute(text(
            "UPDATE procedural_memory SET is_active = TRUE WHERE id = ANY(CAST(:ids AS uuid[]))"),
            dict(ids=target["reactivate"])).rowcount
    db.commit()
    return dict(counts, refused=len(refused), refused_ids=refused, inert=not settings.retrieval_strengthen_writes)


def _warm(db, i, s):
    """Reverse decay's cold move (decay.apply_decay) for one row: the episodic row,
    its chunks and cluster links back from cold storage, with the profile's state."""
    from sqlalchemy import text
    db.execute(text("""
        INSERT INTO episodic_memory
            (id, conversation_id, batch_id, timestamp, topic_tags, intent_tags, context_reliance,
             raw_text, summary_text, embedding, decay_score, access_count, is_archived,
             is_bookmarked, decay_immune, is_private, inject_raw, idempotency_key, source_spans,
             ts_provenance, summary_coverage, representation_verification, abstract_text,
             lossless_flag, is_document, session_id, cluster_id, batch_summary_id)
        SELECT id, conversation_id, batch_id, timestamp, topic_tags, intent_tags, context_reliance,
               raw_text, summary_text, embedding, :score, :access, :archived,
               FALSE, FALSE, is_private, inject_raw, idempotency_key, source_spans,
               ts_provenance, summary_coverage, representation_verification, abstract_text,
               lossless_flag, is_document, session_id,
               (SELECT c.id FROM context_clusters c WHERE c.id = cold_storage.cluster_id),
               (SELECT b.id FROM batch_summaries b WHERE b.id = cold_storage.batch_summary_id)
        FROM cold_storage WHERE id = CAST(:i AS uuid)
    """), dict(i=i, score=s["score"], access=s["access"], archived=s["archived"]))
    db.execute(text("""
        INSERT INTO episodic_chunks (id, turn_id, chunk_index, chunk_text, embedding)
        SELECT id, turn_id, chunk_index, chunk_text, embedding FROM cold_chunks
        WHERE turn_id = CAST(:i AS uuid)
    """), dict(i=i))
    db.execute(text("""
        INSERT INTO episodic_cluster_links (episodic_id, cluster_id)
        SELECT CAST(:i AS uuid), c.id FROM context_clusters c
        WHERE c.id = ANY((SELECT cluster_ids FROM cold_storage WHERE id = CAST(:i AS uuid)))
        ON CONFLICT DO NOTHING
    """), dict(i=i))
    db.execute(text("DELETE FROM cold_chunks WHERE turn_id = CAST(:i AS uuid)"), dict(i=i))
    db.execute(text("DELETE FROM cold_storage WHERE id = CAST(:i AS uuid)"), dict(i=i))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True, help="e.g. cp-25 (store already restored)")
    parser.add_argument("--database", required=True, help="the scratch database holding the restored copy")
    parser.add_argument("--profile", default="default",
                        help=f"one of {sorted(PROFILES)} or a JSON object of overrides")
    parser.add_argument("--validate", action="store_true", help="compare the default target with the store")
    parser.add_argument("--apply", action="store_true", help="write the profile onto the restored copy")
    args = parser.parse_args()
    if os.environ.get("ICE_TEST_DATABASE") != args.database:
        from sqlalchemy.engine import make_url
        from src.api.config import settings as live
        url = make_url(live.database_url).set(database=args.database).render_as_string(hide_password=False)
        env = dict(os.environ, DATABASE_URL=url, ICE_TEST_DATABASE=args.database)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    from src.api.db import SessionLocal
    profile = PROFILES.get(args.profile) if args.profile in PROFILES else json.loads(args.profile)
    events = load_events(args.events, args.checkpoint)
    with SessionLocal() as db:
        episodic, cold, facts = read_store(db)
        if args.validate:
            target = trajectory(events, episodic, cold, facts, {})
            print(json.dumps(validate(target, episodic, cold, facts), default=str, indent=1))
        if args.apply:
            target = trajectory(events, episodic, cold, facts, profile)
            print(json.dumps(apply(db, target, episodic, cold), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
