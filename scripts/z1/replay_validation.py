"""Shared, streaming completeness checks for a v3 historical replay."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Iterable

from scripts.z1.worker_recovery import POLICY, processing_health


def validate_complete_replay(rows: Iterable[dict], expected_turns: dict[str, int]) -> dict:
    """Verify events, rather than trusting the final completion declaration.

    Keep only identities and timestamps; a snapshot need not load every saved
    prompt into memory merely to verify that its replay finished.
    """
    run = complete = pending = None
    written = Counter()
    source_ids = {}
    seen_ids = set()
    probes = set()
    last_time = None
    current_key = current_time = None
    attempts = []
    degraded = []
    instrument_change = False
    for row in rows:
        event = row.get("event")
        if complete is not None:
            raise ValueError("complete replay has trailing events")
        if run is None:
            if (event != "run" or row.get("meta", {}).get("extra", {}).get("version") != "v3"):
                raise ValueError("complete replay needs a single v3 run header")
            run = row
            continue
        if event == "run" or event in {"failed", "maintenance_failed"}:
            raise ValueError("complete replay has repeated run headers or failures")
        if event == "instrument_continuation":
            old, new = row.get("from_identity", {}), row.get("to_identity", {})
            if (instrument_change or pending is not None or row.get("format") != "ice-v3-instrument-continuation-1"
                    or row.get("completed") != dict(written) or not row.get("trace_sha256")
                    or not row.get("archive_manifest_sha256")
                    or old.get("code_sha256") == new.get("code_sha256")
                    or not old.get("code_sha256") or not new.get("code_sha256")
                    or {k: v for k, v in old.items() if k != "code_sha256"}
                       != {k: v for k, v in new.items() if k != "code_sha256"}
                    or set(row.get("code_changes", {})) != {
                        "scripts/z1/worker_recovery.py", "scripts/z1/replay_checkpoint.py"}):
                raise ValueError("unbound or writer-changing instrument continuation")
            instrument_change = True
        if event in {"worker_attempt_failed", "worker_degraded"}:
            extra = run["meta"].get("extra", {})
            key = (row.get("conversation"), row.get("turn"))
            job = row.get("job")
            if (extra.get("worker_failure_policy") != "continue"
                    or extra.get("worker_recovery_policy") != POLICY or row.get("policy") != POLICY
                    or not job or key != (pending if job == "post_flight" else current_key)
                    or row.get("stage") != ("post_flight" if job == "post_flight" else "maintenance")
                    or type(row.get("attempt")) is not int
                    or row.get("attempts") != (2 if job == "post_flight" else 1)
                    or not 1 <= row["attempt"] <= row["attempts"] or not row.get("error_type")):
                raise ValueError("unbound worker failure in complete replay")
            if event == "worker_attempt_failed":
                prior = [a for a in attempts if (a["conversation"], a["turn"], a["job"]) == (*key, job)]
                if row["attempt"] != len(prior) + 1:
                    raise ValueError("worker attempts exceed or disagree with bounded policy")
                attempts.append(row)
            else:
                if (row["attempt"] != row["attempts"] or not row.get("original_source_retained")
                        or not attempts or any(row.get(k) != attempts[-1].get(k)
                            for k in ("conversation", "turn", "job", "attempt", "error_type"))
                        or any((r["conversation"], r["turn"], r["job"]) == (*key, job) for r in degraded)):
                    raise ValueError("worker degradation lacks exhausted attempt evidence")
                degraded.append(row)
        if event == "turn":
            slug, turn = row["conversation"], row["turn"]
            if (pending is not None or slug not in expected_turns
                    or turn != written[slug] + 1 or turn > expected_turns[slug]
                    or row.get("before_turns") != turn - 1):
                raise ValueError("complete replay has missing or unordered turn preflight")
            stamp = datetime.fromisoformat(row["recorded_at"])
            if stamp.tzinfo is None or (last_time is not None and stamp <= last_time):
                raise ValueError("complete replay source timestamps are not strictly ordered")
            pending = (slug, turn)
            current_time = row["recorded_at"]
            last_time = stamp
        elif event == "written":
            key = (row["conversation"], row["turn"])
            if pending != key:
                raise ValueError("complete replay write has no matching turn preflight")
            ids = {row.get("episodic_id"), row.get("batch_id")}
            if any(r["job"] == "post_flight" and (r["conversation"], r["turn"]) == key
                   and (r.get("batch_id") != row.get("batch_id") or row.get("post_flight_complete") is not False)
                   for r in degraded):
                raise ValueError("degraded worker does not match retained original write")
            if None in ids or "" in ids or len(ids) != 2 or ids & seen_ids:
                raise ValueError("complete replay has missing or repeated source identities")
            source_ids[key] = ids
            seen_ids.update(ids)
            written[key[0]] += 1
            current_key = key
            pending = None
        elif event == "as_of_probe":
            key = (row["conversation"], row["split_turn"])
            identity = (row["probe_id"], *key)
            if (pending is not None or key != current_key
                    or row.get("state_turns") != key[1]
                    or row.get("question_time") != current_time):
                raise ValueError("complete replay probe was not captured at its historical cutoff")
            if not identity[0] or identity in probes:
                raise ValueError("complete replay has a missing or duplicate probe identity")
            probes.add(identity)
            gold = row.get("gold_turns", [])
            if (attempts or "memory_processing" in row) and row.get("memory_processing") != processing_health(
                    degraded, len(attempts), key[0], gold):
                raise ValueError("probe worker health disagrees with its historical prefix")
            sources = row.get("gold_sources", [])
            if (len(gold) != len(set(gold)) or len(sources) != len(gold)
                    or {s["turn"] for s in sources} != set(gold)):
                raise ValueError("complete replay gold turns and sources disagree")
            for source in sources:
                if set(source.get("source_ids", [])) != source_ids.get((key[0], source["turn"])):
                    raise ValueError("complete replay gold source does not match its original write")
        elif event == "complete":
            complete = row
    if (run is None or complete is None or pending is not None
            or not complete.get("complete_selected_corpus")):
        raise ValueError("full campaign needs a complete selected-corpus replay")
    if dict(written) != expected_turns or complete.get("turns_by_conversation") != expected_turns:
        raise ValueError("full trace does not contain every selected historical turn exactly once")
    if (attempts or "memory_processing" in complete) and complete.get("memory_processing") != processing_health(degraded, len(attempts)):
        raise ValueError("completion hides or miscounts worker degradation")
    declared = run.get("meta", {}).get("extra", {}).get("planned_probes") or []
    declared_set = {tuple(identity) for identity in declared}
    if (not declared or len(declared_set) != len(declared)
            or declared_set != probes or complete.get("as_of_probes") != len(probes)
            or complete.get("expected_as_of_probes") != len(probes)):
        raise ValueError("complete replay disagrees with its planned probe identities")
    return {"run": run, "complete": complete}
