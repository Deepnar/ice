"""One recorded v3 turn through replay, controls, telemetry and snapshot.

Run via tests/support/disposable_database.py with --out logs/<fresh-name>.
This is an immediate path check, not long-term memory or answer quality.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from scripts.z1 import seed_v3, snapshot
from scripts.z1.answer_as_of import load_probes
from scripts.z1.report_v3_replay import summarize
from src.api.db import SessionLocal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conversation", choices=("355a5709", "ecc64aab"), default="355a5709",
                        help="provider-original or constructed-clock source control")
    args = parser.parse_args()
    seed_v3.isolated_database()
    with patch.object(sys, "argv", ["seed_v3.py", "--conversation", args.conversation,
                                   "--limit", "1", "--probe-panel", "immediate",
                                   "--out", str(args.out)]):
        assert seed_v3.main() == 0
    rows = [json.loads(line) for line in args.out.open()]
    complete = rows[-1]
    assert complete["complete_selected_corpus"] is False
    assert complete["table_counts"] == snapshot.counts()
    assert complete["table_sha256"] == snapshot.fingerprints()
    jobs = [row for row in rows if row["event"] == "maintenance"]
    assert {row["job"] for row in jobs} == set(seed_v3.MEMORY_JOBS)
    assert all("before" in row and "after" in row for row in jobs)
    probe = next(row for row in rows if row["event"] == "as_of_probe")
    provenance = seed_v3.SOURCE_TIMESTAMP_PROVENANCE[args.conversation]
    written = next(row for row in rows if row["event"] == "written")
    assert written["ts_provenance"] == provenance
    assert probe["question_time_provenance"] == provenance
    assert all(source["ts_provenance"] == provenance for source in probe["gold_sources"])
    history = next(row for row in rows if row["event"] == "turn")
    assert history["preflight"]["exposure_writes_enabled"] is True
    stages = [probe["preflight"], *probe["controls"].values()]
    assert all(stage["exposure_writes_enabled"] is False for stage in stages)
    assert all(not stage["lineage"]["future_source_ids"] for stage in stages)
    assert all("ranked_candidates" in stage and "prompt_block_counts" in stage for stage in stages)
    assert set(probe["controls"]) == {"no_codex", "vector_only", "recent_only"}
    assert len(probe["source_storage_at_cutoff"]) == 1
    assert next(iter(probe["source_storage_at_cutoff"].values()))["tier"] == "warm"
    no_codex = probe["controls"]["no_codex"]
    assert not no_codex["generated_by_leg"].get("codex_graph")
    assert not no_codex["generated_by_leg"].get("codex_claims")
    assert all(fragment["type"] not in {"codex", "timeline"}
               for fragment in no_codex["ranked_candidates"])
    assert len(load_probes(args.out, allow_partial=True)) == sum(
        row["event"] == "as_of_probe" for row in rows)
    try:
        load_probes(args.out, allow_partial=False)
    except ValueError:
        pass
    else:
        raise AssertionError("one-turn path control entered the full answer campaign")
    db = SessionLocal()
    try:
        # Neither the full probe nor vector control may earn an extra access.
        assert db.execute(text("SELECT sum(access_count) FROM episodic_memory")).scalar() == 0
        assert db.execute(text("SELECT ts_provenance FROM episodic_memory")).scalar() == provenance
        with tempfile.TemporaryDirectory(prefix="v3-snapshot-path-", dir=args.out.parent) as root:
            with patch.object(snapshot, "SNAPDIR", Path(root)):
                assert snapshot.save("control") == 0
                before = snapshot.fingerprints()
                db.execute(text("UPDATE episodic_memory SET decay_score=0.5"))
                db.commit()
                assert snapshot.fingerprints() != before
                assert snapshot.restore("control") == 0
                assert snapshot.fingerprints() == before
    finally:
        db.close()
    report = summarize(args.out)
    assert not report["complete_corpus_replay"] and not report["failures"]
    assert report["historical_clock"]["historical_turn_scopes"] == 1
    print(f"v3 replay path: one recorded turn ({provenance}), ten real memory jobs, "
          "read-only matched probes, rank/clock trace and full snapshot restore passed; answer quality unmeasured")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
