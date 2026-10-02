"""Two recorded turns exercise repeat observers, not 40-turn retention quality.

Run through tests/support/disposable_database.py with --out logs/<fresh-name>.
The one-turn recent window is a declared instrument fixture, not campaign data.
"""
import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text

from scripts.z1 import seed_v3, snapshot
from scripts.z1.answer_as_of import load_probes
from scripts.z1.development_repeats import POLICY, admit_repeats, packet_hash, repeat_probes
from scripts.z1.replay_checkpoint import atomic_json
from scripts.z1.run_meta import file_digest
from src.api.config import settings
from src.api.db import SessionLocal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seed_v3.isolated_database()
    conversations, _, _, _ = seed_v3.load_plan(include_native=False, include_generated=False)
    slug = "355a5709"
    base = {"probe_id": "instrument-native", "conversation": slug, "split_turn": 1,
            "question": "What did I ask in the first message?", "expected_answer": conversations[slug][0]["prompt"],
            "cutoff_kind": "native_unlabeled_checkpoint", "probe_type": "instrument", "gold_turns": []}
    review = {"gold_turns": [1], "reviewer": "authored instrument fixture",
              "review_scope": "complete_history_through_cutoff", "knowledge_scope": "private_history",
              "task_types": ["episodic_lookup"], "reason": "The question identifies the fixed first message.",
              "reviewed_expected_answer": base["expected_answer"]}
    packet = {"version": "v3", "kind": "development_repeat_review", "recent_window_turns": 1,
              "inputs": [file_digest(p) for p in (seed_v3.CORPUS, seed_v3.UNIFIED, seed_v3.TYPED, seed_v3.DERIVED, seed_v3.GENERATED)],
              "families": [{"base_probe_id": base["probe_id"], "question_family_id": "instrument-first-message",
                  "expected_answer_unchanged": True,
                  "before": {**review, "cutoff_turn": 1, "reviewed_through_turn": 1, "recent_only_answerable": True},
                  "after": {**review, "cutoff_turn": 2, "reviewed_through_turn": 2, "recent_only_answerable": False}}]}
    output = seed_v3.private_output(str(args.out))
    packet_path = output.with_suffix(".repeat-review.json")
    atomic_json(packet_path, packet)
    plans = repeat_probes(packet, conversations, {(slug, 1): [base], (slug, 2): []}, packet["inputs"], 1)
    schedule = {(slug, p["split_turn"]): [p] for p in plans}
    replay_args = SimpleNamespace(out=str(output), limit=2, conversation=slug, probe_panel="existing",
        no_maintenance=False, no_probe_controls=False, resume=False, checkpoint_every=1,
        checkpoint_dir=None, development_repeat_review=str(packet_path))
    original_observation = seed_v3.probe_observation
    observations = []
    @contextmanager
    def observed(db):
        before = snapshot.fingerprints()
        with original_observation(db):
            yield
        assert snapshot.fingerprints() == before
        observations.append(True)
    with patch.object(settings, "recent_window_max_turns", 1), patch.object(seed_v3, "probe_observation", observed):
        assert seed_v3.run(replay_args, conversations, schedule) == 0
    rows = [json.loads(line) for line in output.open()]
    assert not rows[-1]["complete_selected_corpus"]
    probes = load_probes(output, allow_partial=True)
    assert len(admit_repeats(rows[0], probes, plans, packet_hash(packet_path))) == 2
    assert len(observations) == 2
    assert rows[0]["meta"]["extra"]["development_repeat_policy"] == POLICY
    assert len([r for r in rows if r["event"] == "written"]) == 2
    assert {r["job"] for r in rows if r["event"] == "maintenance"} == set(seed_v3.MEMORY_JOBS)
    for probe in probes:
        assert len(probe["gold_sources"]) == 1 and probe["gold_sources"][0]["turn"] == 1
        assert set(probe["controls"]) == {"no_codex", "vector_only", "recent_only"}
        for stage in [probe["preflight"], *probe["controls"].values()]:
            assert stage["exposure_writes_enabled"] is False
            assert not stage["lineage"]["future_source_ids"]
        assert "development_repeat" not in json.dumps(probe["preflight"]["prompt_messages"])
    db = SessionLocal()
    try:
        assert db.execute(text("SELECT count(*) FROM episodic_memory")).scalar() == 2
    finally:
        db.close()
    print("v3 two-turn repeat path passed: actual preflight/writer/ten jobs/four read-only arms, pinned reviews, no diagnostic exposure; one-turn fixture window, no retention/answer quality score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
