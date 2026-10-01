"""Recovery must preserve the committed store/trace and reject a changed run."""
import json
from pathlib import Path

import pytest

from scripts.z1 import replay_checkpoint as recovery


def test_checkpoint_recovery_discards_only_unfinished_tail(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    state = {"rows": ["source-1"]}

    def save(_arm):
        (recovery.snapshot.SNAPDIR / "store.sql").write_text(json.dumps(state))
        return 0

    def restore(_arm):
        state.clear()
        state.update(json.loads((recovery.snapshot.SNAPDIR / "store.sql").read_text()))
        return 0

    monkeypatch.setattr(recovery.snapshot, "save", save)
    monkeypatch.setattr(recovery.snapshot, "restore", restore)
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"model": "pinned"})
    try:
        with trace.open("w") as sink:
            sink.write('{"event":"written","turn":1}\n')
            point.capture(sink, {"completed": {"one": 1}, "cadence": "original"})
            committed = trace.read_bytes()
            sink.write('{"event":"partial_writer"}\n')
        state["rows"].append("half-written-2")
        restored = point.recover()
        assert state["rows"] == ["source-1"]
        assert trace.read_bytes() == committed
        assert restored == {"completed": {"one": 1}, "cadence": "original"}
        assert next(point.root.glob("unfinished-*.jsonl")).read_text() == '{"event":"partial_writer"}\n'
        # A second operator cannot race the original's checkpoint/restore.
        with pytest.raises(RuntimeError, match="active operator"):
            recovery.Checkpoints(point.root, trace, point.identity)
    finally:
        point.close()


def test_failed_checkpoint_never_replaces_previous_pointer(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setattr(recovery.snapshot, "save", lambda _arm: 0)
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"model": "pinned"})
    try:
        with trace.open("w") as sink:
            sink.write("committed\n")
            point.capture(sink, {"completed": 1})
            original = point.pointer.read_bytes()
            monkeypatch.setattr(recovery.snapshot, "save", lambda _arm: 1)
            sink.write("next\n")
            with pytest.raises(RuntimeError, match="snapshot failed"):
                point.capture(sink, {"completed": 2})
        assert point.pointer.read_bytes() == original
    finally:
        point.close()


def test_evaluation_state_survives_rolling_recovery_cleanup(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    value = [1]
    def save(_arm):
        (recovery.snapshot.SNAPDIR / "store.sql").write_text(str(value[0]))
        return 0
    monkeypatch.setattr(recovery.snapshot, "save", save)
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"model": "pinned"})
    try:
        with trace.open("w") as sink:
            sink.write("first\n")
            point.capture(sink, {"completed": 1}, evaluation_key="one-1")
            for i in (2, 3, 4):
                value[0] = i
                sink.write(f"turn-{i}\n")
                point.capture(sink, {"completed": i})
        assert len(list(point.root.glob("generation-*"))) == 2
        assert (point.root / "evaluation/one-1/store.sql").read_text() == "1"
        assert json.loads((point.root / "evaluation/one-1/recovery.json").read_text())["state"] == {"completed": 1}
    finally:
        point.close()


@pytest.mark.parametrize("change", ["identity", "trace"])
def test_changed_identity_or_committed_prefix_blocks_restore(tmp_path, monkeypatch, change):
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setattr(recovery.snapshot, "save", lambda _arm: 0)
    monkeypatch.setattr(recovery.snapshot, "restore", lambda _arm: pytest.fail("unsafe restore called"))
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"model": "pinned"})
    try:
        with trace.open("w") as sink:
            sink.write("committed\n")
            point.capture(sink, {"completed": 1})
        if change == "identity":
            point.identity = {"model": "changed"}
        else:
            trace.write_text("corrupted\n")
        with pytest.raises(ValueError, match="changed"):
            point.recover()
    finally:
        point.close()


def test_finished_resume_is_read_only_and_refuses_modified_store(tmp_path, monkeypatch):
    from scripts.z1 import replay_validation
    trace = tmp_path / "trace.jsonl"
    trace.write_text('{"event":"run"}\n')
    monkeypatch.setattr(recovery.snapshot, "save", lambda _: 0)
    monkeypatch.setattr(recovery.snapshot, "restore", lambda _: pytest.fail("finished run restored"))
    monkeypatch.setattr(replay_validation, "validate_complete_replay", lambda *_:
                        {"complete": {"table_counts": {"source": 1}, "table_sha256": {"source": "original"}}})
    monkeypatch.setattr(recovery.snapshot, "fingerprints", lambda: {"source": "original"})
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"model": "pinned"})
    try:
        with trace.open("a") as sink:
            point.capture(sink, {"completed": 1})
        before = trace.read_bytes()
        assert point.complete_unchanged({"one": 1}, {"source": 1})
        assert trace.read_bytes() == before
        monkeypatch.setattr(recovery.snapshot, "fingerprints", lambda: {"source": "changed"})
        with pytest.raises(ValueError, match="complete seed store changed"):
            point.complete_unchanged({"one": 1}, {"source": 1})
    finally:
        point.close()
