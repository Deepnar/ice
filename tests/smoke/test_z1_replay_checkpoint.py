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


@pytest.mark.parametrize("bad", [None, "settings", "writer", "archive", "prefix", "baseline"])
def test_instrument_continuation_preserves_prefix_and_refuses_unapproved_changes(tmp_path, monkeypatch, bad):
    trace = tmp_path / "trace.jsonl"
    model = {"model": "unchanged", "settings_sha256": "unchanged", "code_sha256": "old"}
    monkeypatch.setattr(recovery, "code_digest", lambda ref=None: "old" if ref else "new")
    changes = {path: {"before": "old", "after": "new"} for path in recovery.INSTRUMENT_CHANGES}
    monkeypatch.setattr(recovery, "instrument_changes", lambda ref: changes)
    monkeypatch.setattr(recovery, "registered_repair", lambda *_: {"reason": "reviewed repair"})
    monkeypatch.setattr(recovery.snapshot, "save", lambda _: (recovery.snapshot.SNAPDIR / "store.sql").write_text("original") and 0)
    restores = []
    monkeypatch.setattr(recovery.snapshot, "restore", lambda _: restores.append(1) or 0)
    point = recovery.Checkpoints(tmp_path / "recovery", trace, model)
    try:
        with trace.open("w") as sink:
            sink.write('{"event":"written","turn":1}\n')
            point.capture(sink, {"completed": {"one": 1}})
        original_pointer = point.pointer.read_bytes()
        directory, original = point.validate()
        original_manifest = (directory / "recovery.json").read_bytes()
        receipt = point.prepare_instrument_continuation("reviewed-baseline")
        assert point.pointer.read_bytes() == original_pointer
        assert (directory / "recovery.json").read_bytes() == original_manifest
        assert restores == [] and receipt["completed"] == {"one": 1}
        assert point.prepare_instrument_continuation("reviewed-baseline") == receipt
        point.identity = receipt["to_identity"]
        if bad == "settings":
            point.identity = {**point.identity, "settings_sha256": "modified"}
        elif bad == "writer":
            monkeypatch.setattr(recovery, "instrument_changes", lambda ref: (_ for _ in ()).throw(ValueError("unapproved writer")))
        elif bad == "baseline":
            monkeypatch.setattr(recovery, "code_digest", lambda ref=None: "other" if ref else "new")
        elif bad == "archive":
            (point.root / receipt["archive"] / "trace-prefix.jsonl").write_text("corrupt\n")
        elif bad == "prefix":
            trace.write_text("corrupt\n")
        if bad:
            with pytest.raises(ValueError):
                point.recover()
            assert restores == []
        else:
            assert point.recover()["completed"] == {"one": 1}
            assert len(restores) == 1
            marker = json.loads(trace.read_text().splitlines()[-1])
            assert marker["event"] == "instrument_continuation"
            with trace.open("a") as sink:
                point.capture(sink, {"completed": {"one": 1}})
            assert point.validate()[1]["identity"] == receipt["to_identity"]
            assert (point.root / receipt["archive"] / "recovery.json").read_bytes() == original_manifest
    finally:
        point.close()


def test_allowed_tool_edits_still_need_exact_registered_hashes(tmp_path, monkeypatch):
    import hashlib
    files = [tmp_path / path for path in sorted(recovery.INSTRUMENT_CHANGES)]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"reviewed-tool")
    monkeypatch.setattr(recovery, "code_files", lambda: (tmp_path, files))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: b"original-tool")
    monkeypatch.setattr(recovery, "code_digest", lambda ref=None: "old" if ref else "reviewed")
    changes = {str(path.relative_to(tmp_path)): {
        "before": hashlib.sha256(b"original-tool").hexdigest(),
        "after": hashlib.sha256(b"reviewed-tool").hexdigest()} for path in files}
    registry = tmp_path / "repairs.json"
    monkeypatch.setattr(recovery, "REPAIR_REGISTRY", registry)
    registry.write_text(json.dumps({"format": "ice-v3-reviewed-instrument-repairs-1", "repairs": [{
        "from_code": "old", "to_code": "reviewed", "code_changes": changes}]}))
    assert recovery.instrument_changes("baseline") == changes
    files[0].write_bytes(b"unreviewed-new-tool")
    with pytest.raises(ValueError, match="registered reviewed"):
        recovery.instrument_changes("baseline")


def test_second_registered_continuation_preserves_first_receipt_and_both_prefixes(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    code = ["middle"]
    monkeypatch.setattr(recovery, "code_digest", lambda ref=None:
                        {"first-baseline": "old", "second-baseline": "middle"}[ref] if ref else code[0])
    changes = {"scripts/z1/replay_checkpoint.py": {"before": "old", "after": "new"}}
    monkeypatch.setattr(recovery, "instrument_changes", lambda _: changes)
    monkeypatch.setattr(recovery, "registered_repair", lambda *_: {"reason": "reviewed repair"})
    monkeypatch.setattr(recovery.snapshot, "save", lambda _: 0)
    monkeypatch.setattr(recovery.snapshot, "restore", lambda _: 0)
    point = recovery.Checkpoints(tmp_path / "recovery", trace, {"code_sha256": "old", "models": "same"})
    try:
        with trace.open("w") as sink:
            sink.write('{"event":"written","turn":1}\n')
            point.capture(sink, {"completed": {"one": 1}})
        first = point.prepare_instrument_continuation("first-baseline")
        # Existing installations have the first receipt under the legacy name.
        point.receipt_path(first["from_identity"], first["to_identity"]).rename(
            point.root / "instrument-continuation.json")
        legacy = (point.root / "instrument-continuation.json").read_bytes()
        point.identity = first["to_identity"]
        point.recover()
        with trace.open("a") as sink:
            sink.write('{"event":"written","turn":2}\n')
            point.capture(sink, {"completed": {"one": 2}})
        original = point.pointer.read_bytes()
        second_prefix = trace.read_bytes()
        code[0] = "latest"
        second = point.prepare_instrument_continuation("second-baseline")
        assert second != first and second["completed"] == {"one": 2}
        assert point.prepare_instrument_continuation("second-baseline") == second
        assert (point.root / "instrument-continuation.json").read_bytes() == legacy
        assert point.pointer.read_bytes() == original and trace.read_bytes() == second_prefix
        point.identity = second["to_identity"]
        with trace.open("a") as sink:
            sink.write('{"event":"failed"}\n')
        point.recover()
        markers = [json.loads(line) for line in trace.read_text().splitlines()
                   if json.loads(line)["event"] == "instrument_continuation"]
        assert [r["to_identity"]["code_sha256"] for r in markers] == ["middle", "latest"]
        assert (point.root / second["archive"] / "trace-prefix.jsonl").read_bytes() == second_prefix
        assert (point.root / first["archive"] / "trace-prefix.jsonl").read_text() == '{"event":"written","turn":1}\n'
    finally:
        point.close()
