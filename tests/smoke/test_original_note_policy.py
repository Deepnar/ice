"""Changing summary verification must not hide unchanged complete originals."""

import pytest

from src.memory.summary_snapshot import bind_snapshot, compose_parts, snapshot_matches


@pytest.mark.parametrize("mode, readable", [("source", True), ("supported", False)])
def test_new_verifier_policy_preserves_only_unchanged_original_cache(mode, readable):
    sources = [dict(id="one", batch_id="batch", fingerprint="source-fingerprint", timestamp=None)]
    parts = [dict(source_ids=["one"], mode=mode, text="The port remains 8391.", recorded_range="unknown")]
    text = compose_parts(parts)
    manifest = bind_snapshot(sources, text, parts=parts)
    manifest["representation_policy"].pop("summary_support")
    assert snapshot_matches(manifest, text, sources) is readable
    assert not snapshot_matches(manifest, text, [{**sources[0], "fingerprint": "changed-source"}])
    assert not snapshot_matches(manifest, text + " Invented addition.", sources)
