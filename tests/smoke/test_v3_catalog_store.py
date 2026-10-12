"""Lossless catalog reuse preserves native identities and observed counts."""

import copy
import gzip
import hashlib
import io
import json

import pytest

from experiments.v3_repair import artifact_inventory as inventory
from experiments.v3_repair import catalog_store as storage
from experiments.v3_repair.prepare_inputs import digest


def catalog():
    schema = {name: dict(primary=["id"], state=list(fields) + ["id"], content=list(fields))
              for name, fields in inventory.CONTENT.items()}
    rows = [dict(table="codex_claims", key=name, logical_identity=["codex_claims", name],
                 stored_content_sha256=digest("same content"), stored_state_sha256=digest("same state"))
            for name in ("claim-a", "claim-b")]
    result = dict(version="v3", policy=inventory.POLICY, schema=schema, rows=rows,
                  eligible_inventory="unobserved", lifetime_write_events="unobserved")
    result["catalog_sha256"] = digest(result)
    inventory.validate(result)
    return result


def test_exact_bytes_repeated_references_preserve_distinct_equal_text_rows(tmp_path):
    value = catalog()
    first = storage.persist(tmp_path, value)
    file = tmp_path / storage.DIRECTORY / (first["sha256"] + ".json.gz")
    before = file.read_bytes(), file.stat().st_mtime_ns
    assert storage.persist(tmp_path, copy.deepcopy(value)) == first
    assert (file.read_bytes(), file.stat().st_mtime_ns) == before
    restored = storage.load(tmp_path, first)
    assert restored == value and len(inventory.validate(restored)) == 2
    assert gzip.decompress(before[0]) == json.dumps(value, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()
    assert len(list((tmp_path / storage.DIRECTORY).iterdir())) == 1


@pytest.mark.parametrize("field", ["stored_content_sha256", "stored_state_sha256"])
def test_changed_payload_or_state_has_new_complete_catalog(tmp_path, field):
    original = catalog()
    changed = copy.deepcopy(original)
    changed["rows"][0][field] = digest("changed")
    changed["catalog_sha256"] = digest({k: v for k, v in changed.items() if k != "catalog_sha256"})
    refs = [storage.persist(tmp_path, value) for value in (original, changed)]
    assert refs[0] != refs[1]
    assert storage.load(tmp_path, refs[0]) == original
    assert storage.load(tmp_path, refs[1]) == changed
    if field == "stored_state_sha256":
        assert changed["rows"][0]["stored_content_sha256"] == original["rows"][0]["stored_content_sha256"]


def test_structural_owners_roundtrip_without_mutating_messages_or_callers(tmp_path):
    value = catalog()
    record = dict(event="chronological_probe", stages={arm: dict(artifact_inventory=value,
        prompt_messages=[dict(role="user", content="artifact_inventory is a word in this question")])
        for arm in ("full", "no_codex", "vector_only", "recent_only")})
    original = copy.deepcopy(record)
    saved = storage.externalize(tmp_path, record)
    assert record == original
    assert all("artifact_inventory" not in stage for stage in saved["stages"].values())
    assert storage.materialize(tmp_path, saved) == record
    assert len(list((tmp_path / storage.DIRECTORY).iterdir())) == 1
    references = storage.verify_trace(tmp_path, [dict(identity=dict(artifact_catalogs=storage.POLICY)), saved])
    assert len(references) == 1


def test_inline_legacy_records_stay_equal_and_write_nothing(tmp_path):
    record = dict(event="source_preflight", preflight=dict(artifact_inventory=catalog()))
    assert storage.materialize(None, record) == record
    assert storage.verify_trace(tmp_path, [dict(identity={}), record]) == {}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["missing", "changed_blob", "size", "hash", "codec", "catalog_identity"])
def test_reference_failure_never_becomes_empty_inventory(tmp_path, mode):
    reference = storage.persist(tmp_path, catalog())
    path = tmp_path / storage.DIRECTORY / (reference["sha256"] + ".json.gz")
    if mode == "missing": path.unlink()
    if mode == "changed_blob": path.write_bytes(b"not the original catalog")
    if mode == "size": reference["uncompressed_bytes"] += 1
    if mode == "hash": reference["stored_sha256"] = "0" * 64
    if mode == "codec": reference["codec"] = "different"
    if mode == "catalog_identity": reference["catalog_sha256"] = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        storage.load(tmp_path, reference)


def test_ambiguous_or_missing_new_capture_is_refused(tmp_path):
    value = catalog()
    reference = storage.persist(tmp_path, value)
    with pytest.raises(ValueError, match="ambiguous"):
        storage.materialize(tmp_path, dict(artifact_inventory=value, artifact_inventory_ref=reference))
    with pytest.raises(ValueError, match="unbound"):
        storage.verify_trace(tmp_path, [dict(identity={}), dict(artifact_inventory_ref=reference)])
    with pytest.raises(ValueError, match="capture missing"):
        storage.verify_trace(tmp_path, [dict(identity=dict(artifact_catalogs=storage.POLICY)),
            dict(event="source_preflight", preflight={})])


def test_durable_blob_precedes_trace_reference_and_corruption_is_not_overwritten(tmp_path, monkeypatch):
    value = catalog()
    checkpoints = []
    native_link = storage.os.link
    def capture(source, destination):
        checkpoints.append(gzip.decompress(source.read_bytes()))
        return native_link(source, destination)
    monkeypatch.setattr(storage.os, "link", capture)
    reference = storage.persist(tmp_path, value)
    assert checkpoints and hashlib.sha256(checkpoints[0]).hexdigest() == reference["sha256"]
    path = tmp_path / storage.DIRECTORY / (reference["sha256"] + ".json.gz")
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="immutable"):
        storage.persist(tmp_path, value)
    assert path.read_bytes() == b"corrupt"


def test_decoder_cache_is_bounded_and_rechecks_changed_files(tmp_path):
    reader = storage.Reader(tmp_path)
    original = catalog()
    references = []
    for i in range(4):
        value = copy.deepcopy(original)
        value["rows"][0]["stored_state_sha256"] = digest(i)
        value["catalog_sha256"] = digest({k: v for k, v in value.items() if k != "catalog_sha256"})
        reference = storage.persist(tmp_path, value)
        references.append(reference)
        assert reader(dict(artifact_inventory_ref=reference)) == value
        assert len(reader.cache) <= 2
    assert len(reader.references) == 4
    last = references[-1]
    path = tmp_path / storage.DIRECTORY / (last["sha256"] + ".json.gz")
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="stored catalog"):
        reader(dict(artifact_inventory_ref=last))


def test_metadata_without_catalog_needs_no_file_sink():
    from experiments.v3_repair.chronological_replay import emit
    sink = io.StringIO()
    emit(sink, event="plain_metadata", original="unchanged")
    assert json.loads(sink.getvalue()) == dict(event="plain_metadata", original="unchanged")
