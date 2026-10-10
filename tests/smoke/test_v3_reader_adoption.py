"""Exact registered reader changes cannot relax frozen writer identities."""

import copy
import hashlib

import pytest

from experiments.v3_repair import reader_adoption as adoption
from experiments.v3_repair.prepare_inputs import digest, fingerprint


@pytest.fixture
def profile_source(tmp_path, monkeypatch):
    from scripts.z1 import replay_checkpoint
    from experiments.v3_repair import run_tuning_contexts

    source = tmp_path / "src/reader.py"
    source.parent.mkdir()
    old, new = b"frozen writer\n", b"approved reader\n"
    source.write_bytes(old)
    record = fingerprint(source, root=tmp_path)
    source.write_bytes(new)
    monkeypatch.setattr(adoption, "ROOT", tmp_path)
    monkeypatch.setattr(adoption, "TRANSITIONS", {"src/reader.py": dict(
        before=hashlib.sha256(old).hexdigest(), after=hashlib.sha256(new).hexdigest(), reason="Test transition.")})
    monkeypatch.setattr(adoption, "before_bytes", lambda _relative: old)
    monkeypatch.setattr(replay_checkpoint, "code_files", lambda: (tmp_path, [source]))
    for name in ("reader_adoption.py", "run_tuning_contexts.py", "compare_tuning_contexts.py", "audit_development.py"):
        path = tmp_path / "experiments/v3_repair" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("controlled instrument\n")
    metric = tmp_path / "scripts/z1/metric_contract.py"
    metric.parent.mkdir(parents=True); metric.write_text("controlled metric\n")
    identity = dict(partition="tuning", preparation="fixed", settings_sha256="fixed-settings", code=[record])
    config = dict(identity=identity, identity_sha256=digest(identity))
    seal = dict(identity=copy.deepcopy(identity), identity_sha256=digest(identity))
    monkeypatch.setattr(adoption, "bundle_bindings", lambda *_args: {"controlled-source": "source-hash"})
    monkeypatch.setattr(run_tuning_contexts, "resolved_baseline", lambda *_a, **_k: None)
    return tmp_path, source, config, seal, old


def test_profile_separates_frozen_native_code_and_current_reader(profile_source):
    root, source, config, seal, old = profile_source
    profile = adoption.build(root, config, seal)
    identity = profile["identity"]
    native = hashlib.sha256(b"src/reader.py" + old).hexdigest()
    assert identity["producer_native_code_sha256"] == native
    assert identity["producer_native_code_sha256"] != hashlib.sha256(b"src/reader.py" + source.read_bytes()).hexdigest()
    assert set(identity["approved_changes"]) == {"src/reader.py"}
    assert not profile["writer_replayed"] and not profile["execution_ready"]
    adoption.validate_runtime(profile, root, config)
    assert adoption.build(root, config, seal) == profile


@pytest.mark.parametrize("fault", ["reader", "unknown", "missing", "extra_native", "old_bytes", "duplicate", "outside"])
def test_unregistered_changes_refuse_before_bundle_inspection(profile_source, monkeypatch, fault):
    from scripts.z1 import replay_checkpoint

    root, source, config, seal, _old = profile_source
    monkeypatch.setattr(adoption, "bundle_bindings", lambda *_args: pytest.fail("invalid code reached bundle"))
    if fault == "reader": source.write_text("different implementation")
    elif fault == "unknown":
        other = root / "src/other.py"; other.write_text("original")
        expected = fingerprint(other, root=root); other.write_text("unapproved")
        config["identity"]["code"].append(expected)
    elif fault == "missing": source.unlink()
    elif fault == "extra_native":
        other = root / "src/other.py"; other.write_text("new module")
        monkeypatch.setattr(replay_checkpoint, "code_files", lambda: (root, [source, other]))
    elif fault == "old_bytes": monkeypatch.setattr(adoption, "before_bytes", lambda _relative: b"wrong historical bytes")
    elif fault == "duplicate": config["identity"]["code"].append(config["identity"]["code"][0])
    else: config["identity"]["code"][0]["path"] = "../outside.py"
    config["identity_sha256"] = digest(config["identity"])
    with pytest.raises((ValueError, FileNotFoundError)):
        adoption.build(root, config, seal)


@pytest.mark.parametrize("fault", ["identity", "source", "instrument", "reservation", "settings"])
def test_changed_profile_or_bound_contract_refuses(profile_source, monkeypatch, fault):
    root, _source, config, seal, _old = profile_source
    profile = adoption.build(root, config, seal)
    if fault == "identity": profile["identity"]["producer_native_code_sha256"] = "forged-current-code"
    elif fault == "source": monkeypatch.setattr(adoption, "bundle_bindings", lambda *_args: {"controlled-source": "changed"})
    elif fault == "instrument": (root / "experiments/v3_repair/run_tuning_contexts.py").write_text("changed after adoption")
    else:
        config["identity"]["partition" if fault == "reservation" else "settings_sha256"] = "validation" if fault == "reservation" else "changed"
        config["identity_sha256"] = digest(config["identity"])
    with pytest.raises(ValueError): adoption.validate(profile, root, config, seal)


def test_old_collector_without_explicit_profile_still_refuses(profile_source, monkeypatch):
    from experiments.v3_repair import run_tuning_contexts

    root, _source, config, _seal, _old = profile_source
    monkeypatch.setattr(run_tuning_contexts, "ROOT", root)
    monkeypatch.setattr(run_tuning_contexts, "fingerprint", lambda path: fingerprint(path, root=root))
    with pytest.raises(ValueError, match="code changed"):
        run_tuning_contexts.current_code(config["identity"])
