"""Actual settings/coordinate inventory remains distinct from a tuning sweep."""

import copy

import pytest

from experiments.v3_repair.export_tuning_ledger import PENDING, build, export
from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from experiments.v3_repair.settings_inventory import build_inventory
from experiments.v3_repair.tuning_plan import ROOT, coordinates
from src.api.config import Settings, settings


@pytest.fixture
def inputs():
    grid = dict(rows=coordinates(settings), blocked_stages=dict(
        relevance_admission="required_content_and_rejection_opponents",
        trust_activation="source_truth", gates_style="style_opponents", lifecycle="aging_targets"))
    return grid, build_inventory(ROOT, Settings, settings)


def test_actual_grid_all_inventory_fields_have_explicit_dispositions(inputs):
    grid, inventory = inputs
    value = build(grid, inventory, {})
    assert value["coordinate_count"] == len(grid["rows"])
    assert value["planned_settings_fields"] == 13
    assert sum(value["disposition_counts"].values()) == len(inventory["settings"])
    rows = {r["name"]: r for r in value["settings"]}
    assert rows["retrieval_rerank_enabled"]["trial_values"] == [False]
    assert rows["procedural_similarity_threshold"]["disposition"] == "retired_no_tuning"
    assert rows["source_support_threshold"]["disposition"] == "held_current_policy"
    assert rows["ltm_decision_threshold"]["dependency"] == "style_opponents"
    assert rows["procedural_min_cited_turns"]["requires_writer_replay"]
    assert rows["turn_summary_coverage_threshold"]["requires_writer_replay"]
    assert not rows["codex_direct_trust_floor"]["requires_writer_replay"]
    assert value["method"] == "named_tuning_dispositions_v3"
    floor = rows["retrieval_rerank_min_score"]
    assert floor["disposition"] == "pending_qualified_stage"
    assert floor["stage"] == "relevance_admission"
    assert floor["dependency"] == "required_content_and_rejection_opponents"
    assert floor["trial_values"] is None and not floor["qualified"]
    assert not floor["requires_writer_replay"]
    for name in ("decay_strengthen_amount", "codex_retention_increment", "codex_retention_cap"):
        assert rows[name]["requires_writer_replay"]
        assert rows[name]["disposition"] == "pending_qualified_stage"
    assert rows["codex_retention_rank_weight"]["stage"] == "lifecycle"
    assert not rows["codex_retention_rank_weight"]["requires_writer_replay"]
    assert value["research_input_cap"]["trial_values"] == [4000, 8000, 16000, 32000]
    for names in PENDING.values():
        assert all(rows[n]["trial_values"] is None and rows[n]["qualified"] is False for n in names)


def test_actual_seal_fixed_profile_resume_and_hash_refusal(inputs, tmp_path, monkeypatch):
    grid, inventory = inputs
    seal = dict(identity=dict(settings_sha256=digest(settings.model_dump(mode="json"))))
    seal["identity_sha256"] = digest(seal["identity"])
    seal["artifacts"] = {"coordinates.json": digest(grid), "consumer-ledger.json": digest(inventory)}
    for name, value in [("manifest.json", seal), ("coordinates.json", grid), ("consumer-ledger.json", inventory)]:
        write_or_verify(tmp_path / name, value)
    first = export(tmp_path)
    output = tmp_path / "first.json"
    write_or_verify(output, first)
    retained = (output.read_bytes(), output.stat().st_mtime_ns)
    inputs_before = {name: (path.read_bytes(), path.stat().st_mtime_ns)
        for name in ("manifest.json", "coordinates.json", "consumer-ledger.json")
        if (path := tmp_path / name).exists()}
    again = export(tmp_path)
    write_or_verify(output, again)
    assert first == again
    assert retained == (output.read_bytes(), output.stat().st_mtime_ns)
    assert inputs_before == {name: (path.read_bytes(), path.stat().st_mtime_ns)
        for name in inputs_before if (path := tmp_path / name).exists()}
    assert first["fixed_profile_controls"]["auto_finetune"]["resolved_value"] is False
    assert first["fixed_profile_controls"]["maintenance_intervals"]["resolved_value"] == settings.maintenance_intervals
    monkeypatch.setattr(settings, "runtime_cycles_cap", settings.runtime_cycles_cap + 1)
    with pytest.raises(ValueError, match="fixed profile differs"): export(tmp_path)
    write_or_verify(tmp_path / "first.json", first)


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "missing_pending", "missing_stage", "empty_dependency"])
def test_incomplete_or_unrecognized_knob_ledger_refuses(inputs, fault):
    grid, inventory = copy.deepcopy(inputs)
    if fault == "duplicate": inventory["settings"].append(inventory["settings"][0])
    if fault == "unknown": grid["rows"][1]["overrides"]["invented_setting"] = 1
    if fault == "missing_pending":
        inventory["settings"] = [r for r in inventory["settings"] if r["name"] != "ltm_decision_threshold"]
    if fault == "missing_stage":
        del grid["blocked_stages"]["relevance_admission"]
    if fault == "empty_dependency":
        grid["blocked_stages"]["relevance_admission"] = ""
    with pytest.raises(ValueError): build(grid, inventory, {})
