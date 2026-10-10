"""Mechanical coordinate deltas must not become semantic or answer grades."""

import copy
import json

import pytest

from experiments.v3_repair import compare_tuning_contexts as comparison
from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from scripts.z1.evidence_content import sha


@pytest.fixture
def matched(tmp_path, monkeypatch):
    root, baseline, candidate = (tmp_path / name for name in ("producer", "baseline", "candidate"))
    root.mkdir(); (root / "seed.jsonl").write_text("controlled-original-trace\n")
    original = "Chosen port7319."
    def stage(text, tokens):
        fragment = dict(text=text, source_row="row1", origin_batches=["batch1"])
        return dict(gate=dict(b2_retrieve=True, final_retrieve=True, source_action="retrieved",
                             routed_model="fixed-answerer", total_budget=4000),
            generated_by_leg={"vector": [fragment]}, ranked_candidates=[fragment],
            budgeted=[fragment], selected=[fragment], prompt_messages=[dict(role="system", content=text)],
            prompt_tokens=tokens, selected_tokens=tokens - 1,
            gold_fragment_coverage=dict(gold_turns=1, selected_or_source_note=1,
                source_turn_rank_at_5=1, source_turn_rank_at_10=1))
    probes = {key: dict(probe_id=key, split_turn=64, question="Which port?", expected_answer="7319",
        metric_contract=dict(lookup_rank_tuning_eligible=key == "q1", task_types=["episodic_lookup" if key == "q1" else "procedural"]),
        gold_sources=[dict(turn=1, prompt=original, response="Recorded reply.")],
        preflight=stage(original, 30), controls={arm: stage(original, 30) for arm in comparison.ARMS if arm != "full"})
        for key in ("q1", "q2")}
    monkeypatch.setattr(comparison, "source_probes", lambda _root: (
        dict(producer_identity="writer", trace_sha256="trace"), probes))
    for directory, is_candidate in ((baseline, False), (candidate, True)):
        directory.mkdir(); (directory / "units").mkdir()
        identity = dict(producer="writer", producer_trace_sha256="trace", coordinate_seal="same-seal",
            code=[], probes=sorted(probes), settings_sha256="same-baseline",
            coordinate=dict(id="smaller" if is_candidate else "baseline"))
        header = dict(identity=identity, identity_sha256=digest(identity), complete=False, score_of_record=False)
        write_or_verify(directory / "run.json", header)
        records = []
        for key, original_probe in sorted(probes.items()):
            current = copy.deepcopy(original_probe)
            if is_candidate:
                # Same source IDs; one value disappears and the other remains negated.
                text = "Recorded configuration." if key == "q1" else "Do NOT use port7319."
                current["preflight"] = stage(text, 15)
                current["controls"] = {arm: stage(text, 15) for arm in comparison.ARMS if arm != "full"}
            unit = dict(probe_id=key, identity_sha256=header["identity_sha256"], source_probe_sha256=digest(original_probe),
                        probe=current, answer_quality="unmeasured")
            unit["unit_sha256"] = digest(unit)
            write_or_verify(directory / "units" / (digest(key) + ".json"), unit)
            records.append(unit)
        result = dict(**header, records=records); result["complete"] = True
        write_or_verify(directory / "contexts.json", result)
    review = tmp_path / "source-units.json"
    start = original.index("7319")
    reviews = [dict(probe_id=key, reviewed_through_turn=64, reason="Controlled exact port source.",
        requirements=[dict(id="port", kind="exact_literal", alternatives=[dict(turn=1, role="user",
            start=start, end=start + 4, source_sha256=sha(original), literal="7319")])]) for key in probes]
    review.write_text(json.dumps(dict(kind="v3_required_evidence_review", trace_sha256="trace", records=reviews)))
    return root, baseline, candidate, review


def test_source_hit_with_missing_value_and_negated_retention_do_not_earn_answer_quality(matched):
    root, baseline, candidate, units = matched
    report = comparison.compare(root, baseline, candidate, units)
    rows = {row["probe_id"]: row for row in report["records"] if row["arm"] == "full"}
    assert rows["q1"]["source_coverage"]["candidate"]["selected_or_source_note"] == 1
    assert rows["q1"]["required_literal_survival"]["candidate"]["port"]["final_input"] == "missing"
    assert rows["q2"]["required_literal_survival"]["candidate"]["port"]["final_input"] == "retained"
    for row in rows.values():
        assert row["semantic_support"] == row["answer_quality"] == "unmeasured"
        assert row["prompt_tokens"]["delta"] == -15 and row["rendered_input_changed"]
        assert not row["selected_sources_added"] and not row["selected_sources_removed"]
    assert rows["q1"]["source_rank_diagnostics"] is not None
    assert rows["q2"]["source_rank_diagnostics"] is None
    assert report["totals"]["full"]["rank_eligible_occurrences"] == 1
    assert report["promotion"] == "not_performed" and report["score_of_record"] is False


def test_no_unit_labels_remains_unmeasured_and_report_resumes_byte_stably(matched, tmp_path):
    root, baseline, candidate, _ = matched
    first = comparison.compare(root, baseline, candidate)
    assert all(r["required_literal_survival"] is None for r in first["records"])
    out = tmp_path / "report.json"; write_or_verify(out, first)
    before = out.read_bytes()
    write_or_verify(out, comparison.compare(root, baseline, candidate))
    assert out.read_bytes() == before
    altered = copy.deepcopy(first); altered["totals"]["full"]["occurrences"] = 99
    with pytest.raises(ValueError, match="immutable"): write_or_verify(out, altered)
    assert out.read_bytes() == before


@pytest.mark.parametrize("fault", ["body", "aggregate", "roster", "metadata", "producer", "reader"])
def test_mismatched_or_corrupt_coordinate_refuses_pooled_metrics(matched, fault):
    root, baseline, candidate, _ = matched
    path = candidate / "units" / (digest("q1") + ".json")
    if fault in {"body", "metadata"}:
        unit = json.loads(path.read_text())
        unit["probe"]["expected_answer"] = "Wrong question reference"
        if fault == "metadata": unit["unit_sha256"] = digest({k: v for k, v in unit.items() if k != "unit_sha256"})
        path.write_text(json.dumps(unit))
    elif fault == "roster": path.unlink()
    elif fault == "aggregate":
        aggregate = json.loads((candidate / "contexts.json").read_text()); aggregate["complete"] = False
        (candidate / "contexts.json").write_text(json.dumps(aggregate))
    else:
        for filename in ("run.json", "contexts.json"):
            value = json.loads((candidate / filename).read_text())
            if fault == "reader": value["identity"]["reader_adoption"] = {"identity_sha256": "different-reader"}
            else: value["identity"]["producer"] = "different-writer"
            value["identity_sha256"] = digest(value["identity"])
            if filename == "contexts.json":
                for unit in value["records"]:
                    unit["identity_sha256"] = value["identity_sha256"]
                    unit["unit_sha256"] = digest({k: v for k, v in unit.items() if k != "unit_sha256"})
                    (candidate / "units" / (digest(unit["probe_id"]) + ".json")).write_text(json.dumps(unit))
            (candidate / filename).write_text(json.dumps(value))
    with pytest.raises(ValueError): comparison.compare(root, baseline, candidate)


@pytest.mark.parametrize("fault", ["future", "offset", "duplicate", "unknown", "semantic", "trace"])
def test_source_requirement_or_grade_transfer_fault_is_refused(matched, fault):
    root, baseline, candidate, review = matched
    packet = json.loads(review.read_text())
    first = packet["records"][0]
    if fault == "future": first["requirements"][0]["alternatives"][0]["turn"] = 100
    elif fault == "offset": first["requirements"][0]["alternatives"][0]["start"] = 0
    elif fault == "duplicate": packet["records"].append(first)
    elif fault == "unknown": first["probe_id"] = "unreviewed-question"
    elif fault == "semantic": first["requirements"][0]["semantic_review"] = {"final_input": {"status": "supported"}}
    else: packet["trace_sha256"] = "wrong-trace"
    review.write_text(json.dumps(packet))
    with pytest.raises(ValueError): comparison.compare(root, baseline, candidate, review)
