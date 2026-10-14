"""Real answer runner/SDK adapter with controlled responses and frozen inputs."""

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest

from experiments.lme.cloud_provider import TextGenerator
from experiments.v3_repair import run_tuning_answers as bridge
from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from scripts.z1 import answer_as_of as native


@pytest.fixture
def packets(tmp_path, monkeypatch):
    root, baseline, candidate, out = (tmp_path / n for n in ("producer", "baseline", "candidate", "answers"))
    root.mkdir()
    meta = dict(event="run", meta=dict(inputs=[], settings_resolved=dict(recent_window_max_turns=40),
                                     extra=dict(clock_policy="controlled-source-time")))
    (root / "seed.jsonl").write_text(json.dumps(meta) + "\n")
    stage = dict(prompt_messages=[dict(role="system", content="Use the supplied memory."),
                                 dict(role="user", content="Which configured port?")],
                 prompt_tokens=12, selected_tokens=0, gold_fragment_coverage=dict(selected=0),
                 lineage=dict(future_source_ids=[]), gate=dict(final_retrieve=False), selected=[])
    probes = {key: dict(event="as_of_probe", probe_id=key, conversation="synthetic", split_turn=64,
        state_turns=64, cutoff_kind="reviewed_development_checkpoint", type="episodic_lookup",
        question="Which configured port?", question_time="2025-01-01T00:00:00+00:00",
        gold_turns=[1], expected_answer="ORACLE EXPECTED CANARY", catalog_expected_answer="ORACLE CATALOG CANARY",
        gold_sources=[dict(turn=1, prompt="ORACLE SOURCE CANARY", response="ORACLE REPLY CANARY",
            recorded_at="2024-01-01T00:00:00+00:00", source_ids=["row1"])],
        label_review=dict(task_types=["episodic_lookup"], knowledge_scope="private_history", reason="Controlled source key.",
                          reviewed_through_turn=64, recent_only_answerable=False, question_family_id="one-dependent-family"),
        metric_contract=dict(lookup_rank_tuning_eligible=True), preflight=copy.deepcopy(stage),
        controls={arm: copy.deepcopy(stage) for arm in bridge.ARMS if arm != "full"}) for key in ("q1", "q2")}
    monkeypatch.setattr(bridge, "source_probes", lambda _root: (dict(producer_identity="writer", trace_sha256="trace"), probes))
    for path, is_candidate in ((baseline, False), (candidate, True)):
        path.mkdir(); (path / "units").mkdir()
        coordinate = dict(id="budget1" if is_candidate else "baseline", stage="budget" if is_candidate else "baseline", requires_writer_replay=False)
        identity = dict(producer="writer", producer_trace_sha256="trace", coordinate_seal="same-seal", code=[],
                        probes=sorted(probes), settings_sha256="same-baseline", coordinate=coordinate)
        header = dict(identity=identity, identity_sha256=digest(identity), complete=False)
        write_or_verify(path / "run.json", header)
        records = []
        for key, probe in sorted(probes.items()):
            unit = dict(probe_id=key, identity_sha256=header["identity_sha256"], source_probe_sha256=digest(probe),
                        probe=copy.deepcopy(probe), answer_quality="unmeasured")
            unit["unit_sha256"] = digest(unit)
            write_or_verify(path / "units" / (digest(key) + ".json"), unit); records.append(unit)
        result = dict(**header, records=records); result["complete"] = True
        write_or_verify(path / "contexts.json", result)
    monkeypatch.setattr(native, "LOGS", tmp_path)
    monkeypatch.setattr(native, "load_selected_env", lambda: None)
    monkeypatch.setenv("PROBE_API_BASE_URL", "https://test.invalid/v1")
    monkeypatch.setattr(native.time, "sleep", lambda _seconds: None)
    return root, baseline, candidate, out, probes


def fake_sdk(monkeypatch, request):
    client = NS(responses=NS(create=request))
    monkeypatch.setattr(native, "TextGenerator", lambda profile: TextGenerator(profile, client=client))


def test_offline_plan_real_parser_and_two_survivor_bound_without_client_or_credentials(packets, monkeypatch):
    root, baseline, candidate, out, _ = packets
    monkeypatch.setattr(native, "TextGenerator", lambda *_a: pytest.fail("planning loaded SDK client"))
    monkeypatch.setattr(native, "load_selected_env", lambda: pytest.fail("planning loaded credentials"))
    manifest = bridge.plan(root, baseline, [candidate], out)
    assert manifest["identity"]["answer_model"] == "gpt-6-luna"
    assert manifest["expected_answers"] == 16 and manifest["bounded_transient_attempts"] == 32
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert bridge.plan(root, baseline, [candidate], out) == manifest
    assert before == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="repeats"): bridge.plan(root, baseline, [candidate] * 3, out)
    with pytest.raises(ValueError, match="repeats"): bridge.plan(root, baseline, [candidate, candidate], out)


def test_actual_sdk_messages_sessions_cache_and_reference_metadata(packets, monkeypatch):
    root, baseline, candidate, out, _ = packets
    calls = []
    def request(**kwargs):
        calls.append(kwargs)
        assert "ORACLE" not in json.dumps(kwargs)
        return dict(status="completed", output_text="Controlled answer.", id=f"response{len(calls)}", usage=dict(input_tokens=12))
    fake_sdk(monkeypatch, request)
    manifest = bridge.plan(root, baseline, [candidate], out)
    assert bridge.execute(out, manifest) == 0 and len(calls) == 16
    assert len({c["extra_headers"]["x-opencode-session"] for c in calls}) == 16
    assert all(c["model"] == "gpt-6-luna" and "temperature" not in c for c in calls)
    assert all(c["input"] == [dict(role="system", content="Use the supplied memory."),
                              dict(role="user", content="Which configured port?")] for c in calls)
    for path in out.rglob("answers-*.json"):
        value = json.loads(path.read_text())
        assert value["diagnostic_only"] and value["score_of_record"] is False
        assert value["processing_complete"] and value["complete"]
        for row in value["records"]:
            assert row["expected_answer"] == "ORACLE EXPECTED CANARY"
            assert "ORACLE SOURCE CANARY" in row["gold_turn_text"]
            assert row["label_review"]["question_family_id"] == "one-dependent-family"
            assert row["prompt_evidence_support"] == "unreviewed" and row["answer_record_sha256"]
    before = {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("answers-*.json")}
    monkeypatch.setattr(native, "TextGenerator", lambda *_a: pytest.fail("completed resume loaded SDK client"))
    assert bridge.execute(out, manifest) == 0
    assert len(calls) == 16 and before == {str(p.relative_to(out)): p.read_bytes() for p in out.rglob("answers-*.json")}
    # The existing independent grader accepts distinct configurations without
    # regenerating full answers or changing their native cache tags.
    from scripts.z1 import score_frozen_answers
    paths = [out / identifier / "grading-full.json" for identifier in manifest["identity"]["collections"]]
    monkeypatch.setattr(score_frozen_answers, "private_path", lambda p: p)
    arms, hashes = score_frozen_answers.collect(paths, dict(judge="controlled-independent"))
    assert set(arms) == {"baseline:full", "budget1:full"} and len(hashes) == 2
    assert all(arms["baseline:full"][key]["grade_key"] == arms["budget1:full"][key]["grade_key"]
               for key in arms["baseline:full"])


@pytest.mark.parametrize("fault", ["interrupt", "isolated", "quota"])
def test_real_cloud_fault_receipts_resume_without_repeating_successes(packets, monkeypatch, fault):
    from experiments.lme.cloud_provider import ProviderAccessError
    root, baseline, _, out, _ = packets
    calls = []
    enabled = {"fault": True}
    def request(**kwargs):
        calls.append(kwargs)
        if enabled["fault"]:
            if fault == "interrupt" and len(calls) == 2: raise KeyboardInterrupt()
            if fault == "isolated" and len(calls) in (2, 3): raise ConnectionError("controlled network fault" * 50)
            if fault == "quota" and len(calls) == 2: raise ProviderAccessError("fixture", "quota", 429, "quota")
        return dict(status="completed", output_text="Controlled answer.", id=f"response{len(calls)}")
    fake_sdk(monkeypatch, request)
    manifest = bridge.plan(root, baseline, [], out)
    if fault == "interrupt":
        with pytest.raises(KeyboardInterrupt): bridge.execute(out, manifest)
    elif fault == "quota":
        with pytest.raises(RuntimeError): bridge.execute(out, manifest)
    else: assert bridge.execute(out, manifest) == 0
    first_path = out / manifest["identity"]["collections"][0] / "answers-full.json"
    saved = json.loads(first_path.read_text())
    first = next(r for r in saved["records"] if r["probe_id"] == "q1")
    assert first["answer"] == "Controlled answer." and first["cloud_attempts"] == 1
    failed = next(r for r in saved["records"] if r["probe_id"] == "q2")
    if fault == "interrupt": assert failed["error_type"] == "UnconfirmedCloudCall" and failed["cloud_attempts"] == 1
    elif fault == "isolated": assert failed["fault_disposition"] == "degraded_final" and failed["cloud_attempts"] == 2
    else: assert failed["error_status"] == 429
    fault_path = first_path.parent / "faults-full.jsonl"
    receipts = [json.loads(line) for line in fault_path.read_text().splitlines()]
    assert len(receipts) == (2 if fault == "isolated" else 1)
    assert all(r["context"]["probe_id"] == "q2" and r["context"]["cutoff"] == 64
               and r["context"]["coordinate"] == "baseline" and r["context"]["arm"] == "full"
               and r["context"]["model"] == "gpt-6-luna" and r["context"]["artifact_id"]
               and r["traceback"] for r in receipts)
    assert [r["context"]["attempt"] for r in receipts] == ([1, 2] if fault == "isolated" else [1])
    if fault == "isolated": assert len(receipts[0]["error"]) > 1000
    fault_bytes = fault_path.read_bytes()
    before = copy.deepcopy(first)
    enabled["fault"] = False
    assert bridge.execute(out, manifest) == 0
    assert fault_path.read_bytes() == fault_bytes
    resumed = json.loads(first_path.read_text())
    assert next(r for r in resumed["records"] if r["probe_id"] == "q1") == before
    assert sum("full-q1" in c["extra_headers"]["x-opencode-session"] for c in calls) == 1
    assert bridge.execute(out, manifest) == 0


def test_changed_fault_journal_refuses_even_complete_answer_cache(packets, monkeypatch):
    root, baseline, _, out, _ = packets
    fake_sdk(monkeypatch, lambda **_k: dict(status="completed", output_text="Controlled answer."))
    manifest = bridge.plan(root, baseline, [], out)
    assert bridge.execute(out, manifest) == 0
    path = out / manifest["identity"]["collections"][0] / "faults-full.jsonl"
    recorder = bridge.FaultRecorder(path, manifest["identity_sha256"], private_root=native.LOGS)
    recorder.record(ValueError("controlled prior fault"), dict(stage="cloud_answer", probe_id="q1"))
    body = json.loads(path.read_text()); body["error"] = "altered"
    path.write_text(json.dumps(body) + "\n")
    monkeypatch.setattr(native, "TextGenerator", lambda *_a: pytest.fail("changed journal loaded SDK"))
    with pytest.raises(ValueError, match="chain changed"): bridge.execute(out, manifest)


def test_credentials_loaded_by_native_runner_after_observer_are_redacted(packets, monkeypatch):
    root, baseline, _, out, _ = packets
    secret = "late-test-provider-key"
    monkeypatch.delenv("PROBE_API_KEY", raising=False)
    monkeypatch.setattr(native, "load_selected_env", lambda: monkeypatch.setenv("PROBE_API_KEY", secret))
    def request(**kwargs):
        if "full-q1" in kwargs["extra_headers"]["x-opencode-session"]:
            raise ConnectionError(f"The problematic value is {secret}")
        return dict(status="completed", output_text="Controlled answer.")
    fake_sdk(monkeypatch, request)
    manifest = bridge.plan(root, baseline, [], out)
    assert bridge.execute(out, manifest) == 0
    path = out / manifest["identity"]["collections"][0] / "faults-full.jsonl"
    assert secret not in path.read_text() and "[REDACTED]" in path.read_text()
    assert len(path.read_text().splitlines()) == 2


@pytest.mark.parametrize("fault", ["body", "metadata"])
def test_changed_cached_answer_or_source_metadata_refuses_before_sdk(packets, monkeypatch, fault):
    root, baseline, _, out, _ = packets
    fake_sdk(monkeypatch, lambda **_k: dict(status="completed", output_text="Controlled answer."))
    manifest = bridge.plan(root, baseline, [], out); assert bridge.execute(out, manifest) == 0
    path = out / manifest["identity"]["collections"][0] / "answers-full.json"
    value = json.loads(path.read_text())
    row = value["records"][0]
    if fault == "body": row["answer"] = "Altered cached response"
    else:
        row["expected_answer"] = "Different question reference"
        row["answer_record_sha256"] = digest({k: v for k, v in row.items() if k != "answer_record_sha256"})
    path.write_text(json.dumps(value))
    monkeypatch.setattr(native, "TextGenerator", lambda *_a: pytest.fail("corruption loaded SDK client"))
    with pytest.raises(ValueError, match="body changed|metadata changed"): bridge.execute(out, manifest)


def full_arm_candidate(tmp_path, probes, name="full-arm", coordinate_id="budget2"):
    """A full-arm collection over the same producer: controls are absent, not copied."""
    from experiments.v3_repair import run_full_arm_coordinates as full_arm
    path = tmp_path / name; path.mkdir(); (path / "units").mkdir()
    identity = dict(producer="writer", producer_trace_sha256="trace", coordinate_seal="same-seal", code=[],
                    probes=sorted(probes), settings_sha256="same-baseline",
                    coordinate=dict(id=coordinate_id, stage="budget", requires_writer_replay=False),
                    arms=["full"], collector_policy=full_arm.POLICY, collector=dict(path="c", sha256="c", bytes=1))
    header = dict(identity=identity, identity_sha256=digest(identity), complete=False)
    write_or_verify(path / "run.json", header)
    records = []
    for key, probe in sorted(probes.items()):
        unit = dict(probe_id=key, identity_sha256=header["identity_sha256"], source_probe_sha256=digest(probe),
                    probe=dict(copy.deepcopy(probe), controls={}), answer_quality="unmeasured")
        unit["unit_sha256"] = digest(unit)
        write_or_verify(path / "units" / (digest(key) + ".json"), unit); records.append(unit)
    result = dict(**header, records=records); result["complete"] = True
    write_or_verify(path / "contexts.json", result)
    return path


def test_full_arm_candidate_answers_only_full_beside_four_arm_baseline(packets, monkeypatch, tmp_path):
    root, baseline, _, out, probes = packets
    candidate = full_arm_candidate(tmp_path, probes)
    calls = []
    def request(**kwargs):
        calls.append(kwargs)
        return dict(status="completed", output_text="Controlled answer.", id=f"response{len(calls)}", usage=dict(input_tokens=12))
    fake_sdk(monkeypatch, request)
    manifest = bridge.plan(root, baseline, [candidate], out)
    assert manifest["identity"]["collection_arms"] == [list(bridge.ARMS), ["full"]]
    assert manifest["expected_answers"] == 10
    assert bridge.execute(out, manifest) == 0 and len(calls) == 10
    directory = out / manifest["identity"]["collections"][1]
    assert sorted(p.name for p in directory.glob("answers-*.json")) == ["answers-full.json"]
    from scripts.z1 import seed_v3
    assert native.PROMPT_ARMS == seed_v3.PROMPT_ARMS == bridge.ARMS
    # A full-arm collection can never stand in for the four-arm baseline.
    with pytest.raises(ValueError, match="all four arms"):
        bridge.plan(root, candidate, [], tmp_path / "other")


def test_three_survivors_in_one_stage_refuse_but_a_baseline_repeat_is_the_noise_control(packets, tmp_path):
    root, baseline, _, out, probes = packets
    three = [full_arm_candidate(tmp_path, probes, f"c{i}", f"budget-{i}") for i in range(3)]
    with pytest.raises(ValueError, match="at most two"):
        bridge.plan(root, baseline, three, out)
    repeat = full_arm_candidate(tmp_path, probes, "repeat", "baseline")
    manifest = bridge.plan(root, baseline, [repeat, *three[:2]], tmp_path / "answers2")
    assert manifest["identity"]["coordinate_ids"] == ["baseline", "baseline-repeat", "budget-0", "budget-1"]
    assert manifest["identity"]["stages"] == ["budget"]


def test_later_stage_can_answer_only_the_baseline_full_arm(packets, tmp_path):
    root, baseline, _, out, probes = packets
    candidate = full_arm_candidate(tmp_path, probes)
    manifest = bridge.plan(root, baseline, [candidate], out, baseline_full_only=True)
    assert manifest["identity"]["collection_arms"] == [["full"], ["full"]]
    assert manifest["expected_answers"] == 4
