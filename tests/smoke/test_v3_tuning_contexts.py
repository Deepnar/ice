"""Coordinate cache guards plus exact SQL prefix/read-only/restore behavior.

The SQL control uses real snapshots and preparation with a controlled encoder
and classifier. It does not measure model or writer semantic quality.
"""

import copy
import hashlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest

from experiments.v3_repair import run_tuning_contexts as runner
from experiments.v3_repair.prepare_inputs import digest, fingerprint, write_or_verify
from experiments.v3_repair.tuning_plan import coordinates
from src.api.config import settings


def save_unit(out, header, probe):
    (out / "units").mkdir(exist_ok=True)
    unit = dict(probe_id=probe["probe_id"], identity_sha256=header["identity_sha256"],
                source_probe_sha256=digest(probe), probe=probe, answer_quality="unmeasured")
    unit["unit_sha256"] = digest(unit)
    path = out / "units" / (digest(probe["probe_id"]) + ".json")
    write_or_verify(path, unit)
    return path


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root, plan, out = (tmp_path / name for name in ("development", "plan", "coordinate"))
    root.mkdir(); plan.mkdir()
    code = [fingerprint(runner.ROOT / "experiments/v3_repair/tuning_plan.py")]
    identity = dict(partition="tuning", preparation="sealed-inputs", code=code,
        source_database=__import__("sqlalchemy").engine.make_url(settings.database_url).database,
        settings_sha256=digest(settings.model_dump(mode="json")))
    config = dict(identity=identity, identity_sha256=digest(identity), database="owned")
    write_or_verify(root / "development-run.json", config)
    seal_id = dict(preparation=identity["preparation"], settings_sha256=identity["settings_sha256"], code=code)
    rows = dict(rows=coordinates(settings))
    seal = dict(identity=seal_id, identity_sha256=digest(seal_id), artifacts={"coordinates.json": digest(rows)})
    write_or_verify(plan / "manifest.json", seal); write_or_verify(plan / "coordinates.json", rows)
    probes = {"q1": dict(probe_id="q1", conversation="history", split_turn=2,
                         question="Which port?", expected_answer="Scorer-only value.")}
    monkeypatch.setattr(runner, "source_probes", lambda _p: (dict(trace_sha256="source-trace"), probes))
    return root, plan, out, config, probes


def test_complete_unit_resumes_byte_stably_but_altered_body_is_refused(sealed):
    root, plan, out, _, probes = sealed
    _, header, _, _ = runner.setup(root, plan, "baseline", out)
    path = save_unit(out, header, probes["q1"])
    before = {p.name: p.read_bytes() for p in (out / "run.json", path)}
    assert runner.setup(root, plan, "baseline", out)[1] == header
    assert not runner.pending_groups(out, probes)
    assert before == {p.name: p.read_bytes() for p in (out / "run.json", path)}
    unit = json.loads(path.read_text()); unit["probe"]["question"] = "Altered cached input"
    path.write_text(json.dumps(unit))
    with pytest.raises(ValueError, match="content"): runner.setup(root, plan, "baseline", out)


@pytest.mark.parametrize("fault", ["producer", "coordinate", "code", "settings", "validation", "source"])
def test_drift_refuses_before_any_model_or_database_call(sealed, monkeypatch, fault):
    root, plan, out, config, probes = sealed
    runner.setup(root, plan, "baseline", out)
    old = (out / "run.json").read_bytes()
    if fault in {"producer", "code", "validation"}:
        if fault == "producer": config["identity"]["preparation"] = "unsealed-change"
        elif fault == "code":
            config["identity"]["code"][0]["sha256"] = "changed"
            config["identity_sha256"] = digest(config["identity"])
        else:
            config["identity"]["partition"] = "validation"
            config["identity_sha256"] = digest(config["identity"])
        (root / "development-run.json").write_text(json.dumps(config))
    elif fault == "coordinate":
        rows = json.loads((plan / "coordinates.json").read_text())
        rows["rows"][0]["overrides"] = dict(retrieval_rrf_k=999)
        (plan / "coordinates.json").write_text(json.dumps(rows))
    elif fault == "settings": monkeypatch.setattr(settings, "retrieval_rrf_k", settings.retrieval_rrf_k + 1)
    else:
        header = json.loads((out / "run.json").read_text())
        save_unit(out, header, probes["q1"])
        probes["q1"]["question"] = "Changed reviewed question"
    with pytest.raises(ValueError): runner.setup(root, plan, "baseline", out)
    assert (out / "run.json").read_bytes() == old


def test_default_plan_never_provisions_or_launches(sealed, monkeypatch, capsys):
    from scripts.z1 import run_v3_campaign
    root, plan, out, config, probes = sealed
    monkeypatch.setattr(runner, "ROOT", root.parent)
    # The CLI private-path requirement is independently enforced; use its logs tree.
    logs = root.parent / "logs"; logs.mkdir()
    for path in (root, plan): path.rename(logs / path.name)
    root, plan, out = logs / root.name, logs / plan.name, logs / out.name
    monkeypatch.setattr(runner, "setup", lambda *_a, **_k: (config, {}, coordinates(settings)[0], probes))
    monkeypatch.setattr(run_v3_campaign, "database_environment", lambda *_a: pytest.fail("planning touched DB"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: pytest.fail("planning launched child"))
    monkeypatch.setattr(sys, "argv", ["run_tuning_contexts.py", "--development-run", str(root),
        "--tuning-plan", str(plan), "--coordinate", "baseline", "--out", str(out)])
    assert runner.main() == 0
    assert json.loads(capsys.readouterr().out)["cloud_calls"] is False


def test_current_baseline_normalizes_owned_database_without_accepting_other_drift(sealed, monkeypatch):
    from sqlalchemy.engine import make_url
    _, _, _, config, _ = sealed
    baseline = runner.resolved_baseline(config)
    monkeypatch.setattr(settings, "database_url", make_url(settings.database_url).set(
        database=config["database"]).render_as_string(hide_password=False))
    monkeypatch.setenv("ICE_TEST_DATABASE", config["database"])
    assert runner.resolved_baseline(config, owned=True) == baseline
    monkeypatch.setattr(settings, "retrieval_rrf_k", settings.retrieval_rrf_k + 1)
    with pytest.raises(ValueError, match="settings differ"): runner.resolved_baseline(config, owned=True)


@pytest.mark.parametrize("adopted", [False, True])
def test_native_prefix_collection_recovers_without_duplicate_units_or_writer_effects(
        tmp_path, monkeypatch, empty_replay_database, adopted):
    from sqlalchemy import text
    from src.api.db import SessionLocal
    from src.classifier import classifier as classifier_module
    from src.classifier.schemas import ClassificationResult
    from src.memory import embedder as embedding_module
    from src.memory.models import Conversation, EpisodicMemory
    from src.memory.source import chat_provenance
    from src.model_registry import registry
    from src.retrieval import ner_utils
    from scripts.z1 import production_parity as pp, replay_checkpoint as recovery, snapshot

    root = tmp_path / "development"; root.mkdir()
    out = tmp_path / "coordinate"; out.mkdir()
    coordinate = next(r for r in coordinates(settings) if r.get("input_prompt_cap") == 4000)
    identity = dict(coordinate=coordinate, source="controlled-native-snapshot-test")
    header = dict(identity=identity, identity_sha256=digest(identity), complete=False,
                  cloud_calls=False, score_of_record=False)
    config = dict(database=snapshot.DB, identity=dict(expected_turns={"history": 3}))
    producer_code = recovery.code_digest()
    if adopted:
        from experiments.v3_repair import reader_adoption

        monkeypatch.setattr(settings, "context_use_serving_window", False)
        monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
        code = [fingerprint(path) for path in recovery.code_files()[1]]
        for record in code:
            if record["path"] in reader_adoption.TRANSITIONS:
                old = reader_adoption.before_bytes(record["path"])
                record.update(sha256=hashlib.sha256(old).hexdigest(), bytes=len(old))
        config["identity"].update(partition="tuning", preparation="controlled-inputs", code=code,
            source_database=snapshot.DB, settings_sha256=digest(settings.model_dump(mode="json")))
        config["identity_sha256"] = digest(config["identity"])
        sealed = dict(identity=copy.deepcopy(config["identity"]), identity_sha256=config["identity_sha256"])
        producer_code = reader_adoption.code_transition((config["identity"], sealed["identity"]))[2]
        (root / "schema.sql").write_text("Controlled native fixture schema marker.\n")
        write_or_verify(root / "schema.json", {"fixture": "existing disposable ORM schema"})
    write_or_verify(root / "development-run.json", config)
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    meta = dict(inputs=[], extra=dict(resolved_background_model="controlled-writer", writer_model_digests={}))
    trace = root / "seed.jsonl"
    trace.write_text(json.dumps(dict(event="run_meta", meta=meta)) + "\n")
    bound = dict(selected_replay_plan=config["identity"], database=snapshot.DB,
        code_sha256=producer_code, tables=snapshot.TABLES, inputs=[],
        models="controlled-writer", writer_model_digests={})
    checkpoints = recovery.Checkpoints(root / "seed.recovery", trace, bound)
    cid = uuid.uuid4()
    times, index, source_ids = {}, {}, []
    try:
        with SessionLocal() as db:
            db.add(Conversation(id=cid)); db.commit()
            for number in (1, 2, 3):
                prompt = "Chosen port7319." if number == 1 else f"Ordinary recorded turn{number}."
                response = "Recorded reply."
                row = EpisodicMemory(conversation_id=cid, batch_id=uuid.uuid4(),
                    idempotency_key=f"controlled-prefix-{number}",
                    raw_text=f"User: {prompt}\n\nAssistant: {response}",
                    source_spans=chat_provenance(prompt, response), context_reliance="Zero_Shot",
                    timestamp=now + timedelta(minutes=number), lossless_flag=True,
                    embedding=[1.0] + [0.0] * (settings.embedding_dim - 1))
                db.add(row); db.commit()
                for key in (str(row.id), str(row.batch_id)):
                    times[key] = row.timestamp.isoformat(); index[key] = f"history:{number}"
                if number == 1: source_ids = [str(row.id), str(row.batch_id)]
                if number in (2, 3):
                    state = dict(completed={"history": number}, seen_source_times=copy.deepcopy(times),
                                 source_turn_by_id=copy.deepcopy(index), conversation_ids={"history": str(cid)})
                    with trace.open("a") as sink:
                        checkpoints.capture(sink, state, evaluation_key=f"history-{number}")
        original_rows = snapshot.fingerprints()
    finally: checkpoints.close()
    if adopted:
        with trace.open("a") as sink:
            sink.write(json.dumps(dict(event="complete", table_counts=snapshot.counts(),
                                       table_sha256=original_rows)) + "\n")
        profile = reader_adoption.build(root, config, sealed)
        header["identity"]["reader_adoption"] = profile
        header["identity_sha256"] = digest(header["identity"])
    write_or_verify(out / "run.json", header)

    encoder = NS(encode=lambda *_a, **_k: [1.0] + [0.0] * (settings.embedding_dim - 1))
    loads = []
    class Classifier:
        def __init__(self, **_kw): self.embedder = encoder; loads.append("classifier")
        def classify(self, prompt, conversation_id=None):
            assert str(conversation_id) == str(cid)
            return ClassificationResult([], [], "Zero_Shot", [0.0] * 27, 0.0, prompt=prompt)
    monkeypatch.setattr(classifier_module, "PyTorchClassifier", Classifier)
    monkeypatch.setattr(embedding_module, "get_embedder", lambda: encoder)
    monkeypatch.setattr(ner_utils, "extract_entities", lambda *_a, **_k: [])
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(pp, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(pp, "find_best_model", lambda *_a: ("controlled-answerer", None))
    probes = {key: dict(event="as_of_probe", probe_id=key, conversation="history", split_turn=cutoff,
        question=f"Which port at{cutoff}, query{key}?", question_time=(now + timedelta(minutes=cutoff)).isoformat(),
        gold_sources=[dict(turn=1, source_ids=source_ids, prompt="Chosen port7319.", response="Recorded reply.")],
        expected_answer="ScorerOnlyMustNotEnterPrompt", metric_contract={"qualified": False},
        preflight={}, controls={arm: {} for arm in ("no_codex", "vector_only", "recent_only")})
        for key, cutoff in (("q1", 2), ("q2", 2), ("q3", 3))}
    before_settings = settings.model_dump()
    built, restored = [], []
    real_build, real_restore = pp.build, snapshot.restore
    fail = {"active": True}
    def observed_build(db, question, *args, **kwargs):
        assert db.execute(text("SHOW transaction_read_only")).scalar() == "on"
        assert settings.retrieval_strengthen_writes is False
        assert kwargs["prompt_cap"] == 4000
        assert db.query(EpisodicMemory).count() == (3 if "at3" in question else 2)
        built.append(question)
        if "queryq2" in question and fail["active"]: raise KeyboardInterrupt()
        return real_build(db, question, *args, **kwargs)
    def observed_restore(arm):
        restored.append(snapshot.SNAPDIR.name)
        return real_restore(arm)
    monkeypatch.setattr(pp, "build", observed_build)
    monkeypatch.setattr(snapshot, "restore", observed_restore)
    with pytest.raises(KeyboardInterrupt): runner.collect(root, out, header, coordinate, probes, config)
    assert snapshot.fingerprints() == original_rows and settings.model_dump() == before_settings
    assert len(list((out / "units").glob("*.json"))) == 1
    assert not (out / "contexts.json").exists()
    first = next((out / "units").glob("*.json")).read_bytes()
    fail["active"] = False
    result = runner.collect(root, out, header, coordinate, probes, config)
    assert result["complete"] and len(result["records"]) == 3
    assert sum("queryq1" in q for q in built) == 1
    assert restored.count("history-2") == 2 and restored.count("history-3") == 1
    assert snapshot.fingerprints() == original_rows
    assert (out / "units" / (digest("q1") + ".json")).read_bytes() == first
    for unit in result["records"]:
        assert unit["probe"]["expected_answer"] == probes[unit["probe_id"]]["expected_answer"]
        for stage in (unit["probe"]["preflight"], *unit["probe"]["controls"].values()):
            assert stage["gate"]["input_prompt_cap"] == 4000
            assert stage["prompt_tokens"] * settings.token_count_safety_margin <= 4000
            assert stage["exposure_writes_enabled"] is False
            assert not stage["lineage"]["future_source_ids"]
            assert "ScorerOnlyMustNotEnterPrompt" not in json.dumps(stage["prompt_messages"])
    aggregate = (out / "contexts.json").read_bytes()
    from experiments.v3_repair.compare_tuning_contexts import load_contexts
    loaded_header, loaded = load_contexts(out, probes)
    assert loaded_header == header and set(loaded) == set(probes)
    assert loaded["q1"]["preflight"]["gate"]["input_prompt_cap"] == 4000
    runner.collect(root, out, header, coordinate, probes, config)
    assert len(loads) == 2 and (out / "contexts.json").read_bytes() == aggregate
    assert snapshot.fingerprints() == original_rows
    # Complete cache must still verify/restore final state; failure cannot report success.
    monkeypatch.setattr(snapshot, "restore", lambda _arm: 1)
    with pytest.raises(RuntimeError, match="final restore failed"):
        runner.collect(root, out, header, coordinate, probes, config)
    assert (out / "contexts.json").read_bytes() == aggregate
    monkeypatch.setattr(snapshot, "restore", observed_restore)
    # A forged prefix may not be queried even though its source bytes exist.
    directory = root / "seed.recovery/evaluation/history-2"
    receipt = json.loads((directory / "recovery.json").read_text())
    receipt["identity"]["models"] = "different-writer"
    (directory / "recovery.json").write_text(json.dumps(receipt))
    (out / "units" / (digest("q2") + ".json")).unlink()
    before_loads = len(loads)
    with pytest.raises(ValueError, match="producer execution|frozen prefix"):
        runner.collect(root, out, header, coordinate, probes, config)
    assert len(loads) == before_loads and snapshot.fingerprints() == original_rows
