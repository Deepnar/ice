"""Whole-history admission, scorer isolation and durable selected replay identity."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from experiments.v3_repair import run_development as dev
from experiments.v3_repair.prepare_inputs import digest
from scripts.z1.metric_contract import metric_contract
from src.api.config import settings


@pytest.fixture
def packet(tmp_path, monkeypatch):
    inputs, run = tmp_path / "inputs", tmp_path / "run"
    inputs.mkdir(); run.mkdir()
    groups, records = {}, []
    for slug, partition in [("tune0001", "tuning"), ("valid001", "validation")]:
        groups[slug] = [dict(turn_number=i, prompt=f"Original {i}.", response="Recorded reply.",
            timestamp=(datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat(),
            timestamp_provenance="original") for i in (1, 2)]
        review = dict(conversation=slug, cutoff_turn=2, probe_id=slug + "-q",
            question="Which value?", expected_answer="Secret scorer value.", reason="Reviewed exact cutoff.",
            reviewed_through_turn=2, recent_only_answerable=False,
            knowledge_scope="private_history", task_types=["episodic_lookup"])
        records.append(dict(review=review, partition=partition, reviewed_answer=review["expected_answer"],
                            gold_turns=[1], metric_contract=metric_contract(review, [1])))
    (inputs / "development-originals.json").write_text(json.dumps(groups))
    (inputs / "development-keys.json").write_text(json.dumps(dict(records=records)))
    manifest = dict(identity_sha256="sealed-fixture", artifacts={
        name: digest(json.loads((inputs / name).read_text())) for name in
        ("development-originals.json", "development-keys.json")})
    monkeypatch.setattr(dev, "prepare", lambda _p: manifest)
    monkeypatch.setattr(dev, "DEVELOPMENT", dict.fromkeys(groups, 2))
    monkeypatch.setattr(settings, "background_model_mode", "shared")
    monkeypatch.setattr(settings, "background_model_name", "fixture-writer")
    monkeypatch.setattr(settings, "codex_extraction_model", "fixture-extractor")
    monkeypatch.setattr(settings, "codex_extraction_ner_tier", "background")
    return inputs, run, records


def test_selected_partition_preserves_keys_without_feeding_them_to_runtime(packet):
    inputs, _, _ = packet
    _, conversations, probes, keys = dev.load_selected(inputs, "tuning")
    assert set(conversations) == {"tune0001"} and len(keys) == 1
    assert all("Secret scorer value." not in r["prompt"] + r["response"]
               for r in conversations["tune0001"])
    probe = probes["tune0001", 2][0]
    assert probe["question"] == "Which value?" and probe["expected_answer"] == "Secret scorer value."
    assert probe["label_review"]["reviewed_through_turn"] == 2
    assert probe["metric_contract"]["lookup_rank_tuning_eligible"]


@pytest.mark.parametrize("fault", ["future", "duplicate", "overlap", "metric"])
def test_source_and_partition_faults_refuse_before_execution(packet, fault):
    inputs, _, records = packet
    if fault == "future": records[0]["gold_turns"] = [3]
    elif fault == "duplicate": records.append(records[0])
    elif fault == "overlap": records[1]["review"]["conversation"] = "tune0001"
    else: records[0]["metric_contract"]["lookup_rank_tuning_eligible"] = False
    (inputs / "development-keys.json").write_text(json.dumps(dict(records=records)))
    with pytest.raises(ValueError): dev.load_selected(inputs, "tuning")


def test_plan_resumes_byte_stably_and_refuses_changed_policy(packet):
    inputs, run, _ = packet
    first, _, _ = dev.planned_run(inputs, run, "tuning", 10)
    before = {p.name: p.read_bytes() for p in run.glob("*.json")}
    assert dev.planned_run(inputs, run, "tuning", 10)[0] == first
    assert before == {p.name: p.read_bytes() for p in run.glob("*.json")}
    with pytest.raises(ValueError, match="identity changed"):
        dev.planned_run(inputs, run, "tuning", 5)
    assert before == {p.name: p.read_bytes() for p in run.glob("*.json")}


def test_owned_child_matches_parent_settings_without_accepting_other_drift(packet, monkeypatch):
    from sqlalchemy.engine import make_url
    inputs, run, _ = packet
    parent = dev.planned_run(inputs, run, "tuning", 10)[0]
    monkeypatch.setattr(settings, "database_url", make_url(settings.database_url).set(
        database=parent["database"]).render_as_string(hide_password=False))
    monkeypatch.setenv("ICE_TEST_DATABASE", parent["database"])
    assert dev.planned_run(inputs, run, "tuning", 10, owned=True)[0] == parent
    monkeypatch.setattr(settings, "retrieval_rrf_k", settings.retrieval_rrf_k + 1)
    with pytest.raises(ValueError, match="identity changed"):
        dev.planned_run(inputs, run, "tuning", 10, owned=True)


def test_recovery_requires_trace_and_checkpoint_together(tmp_path):
    assert not dev.replay_args(tmp_path, 10).resume
    (tmp_path / "seed.jsonl").write_text("unfinished")
    with pytest.raises(ValueError, match="pair is incomplete"):
        dev.replay_args(tmp_path, 10)
    point = tmp_path / "seed.recovery"
    point.mkdir(); (point / "current.json").write_text("{}")
    assert dev.replay_args(tmp_path, 10).resume


@pytest.mark.parametrize("run,partition", [(False, "tuning"), (True, "validation")])
def test_no_call_plan_and_reserved_validation_never_provision_or_launch(tmp_path, monkeypatch, run, partition):
    import sys
    from scripts.z1 import run_v3_campaign
    root = tmp_path / "logs"; root.mkdir()
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    monkeypatch.setattr(dev, "planned_run", lambda *_a, **_k: (
        dict(database="owned", validation_execution="locked"), {"history": [1, 2]}, {("history", 2): [1]}))
    monkeypatch.setattr(run_v3_campaign, "database_environment", lambda *_a: pytest.fail("plan created DB"))
    monkeypatch.setattr(dev.subprocess, "run", lambda *_a, **_k: pytest.fail("plan launched a model child"))
    monkeypatch.setattr(sys, "argv", ["run_development.py", "--inputs", str(root / "inputs"),
        "--run-dir", str(root / "run"), "--partition", partition] + (["--run"] if run else []))
    if run:
        with pytest.raises(ValueError, match="reserved"): dev.main()
    else:
        assert dev.main() == 0


def test_selected_plan_receipt_reaches_checkpoint_identity(monkeypatch):
    from types import SimpleNamespace
    from scripts.z1.replay_checkpoint import run_identity
    meta = dict(inputs=[], extra=dict(resolved_background_model="writer", planned_probes=[],
                                     selected_replay_plan=dict(expected_turns={"dev": 2}, keys="bound")))
    identity = run_identity(meta, SimpleNamespace(resume=True), settings)
    assert identity["selected_replay_plan"] == meta["extra"]["selected_replay_plan"]
    saved = json.loads(json.dumps(identity))
    meta["extra"]["selected_replay_plan"]["keys"] = "different"
    changed = run_identity(meta, SimpleNamespace(resume=True), settings)
    assert saved != changed


@pytest.fixture
def empty_replay_database(monkeypatch):
    """Replay needs a genuinely empty store, even after other smoke fixtures."""
    import os
    import uuid
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from sqlalchemy.orm import sessionmaker
    from scripts.z1 import seed_v3, snapshot
    from src.api import db as database
    from src.memory import embedder as embedding_module
    from src.memory.models import Base
    from types import SimpleNamespace as NS
    controlled = NS(encode=lambda *_a, **_k: [1.0] + [0.0] * (settings.embedding_dim - 1))
    monkeypatch.setattr(embedding_module, "get_embedder", lambda: controlled)
    from src.workers import post_flight
    source = make_url(settings.database_url)
    assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
            and source.database == os.environ["ICE_TEST_DATABASE"])
    name = "ice_test_" + uuid.uuid4().hex
    admin = create_engine(source.set(database="postgres"), isolation_level="AUTOCOMMIT")
    engine = None
    created = False
    try:
        with admin.connect() as conn: conn.execute(text(f'CREATE DATABASE "{name}"'))
        created = True
        target = source.set(database=name)
        engine = create_engine(target)
        with engine.begin() as conn: conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        Base.metadata.create_all(engine)
        sessions = sessionmaker(bind=engine)
        for module in (database, seed_v3, post_flight):
            monkeypatch.setattr(module, "SessionLocal", sessions)
        monkeypatch.setattr(settings, "database_url", target.render_as_string(hide_password=False))
        monkeypatch.setattr(snapshot, "DB", name)
        monkeypatch.setenv("ICE_TEST_DATABASE", name)
        yield
    finally:
        if engine is not None: engine.dispose()
        if created:
            with admin.connect() as conn: conn.execute(text(f'DROP DATABASE "{name}" WITH (FORCE)'))
        admin.dispose()


def test_shared_engine_real_sql_recovers_sources_and_measures_content_loss(tmp_path, monkeypatch, empty_replay_database):
    """Actual preamble/assembly/store/post-flight/checkpoints; controlled models/jobs.

    Periodic job callbacks and chained graph/procedure extraction are controlled
    here. Their semantic quality is not measured by this adapter test.
    """
    import os
    from types import SimpleNamespace as NS
    from sqlalchemy import text
    from sqlalchemy.engine import make_url
    from scripts.z1 import seed_v3, replay_checkpoint as recovery
    from src.api.db import SessionLocal
    from src.classifier import classifier as classifier_module
    from src.classifier.schemas import ClassificationResult
    from src.memory import embedder as embedding_module
    from src.model_registry import registry
    from src.retrieval import ner_utils
    from src.workers import bg_client_factory, post_flight
    from src.workers.runtime import JOBS

    assert (os.environ.get("ICE_TEST_DATABASE", "").startswith("ice_test_")
            and make_url(settings.database_url).database == os.environ["ICE_TEST_DATABASE"])
    with SessionLocal() as db:
        assert all(db.execute(text(f'SELECT count(*) FROM "{n}"')).scalar() == 0
                   for n in seed_v3.Base.metadata.tables), "fixture must start isolated and empty"
    encoder = NS(encode=lambda *_a, **_k: [1.0] + [0.0] * (settings.embedding_dim - 1))
    class Classifier:
        def __init__(self, **_kw): self.embedder = encoder
        def classify(self, prompt, conversation_id=None):
            assert conversation_id
            return ClassificationResult([], [], "Zero_Shot", [0.0] * 27, 0.0, prompt=prompt)
    monkeypatch.setattr(classifier_module, "PyTorchClassifier", Classifier)
    monkeypatch.setattr(embedding_module, "get_embedder", lambda: encoder)
    monkeypatch.setattr(ner_utils, "extract_entities", lambda *_a, **_k: [])
    monkeypatch.setattr(settings, "memory_source_gate_enabled", False)
    monkeypatch.setattr(settings, "context_use_serving_window", False)
    monkeypatch.setattr(registry, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(seed_v3.pp, "find_best_model", lambda *_a: ("controlled-model", None))
    monkeypatch.setattr(seed_v3.pp, "get_model_context_window", lambda *_a: 8192)
    monkeypatch.setattr(bg_client_factory, "get_bg_model_name", lambda: "controlled-writer")
    monkeypatch.setattr(recovery, "background_model_digests", lambda *_a: {"controlled-writer": "fixture-digest"})
    monkeypatch.setattr(post_flight, "extract_codex", lambda **_k: None)
    monkeypatch.setattr(post_flight, "extract_procedural", lambda **_k: None)
    functions, job_calls = {}, []
    for name in seed_v3.MEMORY_JOBS:
        module, function = JOBS[name].path.split(":")
        def called(*_a, _name=name, **_k): job_calls.append(_name)
        functions.setdefault(module, {})[function] = called
    monkeypatch.setattr(seed_v3, "importlib", NS(import_module=lambda n: NS(**functions[n])))
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    conversations = {"fixture": [dict(turn_number=i, timestamp=now + timedelta(minutes=i),
        ts_provenance="original", prompt="Chosen port7319." if i == 1 else "Keep that configuration.",
        response="Recorded existing reply.") for i in (1, 2)]}
    probes = {("fixture", 2): [dict(probe_id="q", probe_type="episodic_lookup",
        question="Which port?", expected_answer="7319", gold_turns=[1], source_split_turn=1,
        cutoff_kind="reviewed", label_status="source_reviewed_valid") ]}
    root = tmp_path / "replay"; root.mkdir()
    monkeypatch.setattr(seed_v3, "private_output", lambda p: __import__("pathlib").Path(p))
    source = root / "input.json"; source.write_text('{"fixture":true}')
    receipt = dict(policy="controlled-short-adapter", expected_turns={"fixture": 2})
    review = dict(probe_id="q", conversation="fixture", cutoff_turn=2, question="Which port?",
        expected_answer="7319", reviewed_through_turn=2, recent_only_answerable=False,
        knowledge_scope="private_history", task_types=["episodic_lookup"])
    contract = metric_contract(review, [1])
    probes["fixture", 2][0].update(label_review=review, metric_contract=contract)
    (root / "development-run.json").write_text(json.dumps(dict(identity=receipt, identity_sha256=digest(receipt))))
    (root / "reviewed-keys.json").write_text(json.dumps(dict(records=[dict(review=review,
        reviewed_answer="7319", gold_turns=[1], metric_contract=contract)])))
    bound_files = [source, root / "development-run.json", root / "reviewed-keys.json"]
    # This also exercises a planned cap through the actual preparation path.
    from experiments.v3_repair.tuning_plan import coordinates
    coordinate = next(r for r in coordinates(settings) if r["overrides"].get("context_budget_max") == 4000)
    for name, value in coordinate["overrides"].items(): monkeypatch.setattr(settings, name, value)
    original = seed_v3.pp.build
    def capped(db, question, *a, **kw):
        return original(db, question, *a, **kw, prompt_cap=coordinate["input_prompt_cap"])
    def interrupted(db, question, *a, **kw):
        if question == "Keep that configuration.": raise KeyboardInterrupt()
        return capped(db, question, *a, **kw)
    monkeypatch.setattr(seed_v3.pp, "build", interrupted)
    with pytest.raises(KeyboardInterrupt):
        seed_v3.run(dev.replay_args(root, 1), conversations, probes,
                    expected_turns={"fixture": 2}, input_files=bound_files, selected_plan=receipt)
    point = json.loads((root / "seed.recovery/current.json").read_text())
    saved = json.loads((root / "seed.recovery" / point["generation"] / "recovery.json").read_text())
    assert saved["state"]["completed"] == {"fixture": 1}
    monkeypatch.setattr(seed_v3.pp, "build", capped)
    assert seed_v3.run(dev.replay_args(root, 1), conversations, probes,
        expected_turns={"fixture": 2}, input_files=bound_files, selected_plan=receipt) == 0
    rows = [json.loads(s) for s in (root / "seed.jsonl").read_text().splitlines()]
    assert [r["turn"] for r in rows if r["event"] == "written"] == [1, 2]
    assert len(set(job_calls)) == 10
    probe = next(r for r in rows if r["event"] == "as_of_probe")
    assert "Chosen port7319." in probe["gold_sources"][0]["prompt"]
    assert "7319" not in json.dumps(probe["preflight"]["prompt_messages"])
    from src.api.context_ledger import effective_memory_budget
    assert probe["preflight"]["gate"]["total_budget"] == effective_memory_budget(4000, "Which port?",
        generation_reserve=settings.context_generation_reserve, floor=settings.context_budget_floor)
    assert probe["preflight"]["gate"]["input_prompt_cap"] == 4000
    assert probe["preflight"]["prompt_tokens"] * settings.token_count_safety_margin <= 4000
    from scripts.z1.replay_validation import validate_complete_replay
    assert validate_complete_replay(iter(rows), {"fixture": 2})
    committed = (root / "seed.jsonl").read_bytes()
    assert seed_v3.run(dev.replay_args(root, 1), conversations, probes,
        expected_turns={"fixture": 2}, input_files=bound_files, selected_plan=receipt) == 0
    assert committed == (root / "seed.jsonl").read_bytes()
    from experiments.v3_repair.audit_development import audit
    report, packet = audit(root)
    assert report["metric_cohorts"]["lookup_rank_tuning_eligible"] == 1
    assert report["semantic_review"]["qualified_occurrences"] == 0
    assert packet["records"][0]["requirements"] == [] and packet["approved_units"] == 0
    from scripts.z1.evidence_content import audit_probe, sha
    original_source = probe["gold_sources"][0]["prompt"]
    start = original_source.index("7319")
    requirement = dict(id="port", kind="exact_literal", alternatives=[dict(turn=1, role="user",
        start=start, end=start + 4, source_sha256=sha(original_source), literal="7319")])
    inspected = audit_probe(probe, dict(reviewed_through_turn=2,
        reason="Controlled exact-value source; smaller-budget final prompt omits it.", requirements=[requirement]))
    assert inspected["requirements"][0]["stages"]["final_input"] == dict(
        literal="missing", semantic_support="not_supported",
        content_sha256=inspected["requirements"][0]["stages"]["final_input"]["content_sha256"])
    changed = json.loads((root / "reviewed-keys.json").read_text())
    changed["records"][0]["reviewed_answer"] = "Modified scorer key"
    (root / "reviewed-keys.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="bytes changed"): audit(root)
