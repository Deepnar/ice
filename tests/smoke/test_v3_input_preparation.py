"""Private input provenance, data separation and crash-safe offline preparation."""

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from experiments.v3_repair import prepare_inputs as prep
from experiments.v3_repair.settings_inventory import build_inventory, candidate_reads, relevant


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def oracle_rows():
    return [dict(question_id=f"q{i}", question="Which port?", answer="7319",
                 question_date="2025/01/02", answer_session_ids=["s1"],
                 haystack_session_ids=["s1"], haystack_dates=["2025/01/01"],
                 haystack_sessions=[[dict(role="user", content="Port7319.")]]) for i in range(500)]


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path
    dev = {"devone01": 2, "devtwo02": 2}
    final = {
        "B": dict(prefix="final001", turns=2, families=1, occurrences=2,
                  checkpoints=2, archives=["b.json"]),
        "C": dict(prefix="final002", turns=2, families=1, occurrences=2,
                  checkpoints=2, archives=["c.json"]),
    }
    monkeypatch.setattr(prep, "DEVELOPMENT", dev)
    monkeypatch.setattr(prep, "FINAL", final)
    rows = [dict(conversation_id=prefix + "-fixture", prompt=f"{prefix} original {i}",
                 response="Recorded reply.", timestamp=f"2025-01-0{i}T00:00:00Z")
            for prefix in list(dev) + [d["prefix"] for d in final.values()] for i in (1, 2)]
    corpus = root / "data/simulation/simulation_full.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("\n".join(json.dumps(row) for row in rows))
    review_rows = [dict(conversation=prefix, probe_id=f"{prefix}-question", cutoff_turn=2,
                        expected_answer="Recorded value.", answer_verdict="valid",
                        reviewed_gold_turns=[1], reviewed_through_turn=2,
                        recent_only_answerable=False, knowledge_scope="private_history",
                        task_types=["episodic_lookup"]) for prefix in dev]
    binding = prep.fingerprint(corpus, root)
    binding["sha256"] = binding["sha256"][:16]
    put(root, "logs/z1-v3-manual-2026-10-04-r4/source-reviewed-labels-2026-10-03-r3.json",
        dict(native=dict(inputs=[binding], records=review_rows),
             source_linked=dict(inputs=[binding], records=[])))
    records = [dict(conversation_id=d["prefix"] + "-fixture", probe_id=label + "-p1",
                    turn_index=i, question="What was recorded?", ground_truth="Recorded value.")
               for label, d in final.items() for i in (1, 2)]
    put(root, "experiments/mature/intermediates/master_results.json",
        dict(evaluation_run_results=records))
    for label, d in final.items():
        originals = [row for row in rows if row["conversation_id"].startswith(d["prefix"])]
        put(root, "experiments/curation_files/" + d["archives"][0],
            dict(historical_context_block=[dict(turn_number=i, user_input=row["prompt"],
                                                ai_response=row["response"])
                                           for i, row in enumerate(originals, 1)],
                 future_reference_block=[]))
    put(root, "experiments/lme/data/longmemeval_oracle", oracle_rows())
    for name in ("src/module.py", "experiments/v3_repair/prepare_inputs.py",
                 "experiments/v3_repair/settings_inventory.py", "scripts/z1/label_review.py",
                 "scripts/z1/metric_contract.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# controlled code identity\n")
    monkeypatch.setattr(prep, "build_inventory", lambda *_a: dict(method="controlled", settings=[]))
    return root, root / "logs/plan", corpus


def test_full_preparation_separates_data_and_resumes_byte_stably(inputs):
    root, out, _ = inputs
    first = prep.prepare(out, root=root)
    assert first["development_original_pairs"] == 4 and first["final_original_pairs"] == 4
    assert first["oracle_questions"] == 500 and not first["score_ready"] and not first["replay_ready"]
    dev = json.loads((out / "development-originals.json").read_text())
    assert set(dev) == {"devone01", "devtwo02"}
    keys = json.loads((out / "development-keys.json").read_text())
    assert {row["partition"] for row in keys["records"]} == {"tuning", "validation"}
    references = json.loads((out / "final-legacy-references.json").read_text())
    assert all(row["status"] == "legacy_unqualified" and row["reviewed_gold_turns"] is None
               for row in references["records"])
    before = {p.name: p.read_bytes() for p in out.glob("*.json")}
    assert prep.prepare(out, root=root) == first
    assert before == {p.name: p.read_bytes() for p in out.glob("*.json")}


def test_interruption_before_seal_recovers_existing_artifacts(inputs, monkeypatch):
    root, out, _ = inputs
    original = prep.write_or_verify

    def interrupt(path, value):
        if path.name == "settings-candidates.json":
            raise KeyboardInterrupt()
        original(path, value)

    monkeypatch.setattr(prep, "write_or_verify", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prep.prepare(out, root=root)
    assert not (out / "manifest.json").exists()
    existing = {p.name: p.read_bytes() for p in out.glob("*.json")}
    monkeypatch.setattr(prep, "write_or_verify", original)
    prep.prepare(out, root=root)
    assert all((out / name).read_bytes() == body for name, body in existing.items())


def test_preparation_preserves_synthesis_but_excludes_it_from_rank_tuning(inputs):
    root, out, _ = inputs
    path = root / 'logs/z1-v3-manual-2026-10-04-r4/source-reviewed-labels-2026-10-03-r3.json'
    review = json.loads(path.read_text())
    review['native']['records'][0]['task_types'] = ['summary_synthesis']
    path.write_text(json.dumps(review))
    prep.prepare(out, root=root)
    keys = json.loads((out / 'development-keys.json').read_text())
    assert len(keys['records']) == 2
    synthesis = next(r for r in keys['records'] if 'summary_synthesis' in r['review']['task_types'])
    lookup = next(r for r in keys['records'] if 'episodic_lookup' in r['review']['task_types'])
    assert not synthesis['metric_contract']['lookup_rank_tuning_eligible']
    assert lookup['metric_contract']['lookup_rank_tuning_eligible']
    assert sum(s['lookup_rank_tuning_eligible'] for s in keys['metric_cohorts'].values()) == 1


@pytest.mark.parametrize('field', ['background_model_name', 'codex_extraction_model',
                                  'source_support_revision'])
def test_actual_preparation_refuses_declared_model_drift_without_overwrites(inputs, monkeypatch, field):
    from src.api.config import settings

    root, out, _ = inputs
    prep.prepare(out, root=root)
    before = {p.name: p.read_bytes() for p in out.glob('*.json')}
    monkeypatch.setattr(settings, field, 'different-declared-audit-identity')
    with pytest.raises(ValueError, match='inputs/code/settings changed'):
        prep.prepare(out, root=root)
    assert before == {p.name: p.read_bytes() for p in out.glob('*.json')}


def test_model_profile_binds_declared_roles_without_credentials_or_weight_claims():
    values = {name: 'controlled-model-or-path' for name in prep.LOCAL_MODEL_FIELDS}
    values.update(background_model_mode='shared', background_model_name=None)
    settings = NS(**values, probe_api_key='SECRET_CANARY_NOT_IN_PROFILE')
    profile = prep.local_model_profile(settings)
    assert not profile['research_writer_pins_resolved']
    assert 'SECRET_CANARY' not in json.dumps(profile)
    assert profile['downloaded_weight_identity'] == profile['runtime_capacity'] == 'unqualified'
    settings.background_model_name = 'declared-background-pin'
    assert prep.local_model_profile(settings)['research_writer_pins_resolved']


def test_stale_source_review_fails_before_writing_products(inputs):
    root, out, corpus = inputs
    corpus.write_text(corpus.read_text().replace("Recorded reply.", "Changed reply."))
    with pytest.raises(ValueError, match="review source/catalog bytes changed"):
        prep.prepare(out, root=root)
    assert not list(out.glob("*.json"))


@pytest.mark.parametrize("change", ["code", "corrupt"])
def test_sealed_drift_or_corruption_never_overwrites(inputs, change):
    root, out, _ = inputs
    prep.prepare(out, root=root)
    before = {p.name: p.read_bytes() for p in out.glob("*.json")}
    if change == "code":
        (root / "src/module.py").write_text("# changed implementation\n")
    else:
        (out / "development-keys.json").write_text("{}")
    expected = {p.name: p.read_bytes() for p in out.glob("*.json")}
    with pytest.raises(ValueError, match="changed|mismatch"):
        prep.prepare(out, root=root)
    assert expected == {p.name: p.read_bytes() for p in out.glob("*.json")}
    assert (out / "manifest.json").read_bytes() == before["manifest.json"]


def test_archive_endpoint_and_disagreement_are_visible(tmp_path):
    original = [dict(prompt="Source.", response="Reply.")]
    path = put(tmp_path, "archive.json", dict(historical_context_block=[
        dict(turn_number=1, user_input="Different source.", ai_response="Reply."),
        dict(turn_number=2, user_input="Future.", ai_response="Future.")], future_reference_block=[]))
    report = prep.archive_coverage(original, [path])
    assert report["exact_matching_pairs"] == 0 and report["excluded_after_endpoint"] == 1
    assert len(report["mismatches"]) == 1 and not report["historical_full_byte_identity_confirmed"]


@pytest.mark.parametrize("fault", ["duplicate", "future_source", "clock_shape", "missing_role"])
def test_oracle_requires_bound_original_source_identities(tmp_path, fault):
    rows = oracle_rows()
    if fault == "duplicate":
        rows[-1]["question_id"] = rows[0]["question_id"]
    elif fault == "future_source":
        rows[0]["answer_session_ids"] = ["missing-session"]
    elif fault == "clock_shape":
        rows[0]["haystack_dates"] = []
    else:
        rows[0]["haystack_sessions"][0][0]["role"] = "unknown"
    path = put(tmp_path, "oracle.json", rows)
    with pytest.raises(ValueError, match="invalid oracle"):
        prep.oracle_keys(path)


def test_official_integer_answers_remain_numbers(tmp_path):
    rows = oracle_rows()
    rows[60]["answer"] = 7319
    keys = prep.oracle_keys(put(tmp_path, "oracle.json", rows))
    assert keys["questions"][60]["official_answer"] == 7319
    assert type(keys["questions"][60]["official_answer"]) is int


def test_static_alias_parameter_and_dynamic_reads_are_unqualified_candidates():
    source = '''from src.api.config import settings as cfg
import src.api.config as configuration
x = cfg.model_copy()
x.retrieval_rrf_k = 12
a = x.retrieval_rrf_k
b = getattr(cfg, "context_budget_max")
c = configuration.settings.retrieval_cluster_top_k
def f(settings, name):
    local = settings or cfg
    return local.ltm_prior_bias, getattr(settings, name)
'''
    reads, dynamic = candidate_reads(source, "fixture.py")
    assert set(reads) == {"retrieval_rrf_k", "context_budget_max", "retrieval_cluster_top_k", "ltm_prior_bias"}
    assert len(reads["retrieval_rrf_k"]) == 1 and len(dynamic) == 1
    assert not relevant("retrieval_api_key", 1, 2)
    assert relevant("retrieval_bonus_keyword_match", .5, 1.)
    assert not relevant("context_budget_max", float("inf"), 32)


def test_inventory_does_not_convert_static_presence_to_live_verdict(tmp_path):
    path = tmp_path / "src/consumer.py"
    path.parent.mkdir()
    path.write_text("from src.api.config import settings\nx = settings.retrieval_rrf_k\n")
    fields = {name: NS(get_default=lambda value=value: value) for name, value in
              [("retrieval_rrf_k", 60), ("ltm_prior_bias", .4), ("retrieval_api_key", "private")]}
    report = build_inventory(tmp_path, NS(model_fields=fields),
                             NS(retrieval_rrf_k=30, ltm_prior_bias=.5, retrieval_api_key="private"))
    rows = {row["name"]: row for row in report["settings"]}
    assert set(rows) == {"retrieval_rrf_k", "ltm_prior_bias"}
    assert rows["retrieval_rrf_k"]["code_default"] == 60 and rows["retrieval_rrf_k"]["resolved_value"] == 30
    assert rows["ltm_prior_bias"]["static_status"] == "unresolved_no_static_read"
    assert all(row["tuning_verdict"] == "unmeasured" and row["live_decision_delta"] is None
               for row in rows.values())


def test_named_budget_and_gate_knobs_change_actual_production_decisions():
    from src.api.config import settings
    from src.api.memory_decision import derive_total_budget, decide_memory_retrieval
    from src.classifier.schemas import ClassificationResult

    baseline = settings.model_copy(update=dict(context_budget_min=0, context_budget_max=30000,
                                               context_generation_reserve=1000))
    low = baseline.model_copy(update=dict(context_input_fraction=.25))
    high = baseline.model_copy(update=dict(context_input_fraction=.75))
    assert derive_total_budget(16384, low) < derive_total_budget(16384, high)
    result = ClassificationResult([], [], "Zero_Shot", [0.] * 27, .99, p_ltm=.5)
    low = settings.model_copy(update=dict(ltm_prior_bias=-5.))
    high = settings.model_copy(update=dict(ltm_prior_bias=5.))
    assert not decide_memory_retrieval(result, turn_count=1, total_tokens=0,
                                      recent_window_tokens=0, settings=low).retrieve
    assert decide_memory_retrieval(result, turn_count=1, total_tokens=0,
                                  recent_window_tokens=0, settings=high).retrieve
