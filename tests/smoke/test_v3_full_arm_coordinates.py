"""Full-arm tuning coordinates keep the four-arm baseline's shared identity.

Controlled bundles only: no model, database restore or cloud call. The actual
production-path check is the repeated `baseline` coordinate compared against
the completed four-arm baseline.
"""

import copy

import pytest

from experiments.v3_repair import compare_tuning_contexts as matched
from experiments.v3_repair import run_full_arm_coordinates as full
from experiments.v3_repair import run_tuning_contexts as runner
from experiments.v3_repair.prepare_inputs import digest, write_or_verify
from test_v3_tuning_contexts import sealed  # noqa: F401 — shared fixture


def save(out, header, original, current):
    (out / "units").mkdir(parents=True, exist_ok=True)
    unit = dict(probe_id=original["probe_id"], identity_sha256=header["identity_sha256"],
                source_probe_sha256=digest(original), probe=current, answer_quality="unmeasured")
    unit["unit_sha256"] = digest(unit)
    path = out / "units" / (digest(original["probe_id"]) + ".json")
    write_or_verify(path, unit)
    return path


def bundle(directory, header, probes, stages, *, controls):
    records = []
    for key, original in sorted(probes.items()):
        current = dict(original, preflight=stages[key],
                       controls={arm: stages[key] for arm in matched.ARMS[1:]} if controls else {})
        save(directory, header, original, current)
        records.append(runner.validate_unit(directory / "units" / (digest(key) + ".json"), header, probes))
    write_or_verify(directory / "run.json", header)
    result = dict(**header, records=records); result["complete"] = True
    write_or_verify(directory / "contexts.json", result)


def stage(text, tokens, source="row1"):
    fragment = dict(text=text, source_row=source, origin_batches=[])
    return dict(gate=dict(b2_retrieve=True, final_retrieve=True, source_action="retrieved",
                          routed_model="fixed-answerer", total_budget=4000),
        selected=[fragment], prompt_messages=[dict(role="system", content=text)],
        prompt_tokens=tokens, selected_tokens=tokens - 1,
        gold_fragment_coverage=dict(source_turn_rank_at_5=1, source_turn_rank_at_10=1))


def test_identity_extends_the_unchanged_collector(sealed):
    root, plan, out, _, _ = sealed
    _, four, _, _ = runner.setup(root, plan, "baseline", out)
    _, one, _, _ = full.setup(root, plan, "baseline", out.parent / "full", reader_adoption=None)
    assert {k: one["identity"][k] for k in four["identity"]} == four["identity"]
    assert one["identity"]["arms"] == ["full"] and one["identity"]["collector_policy"] == full.POLICY
    assert one["identity"]["collector"]["path"] == "experiments/v3_repair/run_full_arm_coordinates.py"
    assert one["identity_sha256"] != four["identity_sha256"]
    # Planning again resumes byte-stably; the scratch collector header never lands in `out`.
    assert full.setup(root, plan, "baseline", out.parent / "full", reader_adoption=None)[1] == one
    assert sorted(p.name for p in (out.parent / "full").iterdir()) == ["run.json"]


def test_full_arm_units_refuse_control_arms(sealed):
    root, plan, out, _, probes = sealed
    _, header, _, _ = full.setup(root, plan, "baseline", out, reader_adoption=None)
    original = probes["q1"]
    path = save(out, header, original, dict(original, preflight=stage("x", 3), controls={}))
    assert full.validate_unit(path, header, probes)["probe"]["controls"] == {}
    path.unlink()
    path = save(out, header, original, dict(original, preflight=stage("x", 3), controls={"recent_only": stage("x", 3)}))
    with pytest.raises(ValueError, match="control arms"):
        full.validate_unit(path, header, probes)


def test_collect_builds_only_the_full_arm_then_restores_the_arm_list(sealed, monkeypatch):
    from scripts.z1 import seed_v3
    root, plan, out, config, probes = sealed
    _, header, coordinate, _ = full.setup(root, plan, "baseline", out, reader_adoption=None)
    seen = []

    def controlled_collect(root_, out_, header_, coordinate_, probes_, config_):
        seen.append(seed_v3.PROMPT_ARMS)
        with pytest.raises(ValueError, match="unknown prompt arm"):
            seed_v3.prepare_with_trace(None, {}, None, arm="no_codex")
        bundle(out_, header_, probes_, {k: stage("x", 3) for k in probes_}, controls=False)

    monkeypatch.setattr(runner, "collect", controlled_collect)
    _, loaded = full.collect(root, out, header, coordinate, probes, config)
    assert seen == [("full",)] and loaded["q1"]["controls"] == {}
    assert seed_v3.PROMPT_ARMS == ("full", "no_codex", "vector_only", "recent_only")


@pytest.fixture
def collections(tmp_path, monkeypatch):
    root, baseline, candidate = (tmp_path / name for name in ("producer", "baseline", "candidate"))
    root.mkdir(); (root / "seed.jsonl").write_text("controlled-original-trace\n")
    # Producer probes carry their replay-time contexts, as the real trace does.
    recorded = stage("Chosen port 7319.", 30)
    probes = {key: dict(probe_id=key, split_turn=64, question="Which port?", expected_answer="7319",
                        metric_contract=dict(lookup_rank_tuning_eligible=key == "q1"),
                        preflight=recorded, controls={arm: recorded for arm in matched.ARMS[1:]})
              for key in ("q1", "q2")}
    monkeypatch.setattr(runner, "source_probes", lambda _root: (dict(producer_identity="writer", trace_sha256="trace"), probes))
    monkeypatch.setattr(matched, "source_probes", runner.source_probes)
    shared = dict(producer="writer", producer_trace_sha256="trace", coordinate_seal="same-seal",
                  code=[dict(path="src/reader.py", sha256="a", bytes=1)], probes=sorted(probes),
                  settings_sha256="same-baseline", reader_adoption=dict(profile="same"))

    def header(coordinate, **extra):
        identity = dict(shared, coordinate=dict(id=coordinate), **extra)
        return dict(identity=identity, identity_sha256=digest(identity), complete=False, score_of_record=False)

    same = {key: stage("Chosen port 7319.", 30) for key in probes}
    bundle(baseline, header("baseline"), probes, same, controls=True)

    def make(name, coordinate, stages, **identity):
        extra = dict(arms=["full"], collector_policy=full.POLICY, collector=dict(path="c", sha256="c", bytes=1))
        extra.update(identity)
        directory = tmp_path / name; directory.mkdir()
        bundle(directory, header(coordinate, **extra), probes, stages, controls=False)
        return directory

    return root, baseline, make, same


def test_repeated_baseline_is_a_zero_change_floor(collections):
    root, baseline, make, same = collections
    repeat = make("repeat", "baseline", copy.deepcopy(same))
    report = full.compare(root, baseline, repeat)
    assert {row["arm"] for row in report["records"]} == {"full"}
    assert report["totals"]["occurrences"] == 2
    assert report["totals"]["rendered_inputs_changed"] == report["totals"]["selected_sources_changed"] == 0
    assert report["promotion"] == "not_performed" and report["score_of_record"] is False


def test_changed_full_context_is_counted_without_answer_quality(collections):
    root, baseline, make, same = collections
    changed = dict(same, q2=stage("Other memory.", 20, source="row2"))
    report = full.compare(root, baseline, make("smaller", "fusion-x", changed))
    rows = {row["probe_id"]: row for row in report["records"]}
    assert report["totals"]["rendered_inputs_changed"] == report["totals"]["selected_sources_changed"] == 1
    assert rows["q2"]["selected_sources_added"] == ["row2"] and rows["q2"]["prompt_tokens"]["delta"] == -10
    assert rows["q1"]["source_rank_diagnostics"] is not None and rows["q2"]["source_rank_diagnostics"] is None
    assert all(r["answer_quality"] == r["semantic_support"] == "unmeasured" for r in rows.values())


@pytest.mark.parametrize("fault", ["code", "profile", "undeclared_arms"])
def test_mismatched_or_undeclared_candidate_is_refused(collections, fault):
    root, baseline, make, same = collections
    if fault == "code":
        candidate = make("drift", "fusion-x", same, code=[dict(path="src/reader.py", sha256="b", bytes=1)])
        match = "same producer/baseline/profile"
    elif fault == "profile":
        candidate = make("drift", "fusion-x", same, reader_adoption=dict(profile="other"))
        match = "same producer/baseline/profile"
    else:
        candidate = make("drift", "fusion-x", same, arms=list(matched.ARMS))
        match = "full-arm collection identity"
    with pytest.raises(ValueError, match=match):
        full.compare(root, baseline, candidate)


def test_leg_profile_rows_scale_one_leg_in_every_intent_profile():
    from src.api.config import settings
    rows = full.supplement_rows(settings)
    assert len(rows) == 8 and len({r["id"] for r in rows}) == 8
    base = settings.retrieval_leg_profiles
    for row in rows:
        _, leg, scale = row["id"].split("-")
        scale = float(scale[1:])
        scaled = row["overrides"]["retrieval_leg_profiles"]
        for intent, weights in base.items():
            for name, value in weights.items():
                assert scaled[intent][name] == (round(value * scale, 6) if name == leg else value)


def test_supplement_keeps_baseline_identity_and_applies_only_declared_rows(sealed):
    from src.api.config import settings
    root, plan, out, _, _ = sealed
    _, four, _, _ = runner.setup(root, plan, "baseline", out)
    row = full.supplement_rows(settings)[0]
    _, one, coordinate, _ = full.setup(root, plan, row["id"], out.parent / "leg", reader_adoption=None, supplement=row)
    assert all(one["identity"][k] == four["identity"][k] for k in matched.COMMON)
    assert one["identity"]["coordinate"] == coordinate == row
    assert one["identity"]["supplement"]["outside_sealed_plan"] is True
    before = settings.retrieval_leg_profiles
    with full.supplement_applied(settings, row):
        assert settings.retrieval_leg_profiles == row["overrides"]["retrieval_leg_profiles"]
    assert settings.retrieval_leg_profiles == before
    with pytest.raises(ValueError, match="declared row"):
        with full.supplement_applied(settings, dict(row, overrides={"retrieval_rrf_k": 1})): pass
    with pytest.raises(ValueError, match="declaration"):
        full.setup(root, plan, row["id"], out.parent / "bad", reader_adoption=None, supplement=dict(row, stage="x"))
