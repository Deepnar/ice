"""Supplemental grading evidence cannot rewrite answers or retrieval gold."""

import copy
import fcntl
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.v3_repair import prepare_scorer_context as adapter
from scripts.z1 import score_frozen_answers as scoring
from scripts.z1.metric_contract import metric_contract


def write(path, data):
    path.write_text(json.dumps(data))


def read(path):
    return json.loads(path.read_text())


def stamps(paths):
    return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}


@pytest.fixture
def packet(monkeypatch):
    with tempfile.TemporaryDirectory(prefix="v3-scorer-test-", dir=adapter.ROOT / "logs") as directory:
        root = Path(directory)
        producer = root / "producer"
        producer.mkdir()
        source_file = root / "development-originals.json"
        sources = [dict(conversation_id="sample01-long-id", turn_number=i,
                        prompt="Keep the old service port.", response="Recorded original reply.",
                        timestamp=f"2025-01-01T00:{i:02d}:00+00:00", timestamp_provenance="original")
                   for i in (1, 2)]
        sources[1]["prompt"] = "Later qualification: use the old port only for the test service."
        write(source_file, {"sample01": sources})
        review = dict(knowledge_scope="private_history", task_types=["episodic_lookup"],
                      reviewed_through_turn=60, recent_only_answerable=False,
                      question_family_id="controlled-family")
        probe = dict(event="as_of_probe", probe_id="question-1", conversation="sample01", split_turn=60,
                     question="Which service uses the old port?", expected_answer="The test service.",
                     gold_turns=[1], label_review=review, metric_contract=metric_contract(review, [1]),
                     question_time="2025-02-01T00:00:00+00:00", question_time_provenance="original",
                     development_repeat=dict(phase="after"),
                     gold_sources=[dict(turn=1, prompt=sources[0]["prompt"], response=sources[0]["response"],
                                        recorded_at=sources[0]["timestamp"], ts_provenance="original")])
        trace = producer / "seed.jsonl"
        trace.write_text(json.dumps(probe) + "\n")
        write(producer / "development-run.json", dict(identity=dict(originals_sha256=adapter.sha(source_file))))
        write(producer / "reviewed-keys.json", dict(records=[review]))
        # Isolate adapter source validation; actual complete-replay audit is separately
        # exercised against the retained producer, not replaced in the CLI.
        monkeypatch.setattr(adapter, "audit", lambda _root: (dict(trace_sha256=adapter.sha(trace)), None))
        scorer_sources = [dict(turn=s["turn_number"], user=s["prompt"], assistant=s["response"],
                               timestamp=s["timestamp"], timestamp_provenance="original",
                               role_sha256={role: hashlib.sha256(s[field].encode()).hexdigest()
                                            for role, field in (("user", "prompt"), ("assistant", "response"))},
                               source_role="original_gold" if i == 0 else "later_qualification_context",
                               inside_recent_turn_window=False) for i, s in enumerate(sources)]
        record = dict(probe_id=probe["probe_id"], conversation=probe["conversation"], cutoff=60,
                      question=probe["question"], expected_answer=probe["expected_answer"],
                      original_retrieval_gold_turns=[1], original_metric_contract=probe["metric_contract"],
                      scorer_sources=scorer_sources)
        bound = [source_file, trace, producer / "development-run.json", producer / "reviewed-keys.json"]
        data = dict(version="v3", kind="supplemental_scorer_context", scorer_only=True, runtime_input=False,
                    retrieval_gold_unchanged=True, records=[record],
                    source_bindings={p.relative_to(adapter.ROOT).as_posix(): dict(sha256=adapter.sha(p)) for p in bound})
        path = root / "scorer-context.json"
        write(path, data)
        arms = []
        for tag, answer in (("full", "The test service."), ("vector", "The production service.")):
            row = {k: copy.deepcopy(v) for k, v in probe.items()
                   if k not in {"event", "gold_sources"}}
            row.update(gold_turn_text=adapter.source_text(probe), gold_source_complete=True,
                       answer=answer, error=None, prompt_tokens=91, answer_model="controlled-answerer",
                       answer_input_messages=[dict(role="user", content=probe["question"])],
                       usage=dict(input_tokens=91, output_tokens=4))
            row["answer_record_sha256"] = adapter.digest(row)
            arm = root / f"{tag}.json"
            write(arm, dict(version="v3", tag=tag, processing_complete=True, records=[row]))
            arms.append(arm)
        yield SimpleNamespace(root=root, producer=producer, source_file=source_file, packet=path,
                              arms=arms, out=root / "projected", probe=probe, bound=bound)


def prepare(p):
    return adapter.prepare(p.producer, p.packet, p.arms, p.out)


def products(p):
    return sorted(p.out.glob("answers-*.json"))


def contract_ledger(p):
    source = read(p.packet)['records'][0]['scorer_sources'][1]['user']
    row = dict(probe_id=p.probe['probe_id'], kind='semantic_propositions',
        requirements=['Identify the test service; preserve the limitation.'],
        qualifications_and_optional_details=['Do not require the optional port number.'],
        original_retrieval_gold_turns=p.probe['gold_turns'],
        original_metric_contract=p.probe['metric_contract'], reviewed_through_turn=60,
        source_spans=[dict(turn=2, role='user', start=0, end=len(source), text=source,
                          source_sha256=hashlib.sha256(source.encode()).hexdigest())],
        observations={'full': 'Do not show this to the judge'},
        unpromoted_candidate_metric_contract={'lookup_rank_tuning_eligible': False})
    path = p.root / 'contracts.json'
    write(path, dict(version='v3', kind='editorial_required_content_ledger',
        promotion_eligible=False, score_of_record=False, retrieval_labels_changed=False,
        runtime_inputs_changed=False, records=[row],
        source_bindings={f.relative_to(adapter.ROOT).as_posix(): dict(sha256=adapter.sha(f))
                         for f in [p.packet, p.producer / 'seed.jsonl']}))
    return path


def test_contracts_reach_native_grader_without_rewriting_original_fields(packet):
    p = packet
    ledger = contract_ledger(p)
    old = stamps([*p.bound, p.packet, *p.arms, ledger])
    original, _ = scoring.collect(p.arms, {'judge': 'controlled'})
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    projected, _ = scoring.collect(products(p), {'judge': 'controlled'})
    for path in products(p):
        row = read(path)['records'][0]
        before = next(read(f)['records'][0] for f in p.arms if read(f)['tag'] == read(path)['tag'])
        for field in before.keys() - {'gold_turn_text', 'answer_record_sha256'}:
            assert row[field] == before[field]
        evidence = scoring.case(row)
        criteria = evidence['editorial_answer_contract']
        assert set(criteria) == {'policy', 'kind', 'qualification', 'requirements', 'qualifications'}
        assert 'Do not show this' not in json.dumps(evidence)
        assert 'lookup_rank_tuning_eligible' not in json.dumps(evidence)
        for kind, order, candidates in [('absolute', None, {'answer': row['answer']}),
                ('preference', 'forward', {'A': 'One', 'B': 'Two'}),
                ('preference', 'reverse', {'A': 'Two', 'B': 'One'})]:
            key, unit = scoring.task(kind, evidence, candidates, {'judge': 'controlled'}, order=order)
            assert json.loads(unit['messages'][1]['content'])['editorial_answer_contract'] == criteria
            assert 'separate from source evidence' in unit['messages'][0]['content']
            altered = copy.deepcopy(evidence)
            altered['editorial_answer_contract']['requirements'] = ['A different required meaning.']
            assert scoring.task(kind, altered, candidates, {'judge': 'controlled'}, order=order)[0] != key
    assert {r['grade_key'] for rows in original.values() for r in rows.values()}.isdisjoint(
        {r['grade_key'] for rows in projected.values() for r in rows.values()})
    saved = stamps(list(p.out.glob('*.json')))
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    assert stamps(saved) == saved and stamps(old) == old


@pytest.mark.parametrize('fault', ['promoted', 'missing', 'duplicate', 'cutoff', 'gold', 'metric',
    'role', 'body', 'offset', 'hash', 'future', 'trace_binding', 'packet_binding', 'requirements', 'qualification'])
def test_bad_contract_refuses_without_outputs(packet, fault):
    p = packet
    path = contract_ledger(p)
    data = read(path)
    row = data['records'][0]
    span = row['source_spans'][0]
    if fault == 'promoted': data['promotion_eligible'] = True
    elif fault == 'missing': data['records'] = []
    elif fault == 'duplicate': data['records'].append(row)
    elif fault == 'cutoff': row['reviewed_through_turn'] = 59
    elif fault == 'gold': row['original_retrieval_gold_turns'] = [1, 2]
    elif fault == 'metric': row['original_metric_contract'] = {}
    elif fault == 'role': span['role'] = 'system'
    elif fault == 'body': span['text'] = 'Invented evidence'
    elif fault == 'offset': span['start'] = -1
    elif fault == 'hash': span['source_sha256'] = 'wrong'
    elif fault == 'future': span['turn'] = 61
    elif fault == 'trace_binding': del data['source_bindings'][(p.producer/'seed.jsonl').relative_to(adapter.ROOT).as_posix()]
    elif fault == 'packet_binding': del data['source_bindings'][p.packet.relative_to(adapter.ROOT).as_posix()]
    elif fault == 'requirements': row['requirements'] = []
    elif fault == 'qualification': row['qualifications_and_optional_details'] = [False]
    write(path, data)
    with pytest.raises(ValueError):
        adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=path)
    assert not p.out.exists()


def test_changed_contract_refuses_resume_and_different_arm_criteria_refuse(packet):
    p = packet
    path = contract_ledger(p)
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=path)
    saved = stamps(list(p.out.glob('*.json')))
    data = read(path)
    data['records'][0]['requirements'] = ['Changed required meaning.']
    write(path, data)
    with pytest.raises(ValueError, match='changed'):
        adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=path)
    assert stamps(saved) == saved
    arm = products(p)[0]
    data = read(arm)
    data['records'][0]['editorial_answer_contract']['requirements'] = ['Different between arms.']
    write(arm, data)
    with pytest.raises(ValueError, match='different occurrences or evidence'):
        scoring.collect(products(p), {'judge': 'controlled'})


def test_uncontracted_grader_payload_is_unchanged(packet):
    evidence = scoring.case(read(packet.arms[0])['records'][0])
    assert 'editorial_answer_contract' not in evidence
    _, unit = scoring.task('absolute', evidence, {'answer': 'A'}, {})
    assert unit['messages'][0]['content'] == scoring.ABSOLUTE
    assert unit['messages'][1]['content'] == json.dumps(
        {**evidence, 'answer': 'A'}, ensure_ascii=False, sort_keys=True)


def test_exact_contract_uses_native_alternatives_and_rejects_unproven_value(packet):
    p = packet
    path = contract_ledger(p)
    data = read(path)
    row = data['records'][0]
    row['kind'] = 'exact_literal'
    span = row['source_spans'][0]
    alternative = {k: v for k, v in span.items() if k != 'text'}
    alternative['literal'] = span['text']
    row['requirements'] = [dict(id='exact-controlled-value', kind='exact_literal', alternatives=[alternative])]
    write(path, data)
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=path)
    projected = read(products(p)[0])['records'][0]
    contract = scoring.case(projected)['editorial_answer_contract']
    assert contract['kind'] == 'exact_literal'
    assert json.dumps([span['text']], ensure_ascii=False) in contract['requirements'][0]
    alternative['literal'] = 'Invented value without source proof'
    write(path, data)
    with pytest.raises(ValueError, match='lacks its validated source proof'):
        adapter.prepare(p.producer, p.packet, p.arms, p.root/'other-projection', contracts=path)


def test_contract_interruption_reuses_outputs_and_refuses_late_binding_drift(packet, monkeypatch):
    p = packet
    ledger = contract_ledger(p)
    native = adapter.write_or_verify
    def stop_after_plan(path, value):
        if path.name.startswith('answers-'):
            raise KeyboardInterrupt
        return native(path, value)
    monkeypatch.setattr(adapter, 'write_or_verify', stop_after_plan)
    with pytest.raises(KeyboardInterrupt):
        adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    assert (p.out/'scorer-context-plan.json').exists() and not products(p)
    monkeypatch.setattr(adapter, 'write_or_verify', native)
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    saved = stamps(list(p.out.glob('*.json')))
    # The original producer source is included transitively in the contract plan.
    p.source_file.write_text(p.source_file.read_text() + ' ')
    with pytest.raises(ValueError, match='binding changed'):
        adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    assert stamps(saved) == saved


@pytest.mark.parametrize('change', ['extra_field', 'wrong_kind', 'empty_requirement', 'false_qualification'])
def test_native_grader_rejects_unqualified_contract_payload(packet, change):
    p = packet
    ledger = contract_ledger(p)
    adapter.prepare(p.producer, p.packet, p.arms, p.out, contracts=ledger)
    row = read(products(p)[0])['records'][0]
    contract = row['editorial_answer_contract']
    if change == 'extra_field': contract['arm'] = 'full'
    elif change == 'wrong_kind': contract['kind'] = 'source_count'
    elif change == 'empty_requirement': contract['requirements'] = ['']
    elif change == 'false_qualification': contract['qualification'] = 'independent human truth'
    with pytest.raises(ValueError, match='invalid editorial'):
        scoring.case(row)


def test_preserves_answers_messages_gold_labels_and_terminal_resume(packet):
    p = packet
    old = stamps([*p.bound, p.packet, *p.arms])
    manifest = prepare(p)
    by_tag = {read(path)["tag"]: read(path) for path in p.arms}
    for path in products(p):
        projected = read(path)
        original = by_tag[projected["tag"]]["records"][0]
        row = projected["records"][0]
        for field in original.keys() - {"gold_turn_text", "answer_record_sha256"}:
            assert row[field] == original[field], field
        assert "Later qualification" in row["gold_turn_text"]
        assert "not an all-required retrieval target" in row["gold_turn_text"]
        assert row["scorer_context_projection"]["original_answer_record_sha256"] == original["answer_record_sha256"]
        assert row["answer_record_sha256"] == adapter.digest({k: v for k, v in row.items() if k != "answer_record_sha256"})
    saved = stamps(list(p.out.glob("*.json")))
    assert prepare(p) == manifest
    assert stamps(saved) == saved
    assert stamps(old) == old
    assert manifest["model_calls"] == 0 and not manifest["score_of_record"]


@pytest.mark.parametrize("fault", ["runtime", "reference", "future", "duplicate_source", "duplicate_case",
                                    "missing_case", "missing_gold", "body", "hash", "clock", "age", "role",
                                    "gold", "metric", "original_binding", "producer_binding"])
def test_bad_packet_refuses_before_outputs(packet, fault):
    p = packet
    data = read(p.packet)
    row = data["records"][0]
    source = row["scorer_sources"][1]
    if fault == "runtime": data["runtime_input"] = True
    elif fault == "reference": row["expected_answer"] = "Changed reference."
    elif fault == "future": source["turn"] = 61
    elif fault == "duplicate_source": row["scorer_sources"].append(source)
    elif fault == "duplicate_case": data["records"].append(row)
    elif fault == "missing_case": data["records"] = []
    elif fault == "missing_gold": row["scorer_sources"] = [source]
    elif fault == "body": source["user"] += " Invented text."
    elif fault == "hash": source["role_sha256"]["assistant"] = "wrong"
    elif fault == "clock": source["timestamp_provenance"] = "synthetic"
    elif fault == "age": source["inside_recent_turn_window"] = True
    elif fault == "role": source["source_role"] = "original_gold"
    elif fault == "gold": row["original_retrieval_gold_turns"] = [1, 2]
    elif fault == "metric": row["original_metric_contract"]["lookup_rank_tuning_eligible"] = False
    elif fault == "original_binding": p.source_file.write_text(p.source_file.read_text() + " ")
    elif fault == "producer_binding": del data["source_bindings"][(p.producer / "reviewed-keys.json").relative_to(adapter.ROOT).as_posix()]
    write(p.packet, data)
    with pytest.raises(ValueError): prepare(p)
    assert not p.out.exists()


@pytest.mark.parametrize("field", ["expected_answer", "question_time", "question_time_provenance", "development_repeat",
                                    "gold_turn_text", "label_review", "answer_record_sha256"])
def test_stale_answer_metadata_refuses_before_outputs(packet, field):
    p = packet
    data = read(p.arms[0])
    data["records"][0][field] = "Changed value"
    write(p.arms[0], data)
    with pytest.raises((ValueError, AttributeError)): prepare(p)
    assert not p.out.exists()


def test_partial_projection_recovers_after_interruption(packet, monkeypatch):
    p = packet
    native = adapter.write_or_verify
    calls = []
    def interrupted(path, value):
        calls.append(path)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return native(path, value)
    monkeypatch.setattr(adapter, "write_or_verify", interrupted)
    with pytest.raises(KeyboardInterrupt): prepare(p)
    assert len(products(p)) == 1 and (p.out / "scorer-context-plan.json").exists()
    assert not (p.out / "scorer-context-complete.json").exists()
    saved = stamps(products(p))
    monkeypatch.setattr(adapter, "write_or_verify", native)
    prepare(p)
    assert len(products(p)) == 2 and stamps(saved) == saved


def test_partial_plan_refuses_changed_unwritten_answer_arm(packet, monkeypatch):
    p = packet
    native = adapter.write_or_verify
    def interrupted(path, value):
        if path.name.startswith("answers-"):
            raise KeyboardInterrupt
        return native(path, value)
    monkeypatch.setattr(adapter, "write_or_verify", interrupted)
    with pytest.raises(KeyboardInterrupt): prepare(p)
    assert not products(p)
    saved = stamps([p.out / "scorer-context-plan.json"])
    data = read(p.arms[1])
    row = data["records"][0]
    row["answer"] = "A revised saved answer."
    row.pop("answer_record_sha256")
    row["answer_record_sha256"] = adapter.digest(row)
    write(p.arms[1], data)
    monkeypatch.setattr(adapter, "write_or_verify", native)
    with pytest.raises(ValueError, match="changed"): prepare(p)
    assert not products(p) and stamps(saved) == saved


def test_changed_packet_refuses_completed_reuse_without_writes(packet):
    p = packet
    prepare(p)
    saved = stamps(list(p.out.glob("*.json")))
    data = read(p.packet)
    data["records"][0]["scorer_sources"][1]["source_role"] = "optional_reference_context"
    write(p.packet, data)
    with pytest.raises(ValueError, match="changed"): prepare(p)
    assert stamps(saved) == saved


def test_original_paths_and_concurrent_preparation_refuse(packet):
    p = packet
    for target in (p.producer, p.producer / "derived", p.root):
        with pytest.raises(ValueError, match="separate"): adapter.prepare(p.producer, p.packet, p.arms, target)
    p.out.mkdir()
    with (p.out / ".scorer-context.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError): prepare(p)
    assert not products(p)


def test_source_drift_during_preparation_is_caught(packet, monkeypatch):
    p = packet
    native = adapter.render
    def changed(record):
        result = native(record)
        p.source_file.write_text(p.source_file.read_text() + " ")
        return result
    monkeypatch.setattr(adapter, "render", changed)
    with pytest.raises(ValueError, match="binding changed"): prepare(p)
    assert not products(p)


def test_failed_answer_is_retained_ungraded_and_pending_answer_refuses(packet):
    p = packet
    data = read(p.arms[0])
    row = data["records"][0]
    row.update(answer="", error="incomplete_completion", error_type="CloudCompletionError",
               cloud_attempts=2, fault_disposition="degraded_final")
    row.pop("answer_record_sha256")
    row["answer_record_sha256"] = adapter.digest(row)
    write(p.arms[0], data)
    prepare(p)
    arms, _ = scoring.collect(products(p), {"controlled": True})
    assert not next(iter(arms["full"].values()))["available"]
    data["processing_complete"] = False
    write(p.arms[0], data)
    with pytest.raises(ValueError, match="terminally processed"): prepare(p)


def test_native_absolute_keys_and_both_preference_orders_use_new_evidence(packet):
    p = packet
    identity = {"judge": "controlled-identity"}
    old_arms, _ = scoring.collect(p.arms, identity)
    old_tasks, _ = scoring.work(old_arms, identity, "absolute", p.root / "old-cache")
    prepare(p)
    arms, _ = scoring.collect(products(p), identity)
    tasks, _ = scoring.work(arms, identity, "absolute", p.root / "new-cache")
    assert len(tasks) == 2 and not set(old_tasks) & set(tasks)
    cache = p.root / "new-cache" / "cache"
    cache.mkdir(parents=True)
    for key, unit in tasks.items():
        evidence = json.loads(unit["messages"][1]["content"])
        assert "Later qualification" in evidence["source"]
        assert not {"tag", "arm", "prompt_tokens", "answer_model"} & evidence.keys()
        # Authored receipts exercise the native cache/preference constructor;
        # these are not model verdicts or a judge-qualification result.
        write(cache / f"{key}.json", dict(key=key, task=unit, attempts=[dict(attempt=1, raw=dict(
            value=dict(grade="correct", reason="Authored control.", basis="source"),
            provider_completion=dict(completion_status="completed")))]))
    preferences, pairs = scoring.work(arms, identity, "preferences", cache.parent)
    assert len(preferences) == 2 and len(pairs) == 1
    payloads = {unit["order"]: json.loads(unit["messages"][1]["content"]) for unit in preferences.values()}
    assert payloads[0]["answer_A"] == payloads[1]["answer_B"]
    assert payloads[0]["answer_B"] == payloads[1]["answer_A"]
    assert payloads[0]["source"] == payloads[1]["source"]
    first_arm = next(iter(arms.values()))
    assert payloads[0]["source"] == next(iter(first_arm.values()))["evidence"]["source"]
