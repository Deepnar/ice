"""No-memory controls (FINAL revision 5): the private histories' step is unchanged, repeats at one
checkpoint see what was already kept, and LoCoMo's grader leaves controls to the blind comparison."""
import json

from experiments.z2 import questions as Q


def fake_models(seen):
    def writer(name):
        def generate(messages, **_):
            seen.append((name, messages[0]["content"], json.loads(messages[1]["content"])))
            body = json.loads(messages[1]["content"])
            if "MESSAGE" in body:
                return type("R", (), dict(text=json.dumps(dict(needs_no_memory=True)), usage={}))()
            return type("R", (), dict(text=json.dumps(dict(prompt=f"msg {len(seen)}")), usage={}))()
        return generate
    return {r: (writer(r), dict(role=r)) for r in Q.WRITERS_CYCLE}


def originals(n):
    return [dict(turn_number=t, prompt=f"user {t}", response=f"bot {t}") for t in range(1, n + 1)]


def test_one_per_checkpoint_keeps_the_writer_prompt_unchanged(tmp_path):
    seen = []
    kept, _ = Q.controls_at("H", originals(30), tmp_path, [10, 20], models=fake_models(seen))
    writes = [(s, b) for _, s, b in seen if "KIND" in b]
    assert len(kept) == 2 and all(s == Q.CONTROL_SYSTEM and "ALREADY_WRITTEN" not in b for s, b in writes)


def test_repeats_at_one_checkpoint_see_the_controls_already_kept(tmp_path):
    seen = []
    kept, _ = Q.controls_at("L", originals(30), tmp_path, [30] * 3, models=fake_models(seen), avoid=True)
    writes = [(s, b) for _, s, b in seen if "KIND" in b]
    assert len(kept) == 3 and "ALREADY_WRITTEN" not in writes[0][1]
    assert writes[2][1]["ALREADY_WRITTEN"] == [k["prompt"] for k in kept[:2]]
    assert writes[2][0] == Q.CONTROL_SYSTEM + Q.CONTROL_AVOID
    assert [k["kind"] for k in kept] == list(Q.CONTROL_KINDS[:3])


def test_rerunning_the_merge_replaces_controls_never_duplicates():
    from experiments.final import no_memory_controls as C
    asks = [dict(id="q1", kind="question"), dict(id="old", kind="control")]
    kept = [dict(id="L:ctl000", checkpoint_turn=9, prompt="hi there", kind="small_talk")]
    once = C.merged(asks, kept)
    assert [a["id"] for a in once] == ["q1", "L:ctl000"] and C.merged(once, kept) == once
    assert once[1] == dict(id="L:ctl000", checkpoint_turn=9, question="hi there", kind="control", type="small_talk")


def test_the_memory_systems_search_the_controls_and_locomo_scoring_skips_them():
    import ast
    from pathlib import Path
    src = Path("experiments/final/baselines/run_baseline.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "questions")
    ns = {}
    exec(compile(ast.Module([fn], []), "q", "exec"), ns)
    asks = [dict(id="q", kind="question"), dict(id="c", kind="control"), dict(id="x", kind="other")]
    assert [a["id"] for a in ns["questions"](asks)] == ["q", "c"]
    assert 'unit["ask"].get("kind", "question") == "question"' in Path("experiments/locomo/grade.py").read_text()


def test_final_waits_for_the_controls_before_reading_or_searching():
    from experiments.final import run_final as F
    stages = {s.key: s for s in F.final_stages()}
    assert "prep-locomo-controls" in stages["local-baseline-search-amem-LOCOMO"].needs
    assert "prep-locomo-controls" in stages["local-read-LOCOMO"].needs
    assert stages["prep-controls-S1"].needs == ("prep-generate-S1",) and "prep-controls-S9" not in stages
    reads = [s for k, s in stages.items() if k.startswith("local-read-S") and "prep-controls-S4" in s.needs]
    assert reads


def _folder(root, condition, asks, texts):
    import json as j
    (root / "units" / condition).mkdir(parents=True, exist_ok=True)
    index, rows = {}, []
    for ask, text in zip(asks, texts):
        (root / "units" / condition / f"{ask['id']}.json").write_text(j.dumps(dict(condition=condition, ask=ask)))
        index[f"{condition}|{ask['id']}"] = f"{condition}-{ask['id']}"
        rows.append(dict(prompt_sha256=f"{condition}-{ask['id']}", finish="complete", text=text))
    (root / "index.json").write_text(j.dumps(dict(index=index)))
    (root / "answers.jsonl").write_text("".join(j.dumps(r) + "\n" for r in rows))


def test_a_memory_system_is_compared_with_ices_no_memory_reply(tmp_path):
    from experiments.locomo import grade as LG
    asks = [dict(id="c1", kind="control", question="capital of peru?"), dict(id="q1", kind="question", question="?")]
    _folder(tmp_path / "mem0", "mem0", asks, ["Lima, and you once said you liked Peru", "x"])
    _folder(tmp_path / "ice", "recent_only", asks[:1], ["Lima."])
    seen = []

    def ask(system, prompt):
        seen.append(json.loads(prompt))
        return dict(better="B" if json.loads(prompt)["A"].startswith("Lima,") else "A")
    LG.no_memory_harm(tmp_path / "mem0", tmp_path / "mem0", tmp_path / "out", ask,
                      no_memory_units=tmp_path / "ice", no_memory_answers=tmp_path / "ice")
    report = json.loads((tmp_path / "out" / "locomo-controls.json").read_text())
    assert len(seen) == 2 and all(s["MESSAGE"] == "capital of peru?" for s in seen)     # the control only, both orders
    assert report["by_condition"]["mem0"]["harm_rate"] == 1.0


def test_answers_take_recent_only_for_the_controls_only(tmp_path):
    from experiments.z2 import answers as A
    asks = [dict(id="c1", kind="control"), dict(id="q1", kind="question")]
    for cond in ("recent_only", "full_default"):
        (tmp_path / "units" / cond).mkdir(parents=True)
        for a in asks:
            (tmp_path / "units" / cond / f"{a['id']}.json").write_text(json.dumps(dict(
                condition=cond, ask=a, stage=dict(prompt_messages=[dict(role="user", content=cond + a["id"])]))))
    got = sorted((c, q) for c, q, _, _ in A.load_units(tmp_path, ["full_default", "recent_only:control"]))
    assert got == [("full_default", "c1"), ("full_default", "q1"), ("recent_only", "c1")]
