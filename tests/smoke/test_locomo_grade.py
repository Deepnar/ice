"""FINAL's LoCoMo grading by Mem0's protocol: J's prompt verbatim with its placeholders, F1 on
word sets, BLEU-1 by NLTK, labels parsed strictly, unlabelled replies left for the next run."""

import json

import pytest

from experiments.locomo import grade as G
from experiments.z2 import read_passes as R


def test_mem0_metrics_and_prompt():
    assert "{question}" in G.J_PROMPT and "{gold_answer}" in G.J_PROMPT and "generous" in G.J_PROMPT
    assert G.f1("The cat is Tofu.", "Tofu") == pytest.approx(2 * 0.25 * 1 / 1.25)        # set of 4 words vs 1
    assert G.f1("", "Tofu") == 0.0
    assert G.bleu1("tofu", "Tofu") == pytest.approx(1.0) and 0 < G.bleu1("the cat is tofu", "tofu") < 1
    assert G.label('{"label": "CORRECT"}') == 1 and G.label('```json\n{"label": "WRONG"}\n```') == 0
    assert G.label("I am not sure") is None


def test_run_grades_answered_contexts_and_waits_on_unlabelled(tmp_path):
    occ = [dict(family="LOCOMO-c:Q000", checkpoint_turn=9, repeat_index=0, type="single_hop", question="Cat?",
                reference=dict(answer="Tofu"))]
    (tmp_path / "occ.json").write_text(json.dumps(occ))
    for cond in ("full_default", "bare_vector"):
        R.write_unit(R.unit_path(tmp_path / "units", cond, "LOCOMO-c:Q000@9#0"),
                     dict(condition=cond, ask=dict(id="LOCOMO-c:Q000@9#0"), stage=dict(prompt_messages=[])))
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "index.json").write_text(json.dumps(dict(index={"full_default|LOCOMO-c:Q000@9#0": "a",
                                                               "bare_vector|LOCOMO-c:Q000@9#0": "b"})))
    (answers / "answers.jsonl").write_text(json.dumps(dict(prompt_sha256="a", finish="complete", text="Tofu")) + "\n"
                                           + json.dumps(dict(prompt_sha256="b", finish="complete", text="No idea")) + "\n")
    replies = iter(['{"label": "CORRECT"}', "hmm"])
    pending = G.run(tmp_path / "units", answers, tmp_path / "occ.json", tmp_path / "out", lambda p: next(replies),
                    say=lambda *_: None)
    assert pending == 1
    rep = json.loads((tmp_path / "out" / "locomo-report.json").read_text())["by_condition"]
    # folder order: bare_vector first — the judge's CORRECT is recorded with Mem0's F1 and BLEU-1 of
    # "No idea" vs "Tofu"; full_default's unlabelled reply is left for the next run
    assert rep == {"bare_vector": {"all": dict(n=1, j=1, f1=0.0, bleu1=0.0), "single_hop": dict(n=1, j=1, f1=0.0, bleu1=0.0)}}


def test_a_cut_or_unlabelled_reply_is_retried_then_unjudged_never_a_crash(tmp_path):
    """Z3 judging run 1 (2026-11-03): a reply cut by its output cap raised and stopped a whole check."""
    from experiments.lme.cloud_provider import CloudCompletionError
    occ = [dict(family="LOCOMO-c:Q000", checkpoint_turn=9, repeat_index=0, type="single_hop", question="Cat?",
                reference=dict(answer="Tofu"))]
    (tmp_path / "occ.json").write_text(json.dumps(occ))
    R.write_unit(R.unit_path(tmp_path / "units", "full_default", "LOCOMO-c:Q000@9#0"),
                 dict(condition="full_default", ask=dict(id="LOCOMO-c:Q000@9#0"), stage=dict(prompt_messages=[])))
    answers = tmp_path / "answers"
    answers.mkdir()
    (answers / "index.json").write_text(json.dumps(dict(index={"full_default|LOCOMO-c:Q000@9#0": "a"})))
    (answers / "answers.jsonl").write_text(json.dumps(dict(prompt_sha256="a", finish="complete", text="Tofu")) + "\n")

    def cut(prompt):
        raise CloudCompletionError("cloud answer completion was not successful: incomplete")
    runs = [G.run(tmp_path / "units", answers, tmp_path / "occ.json", tmp_path / "out", cut, say=lambda *_: None)
            for _ in range(3)]
    assert runs == [1, 1, 0]
    rows = [json.loads(l) for l in (tmp_path / "out/locomo-grades.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "unjudged" and "incomplete" in rows[0]["reason"]
    assert G.OUTPUT_TOKENS == 16_000
