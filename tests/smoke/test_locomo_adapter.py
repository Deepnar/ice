"""FINAL's LoCoMo adapter: consecutive messages pair up with speaker names kept, photos become
text, turns are timed from the session's own date, evidence ids map to turns, category 5 is a
never-said question, token F1 is LoCoMo's, and only the pinned file is accepted."""

import json

import pytest

from experiments.locomo import adapter as A

SAMPLE = dict(sample_id="conv-x", qa=[
    dict(question="What is the cat called?", answer="Tofu", evidence=["D1:1"], category=4),
    dict(question="When did Bo run?", answer="2 June 2023", evidence=["D2:2; D2:1", "D9:9"], category=2),
    dict(question="What did Ann sell?", evidence=["D1:2"], category=5, adversarial_answer="a car")],
    conversation=dict(speaker_a="Ann", speaker_b="Bo",
                      session_1_date_time="1:56 pm on 8 May, 2023",
                      session_1=[dict(speaker="Ann", dia_id="D1:1", text="My cat is Tofu."),
                                 dict(speaker="Bo", dia_id="D1:2", text="Nice!", blip_caption="a cat on a sofa"),
                                 dict(speaker="Ann", dia_id="D1:3", text="Bye.")],
                      session_2_date_time="9:00 am on 2 June, 2023",
                      session_2=[dict(speaker="Bo", dia_id="D2:1", text="I ran 10k."),
                                 dict(speaker="Ann", dia_id="D2:2", text="Well done.")]))


def test_originals_pair_messages_and_keep_speakers_and_dates():
    originals, dia = A.to_originals(SAMPLE)
    assert [o["turn_number"] for o in originals] == [1, 2, 3]
    assert originals[0]["prompt"] == "Ann: My cat is Tofu." and originals[0]["response"] == "Bo: Nice! [shares a photo: a cat on a sofa]"
    assert originals[1]["prompt"] == "Ann: Bye." and originals[1]["response"] == ""          # unpaired last message
    assert originals[0]["timestamp"] == "2023-05-08T13:56:00Z" and originals[2]["timestamp"] == "2023-06-02T09:00:00Z"
    assert originals[0]["timestamp_provenance"] == "locomo_session_date"
    assert dia == {"D1:1": 1, "D1:2": 1, "D1:3": 2, "D2:1": 3, "D2:2": 3}


def test_questions_references_and_never_said():
    originals, dia = A.to_originals(SAMPLE)
    asks, occ, dropped = A.questions(SAMPLE, dia, 3)
    assert dropped == 1                                                      # D9:9 names no message
    assert asks[0]["id"] == "LOCOMO-conv-x:Q000@3#0" and asks[0]["checkpoint_turn"] == 3
    assert occ[1]["reference"]["citations"] == [dict(turn=3)] and occ[1]["type"] == "temporal"
    assert occ[2]["expected_abstention"] and occ[2]["reference"]["required"] == [] and occ[2]["type"] == "adversarial"


def test_token_f1_and_pinned_data(tmp_path):
    assert A.token_f1("The cat is Tofu", "Tofu") == pytest.approx(0.5)            # "cat is tofu": P 1/3, R 1
    assert A.token_f1("nothing", "Tofu") == 0.0
    bad = tmp_path / "locomo10.json"
    bad.write_text(json.dumps([SAMPLE]))
    with pytest.raises(ValueError):
        A.load(bad)


def test_export_writes_everything_the_runner_reads(tmp_path):
    report = A.export(tmp_path, data=[SAMPLE])
    d = tmp_path / "conv-x"
    assert report["conv-x"] == dict(turns=3, questions=3, evidence_ids_dropped=1)
    assert json.loads((d / "ledger.json").read_text()) == dict(entries=[])
    assert {p.name for p in d.iterdir()} == {"originals.json", "asks.json", "occurrences.json", "ledger.json"}
