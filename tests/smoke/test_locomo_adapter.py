"""FINAL's LoCoMo adapter: speakers on fixed sides with names kept, photos become text, turns are
timed from the session's own date, evidence ids map to turns, category 5 is excluded (as memory
papers do), and only the pinned file is accepted."""

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


def test_speaker_a_is_always_the_user_side_even_when_speaker_b_opens():
    sample = dict(SAMPLE, conversation=dict(SAMPLE["conversation"], session_2=[
        dict(speaker="Bo", dia_id="D2:1", text="I ran 10k."), dict(speaker="Ann", dia_id="D2:2", text="Well done.")]))
    originals, dia = A.to_originals(sample)
    opened_by_bo = originals[2]
    assert opened_by_bo["prompt"] == "" and opened_by_bo["response"] == "Bo: I ran 10k."
    assert originals[3]["prompt"] == "Ann: Well done." and originals[3]["response"] == ""
    assert all(o["prompt"].startswith("Ann") or not o["prompt"] for o in originals)
    assert all(o["response"].startswith("Bo") or not o["response"] for o in originals)


def test_originals_pair_messages_and_keep_speakers_and_dates():
    originals, dia = A.to_originals(SAMPLE)
    assert [o["turn_number"] for o in originals] == [1, 2, 3, 4]
    assert originals[0]["prompt"] == "Ann: My cat is Tofu." and originals[0]["response"] == "Bo: Nice! [shares a photo: a cat on a sofa]"
    assert originals[1]["prompt"] == "Ann: Bye." and originals[1]["response"] == ""          # unpaired last message
    assert originals[0]["timestamp"] == "2023-05-08T13:56:00Z" and originals[2]["timestamp"] == "2023-06-02T09:00:00Z"
    assert originals[2]["prompt"] == "" and originals[2]["response"] == "Bo: I ran 10k."      # Bo opens session 2
    assert originals[0]["timestamp_provenance"] == "locomo_session_date"
    assert dia == {"D1:1": 1, "D1:2": 1, "D1:3": 2, "D2:1": 3, "D2:2": 4}


def test_questions_references_and_category_5_excluded():
    originals, dia = A.to_originals(SAMPLE)
    asks, occ, dropped = A.questions(SAMPLE, dia, 4)
    assert len(asks) == len(occ) == 2 and all(o["type"] != "adversarial" for o in occ)
    assert dropped == 1                                                      # D9:9 names no message
    assert asks[0]["id"] == "LOCOMO-conv-x:Q000@4#0" and asks[0]["checkpoint_turn"] == 4
    assert occ[1]["reference"]["citations"] == [dict(turn=3), dict(turn=4)] and occ[1]["type"] == "temporal"


def test_only_the_pinned_file_is_accepted(tmp_path):
    bad = tmp_path / "locomo10.json"
    bad.write_text(json.dumps([SAMPLE]))
    with pytest.raises(ValueError):
        A.load(bad)


def test_export_writes_everything_the_runner_reads(tmp_path):
    report = A.export(tmp_path, data=[SAMPLE])
    d = tmp_path / "conv-x"
    assert report["conv-x"] == dict(turns=4, questions=2, evidence_ids_dropped=1)
    assert json.loads((d / "ledger.json").read_text()) == dict(entries=[])
    assert {p.name for p in d.iterdir()} == {"originals.json", "asks.json", "occurrences.json", "ledger.json"}
