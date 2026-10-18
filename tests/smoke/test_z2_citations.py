"""Z2 citation check (spec Z2_EVALUATION §5 item 2): planted wrong offset, wrong
turn, future turn and paraphrased span all refuse; a true citation passes."""

from experiments.z2 import ledger

ORIGINALS = [
    dict(turn_number=1, prompt="my sister Asha lives in Pune now", response="Noted: Asha is in Pune."),
    dict(turn_number=2, prompt="actually she moved to Goa last week", response="Got it, Goa then."),
    dict(turn_number=3, prompt="what's a good name for a cat", response="How about Miso?"),
]


def cite(turn, role, quote):
    citation = ledger.resolve(ORIGINALS, turn, role, quote)
    assert citation is not None
    return citation


def test_true_citation_passes_and_resolves_offsets():
    c = cite(2, "user", "she moved to Goa")
    assert ORIGINALS[1]["prompt"][c["start"]:c["end"]] == "she moved to Goa"
    assert c["occurrences"] == 1
    assert ledger.problem(ORIGINALS, c, checkpoint=2) is None


def test_planted_wrong_offset_refuses():
    c = dict(cite(1, "user", "Asha lives in Pune"))
    c["start"] += 1
    c["end"] += 1
    assert ledger.problem(ORIGINALS, c, checkpoint=3) == "span_mismatch"


def test_planted_wrong_turn_refuses():
    c = dict(cite(2, "user", "she moved to Goa"), turn=1)
    assert ledger.problem(ORIGINALS, c, checkpoint=3) == "source_hash"


def test_planted_wrong_role_refuses():
    c = dict(cite(1, "user", "Asha"), role="assistant")
    assert ledger.problem(ORIGINALS, c, checkpoint=3) == "source_hash"


def test_planted_future_turn_refuses():
    c = cite(2, "user", "she moved to Goa")
    assert ledger.problem(ORIGINALS, c, checkpoint=1) == "future_turn"


def test_planted_paraphrase_refuses():
    assert ledger.resolve(ORIGINALS, 2, "user", "she relocated to Goa") is None
    c = dict(cite(2, "user", "she moved to Goa"), quote="she relocated to Goa")
    assert ledger.problem(ORIGINALS, c, checkpoint=2) == "span_mismatch"


def test_whitespace_drift_is_located_but_original_bytes_are_stored():
    originals = [dict(turn_number=1, prompt="plan:\n- GRE\tJune\n- IELTS later", response="ok")]
    c = ledger.resolve(originals, 1, "user", "- GRE June - IELTS")
    assert c["quote"] == "- GRE\tJune\n- IELTS"                 # the source's bytes, not the writer's
    assert originals[0]["prompt"][c["start"]:c["end"]] == c["quote"]
    assert ledger.problem(originals, c, checkpoint=1) is None
    forged = dict(c, quote="- GRE June - IELTS")                  # a stored writer text would fail
    assert ledger.problem(originals, forged, checkpoint=1) == "span_mismatch"


def test_case_spelling_and_stitching_drift_is_a_paraphrase():
    assert ledger.resolve(ORIGINALS, 1, "user", "My sister Asha") is None
    assert ledger.resolve(ORIGINALS, 1, "user", "my sistr Asha") is None
    assert ledger.resolve(ORIGINALS, 1, "user", "my sister lives in Pune") is None


def test_check_removes_whole_item_and_never_repairs():
    good = dict(id="a", checkpoint=3, citations=[cite(1, "user", "Asha lives in Pune")])
    mixed = dict(id="b", checkpoint=3, citations=[cite(2, "user", "Goa"),
                                                 dict(cite(3, "user", "cat"), start=0, end=3)])
    uncited = dict(id="c", checkpoint=3, citations=[])
    kept, removed = ledger.check(ORIGINALS, [good, mixed, uncited], lambda i: i["checkpoint"])
    assert [k["id"] for k in kept] == ["a"]
    assert {r["item"]["id"]: r["failures"] for r in removed} == {
        "b": [(1, "span_mismatch")], "c": [(0, "uncited")]}
    assert mixed["citations"][1]["start"] == 0      # the input is not edited


def test_malformed_and_out_of_range_refuse():
    assert ledger.problem(ORIGINALS, {"turn": 1}, checkpoint=3) == "malformed"
    c = dict(cite(1, "user", "Asha"), turn=9)
    assert ledger.problem(ORIGINALS, c, checkpoint=9) == "no_such_source"
    c = dict(cite(1, "user", "Asha"), end=10_000)
    assert ledger.problem(ORIGINALS, c, checkpoint=3) == "offsets"
